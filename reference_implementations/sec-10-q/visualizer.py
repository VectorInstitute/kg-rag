import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from neo4j import GraphDatabase

class GraphVisualizer:
    """Utility class for visualizing Neo4j graphs."""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """Initialize visualizer.
        
        Args:
            uri: Neo4j database URI
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        
    def get_graph_statistics(self) -> Tuple[int, int, List[Tuple[str, int]]]:
        """Get basic statistics about the graph.
        
        Returns:
            Tuple of (node_count, relationship_count, label_counts)
        """
        with self.driver.session(database=self.database) as session:
            # Get total counts
            result = session.run("""
                MATCH (n)
                WITH count(n) as nodes
                MATCH ()-[r]->()
                RETURN nodes, count(r) as rels
            """)
            record = result.single()
            node_count = record["nodes"]
            rel_count = record["rels"]
            
            # Get label counts
            result = session.run("""
                MATCH (n)
                WITH labels(n) as labels
                UNWIND labels as label
                WITH label, count(*) as count
                RETURN label, count
                ORDER BY count DESC
            """)
            label_counts = [(record["label"], record["count"]) for record in result]
            
        return node_count, rel_count, label_counts

    def visualize_subgraph(self, 
                          limit: int = 100,
                          layout: str = "spring",
                          figsize: Tuple[int, int] = (12, 8),
                          node_size: int = 1000,
                          with_labels: bool = True):
        """Visualize a subgraph of the database.
        
        Args:
            limit: Maximum number of nodes to visualize
            layout: NetworkX layout algorithm to use
            figsize: Figure size (width, height)
            node_size: Size of nodes in visualization
            with_labels: Whether to show node labels
        """
        # Get subgraph data
        with self.driver.session(database=self.database) as session:
            result = session.run(f"""
                MATCH (n)-[r]->(m)
                WITH * LIMIT {limit}
                RETURN 
                    id(n) as source_id,
                    labels(n) as source_labels,
                    id(m) as target_id,
                    labels(m) as target_labels,
                    type(r) as relationship_type
            """)
            
            # Create NetworkX graph
            G = nx.DiGraph()
            
            for record in result:
                source = f"{record['source_id']}\n{record['source_labels'][0]}"
                target = f"{record['target_id']}\n{record['target_labels'][0]}"
                rel_type = record['relationship_type']
                
                G.add_edge(source, target, relationship=rel_type)
        
        # Visualize
        plt.figure(figsize=figsize)
        
        if layout == "spring":
            pos = nx.spring_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.random_layout(G)
            
        nx.draw(G, pos,
               with_labels=with_labels,
               node_color='lightblue',
               node_size=node_size,
               arrowsize=20,
               font_size=8)
        
        # Add relationship labels
        edge_labels = nx.get_edge_attributes(G, 'relationship')
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
        
        plt.title(f"Neo4j Graph Visualization (showing {G.number_of_nodes()} nodes)")
        plt.axis('off')
        plt.show()

    def close(self):
        """Close database connection."""
        self.driver.close()