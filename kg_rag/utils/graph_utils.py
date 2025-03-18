"""
Common graph utilities for KG-RAG approaches.
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
from tqdm.auto import tqdm
from langchain_openai import ChatOpenAI
# from langchain_experimental.graph_transformers import LLMGraphTransformer
from kg_rag.utils.graph_transformers import MetadataEnhancedLLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph


def save_graph(graph: nx.DiGraph, file_path: str) -> None:
    """
    Save a NetworkX graph to a pickle file.
    
    Args:
        graph: The graph to save
        file_path: Path to save the graph to
    """
    with open(file_path, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Graph saved to {file_path}")


def load_graph(file_path: str) -> nx.DiGraph:
    """
    Load a NetworkX graph from a pickle file.
    
    Args:
        file_path: Path to the graph pickle file
        
    Returns:
        Loaded NetworkX graph
    """
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)
    print(f"Graph loaded from {file_path}")
    return graph


def save_graph_documents(graph_documents: List[Any], file_path: str) -> None:
    """
    Save graph documents to a pickle file.
    
    Args:
        graph_documents: The graph documents to save
        file_path: Path to save the graph documents to
    """
    with open(file_path, 'wb') as f:
        pickle.dump(graph_documents, f)
    print(f"Graph documents saved to {file_path}")


def load_graph_documents(file_path: str) -> List[Any]:
    """
    Load graph documents from a pickle file.
    
    Args:
        file_path: Path to the graph documents pickle file
        
    Returns:
        Loaded graph documents
    """
    with open(file_path, 'rb') as f:
        graph_documents = pickle.load(f)
    print(f"Graph documents loaded from {file_path}")
    return graph_documents


def create_graph_from_documents(
    documents: List[Document],
    llm: Optional[ChatOpenAI] = None,
    graph_documents_path: Optional[str] = None,
    force_rebuild: bool = False
) -> Tuple[nx.DiGraph, List[Any]]:
    """
    Create a knowledge graph from documents.
    
    Args:
        documents: List of documents to create the graph from
        llm: LLM to use for graph creation (default: ChatOpenAI with gpt-4o)
        graph_documents_path: Optional path to save/load graph documents
        force_rebuild: Whether to force rebuilding the graph documents
        
    Returns:
        Tuple of (NetworkX graph, graph documents)
    """
    # Setup LLM if not provided
    if llm is None:
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4o"
        )
    
    # Create LLM transformer
    llm_transformer = MetadataEnhancedLLMGraphTransformer(llm)
    
    # Check if graph documents can be loaded
    if graph_documents_path and os.path.exists(graph_documents_path) and not force_rebuild:
        graph_documents = load_graph_documents(graph_documents_path)
    else:
        # Convert documents to graph documents
        print("Converting documents to graph documents...")
        graph_documents = llm_transformer.convert_to_graph_documents(tqdm(documents))
        
        # Save graph documents if path is provided
        if graph_documents_path:
            save_graph_documents(graph_documents, graph_documents_path)
    
    # Create the graph
    graph = NetworkxEntityGraph()
    
    # Add nodes to the graph
    for doc in graph_documents:
        for node in doc.nodes:
            graph.add_node(node.id)

    # Add edges to the graph
    for doc in graph_documents:
        for edge in doc.relationships:
            graph._graph.add_edge(
                edge.source.id,
                edge.target.id,
                relation=edge.type,
            )
    
    return graph._graph, graph_documents


def create_graph_from_graph_documents(
    graph_documents: List[Any]
) -> nx.DiGraph:
    """
    Create a NetworkX knowledge graph from graph documents.
    
    Args:
        graph_documents: List of graph documents to create the graph from
        
    Returns:
        NetworkX graph
    """
    graph = nx.DiGraph()
    
    # Add nodes to the graph
    for doc in graph_documents:
        for node in doc.nodes:
            graph.add_node(node.id)

    # Add edges to the graph
    for doc in graph_documents:
        for edge in doc.relationships:
            graph.add_edge(
                edge.source.id,
                edge.target.id,
                relation=edge.type,
            )
    
    return graph


def get_graph_statistics(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Get statistics about a graph.
    
    Args:
        graph: The graph to analyze
        
    Returns:
        Dictionary of graph statistics
    """
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    
    # Get relation types
    relation_types = set()
    for _, _, data in graph.edges(data=True):
        relation = data.get('relation', '')
        if relation:
            relation_types.add(relation)
    
    # Get connected components
    if not nx.is_directed(graph):
        connected_components = list(nx.connected_components(graph))
        largest_component_size = max(len(c) for c in connected_components) if connected_components else 0
    else:
        connected_components = list(nx.weakly_connected_components(graph))
        largest_component_size = max(len(c) for c in connected_components) if connected_components else 0
    
    # Get degree statistics
    degrees = [d for _, d in graph.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0
    
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "relation_type_count": len(relation_types),
        "relation_types": list(relation_types),
        "connected_component_count": len(connected_components),
        "largest_component_size": largest_component_size,
        "average_degree": avg_degree,
        "max_degree": max_degree
    }


def visualize_graph_subset(
    graph: nx.DiGraph,
    start_node: str,
    max_depth: int = 2,
    max_nodes: int = 50
) -> nx.DiGraph:
    """
    Extract a visualization-friendly subset of a graph.
    
    Args:
        graph: The full graph
        start_node: The node to start from
        max_depth: Maximum depth to traverse
        max_nodes: Maximum number of nodes to include
        
    Returns:
        A subgraph suitable for visualization
    """
    subgraph = nx.DiGraph()
    nodes_to_explore = [(start_node, 0)]  # (node, depth)
    visited = set()
    
    while nodes_to_explore and len(subgraph) < max_nodes:
        current_node, current_depth = nodes_to_explore.pop(0)
        
        if current_node in visited:
            continue
            
        visited.add(current_node)
        subgraph.add_node(current_node)
        
        if current_depth < max_depth:
            # Add neighbors
            for neighbor in graph.neighbors(current_node):
                if len(subgraph) < max_nodes:
                    edge_data = graph.get_edge_data(current_node, neighbor)
                    relation = edge_data.get('relation', '')
                    
                    subgraph.add_node(neighbor)
                    subgraph.add_edge(current_node, neighbor, relation=relation)
                    
                    if neighbor not in visited:
                        nodes_to_explore.append((neighbor, current_depth + 1))
    
    return subgraph