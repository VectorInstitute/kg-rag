#!/usr/bin/env python
"""
Script to build a Cypher-based knowledge graph from documents.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kg_rag.utils.document_loader import load_documents
from kg_rag.methods.cypher_based.graph_constructor import CypherGraphConstructor
from langchain_openai import ChatOpenAI


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a Cypher-based knowledge graph from documents"
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        required=True,
        help="Directory containing the documents"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save graph documents"
    )
    parser.add_argument(
        "--graph-docs-name",
        type=str,
        default="cypher_graph_documents",
        help="Name of the graph documents file"
    )
    parser.add_argument(
        "--file-filter",
        type=str,
        default=None,
        help="Optional string to filter filenames"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Size of document chunks"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=24,
        help="Overlap between document chunks"
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://localhost:7687",
        help="URI for Neo4j connection"
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default="neo4j",
        help="Username for Neo4j connection"
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default=None,
        help="Password for Neo4j connection (if not provided, uses environment variable)"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding the graph documents"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    # Load environment variables
    load_dotenv()
    
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Path for saving graph documents
    graph_docs_path = output_dir / f"{args.graph_docs_name}.pkl"
    
    # Set Neo4j password from environment variable if not provided
    neo4j_password = args.neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
    
    # Load documents
    print(f"Loading documents from {args.docs_dir}...")
    documents = load_documents(
        directory_path=args.docs_dir,
        file_filter=args.file_filter,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    print(f"Loaded {len(documents)} document chunks")
    
    # Initialize LLM
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o"
    )
    
    # Create graph constructor
    graph_constructor = CypherGraphConstructor(
        llm=llm,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=neo4j_password,
        verbose=args.verbose
    )
    
    # Process documents and build graph
    print("Processing documents and building Neo4j graph...")
    neo4j_graph = graph_constructor.process_documents(
        documents=documents,
        graph_documents_path=str(graph_docs_path),
        force_rebuild=args.force_rebuild,
        clear_existing=True
    )
    
    # Print graph information
    result = neo4j_graph.query("MATCH (n) RETURN count(n) as nodeCount")
    node_count = result[0]["nodeCount"] if result else 0
    
    result = neo4j_graph.query("MATCH ()-[r]->() RETURN count(r) as relCount")
    rel_count = result[0]["relCount"] if result else 0
    
    result = neo4j_graph.query("MATCH (n) RETURN DISTINCT labels(n) as labels")
    label_count = len(result) if result else 0
    
    result = neo4j_graph.query("MATCH ()-[r]->() RETURN DISTINCT type(r) as types")
    rel_type_count = len(result) if result else 0
    
    print("\nNeo4j Graph Information:")
    print(f"  Nodes: {node_count}")
    print(f"  Relationships: {rel_count}")
    print(f"  Node labels: {label_count}")
    print(f"  Relationship types: {rel_type_count}")
    
    # Print some example relationship types
    if rel_type_count > 0:
        result = neo4j_graph.query(
            "MATCH ()-[r]->() RETURN DISTINCT type(r) as type LIMIT 10"
        )
        print("\nExample relationship types:")
        for row in result:
            print(f"  {row['type']}")
    
    print(f"\nGraph built successfully")
    print(f"Graph documents saved to {graph_docs_path}")


if __name__ == "__main__":
    main()