#!/usr/bin/env python
"""
Script to build an entity-based knowledge graph from documents.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kg_rag.utils.document_loader import load_documents
from kg_rag.utils.graph_utils import create_graph_from_documents, save_graph, get_graph_statistics, save_graph_documents
from langchain_openai import ChatOpenAI


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths for saving data
    graph_path = output_dir / f"{args.graph_name}.pkl"
    graph_docs_path = output_dir / f"{args.graph_name}_documents.pkl"
    
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
    
    # Create graph
    print("Creating graph from documents...")
    graph, graph_documents = create_graph_from_documents(
        documents=documents,
        llm=llm,
        graph_documents_path=str(graph_docs_path) if not args.force_rebuild else None,
        force_rebuild=args.force_rebuild
    )
    
    # Save graph and documents
    save_graph(graph, str(graph_path))
    save_graph_documents(graph_documents, str(graph_docs_path))
    
    # Print graph statistics
    stats = get_graph_statistics(graph)
    print("\nGraph statistics:")
    print(f"  Nodes: {stats['node_count']}")
    print(f"  Edges: {stats['edge_count']}")
    print(f"  Relation types: {stats['relation_type_count']}")
    print(f"  Connected components: {stats['connected_component_count']}")
    print(f"  Largest component size: {stats['largest_component_size']}")
    print(f"  Average degree: {stats['average_degree']:.2f}")
    print(f"  Max degree: {stats['max_degree']}")
    
    # Print some example relation types
    print("\nExample relation types:")
    for rel_type in list(stats['relation_types'])[:10]:
        print(f"  {rel_type}")
    
    print(f"\nGraph built successfully and saved to {graph_path}")
    print(f"Graph documents saved to {graph_docs_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build an entity-based knowledge graph from documents"
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
        help="Directory to save the graph to"
    )
    parser.add_argument(
        "--graph-name",
        type=str,
        default="entity_graph",
        help="Name of the graph file"
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


if __name__ == "__main__":
    main()