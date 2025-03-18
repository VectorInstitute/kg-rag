#!/usr/bin/env python
"""
Script to build a vector store for baseline RAG approaches.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kg_rag.methods.baseline_rag.embedder import OpenAIEmbedding
from kg_rag.methods.baseline_rag.document_processor import DocumentProcessor
from kg_rag.methods.baseline_rag.vector_store import ChromaDBManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a vector store for baseline RAG approaches"
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        required=True,
        help="Directory containing the documents"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="sec_10q",
        help="Name of the ChromaDB collection"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="chroma_db",
        help="Directory to store ChromaDB files"
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
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model to use"
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
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)
    
    # Initialize components
    embedder = OpenAIEmbedding(
        api_key=openai_api_key,
        model=args.embedding_model,
        verbose=args.verbose
    )
    
    processor = DocumentProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        verbose=args.verbose
    )
    
    db_manager = ChromaDBManager(
        collection_name=args.collection_name,
        persist_directory=args.persist_dir,
        verbose=args.verbose
    )
    
    # Process documents and build vector store
    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        print(f"Error: Documents directory '{docs_dir}' does not exist.")
        sys.exit(1)
        
    print(f"Processing documents from {docs_dir}...")
    chunks = processor.process_directory(docs_dir)
    
    print(f"Adding {len(chunks)} chunks to vector store...")
    db_manager.add_documents(chunks, embedder)
    
    # Print statistics
    stats = db_manager.get_collection_stats()
    print("\nVector Store Statistics:")
    print(f"  Collection: {stats['collection_name']}")
    print(f"  Documents: {stats['document_count']}")
    
    print(f"\nVector store built successfully at {args.persist_dir}")


if __name__ == "__main__":
    main()