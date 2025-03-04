#!/usr/bin/env python
"""
Script to build GraphRAG artifacts from documents.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kg_rag.utils.document_loader import load_documents
from kg_rag.methods.graphrag_based.indexer import GraphRAGIndexer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build GraphRAG artifacts from documents"
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
        help="Directory to save artifacts"
    )
    parser.add_argument(
        "--artifacts-name",
        type=str,
        default="graphrag_artifacts",
        help="Name of the artifacts file"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Directory for caching"
    )
    parser.add_argument(
        "--vector-store-dir",
        type=str,
        default="vector_stores",
        help="Directory for vector stores"
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
        "--llm-model",
        type=str,
        default="gpt-4o",
        help="LLM model to use"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model to use"
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