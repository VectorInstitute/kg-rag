#!/usr/bin/env python
"""Script to run the standard baseline RAG system."""

import argparse
import json
import os
import sys

from dotenv import load_dotenv


# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kg_rag.methods.baseline_rag.kg_rag import BaselineRAG


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the standard baseline RAG system")
    parser.add_argument(
        "--collection-name",
        type=str,
        default="sec_10q",
        help="Name of the ChromaDB collection",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="chroma_db",
        help="Directory with ChromaDB files",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="OpenAI model to use for generation"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model to use",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top results to retrieve"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query to run (if not provided, will use interactive mode)",
    )
    parser.add_argument(
        "--output-file", type=str, default=None, help="Optional file to save results to"
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    return parser.parse_args()


def main():
    """Run the standard baseline RAG system."""
    # Load environment variables
    load_dotenv()

    args = parse_args()

    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    # Initialize the baseline RAG system
    rag_system = BaselineRAG(
        collection_name=args.collection_name,
        chroma_persist_dir=args.persist_dir,
        model_name=args.model,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        verbose=args.verbose,
    )

    # Run in interactive mode or process a single query
    if args.query is None:
        run_interactive(rag_system, args.output_file)
    else:
        result = process_query(rag_system, args.query, args.output_file)
        print_result(result)


def run_interactive(rag_system, output_file=None):
    """Run in interactive mode."""
    print("\nStandard Baseline RAG Interactive Mode")
    print("Enter 'exit' or 'quit' to end the session")

    results = []

    while True:
        query = input("\nEnter your query: ")

        if query.lower() in ["exit", "quit"]:
            break

        result = process_query(rag_system, query, None)  # Don't save individual results
        print_result(result)

        results.append({"query": query, "result": result})

    # Save all results if output file is specified
    if output_file and results:
        save_results(results, output_file)
        print(f"All results saved to {output_file}")


def process_query(rag_system, query, output_file=None):
    """Process a single query."""
    print(f"Processing query: {query}")

    result = rag_system.query(query)

    # Save result if output file is specified
    if output_file:
        save_results([{"query": query, "result": result}], output_file)
        print(f"Result saved to {output_file}")

    return result


def print_result(result):
    """Print the result in a formatted way."""
    print("\nResult:")
    print(f"Answer: {result.get('answer', 'N/A')}")
    print("\nReasoning:")
    print(result.get("reasoning", "No reasoning provided"))


def save_results(results, output_file):
    """Save results to a file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
