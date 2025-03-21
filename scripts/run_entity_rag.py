#!/usr/bin/env python
"""Script to run the entity-based KG-RAG system."""

import argparse
import json
import os
import sys


# # get env variables
# from dotenv import load_dotenv
# # print current working directory
# load_dotenv("../")

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI

from kg_rag.methods.entity_based.kg_rag import EntityBasedKGRAG
from kg_rag.utils.document_loader import load_documents, load_graph_documents
from kg_rag.utils.graph_utils import create_graph_from_graph_documents


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the simplified KG-RAG system")
    parser.add_argument(
        "--graph-documents-pkl-path",
        type=str,
        default="data/sec-10-q/graphs/graph_documents.pkl",
        help="Path to the graph documents pickle file",
    )
    parser.add_argument(
        "--documents-pkl-path",
        type=str,
        default="data/sec-10-q/graphs/documents.pkl",
        help="Path to the chunked documents pickle file",
    )
    parser.add_argument(
        "--documents-path",
        type=str,
        default="data/sec-10-q/docs",
        help="Path to the document files",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query to run (if not provided, will use interactive mode)",
    )
    parser.add_argument(
        "--top-k-nodes", type=int, default=10, help="Number of top nodes to retrieve"
    )
    parser.add_argument(
        "--top-k-chunks",
        type=int,
        default=5,
        help="Number of top chunks to include in context",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Minimum similarity score for node matching",
    )
    parser.add_argument(
        "--node-freq-weight",
        type=float,
        default=0.4,
        help="Weight for node frequency in chunk scoring (0.0 to 1.0)",
    )
    parser.add_argument(
        "--node-sim-weight",
        type=float,
        default=0.6,
        help="Weight for node similarity in chunk scoring (0.0 to 1.0)",
    )
    parser.add_argument(
        "--use-cot", action="store_true", help="Use Chain-of-Thought prompting"
    )
    parser.add_argument(
        "--numerical-answer",
        action="store_true",
        help="Format answers as numerical values only",
    )
    parser.add_argument(
        "--output-file", type=str, default=None, help="Optional file to save results to"
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    return parser.parse_args()


def main():
    """Run the entity-based KG-RAG system."""
    args = parse_args()

    # Load the documents
    print(f"Loading documents from {args.documents_pkl_path}")
    documents = load_documents(
        directory_path=args.documents_path,
        pickle_path=args.documents_pkl_path,
    )

    # Load the graph
    print(
        f"Converting graph documents from {args.graph_documents_pkl_path} to graph..."
    )
    graph_documents = load_graph_documents(args.graph_documents_pkl_path)

    graph = create_graph_from_graph_documents(graph_documents)

    # Initialize LLM
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

    # Create KG-RAG system
    print("Initializing KG-RAG system...")
    kg_rag = EntityBasedKGRAG(
        graph=graph,
        graph_documents=graph_documents,
        document_chunks=documents,
        llm=llm,
        top_k_nodes=args.top_k_nodes,
        top_k_chunks=args.top_k_chunks,
        similarity_threshold=args.similarity_threshold,
        node_freq_weight=0.4,
        node_sim_weight=0.6,
        use_cot=args.use_cot,
        numerical_answer=args.numerical_answer,
        verbose=args.verbose,
    )

    # Run in interactive mode or process a single query
    if args.query is None:
        run_interactive(kg_rag, args.output_file)
    else:
        result = process_query(kg_rag, args.query, args.output_file)
        print_result(result, use_cot=args.use_cot)


def run_interactive(kg_rag, output_file=None):
    """Run in interactive mode."""
    print("\nEntity-based KG-RAG Interactive Mode")
    print("Enter 'exit' or 'quit' to end the session")

    results = []

    while True:
        query = input("\nEnter your query: ")

        if query.lower() in ["exit", "quit"]:
            break

        result = process_query(kg_rag, query, None)  # Don't save individual results
        print_result(result)

        results.append({"query": query, "result": result})

    # Save all results if output file is specified
    if output_file and results:
        save_results(results, output_file)
        print(f"All results saved to {output_file}")


def process_query(kg_rag, query, output_file=None):
    """Process a single query."""
    result = kg_rag.query(query)

    # Save result if output file is specified
    if output_file:
        save_results([{"query": query, "result": result}], output_file)
        print(f"Result saved to {output_file}")

    return result


def print_result(result, use_cot=False):
    """Print the result in a formatted way."""
    print("\nResult:")
    if use_cot:
        print(f"Answer: {result.get('answer', 'N/A')}")
        print("\nReasoning:")
        print(result.get("reasoning", "No reasoning provided"))
    else:
        print(result)


def save_results(results, output_file):
    """Save results to a file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
