#!/usr/bin/env python
"""
Script to run the entity-based KG-RAG system.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# get env variables
from dotenv import load_dotenv
load_dotenv()

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kg_rag.utils.graph_utils import load_graph
from kg_rag.methods.entity_based.kg_rag import EntityBasedKGRAG
from langchain_openai import ChatOpenAI


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the entity-based KG-RAG system"
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        required=True,
        help="Path to the graph pickle file"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query to run (if not provided, will use interactive mode)"
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=10,
        help="Width of the beam for search"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Maximum depth of the search"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top results to return"
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=2,
        help="Number of chains to extract"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.7,
        help="Minimum score for considering a match"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional file to save results to"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Load the graph
    print(f"Loading graph from {args.graph_path}...")
    graph = load_graph(args.graph_path)
    
    # Initialize LLM
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o"
    )
    llm = llm.bind(response_format={"type": "json_object"})
    
    # Create KG-RAG system
    print("Initializing KG-RAG system...")
    kg_rag = EntityBasedKGRAG(
        graph=graph,
        llm=llm,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        top_k=args.top_k,
        num_chains=args.num_chains,
        min_score=args.min_score,
        chunk_scoring_method = 'weighted_frequency',
        expand_context = True,
        chunk_expansion_size = 1,
        use_cot = True,
        numerical_answer = True,
        verbose=args.verbose
    )
    
    # Run in interactive mode or process a single query
    if args.query is None:
        run_interactive(kg_rag, args.output_file)
    else:
        result = process_query(kg_rag, args.query, args.output_file)
        print_result(result)


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
        
        results.append({
            "query": query,
            "result": result
        })
    
    # Save all results if output file is specified
    if output_file and results:
        save_results(results, output_file)
        print(f"All results saved to {output_file}")


def process_query(kg_rag, query, output_file=None):
    """Process a single query."""
    print(f"Processing query: {query}")
    
    result = kg_rag.query(query)
    
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
    print(result.get('reasoning', 'No reasoning provided'))


def save_results(results, output_file):
    """Save results to a file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()