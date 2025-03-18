#!/usr/bin/env python
"""
Script to run the Cypher-based KG-RAG system.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph

from kg_rag.methods.cypher_based.kg_rag import CypherBasedKGRAG


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Cypher-based KG-RAG system"
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
        "--query",
        type=str,
        default=None,
        help="Query to run (if not provided, will use interactive mode)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum depth for graph exploration"
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=3,
        help="Maximum number of hops for graph exploration"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional file to save results to"
    )
    parser.add_argument(
        "--cypher-mode",
        action="store_true",
        help="Run in Cypher mode (show Cypher queries)"
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
    
    # Set Neo4j password from environment variable if not provided
    neo4j_password = args.neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
    
    # Connect to Neo4j
    print(f"Connecting to Neo4j at {args.neo4j_uri}...")
    graph = Neo4jGraph(
        url=args.neo4j_uri,
        username=args.neo4j_user,
        password=neo4j_password
    )
    
    # Initialize LLMs
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o"
    )
    
    # Create KG-RAG system
    print("Initializing KG-RAG system...")
    kg_rag = CypherBasedKGRAG(
        graph=graph,
        llm=llm,
        max_depth=args.max_depth,
        max_hops=args.max_hops,
        verbose=args.verbose
    )
    
    # Run in interactive mode or process a single query
    if args.query is None:
        run_interactive(kg_rag, args.cypher_mode, args.output_file)
    else:
        if args.cypher_mode:
            result = process_cypher_query(kg_rag, args.query, args.output_file)
            print_cypher_result(result)
        else:
            result = process_query(kg_rag, args.query, args.output_file)
            print_result(result)


def run_interactive(kg_rag, cypher_mode=False, output_file=None):
    """Run in interactive mode."""
    mode_name = "Cypher" if cypher_mode else "QA"
    print(f"\nCypher-based KG-RAG Interactive Mode ({mode_name})")
    print("Enter 'exit' or 'quit' to end the session")
    print("Enter 'mode' to toggle between QA and Cypher modes")
    
    results = []
    current_mode = "cypher" if cypher_mode else "qa"
    
    while True:
        mode_prefix = "[Cypher]" if current_mode == "cypher" else "[QA]"
        query = input(f"\n{mode_prefix} Enter your query: ")
        
        if query.lower() in ["exit", "quit"]:
            break
        
        if query.lower() == "mode":
            current_mode = "qa" if current_mode == "cypher" else "cypher"
            print(f"Switched to {current_mode.upper()} mode")
            continue
        
        if current_mode == "cypher":
            result = process_cypher_query(kg_rag, query, None)  # Don't save individual results
            print_cypher_result(result)
        else:
            result = process_query(kg_rag, query, None)  # Don't save individual results
            print_result(result)
        
        results.append({
            "mode": current_mode,
            "query": query,
            "result": result
        })
    
    # Save all results if output file is specified
    if output_file and results:
        save_results(results, output_file)
        print(f"All results saved to {output_file}")


def process_query(kg_rag, query, output_file=None):
    """Process a single query in QA mode."""
    print(f"Processing query: {query}")
    
    result = kg_rag.query(query)
    
    # Save result if output file is specified
    if output_file:
        save_results([{"mode": "qa", "query": query, "result": result}], output_file)
        print(f"Result saved to {output_file}")
    
    return result


def process_cypher_query(kg_rag, query, output_file=None):
    """Process a single query in Cypher mode."""
    print(f"Generating Cypher for query: {query}")
    
    cypher = kg_rag.get_explicit_cypher(query)
    print(f"Generated Cypher: {cypher}")
    
    print("Executing Cypher query...")
    result = kg_rag.run_cypher(cypher)
    
    # Save result if output file is specified
    if output_file:
        save_results([{"mode": "cypher", "query": query, "cypher": cypher, "result": result}], output_file)
        print(f"Result saved to {output_file}")
    
    return {"cypher": cypher, "result": result}


def print_result(result):
    """Print the result in a formatted way."""
    print("\nResult:")
    print(f"Answer: {result.get('answer', 'N/A')}")
    print("\nReasoning:")
    print(result.get('reasoning', 'No reasoning provided'))


def print_cypher_result(result):
    """Print the Cypher result in a formatted way."""
    print("\nCypher Query:")
    print(result.get('cypher', 'No Cypher query generated'))
    
    print("\nResult:")
    raw_result = result.get('result', [])
    
    if isinstance(raw_result, list) and raw_result:
        if len(raw_result) == 1:
            # Single row result
            print(json.dumps(raw_result[0], indent=2))
        else:
            # Multiple row result
            print(f"Found {len(raw_result)} results:")
            for i, row in enumerate(raw_result[:10]):  # Show at most 10 results
                print(f"Result {i+1}:")
                print(json.dumps(row, indent=2))
            
            if len(raw_result) > 10:
                print(f"...and {len(raw_result) - 10} more results")
    else:
        print("No results returned")


def save_results(results, output_file):
    """Save results to a file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()