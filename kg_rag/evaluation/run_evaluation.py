#!/usr/bin/env python
"""
Script to run evaluation across different KG-RAG methods.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kg_rag.utils.graph_utils import load_graph
from kg_rag.utils.evaluator import Evaluator

# Import RAG systems
from kg_rag.methods.entity_based.kg_rag import EntityBasedKGRAG
from kg_rag.methods.baseline_rag.kg_rag import BaselineRAG
from kg_rag.methods.cypher_based.kg_rag import CypherBasedKGRAG
from kg_rag.methods.graphrag_based.kg_rag import GraphRAGBasedKGRAG, create_graphrag_system


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate KG-RAG methods"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to evaluation dataset CSV"
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        help="Path to the graph pickle file (required for graph-based methods)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["entity", "cypher", "graphrag", "baseline", "all"],
        default="all",
        help="KG-RAG method to evaluate"
    )
    parser.add_argument(
        "--use-cot",
        action="store_true",
        help="Use Chain-of-Thought prompting"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--question-col",
        type=str,
        default="New Question",
        help="Column name for questions in the dataset"
    )
    parser.add_argument(
        "--answer-col",
        type=str,
        default="New Answer",
        help="Column name for answers in the dataset"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="sec_10q",
        help="Name of the ChromaDB collection (for baseline methods)"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="chroma_db",
        help="Directory with ChromaDB files (for baseline methods)"
    )
    parser.add_argument(
        "--graphrag-artifacts",
        type=str,
        default=None,
        help="Path to GraphRAG artifacts (for GraphRAG-based method)"
    )
    parser.add_argument(
        "--vector-store-dir",
        type=str,
        default="vector_stores",
        help="Directory with vector stores (for GraphRAG-based method)"
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://localhost:7687",
        help="URI for Neo4j connection (for Cypher-based method)"
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default="neo4j",
        help="Username for Neo4j connection (for Cypher-based method)"
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default=None,
        help="Password for Neo4j connection (for Cypher-based method)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from a JSON file."""
    if not config_path:
        return {}
        
    with open(config_path, 'r') as f:
        return json.load(f)


def create_entity_rag(graph, config, use_cot=False, verbose=False):
    """Create an entity-based KG-RAG system."""
    llm = ChatOpenAI(
        temperature=0,
        model_name=config.get("model_name", "gpt-4o")
    )
    llm = llm.bind(response_format={"type": "json_object"})
    
    rag_system = EntityBasedKGRAG(
        graph=graph,
        llm=llm,
        beam_width=config.get("beam_width", 10),
        max_depth=config.get("max_depth", 8),
        top_k=config.get("top_k", 50),
        num_chains=config.get("num_chains", 2),
        min_score=config.get("min_score", 0.7),
        use_cot=use_cot,
        verbose=verbose
    )
    
    return rag_system


def create_baseline_rag(config, use_cot=False, verbose=False):
    """Create a standard baseline RAG system."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    rag_system = BaselineRAG(
        openai_api_key=openai_api_key,
        collection_name=config.get("collection_name", "sec_10q"),
        chroma_persist_dir=config.get("persist_dir", "chroma_db"),
        model_name=config.get("model_name", "gpt-4o"),
        embedding_model=config.get("embedding_model", "text-embedding-3-small"),
        top_k=config.get("top_k", 5),
        use_cot=use_cot,
        verbose=verbose
    )
    
    return rag_system


def create_cypher_rag(graph, config, use_cot=False, verbose=False):
    """Create a Cypher-based KG-RAG system."""
    from langchain_neo4j import Neo4jGraph
    
    # Create Neo4j connection
    neo4j_password = config.get("neo4j_password", os.getenv("NEO4J_PASSWORD", "password"))
    neo4j_uri = config.get("neo4j_uri", "bolt://localhost:7687")
    neo4j_user = config.get("neo4j_user", "neo4j")
    
    neo4j_graph = Neo4jGraph(
        url=neo4j_uri,
        username=neo4j_user,
        password=neo4j_password
    )
    
    # Create LLM
    llm = ChatOpenAI(
        temperature=0,
        model_name=config.get("model_name", "gpt-4o")
    )
    
    # Create Cypher-based KG-RAG
    rag_system = CypherBasedKGRAG(
        graph=neo4j_graph,
        llm=llm,
        max_depth=config.get("max_depth", 2),
        max_hops=config.get("max_hops", 3),
        use_cot=use_cot,
        verbose=verbose
    )
    
    return rag_system


def create_graphrag_rag(config, use_cot=False, verbose=False):
    """Create a GraphRAG-based KG-RAG system."""
    # Get configuration
    artifacts_path = config.get("artifacts_path")
    if not artifacts_path:
        raise ValueError("artifacts_path is required for GraphRAG-based method")
    
    vector_store_dir = config.get("vector_store_dir", "vector_stores")
    llm_model = config.get("model_name", "gpt-4o")
    search_strategy = config.get("search_strategy", "local")
    community_level = config.get("community_level", 2)
    
    # Create GraphRAG-based KG-RAG
    rag_system = create_graphrag_system(
        artifacts_path=artifacts_path,
        vector_store_dir=vector_store_dir,
        llm_model=llm_model,
        search_strategy=search_strategy,
        community_level=community_level,
        verbose=verbose
    )
    
    # Update use_cot setting
    rag_system.use_cot = use_cot
    
    return rag_system


def main():
    """Main entry point for the script."""
    # Load environment variables
    load_dotenv()
    
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config_path) or {}
    
    # Add command line arguments to config
    if args.collection_name:
        config.setdefault("baseline", {})["collection_name"] = args.collection_name
    
    if args.persist_dir:
        config.setdefault("baseline", {})["persist_dir"] = args.persist_dir
    
    if args.neo4j_uri:
        config.setdefault("cypher", {})["neo4j_uri"] = args.neo4j_uri
    
    if args.neo4j_user:
        config.setdefault("cypher", {})["neo4j_user"] = args.neo4j_user
    
    if args.neo4j_password:
        config.setdefault("cypher", {})["neo4j_password"] = args.neo4j_password
    
    if args.graphrag_artifacts:
        config.setdefault("graphrag", {})["artifacts_path"] = args.graphrag_artifacts
    
    if args.vector_store_dir:
        config.setdefault("graphrag", {})["vector_store_dir"] = args.vector_store_dir
    
    # Load graph for graph-based methods if needed
    graph = None
    if args.method in ["entity", "all"]:
        if not args.graph_path:
            print("Error: --graph-path is required for entity-based method")
            sys.exit(1)
        print(f"Loading graph from {args.graph_path}...")
        graph = load_graph(args.graph_path)
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    
    # Determine which methods to evaluate
    methods = []
    if args.method == "all":
        methods = ["entity", "baseline", "cypher", "graphrag"]
    else:
        methods = [args.method]
    
    # Filter out methods missing required components
    if "graphrag" in methods and not args.graphrag_artifacts:
        print("Warning: GraphRAG-based method requires --graphrag-artifacts")
        methods.remove("graphrag")
    
    # Suffix for CoT evaluation
    cot_suffix = "-cot" if args.use_cot else ""
    
    results = {}
    
    # Evaluate each method
    for method in methods:
        print(f"\nEvaluating {method}-based KG-RAG{cot_suffix}...")
        
        try:
            # Create RAG system
            if method == "entity":
                rag_system = create_entity_rag(
                    graph, 
                    config.get("entity", {}), 
                    args.use_cot,
                    args.verbose
                )
            elif method == "baseline":
                rag_system = create_baseline_rag(
                    config.get("baseline", {}), 
                    args.use_cot,
                    args.verbose
                )
            elif method == "cypher":
                rag_system = create_cypher_rag(
                    graph, 
                    config.get("cypher", {}), 
                    args.use_cot,
                    args.verbose
                )
            elif method == "graphrag":
                rag_system = create_graphrag_rag(
                    config.get("graphrag", {}), 
                    args.use_cot,
                    args.verbose
                )
            
            # Create evaluator
            method_name = f"{method}{cot_suffix}"
            evaluator = Evaluator(
                rag_system=rag_system,
                output_dir=output_dir,
                experiment_name=f"{method_name}_rag",
                verbose=args.verbose
            )
            
            # Run evaluation
            method_results = evaluator.evaluate(
                data_path=df,
                question_col=args.question_col,
                answer_col=args.answer_col,
                max_samples=args.max_samples
            )
            
            results[method_name] = method_results
            
        except NotImplementedError as e:
            print(f"Skipping {method}-based KG-RAG: {str(e)}")
        except Exception as e:
            print(f"Error evaluating {method}-based KG-RAG: {str(e)}")
    
    # Save combined results
    combined_results_path = output_dir / f"combined_results{cot_suffix}.json"
    with open(combined_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nEvaluation Results Summary:")
    for method, result in results.items():
        print(f"{method} KG-RAG: {result['accuracy']:.2%} accuracy")
    
    print(f"\nDetailed results saved to {output_dir}")


if __name__ == "__main__":
    main()