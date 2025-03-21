#!/usr/bin/env python
"""Run hyperparameter search for KG-RAG methods."""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI


# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kg_rag.methods.baseline_rag.kg_rag import BaselineRAG
from kg_rag.methods.cypher_based.kg_rag import CypherBasedKGRAG

# Import RAG systems
from kg_rag.methods.entity_based.kg_rag import EntityBasedKGRAG
from kg_rag.methods.graphrag_based.kg_rag import (
    create_graphrag_system,
)
from kg_rag.utils.evaluator import run_hyperparameter_search
from kg_rag.utils.graph_utils import load_graph


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run hyperparameter search for KG-RAG methods"
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to evaluation dataset CSV"
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        help="Path to the graph pickle file (required for graph-based methods)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["entity", "cypher", "graphrag", "baseline"],
        default="entity",
        help="KG-RAG method to tune",
    )
    parser.add_argument(
        "--use-cot", action="store_true", help="Use Chain-of-Thought prompting"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hyperparameter_search",
        help="Directory to save results",
    )
    parser.add_argument(
        "--configs-path",
        type=str,
        required=True,
        help="Path to hyperparameter configurations JSON file",
    )
    parser.add_argument(
        "--question-col",
        type=str,
        default="New Question",
        help="Column name for questions in the dataset",
    )
    parser.add_argument(
        "--answer-col",
        type=str,
        default="New Answer",
        help="Column name for answers in the dataset",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="sec_10q",
        help="Name of the ChromaDB collection (for baseline methods)",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="chroma_db",
        help="Directory with ChromaDB files (for baseline methods)",
    )
    parser.add_argument(
        "--graphrag-artifacts",
        type=str,
        default=None,
        help="Path to GraphRAG artifacts (for GraphRAG-based method)",
    )
    parser.add_argument(
        "--vector-store-dir",
        type=str,
        default="vector_stores",
        help="Directory with vector stores (for GraphRAG-based method)",
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://localhost:7687",
        help="URI for Neo4j connection (for Cypher-based method)",
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default="neo4j",
        help="Username for Neo4j connection (for Cypher-based method)",
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default=None,
        help="Password for Neo4j connection (for Cypher-based method)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    return parser.parse_args()


def load_configs(configs_path):
    """Load hyperparameter configurations from a JSON file."""
    with open(configs_path) as f:
        return json.load(f)


def create_entity_rag_factory(graph, use_cot=False):
    """Create a factory function for entity-based KG-RAG."""

    def factory(**kwargs):
        llm = ChatOpenAI(temperature=0, model_name=kwargs.pop("model_name", "gpt-4o"))
        llm = llm.bind(response_format={"type": "json_object"})  # type: ignore

        return EntityBasedKGRAG(graph=graph, llm=llm, use_cot=use_cot, **kwargs)

    return factory


def create_baseline_rag_factory(collection_name, persist_dir, use_cot=False):
    """Create a factory function for standard baseline RAG."""

    def factory(**kwargs):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        return BaselineRAG(
            collection_name=collection_name,
            chroma_persist_dir=persist_dir,
            use_cot=use_cot,
            **kwargs,
        )

    return factory


def create_cypher_rag_factory(neo4j_uri, neo4j_user, neo4j_password, use_cot=False):
    """Create a factory function for Cypher-based KG-RAG."""

    def factory(**kwargs):
        # Create Neo4j connection
        neo4j_graph = Neo4jGraph(
            url=neo4j_uri, username=neo4j_user, password=neo4j_password
        )

        # Create LLM
        llm = ChatOpenAI(temperature=0, model_name=kwargs.pop("model_name", "gpt-4o"))

        # Create Cypher-based KG-RAG
        return CypherBasedKGRAG(graph=neo4j_graph, llm=llm, use_cot=use_cot, **kwargs)

    return factory


def create_graphrag_rag_factory(artifacts_path, vector_store_dir, use_cot=False):
    """Create a factory function for GraphRAG-based KG-RAG."""

    def factory(**kwargs):
        # Get standard GraphRAG system
        rag_system = create_graphrag_system(
            artifacts_path=artifacts_path, vector_store_dir=vector_store_dir, **kwargs
        )

        # Update use_cot setting
        rag_system.use_cot = use_cot

        return rag_system

    return factory


def main():
    """Run hyperparameter search for KG-RAG methods."""
    # Load environment variables
    load_dotenv()

    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir) / args.method
    if args.use_cot:
        output_dir = output_dir / "cot"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load hyperparameter configurations
    configs = load_configs(args.configs_path)

    # Get configurations for the specified method
    method_configs = configs.get(args.method, [])
    if not method_configs:
        print(f"No configurations found for {args.method} method")
        return

    # Create factory function for the specified method
    try:
        if args.method == "entity":
            if not args.graph_path:
                print("Error: --graph-path is required for entity method")
                sys.exit(1)

            # Load graph
            print(f"Loading graph from {args.graph_path}...")
            graph = load_graph(args.graph_path)
            rag_factory = create_entity_rag_factory(graph, args.use_cot)

        elif args.method == "baseline":
            rag_factory = create_baseline_rag_factory(
                args.collection_name, args.persist_dir, args.use_cot
            )

        elif args.method == "cypher":
            # Get Neo4j password
            neo4j_password = args.neo4j_password or os.getenv(
                "NEO4J_PASSWORD", "password"
            )

            rag_factory = create_cypher_rag_factory(
                args.neo4j_uri, args.neo4j_user, neo4j_password, args.use_cot
            )

        elif args.method == "graphrag":
            if not args.graphrag_artifacts:
                print("Error: --graphrag-artifacts is required for graphrag method")
                sys.exit(1)

            rag_factory = create_graphrag_rag_factory(
                args.graphrag_artifacts, args.vector_store_dir, args.use_cot
            )

    except NotImplementedError as e:
        print(f"Error: {str(e)}")
        return

    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    df = pd.read_csv(args.data_path)

    # Sample if needed
    if args.max_samples is not None and args.max_samples < len(df):
        df = df.sample(args.max_samples, random_state=42)

    # Run hyperparameter search
    print(f"Running hyperparameter search for {args.method} method...")
    print(f"Testing {len(method_configs)} configurations")

    results = run_hyperparameter_search(
        rag_system_factory=rag_factory,
        param_configs=method_configs,
        data_path=df,
        output_dir=output_dir,
        question_col=args.question_col,
        answer_col=args.answer_col,
        verbose=args.verbose,
    )

    # Print best configuration
    print("\nHyperparameter Search Results:")
    sorted_configs = sorted(
        results.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )

    print("\nTop 3 Configurations:")
    for i, (name, result) in enumerate(sorted_configs[:3]):
        print(f"{i + 1}. {name}: {result['accuracy']:.2%}")
        print(f"   Parameters: {result['params']}")

    print(f"\nDetailed results saved to {output_dir}")


if __name__ == "__main__":
    main()
