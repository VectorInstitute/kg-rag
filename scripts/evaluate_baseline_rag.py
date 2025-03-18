#!/usr/bin/env python
"""
Script to evaluate baseline RAG approaches on test datasets.
"""

import os
import sys
import argparse
import json
import datetime
import re
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kg_rag.methods.baseline_rag.kg_rag import BaselineRAG
from kg_rag.methods.baseline_rag.kg_rag_cot import BaselineRAGCoT


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate baseline RAG approaches on test datasets"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to evaluation dataset CSV"
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
        help="Directory with ChromaDB files"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["standard", "cot"],
        default="cot",
        help="RAG method to evaluate (standard or chain-of-thought)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use for generation"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model to use"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to retrieve"
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
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    return parser.parse_args()


def extract_number(text):
    """Extract a numerical value from text."""
    # Handle direct numerical values
    if isinstance(text, (int, float)):
        return float(text)
    
    try:
        # Try direct conversion to float
        return float(text)
    except (ValueError, TypeError):
        # Try to find a number in the string
        if isinstance(text, str):
            # Pattern for numbers with optional decimal points
            # Remove commas first to handle formatted numbers
            match = re.search(r'(-?\d+(?:\.\d+)?)', text.replace(',', ''))
            if match:
                return float(match.group(1))
    
    return None


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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the appropriate RAG system
    if args.method == "standard":
        rag_system = BaselineRAG(
            openai_api_key=openai_api_key,
            collection_name=args.collection_name,
            chroma_persist_dir=args.persist_dir,
            model_name=args.model,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
            verbose=args.verbose
        )
        method_name = "standard"
    else:  # Chain-of-Thought
        rag_system = BaselineRAGCoT(
            openai_api_key=openai_api_key,
            collection_name=args.collection_name,
            chroma_persist_dir=args.persist_dir,
            model_name=args.model,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
            verbose=args.verbose
        )
        method_name = "cot"
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    
    # Sample if needed
    if args.max_samples is not None and args.max_samples < len(df):
        df = df.sample(args.max_samples, random_state=42)
        print(f"Sampled {args.max_samples} questions from dataset")
    
    # Setup for evaluation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"evaluation_{method_name}_{timestamp}.txt"
    csv_output_file = output_dir / f"evaluation_{method_name}_details_{timestamp}.csv"
    
    results = []
    correct = 0
    total = len(df)
    
    # Start evaluation
    with open(output_file, 'w') as f:
        # Write header
        f.write(f"Baseline {method_name.upper()} RAG Evaluation Results\n")
        f.write(f"Evaluation Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Questions: {total}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Embedding Model: {args.embedding_model}\n")
        f.write(f"Top-K: {args.top_k}\n")
        f.write("=" * 80 + "\n\n")
        
        # Process each question
        for i, row in tqdm(df.iterrows(), total=total, desc="Evaluating questions"):
            question = row[args.question_col]
            expected_answer = row[args.answer_col]
            
            if args.verbose:
                print(f"\nProcessing question {i+1}/{total}:")
                print(f"Question: {question}")
                print(f"Expected answer: {expected_answer}")
            
            try:
                # Get response from RAG system
                response = rag_system.query(question)
                
                # Extract answer and reasoning
                model_answer = response.get("answer", "")
                model_reasoning = response.get("reasoning", "")
                
                # Compare answers
                expected_num = extract_number(expected_answer)
                model_num = extract_number(model_answer)
                
                is_correct = (expected_num is not None 
                            and model_num is not None 
                            and abs(expected_num - model_num) < 0.001)
                
                if is_correct:
                    correct += 1
                    
            except Exception as e:
                model_answer = f"ERROR: {str(e)}"
                model_reasoning = "Error occurred during processing"
                is_correct = False
            
            # Write results for this question
            f.write(f"Question {i+1}/{total}:\n")
            f.write(f"Question: {question}\n")
            f.write(f"Expected Answer: {expected_answer}\n")
            f.write(f"Model Answer: {model_answer}\n")
            f.write(f"Reasoning:\n{model_reasoning}\n")
            f.write(f"Correct: {is_correct}\n")
            f.write("-" * 80 + "\n\n")
            
            # Store result
            results.append({
                'question_id': i+1,
                'question': question,
                'expected': expected_answer,
                'answer': model_answer,
                'reasoning': model_reasoning,
                'correct': is_correct
            })
        
        # Calculate and write summary
        accuracy = correct / total
        f.write("\nEvaluation Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Questions: {total}\n")
        f.write(f"Correct Answers: {correct}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
    
    # Save detailed results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_output_file, index=False)
    
    print(f"\nEvaluation complete. Accuracy: {accuracy:.2%}")
    print(f"Results saved to {output_file}")
    print(f"Detailed results saved to {csv_output_file}")


if __name__ == "__main__":
    main()