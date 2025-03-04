"""
Evaluation framework for KG-RAG approaches.
"""

import datetime
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Protocol, Union, Callable

import pandas as pd
from tqdm.auto import tqdm


class RagSystem(Protocol):
    """Protocol defining the interface for RAG systems."""
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a question and return a structured response.
        
        Args:
            question: The question to process
            
        Returns:
            Structured response with at least 'answer' and 'reasoning' keys
        """
        ...


def extract_number(text: str) -> Optional[float]:
    """
    Extract a number from text, typically from JSON responses.
    
    Args:
        text: Text to extract number from
        
    Returns:
        Extracted number or None if no number found
    """
    if isinstance(text, (int, float)):
        return float(text)
        
    # Try direct conversion
    try:
        return float(text)
    except (ValueError, TypeError):
        pass
    
    # Try to find a number in the string
    if isinstance(text, str):
        # Pattern for numbers with optional decimal points
        match = re.search(r'(-?\d+(?:\.\d+)?)', text.replace(',', ''))
        if match:
            return float(match.group(1))
            
    return None


class Evaluator:
    """Evaluates RAG systems on QA datasets."""
    
    def __init__(
        self,
        rag_system: RagSystem,
        output_dir: Optional[Union[str, Path]] = None,
        experiment_name: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the evaluator.
        
        Args:
            rag_system: The RAG system to evaluate
            output_dir: Directory to save evaluation results
            experiment_name: Name of the experiment for result filenames
            verbose: Whether to print verbose output
        """
        self.rag_system = rag_system
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name or "evaluation"
        self.verbose = verbose
        
    def evaluate(
        self,
        data_path: Union[str, pd.DataFrame],
        question_col: str = "New Question",
        answer_col: str = "New Answer",
        max_samples: Optional[int] = None,
        normalize_answers: bool = True,
        answer_processor: Optional[Callable[[Any], Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the RAG system on a dataset.
        
        Args:
            data_path: Path to CSV dataset or DataFrame
            question_col: Column name for questions
            answer_col: Column name for ground truth answers
            max_samples: Maximum number of samples to evaluate
            normalize_answers: Whether to normalize answers for comparison
            answer_processor: Optional function to process extracted answers
            
        Returns:
            Dictionary of evaluation results
        """
        # Load data
        if isinstance(data_path, str):
            df = pd.read_csv(data_path)
        else:
            df = data_path
            
        # Sample if needed
        if max_samples is not None and max_samples < len(df):
            df = df.sample(max_samples, random_state=42)
        
        # Setup for results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{self.experiment_name}_{timestamp}.txt"
        
        results = []
        correct = 0
        total = len(df)
        
        # Start evaluation
        with open(output_file, 'w') as f:
            # Write header
            f.write(f"{self.experiment_name} Evaluation Results\n")
            f.write(f"Evaluation Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Questions: {total}\n")
            f.write("=" * 80 + "\n\n")
            
            # Process each question
            for i, row in tqdm(df.iterrows(), total=total, desc="Evaluating questions"):
                question = row[question_col]
                expected_answer = row[answer_col]
                
                if self.verbose:
                    print(f"\nProcessing question {i+1}/{total}:")
                    print(f"Question: {question}")
                    print(f"Expected answer: {expected_answer}")
                
                try:
                    # Get response from RAG system
                    response = self.rag_system.query(question)
                    
                    # Extract answer
                    if isinstance(response, dict):
                        model_answer = response.get("answer", "")
                        model_reasoning = response.get("reasoning", "")
                    else:
                        model_answer = str(response)
                        model_reasoning = "No reasoning provided"
                    
                    # Process answers for comparison
                    if normalize_answers:
                        expected_num = extract_number(expected_answer)
                        model_num = extract_number(model_answer)
                        
                        # Apply custom processor if provided
                        if answer_processor:
                            expected_num = answer_processor(expected_num)
                            model_num = answer_processor(model_num)
                            
                        is_correct = (expected_num is not None 
                                      and model_num is not None 
                                      and expected_num == model_num)
                    else:
                        is_correct = str(model_answer).strip() == str(expected_answer).strip()
                    
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
        results_df.to_csv(
            self.output_dir / f"{self.experiment_name}_details_{timestamp}.csv", 
            index=False
        )
        
        # Return summary results
        summary = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "timestamp": timestamp,
            "output_file": str(output_file),
            "details_file": str(self.output_dir / f"{self.experiment_name}_details_{timestamp}.csv")
        }
        
        if self.verbose:
            print(f"\nEvaluation complete. Accuracy: {accuracy:.2%}")
            print(f"Results saved to {output_file}")
        
        return summary


def run_hyperparameter_search(
    rag_system_factory: Callable[..., RagSystem],
    param_configs: List[Dict[str, Any]],
    data_path: Union[str, pd.DataFrame],
    output_dir: Optional[Union[str, Path]] = None,
    question_col: str = "New Question",
    answer_col: str = "New Answer",
    max_samples: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run hyperparameter search for a RAG system.
    
    Args:
        rag_system_factory: Function that creates RAG system instances from params
        param_configs: List of parameter configurations to test
        data_path: Path to CSV dataset or DataFrame
        output_dir: Directory to save evaluation results
        question_col: Column name for questions
        answer_col: Column name for ground truth answers
        max_samples: Maximum number of samples to evaluate
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary mapping configuration names to evaluation results
    """
    output_dir = Path(output_dir) if output_dir else Path("hyperparameter_search")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    if isinstance(data_path, str):
        df = pd.read_csv(data_path)
    else:
        df = data_path
        
    # Sample if needed
    if max_samples is not None and max_samples < len(df):
        df = df.sample(max_samples, random_state=42)
    
    results = {}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for config in param_configs:
        config_name = config.pop("name", f"config_{len(results)}")
        
        if verbose:
            print(f"\nEvaluating configuration: {config_name}")
            print(f"Parameters: {config}")
        
        # Create RAG system with this configuration
        rag_system = rag_system_factory(**config)
        
        # Create evaluator
        evaluator = Evaluator(
            rag_system=rag_system,
            output_dir=output_dir,
            experiment_name=config_name,
            verbose=verbose
        )
        
        # Run evaluation
        config_results = evaluator.evaluate(
            data_path=df,
            question_col=question_col,
            answer_col=answer_col,
            max_samples=None  # Already sampled if needed
        )
        
        # Store results
        results[config_name] = {
            "params": config,
            "accuracy": config_results["accuracy"],
            "correct": config_results["correct"],
            "total": config_results["total"]
        }
    
    # Save overall results
    with open(output_dir / f"hyperparameter_search_results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print("\nHyperparameter search complete.")
        print(f"Results saved to {output_dir}/hyperparameter_search_results_{timestamp}.json")
        
        # Print sorted results
        print("\nConfigurations sorted by accuracy:")
        sorted_configs = sorted(
            results.items(), 
            key=lambda x: x[1]["accuracy"], 
            reverse=True
        )
        for name, result in sorted_configs:
            print(f"{name}: {result['accuracy']:.2%}")
    
    return results