"""
Main implementation of standard baseline RAG approach.
"""

import json
import re
from typing import Dict, Any, Optional, List

import openai

from .embedder import OpenAIEmbedding
from .document_processor import DocumentProcessor
from .vector_store import ChromaDBManager
from kg_rag.utils.prompts import create_query_prompt


class BaselineRAG:
    """Main class implementing the standard baseline RAG system."""
    
    def __init__(
        self,
        openai_api_key: str, 
        collection_name: str = "document_collection",
        chroma_persist_dir: str = "chroma_db",
        model_name: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 5,
        use_cot: bool = False,
        numerical_answer: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the standard baseline RAG system.
        
        Args:
            openai_api_key: OpenAI API key
            collection_name: Name of the ChromaDB collection
            chroma_persist_dir: Directory to store ChromaDB files
            model_name: Name of the OpenAI model to use for generation
            embedding_model: Name of the OpenAI model to use for embeddings
            top_k: Number of top results to retrieve
            use_cot: Whether to use Chain-of-Thought prompting
            numerical_answer: Whether to format answers as numerical values only
            verbose: Whether to print verbose output
        """
        self.embedder = OpenAIEmbedding(
            api_key=openai_api_key,
            model=embedding_model,
            verbose=verbose
        )
        self.processor = DocumentProcessor(verbose=verbose)
        self.db_manager = ChromaDBManager(
            collection_name=collection_name,
            persist_directory=chroma_persist_dir,
            verbose=verbose
        )
        self.client = openai.Client(api_key=openai_api_key)
        self.model_name = model_name
        self.top_k = top_k
        self.use_cot = use_cot
        self.numerical_answer = numerical_answer
        self.verbose = verbose
    
    def generate_answer(self, query: str, context: str) -> Dict[str, Any]:
        """
        Generate an answer using the OpenAI API.
        
        Args:
            query: User query
            context: Retrieved context from vector store
            
        Returns:
            Dictionary with answer and reasoning
        """
        # Create the prompt using the prompts module
        messages = create_query_prompt(
            question=query,
            context=context,
            system_type="baseline",
            use_cot=self.use_cot,
            numerical_answer=self.numerical_answer
        )
        
        # Set response format to JSON if using CoT or numerical answer
        response_format = {"type": "json_object"} if (self.use_cot or self.numerical_answer) else None
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            response_format=response_format
        )
        
        # Process the response
        content = response.choices[0].message.content
        
        if self.use_cot or self.numerical_answer:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                if self.verbose:
                    print(f"Error parsing JSON response: {content}")
                return {
                    "reasoning": "Error parsing the response from the language model.",
                    "answer": self._extract_answer_from_text(content)
                }
        else:
            # For standard mode, wrap the answer in our consistent format
            return {
                "answer": content,
                "reasoning": f"Retrieved {self.top_k} documents and generated an answer based on them."
            }
    
    def query(self, question: str, n_results: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a query and return a structured response.
        
        Args:
            question: User question
            n_results: Optional number of results to retrieve (defaults to self.top_k)
            
        Returns:
            Dictionary with 'answer' and 'reasoning' keys
        """
        n_results = n_results or self.top_k
        
        if self.verbose:
            print(f"Processing query: {question}")
            
        # Retrieve relevant documents
        results = self.db_manager.query(
            query_text=question, 
            embedding_function=self.embedder, 
            n_results=n_results
        )
        
        # Format context from retrieved documents
        context = "\n\n".join([
            f"[From {meta['filename']}]:\n{doc}"
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ])
        
        if self.verbose:
            print(f"Retrieved {len(results['documents'][0])} documents")
            
        # Generate answer
        result = self.generate_answer(question, context)
        
        if self.verbose:
            print(f"Generated answer: {result['answer']}")
            
        return result
    
    def _extract_answer_from_text(self, text: str) -> str:
        """
        Extract the main answer from text response.
        
        Args:
            text: Response text to extract answer from
            
        Returns:
            Extracted answer
        """
        if self.numerical_answer:
            # For numerical answers, try to extract a number
            number_patterns = [
                r'answer\s*(?:is|:)\s*(-?\d+(?:\.\d+)?)',  # "answer is 42" or "answer: 42"
                r'(-?\d+(?:\.\d+)?)\s*%',  # "42%"
                r'(-?\d+(?:\.\d+)?)\s*(?:million|billion|dollars|USD)',  # "42 million" or "42 dollars"
                r'(?:value|amount|total)\s*(?:of|is|:)\s*(-?\d+(?:\.\d+)?)',  # "value is 42" or "amount: 42"
                r'(\d+(?:\.\d+)?)'  # Any number as a fallback
            ]
            
            for pattern in number_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1)
        
        # For general answers, look for common answer patterns
        answer_patterns = [
            r'(?:answer|conclusion)(?:\s+is|:)\s+(.*?)(?:\.|$)',  # "The answer is..." or "Answer: ..."
            r'(?:in\s+conclusion|therefore)[,:\s]+(.*?)(?:\.|$)',  # "In conclusion..." or "Therefore..."
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no patterns match, return the text as is
        return text