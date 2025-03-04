"""
Main implementation of GraphRAG-based KG-RAG approach.
"""

import json
import re
from typing import Dict, Any, Optional, Union, List

from langchain_openai import ChatOpenAI
from langchain_graphrag.utils import TiktokenCounter
from langchain_graphrag.query.local_search._system_prompt import LOCAL_SEARCH_SYSTEM_PROMPT

from kg_rag.utils.prompts import get_graphrag_prompt_suffix
from .search_strategies import LocalSearchStrategy, GlobalSearchStrategy


class GraphRAGBasedKGRAG:
    """Main class implementing the GraphRAG-based KG-RAG system."""
    
    def __init__(
        self,
        artifacts: Dict[str, Any],
        entities_vector_store,
        llm: Optional[ChatOpenAI] = None,
        search_strategy: str = "local",
        community_level: int = 2,
        show_references: bool = True,
        use_cot: bool = False,
        numerical_answer: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the GraphRAG-based KG-RAG system.
        
        Args:
            artifacts: GraphRAG artifacts
            entities_vector_store: Vector store for entities
            llm: LLM to use for search
            search_strategy: Search strategy to use ("local" or "global")
            community_level: Community level to use
            show_references: Whether to show references in response
            use_cot: Whether to use Chain-of-Thought prompting
            numerical_answer: Whether to format answers as numerical values only
            verbose: Whether to print verbose output
        """
        # Set up LLM if not provided
        if llm is None:
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0
            )
        else:
            self.llm = llm
            
        # Configure LLM with response format if using CoT or numerical answer
        if use_cot or numerical_answer:
            self.llm = self.llm.bind(response_format={"type": "json_object"})
        
        # Store artifacts
        self.artifacts = artifacts
        self.entities_vector_store = entities_vector_store
        
        # Create token counter
        self.token_counter = TiktokenCounter()
        
        # Set up system prompt with appropriate extensions
        prompt_suffix = get_graphrag_prompt_suffix(numerical_answer)
        system_prompt = LOCAL_SEARCH_SYSTEM_PROMPT + prompt_suffix
        
        # Create search strategies
        self.local_search = LocalSearchStrategy(
            llm=self.llm,
            entities_vector_store=entities_vector_store,
            artifacts=artifacts,
            community_level=community_level,
            show_references=show_references,
            verbose=verbose,
            system_prompt=system_prompt,
            strict_output_format=(use_cot or numerical_answer)
        )
        
        self.global_search = GlobalSearchStrategy(
            llm=self.llm,
            artifacts=artifacts,
            community_level=community_level,
            token_counter=self.token_counter,
            show_references=show_references,
            verbose=verbose,
            strict_output_format=(use_cot or numerical_answer)
        )
        
        # Set search strategy
        self.search_strategy = search_strategy.lower()
        if self.search_strategy not in ["local", "global"]:
            if verbose:
                print(f"Invalid search strategy: {search_strategy}. Using local search.")
            self.search_strategy = "local"
        
        # Other settings
        self.verbose = verbose
        self.use_cot = use_cot
        self.numerical_answer = numerical_answer
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process query and return structured response.
        
        Args:
            question: User question to process
            
        Returns:
            Dictionary with 'answer' and 'reasoning' keys
        """
        if self.verbose:
            print(f"Processing query: {question}")
            print(f"Using {self.search_strategy} search strategy")
        
        # Perform search
        if self.search_strategy == "local":
            raw_response = self.local_search.search(question)
        else:
            raw_response = self.global_search.search(question)
        
        # Parse response
        try:
            # Try to parse as JSON first for CoT or numerical responses
            if self.use_cot or self.numerical_answer:
                result = self._parse_json_response(raw_response)
                if result:
                    return result
                    
                # Fallback for structured format
                return {
                    "answer": self._extract_number_from_text(raw_response) if self.numerical_answer else raw_response,
                    "reasoning": f"Failed to parse JSON response. Raw response: {raw_response[:500]}..."
                }
            else:
                # For standard mode, extract info from text
                return self._extract_info_from_text(raw_response, question)
                
        except Exception as e:
            if self.verbose:
                print(f"Error processing response: {str(e)}")
            
            return {
                "answer": raw_response,
                "reasoning": "Error processing the response."
            }
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON response.
        
        Args:
            response: Response string to parse
            
        Returns:
            Parsed response or None if parsing failed
        """
        # Extract JSON if surrounded by backticks or other markers
        json_pattern = r'```json\s*([\s\S]*?)\s*```|```\s*([\s\S]*?)\s*```|(\{[\s\S]*\})'
        match = re.search(json_pattern, response)
        if match:
            json_str = match.group(1) or match.group(2) or match.group(3)
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return parsed
            except:
                pass
        
        # Try parsing the entire response as JSON
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return parsed
        except:
            pass
        
        return None
    
    def _extract_info_from_text(
        self, 
        text: str, 
        question: str
    ) -> Dict[str, Any]:
        """
        Extract information from text response.
        
        Args:
            text: Response text
            question: Original question
            
        Returns:
            Structured response
        """
        # Extract number for answer if numerical answer is expected
        if self.numerical_answer:
            answer = self._extract_number_from_text(text) or text
        else:
            # For qualitative answers, try to find the direct answer
            answer = self._extract_answer_from_text(text) or text
        
        # Use parts of the text as reasoning
        reasoning_lines = []
        for line in text.split('\n'):
            if "answer:" in line.lower() or "conclusion:" in line.lower():
                continue
            if line and not line.isspace():
                reasoning_lines.append(line)
        
        reasoning = "\n".join(reasoning_lines) if reasoning_lines else text
        
        return {
            "answer": answer,
            "reasoning": reasoning
        }
    
    def _extract_number_from_text(self, text: str) -> Optional[str]:
        """
        Extract number from text.
        
        Args:
            text: Text to extract number from
            
        Returns:
            Extracted number or None if no number found
        """
        # Try to find patterns like "The answer is 42" or "42%"
        patterns = [
            r'answer\s*(?:is|:)\s*(-?\d+(?:\.\d+)?)',  # "answer is 42" or "answer: 42"
            r'(-?\d+(?:\.\d+)?)\s*%',  # "42%"
            r'(-?\d+(?:\.\d+)?)\s*(?:million|billion|dollars|USD)',  # "42 million" or "42 dollars"
            r'(?:value|amount|total)\s*(?:of|is|:)\s*(-?\d+(?:\.\d+)?)',  # "value is 42" or "amount: 42"
            r'(\d+(?:\.\d+)?)'  # Any number as a fallback
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_answer_from_text(self, text: str) -> Optional[str]:
        """
        Extract direct answer from text for qualitative responses.
        
        Args:
            text: Text to extract answer from
            
        Returns:
            Extracted answer or None if no clear answer found
        """
        # Try to find common patterns for answers
        patterns = [
            r'(?:answer|conclusion)(?:\s+is|:)\s+(.*?)(?:\.|$)',  # "The answer is..." or "Answer: ..."
            r'(?:in\s+conclusion|therefore)[,:\s]+(.*?)(?:\.|$)',  # "In conclusion..." or "Therefore..."
            r'(?:based\s+on\s+the\s+(?:context|information))[,:\s]+(.*?)(?:\.|$)'  # "Based on the context..."
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                if answer:
                    return answer
        
        return None


# Helper function to create a complete GraphRAG system
def create_graphrag_system(
    artifacts_path: str,
    vector_store_dir: str,
    llm_model: str = "gpt-4o",
    search_strategy: str = "local",
    community_level: int = 2,
    use_cot: bool = False,
    numerical_answer: bool = False,
    verbose: bool = False
) -> GraphRAGBasedKGRAG:
    """
    Create a complete GraphRAG-based KG-RAG system.
    
    Args:
        artifacts_path: Path to GraphRAG artifacts
        vector_store_dir: Directory for vector stores
        llm_model: LLM model to use
        search_strategy: Search strategy to use
        community_level: Community level to use
        use_cot: Whether to use Chain-of-Thought prompting
        numerical_answer: Whether to format answers as numerical values only
        verbose: Whether to print verbose output
        
    Returns:
        GraphRAGBasedKGRAG instance
    """
    import pickle
    from langchain_openai import ChatOpenAI
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    
    # Load artifacts
    with open(artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)
    
    # Set up embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    entities_vector_store = Chroma(
        collection_name="entities",
        persist_directory=vector_store_dir,
        embedding_function=embeddings
    )
    
    # Create LLM
    llm = ChatOpenAI(
        model=llm_model,
        temperature=0
    )
    
    # Create and return GraphRAG system
    return GraphRAGBasedKGRAG(
        artifacts=artifacts,
        entities_vector_store=entities_vector_store,
        llm=llm,
        search_strategy=search_strategy,
        community_level=community_level,
        use_cot=use_cot,
        numerical_answer=numerical_answer,
        verbose=verbose
    )