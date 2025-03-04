"""
Chain extractor for entity-based KG-RAG approach.
"""

import json
from typing import List, Optional
from langchain_openai import ChatOpenAI


class ChainExtractor:
    """Extracts candidate chains from queries using LLM."""
    
    def __init__(
        self, 
        llm: Optional[ChatOpenAI] = None,
        num_chains: int = 1, 
        verbose: bool = False
    ):
        """
        Initialize the chain extractor.
        
        Args:
            llm: LLM to use for extraction (default: ChatOpenAI with gpt-4o)
            num_chains: Number of chains to extract
            verbose: Whether to print verbose output
        """
        if llm is None:
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
            )
            self.llm = self.llm.bind(response_format={"type": "json_object"})
        else:
            self.llm = llm
            
        self.num_chains = num_chains
        self.verbose = verbose
        
    def extract_chains(self, query: str) -> List[List[str]]:
        """
        Extract candidate entity-relationship chains from query.
        
        Args:
            query: User query to extract chains from
            
        Returns:
            List of extracted chains, each a list of alternating entities and relationships
        """
        prompt = f"""Given this question, identify {self.num_chains} possible chain(s) of entities and relationships that would answer it.
        Express each chain as: entity1 -> relation1 -> entity2 -> relation2 -> ...
        Return JSON object with key "chains" containing array of chains.
        
        Question: {query}
        
        Example output:
        {{"chains": [
            ["Apple Inc", "REPORTED", "Revenue", "VALUE_ON", "Q3 2022"],
            ["Apple Inc", "HAS", "Products", "MARGIN_PERCENTAGE", "Q3 2022"]
        ]}}
        """
        
        response = self.llm.invoke(prompt)
        try:
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = response
                
            chains = json.loads(content)["chains"]
            
            if self.verbose:
                print("Extracted chains:")
                for chain in chains:
                    print(f"  {' -> '.join(chain)}")
                    
            return chains
        except Exception as e:
            if self.verbose:
                print(f"Failed to parse LLM output as JSON: {str(e)}")
                print(f"Response: {response}")
            return []