"""
Updated main implementation of entity-based KG-RAG approach with entity-chunk mapping.
"""

import json
import re
from typing import Dict, Any, Optional, List, Tuple, Union, Literal

import networkx as nx
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from kg_rag.methods.entity_based.embedding_handler import EmbeddingHandler
from kg_rag.utils.prompts import create_query_prompt
from kg_rag.methods.entity_based.beam_search import BeamSearchExplorer, KnowledgeChain
from kg_rag.methods.entity_based.chain_extractor import ChainExtractor
from kg_rag.methods.entity_based.entity_matcher import EntityMatcher
from kg_rag.methods.entity_based.entity_chunk_mapper import EntityChunkMapper


class EntityBasedKGRAG:
    """Main class implementing the enhanced entity-based KG-RAG system with chunk retrieval."""
    
    def __init__(
        self,
        graph: nx.DiGraph,
        document_chunks: List[Document],
        llm: Optional[ChatOpenAI] = None,
        beam_width: int = 10,
        max_depth: int = 8,
        top_k: int = 5,
        num_chains: int = 2,
        min_score: float = 0.7,
        chunk_scoring_method: str = 'weighted_frequency',
        expand_context: bool = True,
        chunk_expansion_size: int = 1,
        use_cot: bool = False,
        numerical_answer: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the enhanced entity-based KG-RAG system.
        
        Args:
            graph: NetworkX graph
            document_chunks: List of document chunks for entity mapping
            llm: LLM for text generation
            beam_width: Width of the beam for search
            max_depth: Maximum depth of the search
            top_k: Number of top chunks to return
            num_chains: Number of chains to extract
            min_score: Minimum score for considering a match
            chunk_scoring_method: Method for scoring chunks ('frequency', 'max_score', 
                                  'weighted_frequency', or 'coverage')
            expand_context: Whether to expand context with neighboring chunks
            chunk_expansion_size: Number of neighboring chunks to include in expansion
            use_cot: Whether to use Chain-of-Thought prompting
            numerical_answer: Whether to format answers as numerical values only
            verbose: Whether to print verbose output
        """
        # Set up LLM if not provided
        if llm is None:
            self.llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-4o"
            )
        else:
            self.llm = llm
            
        # Configure LLM with response format if using CoT or numerical answer
        if use_cot or numerical_answer:
            self.llm = self.llm.bind(response_format={"type": "json_object"})
        
        # Store graph and document chunks
        self.graph = graph
        self.document_chunks = document_chunks
        
        # Initialize embedding handler
        self.embedding_handler = EmbeddingHandler(verbose=verbose)
        
        # Generate embeddings for the graph
        self.embedding_handler.embed_graph(graph)
        
        # Create components
        self.chain_extractor = ChainExtractor(
            llm=self.llm,
            num_chains=num_chains,
            verbose=verbose
        )
        
        self.entity_matcher = EntityMatcher(
            embedding_handler=self.embedding_handler,
            verbose=verbose
        )
        
        self.beam_search = BeamSearchExplorer(
            graph=graph,
            embedding_handler=self.embedding_handler,
            beam_width=beam_width,
            max_depth=max_depth,
            min_score=min_score,
            verbose=verbose
        )
        
        # Create entity-chunk mapper
        self.entity_chunk_mapper = EntityChunkMapper(
            graph=graph,
            document_chunks=document_chunks,
            chunk_expansion_size=chunk_expansion_size,
            verbose=verbose
        )
        
        # Other settings
        self.top_k = top_k
        self.chunk_scoring_method = chunk_scoring_method
        self.expand_context = expand_context
        self.use_cot = use_cot
        self.numerical_answer = numerical_answer
        self.verbose = verbose
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a query and return a structured response.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with 'answer' and 'reasoning' keys
        """
        if self.verbose:
            print(f"Processing query: {question}")
        
        # Extract candidate chains from query
        chains = self.chain_extractor.extract_chains(question)
        
        if not chains:
            if self.verbose:
                print("No chains extracted from query.")
            return {
                "answer": "I couldn't find relevant information to answer this question.",
                "reasoning": "Failed to extract knowledge patterns from the query."
            }
        
        # Process each chain and collect findings
        all_findings = []
        
        for chain_pattern in chains:
            findings = self._process_chain(chain_pattern, question)
            all_findings.extend(findings)
        
        # Sort findings by score
        all_findings.sort(key=lambda x: x[1], reverse=True)
        
        if self.verbose:
            print(f"Found {len(all_findings)} total path matches")
            
        # Limit to meaningful findings
        top_findings = [(chain, score) for chain, score in all_findings 
                        if score >= self.beam_search.min_score]
            
        if not top_findings:
            if self.verbose:
                print(f"No findings with score >= {self.beam_search.min_score}")
            return {
                "answer": "I couldn't find relevant information to answer this question.",
                "reasoning": "No high-confidence knowledge paths found in the graph."
            }
        
        # Get context chunks based on findings
        context_chunks = self.entity_chunk_mapper.get_expanded_context(
            chains=top_findings,
            scoring_method=self.chunk_scoring_method,
            top_k=self.top_k,
            expand_context=self.expand_context
        )
        
        if self.verbose:
            print(f"Retrieved {len(context_chunks)} context chunks")
            
        if not context_chunks:
            if self.verbose:
                print("No relevant context chunks found")
            return {
                "answer": "I couldn't find relevant information to answer this question.",
                "reasoning": "No relevant document context found."
            }
        
        # Format chunks and path information as context
        context = self._format_context(top_findings, context_chunks)
        
        # Create the prompt based on the settings
        messages = create_query_prompt(
            question=question,
            context=context,
            system_type="entity",
            use_cot=self.use_cot,
            numerical_answer=self.numerical_answer
        )
        
        # Generate response
        response = self.llm.invoke(messages)
        
        # Process the response
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = response
            
        if self.use_cot or self.numerical_answer:
            try:
                # Try to parse as JSON
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                if self.verbose:
                    print(f"Error parsing JSON response: {content}")
                # Fallback for parsing errors
                return {
                    "answer": self._extract_answer_from_text(content),
                    "reasoning": f"Error parsing JSON response. Raw response: {content[:200]}..."
                }
        else:
            # For standard mode without structured output
            return {
                "answer": self._extract_answer_from_text(content),
                "reasoning": f"Answer generated based on knowledge graph exploration with {len(top_findings)} relevant paths and {len(context_chunks)} document chunks."
            }
    
    def _process_chain(self, chain_pattern: List[str], question: str) -> List[Tuple[KnowledgeChain, float]]:
        """
        Process a single chain pattern to find relevant paths in the graph.
        
        Args:
            chain_pattern: Chain pattern to follow (e.g., ["Apple", "REPORTED", "Revenue"])
            question: Original question for context
            
        Returns:
            List of tuples with (knowledge_chain, score)
        """
        # Get query entities (even indices in the chain)
        query_entities = chain_pattern[::2]
        
        # Match query entities to graph entities
        entity_matches = {}
        for entity in query_entities:
            # Get or create embedding for this entity
            if entity not in self.embedding_handler.entity_embeddings:
                embedding = self.embedding_handler.embedder.embed_query(entity)
                self.embedding_handler.entity_embeddings[entity] = embedding
        
        # Get start entities (top matches for the first entity in the chain)
        start_entity_embedding = self.embedding_handler.entity_embeddings[query_entities[0]]
        start_entity_matches = self.embedding_handler.get_top_entity_matches(
            start_entity_embedding, 
            self.beam_search.beam_width
        )
        start_entities = [entity for entity, _ in start_entity_matches]
        
        # Perform beam search
        chains = self.beam_search.explore(start_entities, chain_pattern)
        
        # Filter chains by minimum score
        findings = [(chain, chain.score) for chain in chains if chain.score >= self.beam_search.min_score]
        
        if self.verbose:
            print(f"Found {len(findings)} chains with scores >= {self.beam_search.min_score}")
            
        return findings
    
    def _format_context(
        self, 
        findings: List[Tuple[KnowledgeChain, float]],
        context_chunks: List[Document]
    ) -> str:
        """
        Format findings and context chunks for the LLM.
        
        Args:
            findings: List of (knowledge_chain, score) tuples
            context_chunks: List of document chunks
            
        Returns:
            Formatted context string
        """
        if not findings or not context_chunks:
            return "No relevant information found."
            
        context_lines = ["Found the following relevant information:"]
        
        # Add knowledge paths
        context_lines.append("\n=== KNOWLEDGE PATHS ===")
        for i, (chain, score) in enumerate(findings[:5]):  # Limit to top 5 for clarity
            # Format the path as a readable string
            path_str = " -> ".join(chain.path)
            
            # Add to context
            context_lines.append(f"Path {i+1} (score: {score:.3f}): {path_str}")
        
        # Add document chunks
        context_lines.append("\n=== DOCUMENT CHUNKS ===")
        for i, chunk in enumerate(context_chunks):
            # Extract content and metadata
            content = chunk.page_content
            source = chunk.metadata.get('source', chunk.metadata.get('file_path', 'unknown'))
            page = chunk.metadata.get('page', '')
            
            # Format source information
            source_info = f"[Source: {source}"
            if page:
                source_info += f", Page: {page}"
            source_info += "]"
            
            # Add to context
            context_lines.append(f"Chunk {i+1}: {source_info}")
            context_lines.append(content)
            context_lines.append("---")
        
        return "\n".join(context_lines)
    
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