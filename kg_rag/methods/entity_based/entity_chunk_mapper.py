"""
Entity-chunk mapping module for entity-based KG-RAG approach.
"""

from typing import Dict, List, Tuple, Any, Set, Optional
import networkx as nx
from collections import Counter, defaultdict
import numpy as np
from heapq import nlargest

from langchain_core.documents import Document


class EntityChunkMapper:
    """Maps entities and paths in knowledge graphs to document chunks."""
    
    def __init__(
        self,
        graph: nx.DiGraph,
        document_chunks: List[Document],
        chunk_expansion_size: int = 1,
        verbose: bool = False
    ):
        """
        Initialize the entity-chunk mapper.
        
        Args:
            graph: NetworkX graph representing the knowledge graph
            document_chunks: List of document chunks to map entities to
            chunk_expansion_size: Number of neighboring chunks to include in context expansion
            verbose: Whether to print verbose output
        """
        self.graph = graph
        self.document_chunks = document_chunks
        self.chunk_expansion_size = chunk_expansion_size
        self.verbose = verbose
        
        # Maps entities to chunk indices
        self.entity_chunk_map: Dict[str, List[int]] = defaultdict(list)
        
        # Maps chunk indices to entities
        self.chunk_entity_map: Dict[int, List[str]] = defaultdict(list)
        
        # Store neighbor chunks for context expansion
        self.chunk_neighbors: Dict[int, List[int]] = {}
        
        # Initialize maps
        self._initialize_maps()
        
    def _initialize_maps(self) -> None:
        """Initialize entity-chunk and chunk-neighbor mappings."""
        if self.verbose:
            print("Initializing entity-chunk mappings...")
        
        # Process each document chunk
        for i, chunk in enumerate(self.document_chunks):
            # Extract content and metadata
            content = chunk.page_content
            metadata = chunk.metadata
            
            # Find entities in this chunk by checking for graph nodes
            chunk_entities = self._extract_entities_from_chunk(content)
            
            # Update mappings
            for entity in chunk_entities:
                self.entity_chunk_map[entity].append(i)
                self.chunk_entity_map[i].append(entity)
        
        # Create neighbor mappings for context expansion
        self._initialize_chunk_neighbors()
        
        if self.verbose:
            print(f"Mapped {len(self.entity_chunk_map)} entities to {len(self.chunk_entity_map)} chunks")
    
    def _extract_entities_from_chunk(self, content: str) -> Set[str]:
        """
        Extract entities from chunk content by matching graph nodes.
        
        Args:
            content: Content of the document chunk
            
        Returns:
            Set of entity names found in the chunk
        """
        entities = set()
        
        # Get all node names from the graph
        graph_nodes = set(self.graph.nodes())
        
        # Simple approach: check if entity name appears in content
        # This could be enhanced with more sophisticated NER or entity linking
        for node in graph_nodes:
            # Skip very short entity names (likely to cause false positives)
            if len(str(node)) < 3:
                continue
                
            if str(node) in content:
                entities.add(node)
        
        return entities
    
    def _initialize_chunk_neighbors(self) -> None:
        """Initialize chunk neighbor mappings for context expansion."""
        if self.verbose:
            print("Initializing chunk neighbors for context expansion...")
        
        # Group chunks by document to find neighbors within the same document
        document_chunks = defaultdict(list)
        
        for i, chunk in enumerate(self.document_chunks):
            doc_id = chunk.metadata.get('source', chunk.metadata.get('file_path', 'unknown'))
            document_chunks[doc_id].append(i)
        
        # Create neighbor mappings
        for doc_id, chunk_indices in document_chunks.items():
            # Sort indices to ensure correct neighbor identification
            chunk_indices.sort()
            
            for i, chunk_idx in enumerate(chunk_indices):
                # Get neighboring chunks within the same document
                neighbors = []
                
                # Get preceding neighbors
                for j in range(1, self.chunk_expansion_size + 1):
                    if i - j >= 0:
                        neighbors.append(chunk_indices[i - j])
                
                # Get following neighbors
                for j in range(1, self.chunk_expansion_size + 1):
                    if i + j < len(chunk_indices):
                        neighbors.append(chunk_indices[i + j])
                
                self.chunk_neighbors[chunk_idx] = neighbors
    
    def get_chunks_for_entity(self, entity: str) -> List[int]:
        """
        Get chunk indices containing a specific entity.
        
        Args:
            entity: Entity to find chunks for
            
        Returns:
            List of chunk indices containing the entity
        """
        return self.entity_chunk_map.get(entity, [])
    
    def get_entities_for_chunk(self, chunk_idx: int) -> List[str]:
        """
        Get entities contained in a specific chunk.
        
        Args:
            chunk_idx: Index of the chunk to get entities for
            
        Returns:
            List of entities contained in the chunk
        """
        return self.chunk_entity_map.get(chunk_idx, [])
    
    def get_chunks_for_path(self, path: List[str]) -> List[int]:
        """
        Get chunk indices containing entities from a path.
        
        Args:
            path: Path of entities from the knowledge graph
            
        Returns:
            List of chunk indices containing entities from the path
        """
        # Extract entities from the path (even indices)
        entities = path[::2]
        
        # Get chunks for each entity
        chunk_sets = [set(self.get_chunks_for_entity(entity)) for entity in entities]
        
        # Return chunks that contain any of the entities
        if not chunk_sets:
            return []
            
        # Union of all chunks
        all_chunks = set().union(*chunk_sets)
        
        return list(all_chunks)
    
    def score_chunks_for_chains(
        self, 
        chains: List[Tuple[Any, float]], 
        method: str = 'frequency',
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Score chunks based on knowledge chains.
        
        Args:
            chains: List of (knowledge_chain, score) tuples
            method: Scoring method ('frequency', 'max_score', 'weighted_frequency', or 'coverage')
            top_k: Number of top chunks to return
            
        Returns:
            List of (chunk_idx, score) tuples for top-k chunks
        """
        if not chains:
            return []
        
        # Extract paths and scores from chains
        paths = [chain[0].path for chain in chains]
        chain_scores = [chain[1] for chain in chains]
        
        # Get chunks for each path
        path_chunks = [set(self.get_chunks_for_path(path)) for path in paths]
        
        # Dictionary to store chunk scores
        chunk_scores = defaultdict(float)
        
        if method == 'frequency':
            # Score based on frequency of chunk across paths
            chunk_counter = Counter()
            for chunks in path_chunks:
                chunk_counter.update(chunks)
            
            # Convert to score
            for chunk_idx, count in chunk_counter.items():
                chunk_scores[chunk_idx] = count
                
        elif method == 'max_score':
            # Score based on max chain score for each chunk
            for i, chunks in enumerate(path_chunks):
                for chunk_idx in chunks:
                    chunk_scores[chunk_idx] = max(chunk_scores[chunk_idx], chain_scores[i])
        
        elif method == 'weighted_frequency':
            # Score based on frequency weighted by chain scores
            for i, chunks in enumerate(path_chunks):
                for chunk_idx in chunks:
                    chunk_scores[chunk_idx] += chain_scores[i]
        
        elif method == 'coverage':
            # Score based on entity coverage in chunks
            for i, path in enumerate(paths):
                entities = path[::2]  # Extract entities from path
                
                for chunk_idx in path_chunks[i]:
                    chunk_entities = set(self.get_entities_for_chunk(chunk_idx))
                    # Score based on percentage of path entities in chunk
                    coverage = len(chunk_entities.intersection(entities)) / len(entities)
                    # Weighted by chain score
                    chunk_scores[chunk_idx] += coverage * chain_scores[i]
        
        else:
            raise ValueError(f"Unknown scoring method: {method}")
        
        # Get top-k chunks
        top_chunks = nlargest(top_k, chunk_scores.items(), key=lambda x: x[1])
        
        return top_chunks
    
    def expand_chunks(self, chunk_indices: List[int]) -> List[int]:
        """
        Expand a list of chunks to include their neighbors.
        
        Args:
            chunk_indices: List of chunk indices to expand
            
        Returns:
            Expanded list of chunk indices including neighbors
        """
        if not chunk_indices:
            return []
            
        # Create a set of all chunks, including the original chunks and their neighbors
        expanded_chunks = set(chunk_indices)
        
        for chunk_idx in chunk_indices:
            # Add neighbors for this chunk
            neighbors = self.chunk_neighbors.get(chunk_idx, [])
            expanded_chunks.update(neighbors)
        
        return list(expanded_chunks)
    
    def get_expanded_context(
        self, 
        chains: List[Tuple[Any, float]], 
        scoring_method: str = 'weighted_frequency',
        top_k: int = 5,
        expand_context: bool = True
    ) -> List[Document]:
        """
        Get expanded context from knowledge chains.
        
        Args:
            chains: List of (knowledge_chain, score) tuples
            scoring_method: Method to score chunks
            top_k: Number of top chunks to return before expansion
            expand_context: Whether to expand context with neighboring chunks
            
        Returns:
            List of document chunks as context
        """
        # Score chunks based on chains
        top_chunks = self.score_chunks_for_chains(
            chains=chains,
            method=scoring_method,
            top_k=top_k
        )
        
        if self.verbose:
            print(f"Found {len(top_chunks)} top chunks using method '{scoring_method}'")
        
        # Extract chunk indices
        chunk_indices = [chunk_idx for chunk_idx, _ in top_chunks]
        
        # Expand context if requested
        if expand_context:
            # Expand to include neighboring chunks
            expanded_indices = self.expand_chunks(chunk_indices)
            
            if self.verbose:
                print(f"Expanded context from {len(chunk_indices)} to {len(expanded_indices)} chunks")
                
            chunk_indices = expanded_indices
        
        # Get actual document chunks
        context_chunks = [self.document_chunks[idx] for idx in chunk_indices if idx < len(self.document_chunks)]
        
        return context_chunks