"""
Beam search explorer for entity-based KG-RAG approach.
"""

import heapq
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import networkx as nx
import numpy as np

from kg_rag.methods.entity_based.embedding_handler import EmbeddingHandler


@dataclass
class KnowledgeChain:
    """Represents a chain of entities and relationships in the graph."""
    path: List[str]  # Alternating entities and relationships
    score: float
    terminal_node: str


class BeamSearchExplorer:
    """Explores graph using beam search with embedding-based similarity."""
    
    def __init__(
        self, 
        graph: nx.DiGraph, 
        embedding_handler: EmbeddingHandler,
        beam_width: int = 3, 
        max_depth: int = 5, 
        min_score: float = 0.7, 
        verbose: bool = False
    ):
        """
        Initialize the beam search explorer.
        
        Args:
            graph: NetworkX graph to explore
            embedding_handler: Handler for embeddings
            beam_width: Width of the beam for search
            max_depth: Maximum depth of the search
            min_score: Minimum score for considering a match
            verbose: Whether to print verbose output
        """
        self.graph = graph
        self.embedding_handler = embedding_handler
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.min_score = min_score
        self.verbose = verbose
        
    def explore(
        self, 
        start_entities: List[str], 
        target_chain: List[str]
    ) -> List[KnowledgeChain]:
        """
        Perform beam search from start entities following target chain pattern.
        
        Args:
            start_entities: List of entities to start from
            target_chain: Target chain pattern to follow
            
        Returns:
            List of knowledge chains found
        """
        beam = [(0, [entity], entity) for entity in start_entities]
        chains = []
        
        # Get embeddings for target relationships (odd indices in target_chain)
        target_relations = target_chain[1::2]
        _, target_relation_embeddings = self.embedding_handler.embed_queries([], target_relations)
        
        if not isinstance(self.beam_width, int):
            if self.verbose:
                print("Warning: Invalid beam width, using default of 3")
            self.beam_width = 3
            
        if self.verbose:
            print(f"\nStarting beam search (width={self.beam_width}, max_depth={self.max_depth})")
            print(f"Target pattern: {' -> '.join(target_chain)}")
        
        for depth in range(self.max_depth):
            if not beam:
                if self.verbose:
                    print(f"  Depth {depth}: No more candidates to explore")
                break
                
            if self.verbose:
                print(f"\n  Depth {depth}:")
                
            candidates = []
            for score, path, current in beam:
                if self.verbose:
                    print(f"    Exploring from: {current}")
                edges = list(self.graph.edges(current, data=True))
                if self.verbose:
                    print(f"    Found {len(edges)} outgoing edges")
                
                for _, neighbor, rel_data in edges:
                    relation = rel_data.get('relation', '')
                    new_path = path + [relation, neighbor]
                    
                    # Calculate similarity score using embeddings
                    chain_score = self._score_chain_match(
                        new_path, 
                        target_chain, 
                        target_relation_embeddings
                    )
                    
                    if self.verbose and chain_score > self.min_score:
                        print(f"      {' -> '.join(new_path)} (score: {chain_score:.3f})")
                    
                    candidates.append((chain_score, new_path, neighbor))
                    chains.append(KnowledgeChain(
                        path=new_path, 
                        score=chain_score, 
                        terminal_node=neighbor
                    ))
            
            beam = heapq.nlargest(self.beam_width, candidates, key=lambda x: x[0])
            if self.verbose:
                print(f"\n    Selected top {len(beam)} candidates for next iteration")
        
        if self.verbose:
            print(f"\nExploration complete. Found {len(chains)} total chains")
        return chains
    
    def _score_chain_match(
        self, 
        path: List[str], 
        target: List[str], 
        target_relation_embeddings: Dict[str, np.ndarray]
    ) -> float:
        """
        Score chain match using embedding similarity.
        
        Args:
            path: Path from the graph
            target: Target pattern
            target_relation_embeddings: Embeddings of target relations
            
        Returns:
            Similarity score between path and target
        """
        score = 0
        min_len = min(len(path), len(target))
        
        # Process relationships (odd indices)
        for i in range(1, min_len, 2):
            if i >= len(path) or i >= len(target):
                break
                
            path_relation = path[i]
            target_relation = target[i]
            
            # Get or compute path relation embedding
            if path_relation not in self.embedding_handler.relation_embeddings:
                embedding = self.embedding_handler.embedder.embed_query(str(path_relation))
                self.embedding_handler.relation_embeddings[path_relation] = np.array(embedding)
            path_embedding = self.embedding_handler.relation_embeddings[path_relation]
            
            # Get target relation embedding
            target_embedding = target_relation_embeddings[target_relation]
            
            # Compute similarity
            similarity = self.embedding_handler.compute_similarity(path_embedding, target_embedding)
            score += similarity
        
        # Process entities (even indices)
        for i in range(0, min_len, 2):
            if i >= len(path) or i >= len(target):
                break
                
            path_entity = path[i]
            target_entity = target[i]
            
            # Get or compute path entity embedding
            if path_entity not in self.embedding_handler.entity_embeddings:
                embedding = self.embedding_handler.embedder.embed_query(str(path_entity))
                self.embedding_handler.entity_embeddings[path_entity] = np.array(embedding)
            path_embedding = self.embedding_handler.entity_embeddings[path_entity]
            
            # Get or compute target entity embedding
            if target_entity not in self.embedding_handler.entity_embeddings:
                embedding = self.embedding_handler.embedder.embed_query(str(target_entity))
                self.embedding_handler.entity_embeddings[target_entity] = np.array(embedding)
            target_embedding = self.embedding_handler.entity_embeddings[target_entity]
            
            # Compute similarity
            similarity = self.embedding_handler.compute_similarity(path_embedding, target_embedding)
            score += similarity
            
        return score / max(len(path), len(target))