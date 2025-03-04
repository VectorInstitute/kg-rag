"""
Entity matcher for entity-based KG-RAG approach.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from kg_rag.methods.entity_based.embedding_handler import EmbeddingHandler


@dataclass(order=True)
class EntityMatch:
    """Represents a matched entity with similarity score."""
    similarity: float
    graph_entity: str = field(compare=False)
    query_entity: str = field(compare=False)


class EntityMatcher:
    """Matches query entities to graph entities using embedding-based similarity."""
    
    def __init__(
        self, 
        embedding_handler: EmbeddingHandler, 
        verbose: bool = False
    ):
        """
        Initialize the entity matcher.
        
        Args:
            embedding_handler: Handler for embeddings
            verbose: Whether to print verbose output
        """
        self.embedding_handler = embedding_handler
        self.verbose = verbose
        
    def get_matches(
        self, 
        query_entities: List[str], 
        graph_entities: List[str], 
        beam_width: int
    ) -> Dict[str, List[EntityMatch]]:
        """
        Find top matches for each query entity using embedding similarity.
        
        Args:
            query_entities: List of entities from the query
            graph_entities: List of entities from the graph
            beam_width: Number of top matches to return per query entity
            
        Returns:
            Dictionary mapping query entities to lists of matches
        """
        matches = {}
        
        # Get embeddings for query entities
        query_embeddings, _ = self.embedding_handler.embed_queries(query_entities, [])
        
        for query_entity in query_entities:
            if self.verbose:
                print(f"\nMatching entity: {query_entity}")
            
            query_embedding = query_embeddings[query_entity]
            top_matches = self.embedding_handler.get_top_entity_matches(query_embedding, beam_width)
            
            matches[query_entity] = []
            if self.verbose:
                print("Top matches:")
            
            for graph_entity, similarity in top_matches:
                match = EntityMatch(
                    graph_entity=graph_entity,
                    query_entity=query_entity,
                    similarity=similarity
                )
                if self.verbose:
                    print(f"  {match.graph_entity} (similarity: {match.similarity:.3f})")
                matches[query_entity].append(match)
                
        return matches