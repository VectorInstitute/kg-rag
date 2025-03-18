"""
Embedding generation and handling utilities for KG-RAG approaches with caching capabilities.
"""

import time
import os
import pickle
from typing import Dict, List, Iterator, Tuple, Any, Optional
import numpy as np
import networkx as nx
from tqdm.auto import tqdm
from langchain_openai import OpenAIEmbeddings


class EmbeddingHandler:
    """Handles embedding generation and similarity calculations for entities and relationships with caching support."""
    
    def __init__(
        self, 
        model: str = "text-embedding-3-small", 
        batch_size: int = 100, 
        max_retries: int = 3,
        verbose: bool = False,
        cache_dir: str = "data/sec-10-q/graphs/",
        use_cache: bool = True
    ):
        """
        Initialize the embedding handler.
        
        Args:
            model: Name of the embedding model to use
            batch_size: Number of items to embed in a single batch
            max_retries: Maximum number of retries for embedding attempts
            verbose: Whether to print verbose output
            cache_dir: Directory to store embedding cache files
            use_cache: Whether to use cached embeddings if available
        """
        self.embedder = OpenAIEmbeddings(model=model)
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self.relation_embeddings: Dict[str, np.ndarray] = {}
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.verbose = verbose
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # Define cache file paths
        self.entity_cache_path = os.path.join(self.cache_dir, "entity_embedding_cache.pkl")
        self.relation_cache_path = os.path.join(self.cache_dir, "relation_embedding_cache.pkl")
        
        # Load cached embeddings if use_cache is True
        if self.use_cache:
            self._load_cached_embeddings()
        
    def _batch_items(self, items: List[str]) -> Iterator[List[str]]:
        """Split items into batches."""
        for i in range(0, len(items), self.batch_size):
            yield items[i:i + self.batch_size]
            
    def _embed_with_retry(self, texts: List[str], retry_count: int = 0) -> List[List[float]]:
        """Attempt to embed texts with retry logic."""
        try:
            return self.embedder.embed_documents(list(map(str, texts)))
        except Exception as e:
            if retry_count < self.max_retries:
                retry_count += 1
                if self.verbose:
                    self._log(f"Embedding attempt {retry_count} failed. Retrying... Error: {str(e)}")
                time.sleep(min(2 ** retry_count, 8))  # Exponential backoff
                return self._embed_with_retry(texts, retry_count)
            else:
                if self.verbose:
                    self._log(f"Max retries ({self.max_retries}) exceeded. Error: {str(e)}")
                raise
            
    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[EmbeddingHandler] {message}")
            
    def _ensure_cache_dir_exists(self) -> None:
        """Ensure cache directory exists."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            self._log(f"Created cache directory at {self.cache_dir}")
    
    def _save_cached_embeddings(self) -> None:
        """Save embeddings to cache files."""
        self._ensure_cache_dir_exists()
        
        # Save entity embeddings
        if self.entity_embeddings:
            with open(self.entity_cache_path, 'wb') as f:
                pickle.dump(self.entity_embeddings, f)
                self._log(f"Saved {len(self.entity_embeddings)} entity embeddings to {self.entity_cache_path}")
        
        # Save relation embeddings
        if self.relation_embeddings:
            with open(self.relation_cache_path, 'wb') as f:
                pickle.dump(self.relation_embeddings, f)
                self._log(f"Saved {len(self.relation_embeddings)} relation embeddings to {self.relation_cache_path}")
    
    def _load_cached_embeddings(self) -> None:
        """Load embeddings from cache files if they exist."""
        # Load entity embeddings
        if os.path.exists(self.entity_cache_path):
            try:
                with open(self.entity_cache_path, 'rb') as f:
                    self.entity_embeddings = pickle.load(f)
                    self._log(f"Loaded {len(self.entity_embeddings)} entity embeddings from cache")
            except Exception as e:
                self._log(f"Error loading entity embeddings from cache: {str(e)}")
        else:
            self._log(f"Entity embedding cache not found at {self.entity_cache_path}")
        
        # Load relation embeddings
        if os.path.exists(self.relation_cache_path):
            try:
                with open(self.relation_cache_path, 'rb') as f:
                    self.relation_embeddings = pickle.load(f)
                    self._log(f"Loaded {len(self.relation_embeddings)} relation embeddings from cache")
            except Exception as e:
                self._log(f"Error loading relation embeddings from cache: {str(e)}")
        else:
            self._log(f"Relation embedding cache not found at {self.relation_cache_path}")
            
    def embed_graph(self, graph: nx.DiGraph, save_cache: bool = True) -> None:
        """
        Embed all entities and relationships in the graph.
        
        Args:
            graph: NetworkX graph with nodes as entities and edges with 'relation' attribute
            save_cache: Whether to save embeddings to cache after processing
        """
        start_time = time.time()
        self._log("Starting graph embedding process...")
        
        # Collect unique nodes and relationships
        nodes = list(graph.nodes())
        relationships = set()
        for _, _, rel_data in graph.edges(data=True):
            relation = rel_data.get('relation', '')
            if relation:
                relationships.add(relation)
        relationships = list(relationships)
        
        self._log(f"Found {len(nodes)} unique nodes and {len(relationships)} unique relationships")
        
        # Embed nodes in batches
        if nodes:
            self._log("Embedding nodes...")
            for batch in tqdm(self._batch_items(nodes), 
                            total=(len(nodes) + self.batch_size - 1) // self.batch_size,
                            desc="Embedding nodes",
                            disable=not self.verbose):
                # Filter out nodes that are already embedded
                batch_to_embed = [node for node in batch if node not in self.entity_embeddings]
                if batch_to_embed:
                    embeddings = self._embed_with_retry(batch_to_embed)
                    for node, embedding in zip(batch_to_embed, embeddings):
                        self.entity_embeddings[node] = np.array(embedding)
                
                if self.verbose:
                    batch_size = len(batch)
                    total_embedded = len(self.entity_embeddings)
                    self._log(f"Processed batch of {batch_size} nodes. Total nodes embedded: {total_embedded}")
        
        # Embed relationships in batches
        if relationships:
            self._log("Embedding relationships...")
            for batch in tqdm(self._batch_items(relationships),
                            total=(len(relationships) + self.batch_size - 1) // self.batch_size,
                            desc="Embedding relationships",
                            disable=not self.verbose):
                # Filter out relationships that are already embedded
                batch_to_embed = [rel for rel in batch if rel not in self.relation_embeddings]
                if batch_to_embed:
                    embeddings = self._embed_with_retry(batch_to_embed)
                    for rel, embedding in zip(batch_to_embed, embeddings):
                        self.relation_embeddings[rel] = np.array(embedding)
                
                if self.verbose:
                    batch_size = len(batch)
                    total_embedded = len(self.relation_embeddings)
                    self._log(f"Processed batch of {batch_size} relationships. Total relationships embedded: {total_embedded}")
        
        end_time = time.time()
        duration = end_time - start_time
        self._log(f"Graph embedding completed in {duration:.2f} seconds")
        self._log(f"Final counts - Nodes: {len(self.entity_embeddings)}, Relationships: {len(self.relation_embeddings)}")
        
        # Save to cache if requested
        if save_cache:
            self._save_cached_embeddings()
    
    def embed_queries(self, entities: List[str], relations: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Embed query entities and relations in batches.
        
        Args:
            entities: List of entity strings to embed
            relations: List of relation strings to embed
            
        Returns:
            Tuple of (entity_embeddings, relation_embeddings) dictionaries
        """
        query_entity_embeddings = {}
        query_relation_embeddings = {}
        
        if self.verbose:
            self._log(f"Processing {len(entities)} query entities and {len(relations)} query relations")
        
        # Embed query entities
        if entities:
            self._log("Embedding query entities...")
            for batch in tqdm(self._batch_items(entities),
                            total=(len(entities) + self.batch_size - 1) // self.batch_size,
                            desc="Embedding query entities",
                            disable=not self.verbose):
                embeddings = self._embed_with_retry(batch)
                for entity, embedding in zip(batch, embeddings):
                    query_entity_embeddings[entity] = np.array(embedding)
        
        # Embed query relations
        if relations:
            self._log("Embedding query relations...")
            for batch in tqdm(self._batch_items(relations),
                            total=(len(relations) + self.batch_size - 1) // self.batch_size,
                            desc="Embedding query relations",
                            disable=not self.verbose):
                embeddings = self._embed_with_retry(batch)
                for relation, embedding in zip(batch, embeddings):
                    query_relation_embeddings[relation] = np.array(embedding)
        
        if self.verbose:
            self._log("Query embedding completed")
            
        return query_entity_embeddings, query_relation_embeddings
    
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (between -1 and 1)
        """
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def get_top_entity_matches(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """
        Get top-k entity matches by cosine similarity.
        
        Args:
            query_embedding: Query embedding to match against
            top_k: Number of top matches to return
            
        Returns:
            List of (entity, similarity_score) tuples
        """
        if self.verbose:
            self._log(f"Finding top {top_k} entity matches...")
            
        similarities = []
        for entity, embedding in self.entity_embeddings.items():
            similarity = self.compute_similarity(query_embedding, embedding)
            similarities.append((entity, similarity))
        
        top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        
        if self.verbose:
            self._log(f"Found {len(top_matches)} entity matches")
            for entity, score in top_matches[:3]:  # Show top 3 for verbose output
                self._log(f"  {entity}: {score:.3f}")
        
        return top_matches
    
    def get_top_relation_matches(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """
        Get top-k relation matches by cosine similarity.
        
        Args:
            query_embedding: Query embedding to match against
            top_k: Number of top matches to return
            
        Returns:
            List of (relation, similarity_score) tuples
        """
        if self.verbose:
            self._log(f"Finding top {top_k} relation matches...")
            
        similarities = []
        for relation, embedding in self.relation_embeddings.items():
            similarity = self.compute_similarity(query_embedding, embedding)
            similarities.append((relation, similarity))
        
        top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        
        if self.verbose:
            self._log(f"Found {len(top_matches)} relation matches")
            for relation, score in top_matches[:3]:  # Show top 3 for verbose output
                self._log(f"  {relation}: {score:.3f}")
        
        return top_matches
        
    def clear_cache(self, delete_files: bool = False) -> None:
        """
        Clear embedding caches from memory and optionally delete cache files.
        
        Args:
            delete_files: Whether to delete the cache files from disk
        """
        # Clear memory caches
        self.entity_embeddings = {}
        self.relation_embeddings = {}
        self._log("Cleared embedding caches from memory")
        
        # Delete cache files if requested
        if delete_files:
            try:
                if os.path.exists(self.entity_cache_path):
                    os.remove(self.entity_cache_path)
                    self._log(f"Deleted entity cache file: {self.entity_cache_path}")
                    
                if os.path.exists(self.relation_cache_path):
                    os.remove(self.relation_cache_path)
                    self._log(f"Deleted relation cache file: {self.relation_cache_path}")
            except Exception as e:
                self._log(f"Error deleting cache files: {str(e)}")