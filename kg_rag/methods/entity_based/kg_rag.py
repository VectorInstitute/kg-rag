"""Simplified Knowledge Graph RAG system with direct query-to-node similarity and streamlined retrieval."""

import json
from typing import Any

import networkx as nx
import numpy as np
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from kg_rag.methods.entity_based.embedding_handler import EmbeddingHandler
from kg_rag.utils.prompts import create_query_prompt


class EntityBasedKGRAG:
    """Entity-based KG-RAG system with direct query-to-node embedding similarity."""

    def __init__(
        self,
        graph: nx.DiGraph,
        graph_documents: list[GraphDocument],
        document_chunks: list[Document],
        llm: ChatOpenAI | None = None,
        top_k_nodes: int = 10,
        top_k_chunks: int = 5,
        similarity_threshold: float = 0.7,
        node_freq_weight: float = 0.4,
        node_sim_weight: float = 0.6,
        use_cot: bool = True,
        numerical_answer: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the entity-based KG-RAG system.

        Args:
            graph: NetworkX graph
            graph_documents: List of graph documents for entity-chunk mapping
            document_chunks: List of document chunks for context retrieval
            llm: LLM for answer generation
            top_k_nodes: Number of top nodes to retrieve
            top_k_chunks: Number of top chunks to return
            similarity_threshold: Minimum similarity score for nodes
            node_freq_weight: Weight for node frequency in chunk scoring (0.0 to 1.0)
            node_sim_weight: Weight for node similarity in chunk scoring (0.0 to 1.0)
            use_cot: Whether to use Chain-of-Thought prompting
            numerical_answer: Whether to format answers as numerical values only
            verbose: Whether to print verbose output
        """
        # Validate weights
        assert 0.0 <= node_freq_weight <= 1.0, (
            "node_freq_weight must be between 0.0 and 1.0"
        )
        assert 0.0 <= node_sim_weight <= 1.0, (
            "node_sim_weight must be between 0.0 and 1.0"
        )
        assert abs(node_freq_weight + node_sim_weight - 1.0) < 1e-6, (
            "Weights must sum to 1.0"
        )

        # Set up LLM
        if llm is None:
            self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
        else:
            self.llm = llm

        # Configure LLM with response format if using CoT or numerical answer
        if use_cot:
            self.llm = self.llm.bind(response_format={"type": "json_object"})  # type: ignore

        # Initialize embedding handler
        self.embedding_handler = EmbeddingHandler(verbose=verbose)

        # Store inputs
        self.graph = graph
        self.document_chunks = document_chunks
        self.top_k_nodes = top_k_nodes
        self.top_k_chunks = top_k_chunks
        self.similarity_threshold = similarity_threshold
        self.node_freq_weight = node_freq_weight
        self.node_sim_weight = node_sim_weight
        self.use_cot = use_cot
        self.numerical_answer = numerical_answer
        self.verbose = verbose

        # Create entity-chunk mapping
        self.entity_chunk_map = self._build_entity_chunk_map(graph_documents)

        # Initialize node embeddings
        self.embedding_handler.embed_graph(graph)

    def _build_entity_chunk_map(
        self, graph_documents: list[GraphDocument]
    ) -> dict[str, list[int]]:
        """
        Build a simple entity-to-chunk mapping from graph documents.

        Args:
            graph_documents: List of graph documents connecting entities to source chunks

        Returns
        -------
            Dictionary mapping entity IDs to lists of chunk indices
        """
        entity_chunk_map: dict[str, list[int]] = {}

        if self.verbose:
            print("Building entity-chunk mapping...")

        for chunk_idx, graph_doc in enumerate(graph_documents):
            # Extract entities from nodes
            if hasattr(graph_doc, "nodes"):
                for node in graph_doc.nodes:
                    # Get entity ID
                    entity_id = None
                    if hasattr(node, "id"):
                        entity_id = node.id
                    elif isinstance(node, dict) and "id" in node:
                        entity_id = node["id"]

                    if entity_id:
                        if isinstance(entity_id, int):
                            entity_id = str(entity_id)
                        # Add chunk to entity's list
                        if entity_id not in entity_chunk_map:
                            entity_chunk_map[entity_id] = []
                        if chunk_idx not in entity_chunk_map[entity_id]:
                            entity_chunk_map[entity_id].append(chunk_idx)

        if self.verbose:
            print(f"Built mapping for {len(entity_chunk_map)} entities")

        return entity_chunk_map

    def _get_similar_nodes(
        self, query_embedding: np.ndarray
    ) -> list[tuple[str, float]]:
        """
        Get nodes similar to query.

        Args:
            query_embedding: Query embedding

        Returns
        -------
            List of (node, similarity_score) tuples
        """
        similarities = []

        # Calculate similarity for all nodes
        for node, embedding in self.embedding_handler.entity_embeddings.items():
            similarity = self.embedding_handler.compute_similarity(
                query_embedding, embedding
            )
            if similarity >= self.similarity_threshold:
                similarities.append((node, similarity))

        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Limit to top-k
        return similarities[: self.top_k_nodes]

    def _get_subgraph(self, nodes: list[str], max_hops: int = 1) -> nx.DiGraph:
        """
        Get subgraph containing nodes and their neighborhoods.

        Args:
            nodes: List of nodes to include
            max_hops: Maximum hops from seed nodes

        Returns
        -------
            NetworkX subgraph
        """
        if not nodes:
            return nx.DiGraph()

        # Start with seed nodes
        nodes_to_include = set(nodes)

        # Add neighbors up to max_hops
        for _ in range(max_hops):
            new_nodes = set()
            for node in nodes_to_include:
                if node in self.graph:
                    # Add incoming neighbors (predecessors)
                    new_nodes.update(self.graph.predecessors(node))
                    # Add outgoing neighbors (successors)
                    new_nodes.update(self.graph.successors(node))
            nodes_to_include.update(new_nodes)

        # Create subgraph
        return self.graph.subgraph(nodes_to_include)

    def _get_chunks_for_nodes(self, nodes: list[str]) -> list[int]:
        """
        Get unique chunks for a list of nodes.

        Args:
            nodes: List of nodes (entities)

        Returns
        -------
            List of unique chunk indices
        """
        chunk_indices = set()

        for node in nodes:
            if node in self.entity_chunk_map:
                chunk_indices.update(self.entity_chunk_map[node])

        return list(chunk_indices)

    def _get_path_descriptions(self, subgraph: nx.DiGraph) -> list[str]:
        """
        Extract meaningful path descriptions from subgraph.

        Args:
            subgraph: NetworkX subgraph to describe

        Returns
        -------
            List of path descriptions
        """
        if not subgraph.nodes:
            return []

        paths = []

        # Get top nodes by degree (more connected nodes are likely more important)
        top_nodes = sorted(
            subgraph.nodes, key=lambda n: subgraph.degree(n), reverse=True
        )[:5]

        # Describe paths from each top node
        for node in top_nodes:
            # Get outgoing edges
            for _, neighbor, data in subgraph.out_edges(node, data=True):
                relation = data.get("relation", "related_to")
                path = f"{node} -> {relation} -> {neighbor}"
                paths.append(path)

                # Add second-hop paths
                for _, next_neighbor, next_data in subgraph.out_edges(
                    neighbor, data=True
                ):
                    next_relation = next_data.get("relation", "related_to")
                    second_hop_path = f"{node} -> {relation} -> {neighbor} -> {next_relation} -> {next_neighbor}"
                    paths.append(second_hop_path)

        return paths[:10]  # Limit to top 10 most relevant paths

    def _score_chunks_by_nodes(
        self, chunks: list[Document], node_info: list[tuple[str, float]]
    ) -> list[tuple[Document, float]]:
        """
        Score chunks based on simplified scoring approach.

        Args:
            chunks: List of document chunks
            node_info: List of (node, similarity_score) tuples from query matching

        Returns
        -------
            List of (chunk, score) tuples
        """
        # Create dictionary of nodes and their similarity scores
        node_scores = dict(node_info)
        nodes = list(node_scores.keys())

        # Track which nodes reference each chunk
        chunk_to_nodes: dict[int, set[str]] = {id(chunk): set() for chunk in chunks}

        # Map chunks to nodes
        for _, chunk in enumerate(chunks):
            chunk_id = id(chunk)
            # Find original index in document_chunks
            orig_idx = None
            for i, orig_chunk in enumerate(self.document_chunks):
                if chunk is orig_chunk:
                    orig_idx = i
                    break

            if orig_idx is not None:
                # Find all nodes that map to this chunk
                for node in nodes:
                    if (
                        node in self.entity_chunk_map
                        and orig_idx in self.entity_chunk_map[node]
                    ):
                        chunk_to_nodes[chunk_id].add(node)

        # Score each chunk
        chunk_scores = []
        for chunk in chunks:
            chunk_id = id(chunk)
            referencing_nodes = chunk_to_nodes[chunk_id]

            if not referencing_nodes:
                chunk_scores.append((chunk, 0.0))
                continue

            # Calculate frequency score - what fraction of relevant nodes point to this chunk
            freq_score = len(referencing_nodes) / len(nodes) if nodes else 0

            # Calculate similarity score - average similarity of nodes that point to this chunk
            sim_score = sum(
                node_scores.get(node, 0) for node in referencing_nodes
            ) / len(referencing_nodes)

            # Combined weighted score
            combined_score = (self.node_freq_weight * freq_score) + (
                self.node_sim_weight * sim_score
            )

            chunk_scores.append((chunk, combined_score))

        return chunk_scores

    def _format_context(self, paths: list[str], chunks: list[Document]) -> str:
        """
        Format paths and chunks into context for the LLM.

        Args:
            paths: List of path descriptions
            chunks: List of document chunks

        Returns
        -------
            Formatted context string
        """
        context_lines = []

        # Add knowledge paths
        if paths:
            context_lines.append("=== KNOWLEDGE GRAPH INFORMATION ===")
            for i, path in enumerate(paths):
                context_lines.append(f"Path {i + 1}: {path}")
            context_lines.append("")

        # Add document chunks
        if chunks:
            context_lines.append("=== DOCUMENT CHUNKS ===")
            for i, chunk in enumerate(chunks):
                # Extract content and metadata
                content = chunk.page_content
                source = chunk.metadata.get(
                    "source", chunk.metadata.get("file_path", "unknown")
                )
                page = chunk.metadata.get("page", "")

                # Format source information
                source_info = f"[Source: {source}"
                if page:
                    source_info += f", Page: {page}"
                source_info += "]"

                # Add to context
                context_lines.append(f"Chunk {i + 1}: {source_info}")
                context_lines.append(content)
                context_lines.append("---")

        return "\n".join(context_lines)

    def _extract_answer_from_text(self, text: str) -> str:
        """
        Extract the main answer from text response.

        Args:
            text: Response text to extract answer from

        Returns
        -------
            Extracted answer
        """
        import re

        if self.numerical_answer:
            # For numerical answers, try to extract a number
            number_patterns = [
                r"answer\s*(?:is|:)\s*(-?\d+(?:\.\d+)?)",  # "answer is 42" or "answer: 42"
                r"(-?\d+(?:\.\d+)?)\s*%",  # "42%"
                r"(-?\d+(?:\.\d+)?)\s*(?:million|billion|dollars|USD)",  # "42 million" or "42 dollars"
                r"(?:value|amount|total)\s*(?:of|is|:)\s*(-?\d+(?:\.\d+)?)",  # "value is 42" or "amount: 42"
                r"(\d+(?:\.\d+)?)",  # Any number as a fallback
            ]

            for pattern in number_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1)

        # For general answers, look for common answer patterns
        answer_patterns = [
            r"(?:answer|conclusion)(?:\s+is|:)\s+(.*?)(?:\.|$)",  # "The answer is..." or "Answer: ..."
            r"(?:in\s+conclusion|therefore)[,:\s]+(.*?)(?:\.|$)",  # "In conclusion..." or "Therefore..."
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no patterns match, return the text as is
        return text

    def _process_results(
        self, response: Any, nodes: Any, paths: Any, top_chunks: Any
    ) -> Any:
        """
        Process the response from the KG-RAG system.

        Args:
            response: Response from the KG-RAG system

        Returns
        -------
            Processed response with 'answer' and 'reasoning' keys
        """
        content = response.content if hasattr(response, "content") else response
        if self.use_cot or self.numerical_answer:
            try:
                # Try to parse as JSON
                return json.loads(content)
            except json.JSONDecodeError:
                if self.verbose:
                    print(f"Error parsing JSON response: {content}")
                # Fallback for parsing errors
                return {
                    "reasoning": f"Error parsing response. Raw output: {content[:200]}...",
                    "answer": self._extract_answer_from_text(content),
                }
        else:
            # For standard mode without structured output
            return {
                "reasoning": f"Answer generated based on {len(nodes)} relevant entities, {len(paths)} knowledge paths, and {len(top_chunks)} document chunks.",
                "answer": content,
            }

    def query(self, question: str) -> Any:
        """
        Process a query and return relevant information.

        Args:
            question: User question

        Returns
        -------
            Dictionary with answer and reasoning
        """
        if self.verbose:
            print(f"Processing query: {question}")

        # Get query embedding
        query_embedding = np.array(
            self.embedding_handler.embedder.embed_query(question)
        )

        # Find similar nodes in the graph
        similar_nodes = self._get_similar_nodes(query_embedding)

        if self.verbose:
            print(f"Found {len(similar_nodes)} similar nodes")
            for node, score in similar_nodes[:5]:
                print(f"  {node}: {score:.3f}")

        if not similar_nodes:
            return {
                "reasoning": "No relevant entities found in the knowledge graph.",
                "answer": "I couldn't find relevant information to answer this question.",
            }

        # Extract node entities
        nodes = [node for node, _ in similar_nodes]

        # Get relevant subgraph
        subgraph = self._get_subgraph(nodes, max_hops=1)

        if self.verbose:
            print(
                f"Generated subgraph with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges"
            )

        # Get relevant path descriptions
        paths = self._get_path_descriptions(subgraph)

        if self.verbose:
            print(f"Generated {len(paths)} path descriptions")
            for path in paths[:5]:
                print(f"  {path}")

        # Get relevant chunk indices
        chunk_indices = self._get_chunks_for_nodes(nodes)

        if self.verbose:
            print(f"Found {len(chunk_indices)} relevant chunks")

        # Get actual document chunks
        context_chunks = [
            self.document_chunks[idx]
            for idx in chunk_indices
            if idx < len(self.document_chunks)
        ]

        # Score chunks based on node information
        chunk_scores = self._score_chunks_by_nodes(context_chunks, similar_nodes)

        # Sort by score (descending)
        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        # Get top-k chunks
        top_chunks = [chunk for chunk, _ in chunk_scores[: self.top_k_chunks]]

        if self.verbose:
            print(f"Selected top {len(top_chunks)} chunks")

        if not top_chunks:
            return {
                "reasoning": "No relevant document chunks found.",
                "answer": "I couldn't find relevant information to answer this question.",
            }

        # Format context
        context = self._format_context(paths, top_chunks)

        # Create prompt using create_query_prompt
        messages = create_query_prompt(
            question=question,
            context=context,
            system_type="entity",  # Using entity-based system type
            use_cot=self.use_cot,
            numerical_answer=self.numerical_answer,
        )

        # Generate answer using LLM
        response = self.llm.invoke(messages)

        return self._process_results(response, nodes, paths, top_chunks)
