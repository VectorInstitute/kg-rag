# Entity-Based Knowledge Graph RAG

<p align="center">
  <img src="../../../assets/entity.png" />
</p>

## Overview

The Entity-Based Knowledge Graph RAG (KG-RAG) method implements a retrieval-augmented generation approach that leverages knowledge graphs to improve context retrieval for question answering tasks. This approach stands apart from traditional vector-based RAG methods by incorporating structured relationship information between entities to provide more precise and relevant context for answers.

## How It Works

The entity-based method follows a six-stage pipeline as illustrated in the workflow diagram:

1. **Query Embedding**
2. **Entity Similarity Matching**
3. **Subgraph Exploration**
4. **Chunk Selection**
5. **Context Assembly**
6. **LLM Generation**

### 1. Query Embedding

The input query is first converted into a dense vector representation using an embedding model (typically OpenAI's `text-embedding-3-small`).

```python
query_embedding = np.array(self.embedding_handler.embedder.embed_query(question))
```

This initial embedding captures the semantic meaning of the user's question and serves as the foundation for identifying relevant entities in the knowledge graph.

### 2. Entity Similarity Matching

The query embedding is compared against pre-computed embeddings of all entities in the knowledge graph using cosine similarity. Entities that exceed a configurable similarity threshold are selected as the initial set of relevant nodes.

```python
def _get_similar_nodes(self, query_embedding: np.ndarray) -> list[tuple[str, float]]:
    similarities = []
    for node, embedding in self.embedding_handler.entity_embeddings.items():
        similarity = self.embedding_handler.compute_similarity(query_embedding, embedding)
        if similarity >= self.similarity_threshold:
            similarities.append((node, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:self.top_k_nodes]
```

Each entity is scored based on its semantic similarity to the query, and the top-k entities (controlled by the `top_k_nodes` parameter) are selected for further exploration.

### 3. Subgraph Exploration

Starting from the identified high-similarity entities, the system explores the local neighborhood in the knowledge graph to uncover related entities and relationships. This exploration creates a query-specific subgraph that captures the relevant knowledge context.

```python
def _get_subgraph(self, nodes: list[str], max_hops: int = 1) -> nx.DiGraph:
    if not nodes:
        return nx.DiGraph()

    nodes_to_include = set(nodes)
    for _ in range(max_hops):
        new_nodes = set()
        for node in nodes_to_include:
            if node in self.graph:
                new_nodes.update(self.graph.predecessors(node))
                new_nodes.update(self.graph.successors(node))
        nodes_to_include.update(new_nodes)

    return self.graph.subgraph(nodes_to_include)
```

The subgraph exploration discovers important entity relationships that might not be directly captured by simple embedding similarity, enabling multi-hop reasoning.

### 4. Chunk Selection

The system maintains an entity-to-chunk mapping that connects graph entities to the document chunks where they are mentioned. Using this mapping, the system identifies document chunks that contain the entities in our subgraph.

```python
def _get_chunks_for_nodes(self, nodes: list[str]) -> list[int]:
    chunk_indices = set()
    for node in nodes:
        if node in self.entity_chunk_map:
            chunk_indices.update(self.entity_chunk_map[node])
    return list(chunk_indices)
```

These chunks are then scored using a weighted combination of:
1. **Node frequency score**: What fraction of relevant entities are mentioned in the chunk
2. **Node similarity score**: Average similarity of entities mentioned in the chunk to the query

```python
def _score_chunks_by_nodes(self, chunks: list[Document], node_info: list[tuple[str, float]]) -> list[tuple[Document, float]]:
    node_scores = dict(node_info)
    nodes = list(node_scores.keys())

    chunk_to_nodes: dict[int, set[str]] = {id(chunk): set() for chunk in chunks}

    # Map chunks to nodes
    for _, chunk in enumerate(chunks):
        chunk_id = id(chunk)
        orig_idx = None
        for i, orig_chunk in enumerate(self.document_chunks):
            if chunk is orig_chunk:
                orig_idx = i
                break

        if orig_idx is not None:
            for node in nodes:
                if (node in self.entity_chunk_map and
                    orig_idx in self.entity_chunk_map[node]):
                    chunk_to_nodes[chunk_id].add(node)

    # Score each chunk
    chunk_scores = []
    for chunk in chunks:
        chunk_id = id(chunk)
        referencing_nodes = chunk_to_nodes[chunk_id]

        if not referencing_nodes:
            chunk_scores.append((chunk, 0.0))
            continue

        # Calculate frequency score
        freq_score = len(referencing_nodes) / len(nodes) if nodes else 0

        # Calculate similarity score
        sim_score = sum(node_scores.get(node, 0) for node in referencing_nodes) / len(referencing_nodes)

        # Combined weighted score
        combined_score = (self.node_freq_weight * freq_score) + (self.node_sim_weight * sim_score)

        chunk_scores.append((chunk, combined_score))

    return chunk_scores
```

The top-k chunks with the highest scores are selected as the most relevant context for answering the query.

### 5. Context Assembly

The system assembles a multi-faceted context that includes both knowledge graph information (in the form of path descriptions) and the selected document chunks.

```python
def _format_context(self, paths: list[str], chunks: list[Document]) -> str:
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
            content = chunk.page_content
            source = chunk.metadata.get("source", chunk.metadata.get("file_path", "unknown"))
            page = chunk.metadata.get("page", "")

            source_info = f"[Source: {source}"
            if page:
                source_info += f", Page: {page}"
            source_info += "]"

            context_lines.append(f"Chunk {i + 1}: {source_info}")
            context_lines.append(content)
            context_lines.append("---")

    return "\n".join(context_lines)
```

This hybridized context provides both the structured knowledge from the graph (showing how entities relate to each other) and the textual information from the document chunks.

### 6. LLM Generation

Finally, the assembled context is provided to a large language model (typically ChatGPT/GPT-4) through a carefully designed prompt that instructs the model to use both the knowledge graph information and document chunks to answer the query.

```python
def query(self, question: str) -> Any:
    # ... (steps 1-5 as described above)

    # Format context
    context = self._format_context(paths, top_chunks)

    # Create prompt
    messages = create_query_prompt(
        question=question,
        context=context,
        system_type="entity",
        use_cot=self.use_cot,
        numerical_answer=self.numerical_answer,
    )

    # Generate answer using LLM
    response = self.llm.invoke(messages)

    return self._process_results(response, nodes, paths, top_chunks)
```

The response can be generated in different formats, including standard text, Chain-of-Thought (CoT) reasoning, or specifically formatted numerical answers, depending on the configuration.

## Key Components

### EmbeddingHandler

The `EmbeddingHandler` class manages embedding generation and caching for entities and relationships in the knowledge graph. It supports batched embedding generation with retry logic for API failures and maintains a persistent cache to avoid regenerating embeddings.

### EntityBasedKGRAG

The `EntityBasedKGRAG` class is the main implementation of the entity-based approach. It orchestrates the entire workflow from query embedding to answer generation and provides configuration options for tuning the system's behavior.

## Configuration Parameters

The entity-based KG-RAG system can be configured with several parameters:

* `top_k_nodes`: Number of most similar entities to consider (default: 10)
* `top_k_chunks`: Number of top-scoring document chunks to include in context (default: 5)
* `similarity_threshold`: Minimum similarity score for considering an entity relevant (default: 0.7)
* `node_freq_weight`: Weight for entity frequency in chunk scoring (default: 0.4)
* `node_sim_weight`: Weight for entity similarity in chunk scoring (default: 0.6)
* `use_cot`: Whether to use Chain-of-Thought prompting (default: False)
* `numerical_answer`: Whether to format answers as numerical values only (default: False)

## Advantages of the Entity-Based Approach

1. **Structure-aware retrieval**: Unlike pure vector-based methods, the entity-based approach leverages the structured relationships in the knowledge graph, allowing it to capture complex connections between entities.

2. **Multi-hop reasoning**: By exploring the subgraph around high-similarity entities, the system can discover relevant information that might be multiple hops away from the initial entities.

3. **Entity-grounded context**: The scoring of document chunks based on entity mentions ensures that the retrieved context is grounded in entities relevant to the query.

4. **Hybrid context assembly**: By combining knowledge graph paths with document chunks, the system provides the LLM with both structured relational information and detailed textual content.

5. **Configurable system behavior**: The various parameters allow tuning the system to optimize for different types of queries and knowledge domains.

## Usage Example

```python
from langchain_openai import ChatOpenAI
from kg_rag.methods.entity_based.kg_rag import EntityBasedKGRAG
from kg_rag.utils.document_loader import load_documents, load_graph_documents
from kg_rag.utils.graph_utils import create_graph_from_graph_documents

# Load documents and graph
documents = load_documents(
    directory_path="data/docs",
    pickle_path="data/graphs/documents.pkl",
)
graph_documents = load_graph_documents("data/graphs/graph_documents.pkl")
graph = create_graph_from_graph_documents(graph_documents)

# Initialize LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# Create KG-RAG system
kg_rag = EntityBasedKGRAG(
    graph=graph,
    graph_documents=graph_documents,
    document_chunks=documents,
    llm=llm,
    top_k_nodes=10,
    top_k_chunks=5,
    similarity_threshold=0.7,
    node_freq_weight=0.4,
    node_sim_weight=0.6,
    use_cot=True,
    numerical_answer=False,
    verbose=True,
)

# Query the system
result = kg_rag.query("What was Apple's revenue in Q3 2023?")
print(f"Answer: {result['answer']}")
print(f"Reasoning: {result['reasoning']}")
```

## Performance Considerations

- **Embedding Cache**: To improve performance, embeddings are cached to avoid regenerating them for the same entities.
- **Batch Processing**: Entity embeddings are generated in batches to optimize API usage.
- **Selective Subgraph Exploration**: The exploration is limited to a configurable number of hops to prevent excessive graph traversal.
- **Weighted Scoring**: The weighted scoring mechanism balances entity frequency and similarity for optimal chunk selection.
