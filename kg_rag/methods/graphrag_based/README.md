# GraphRAG-based KG-RAG

<p align="center">
  <img src="../../../assets/graphrag.png" />
</p>

## Overview

GraphRAG is a knowledge graph-based retrieval augmented generation approach that combines the power of graph-structured knowledge with embedding-based retrieval and community detection algorithms to provide more relevant and comprehensive context for question answering.

The GraphRAG-based KG-RAG implementation in this repository builds upon the methodology described in the LangChain GraphRAG library, extending it with additional capabilities for handling specialized types of queries and integrating with the broader KG-RAG framework.

## Architecture

The architecture of the GraphRAG-based KG-RAG system consists of several key components:

1. **Indexing**: Transform documents into graph structures with nodes, edges, and community detection
2. **Query Processing**: Process user queries to find relevant information
3. **Search Strategies**: Two approaches (Local and Global) for searching the graph
4. **Context Building**: Assembling relevant context from the graph
5. **Response Generation**: Using an LLM to provide the final answer

## Workflow

The workflow of the GraphRAG-based KG-RAG method follows these stages:

![GraphRAG Workflow](../assets/graphrag_workflow.png)

### 1. Query Processing

When a user submits a query, the system begins by processing it to identify:
- Relevant entities
- Related text units
- Important relationships

```python
# The query is processed to find similar entities in the vector store
similar_entities = entities_vector_store.similarity_search(query, k=top_k)
```

### 2. Entity and Relationship Selection

The system selects the most relevant entities and relationships based on:
- Embedding similarity to the query
- Relevance scores
- Entity relationships

### 3. Community Detection

A critical aspect of GraphRAG is its use of community detection algorithms to identify densely connected subgraphs that represent coherent knowledge areas:

```python
# Community detection is performed during indexing
self.community_detector = HierarchicalLeidenCommunityDetector()
```

The community detection process:
1. Creates hierarchical communities at different levels of granularity
2. Enables more focused retrieval based on knowledge clusters
3. Improves the relevance of retrieved information

Communities are detected at multiple levels:
- Level 1: Fine-grained communities with highly specific information
- Level 2: Medium-sized communities with related concepts
- Level 3: Broader communities encompassing larger knowledge domains

### 4. Search Strategies

GraphRAG implements two distinct search strategies:

#### Local Search

Local search focuses on finding the most relevant information within the immediate neighborhood of matched entities:

```python
# Local search implementation
self.local_search = LocalSearchStrategy(
    llm=self.llm,
    entities_vector_store=entities_vector_store,
    artifacts=artifacts,
    community_level=community_level,
    show_references=show_references,
    verbose=verbose,
    system_prompt=system_prompt,
    strict_output_format=(use_cot or numerical_answer),
)
```

The local search process:
1. Finds the most similar entities to the query
2. Retrieves the communities that contain these entities
3. Explores the local neighborhood within these communities
4. Ranks and selects the most relevant context pieces

#### Global Search

Global search takes a broader approach by considering information across the entire graph:

```python
# Global search implementation
self.global_search = GlobalSearchStrategy(
    llm=self.llm,
    artifacts=artifacts,
    community_level=community_level,
    token_counter=self.token_counter,
    show_references=show_references,
    verbose=verbose,
    strict_output_format=(use_cot or numerical_answer),
)
```

The global search process:
1. Identifies relevant communities across the entire graph
2. Generates key points from each community
3. Aggregates key points from all communities
4. Creates a comprehensive context that captures broader relationships

### 5. Context Building

Once relevant information is retrieved, the system builds a context that:
- Includes relevant text passages
- Highlights entity relationships
- Provides community information
- Organizes information in a structured way

The context builder optimizes for:
- Relevance to the query
- Information coherence
- Context length constraints
- Diversity of information

```python
# Context building
self.context_builder = ContextBuilder.build_default(
    token_counter=TiktokenCounter(),
)
```

### 6. Response Generation

Finally, the prepared context is sent to a large language model (LLM) for answer generation:

```python
# Generate the final answer
response = self.llm.invoke(messages)
```

The LLM process:
- Reasons over the provided context
- Extracts relevant information
- Generates a coherent answer
- Can provide references to support the answer

## Key Components

### IndexerArtifacts

The system relies on a set of indexed artifacts that include:
- Entity information and embeddings
- Relationship data
- Community structure at different levels
- Document text units and their mappings

```python
# Artifacts structure generated during indexing
artifacts = {
    "entities": {...},         # Entity data
    "relationships": {...},    # Relationship data
    "communities": {...},      # Community data at multiple levels
    "text_units": {...}        # Document text units
}
```

### Search Strategies Implementation

The two search strategies are implemented as separate classes:

#### LocalSearchStrategy

```python
class LocalSearchStrategy:
    """Local search strategy for GraphRAG."""

    def __init__(
        self,
        llm: ChatOpenAI,
        entities_vector_store: Chroma,
        artifacts: IndexerArtifacts,
        community_level: int = 2,
        show_references: bool = True,
        verbose: bool = False,
        system_prompt: str | None = None,
        strict_output_format: bool = False,
    ):
        ...

    def search(self, query: str) -> Any:
        """Perform local search."""
        ...
```

#### GlobalSearchStrategy

```python
class GlobalSearchStrategy:
    """Global search strategy for GraphRAG."""

    def __init__(
        self,
        llm: ChatOpenAI,
        artifacts: IndexerArtifacts,
        community_level: int = 2,
        token_counter: TiktokenCounter | None = None,
        show_references: bool = True,
        verbose: bool = False,
        strict_output_format: bool = False,
    ):
        ...

    def search(self, query: str) -> str:
        """Perform global search."""
        ...
```

## Advanced Features

### Chain-of-Thought Reasoning

The system supports Chain-of-Thought (CoT) reasoning for more complex queries:

```python
# Enable Chain-of-Thought reasoning
if use_cot or numerical_answer:
    self.llm = self.llm.bind(response_format={"type": "json_object"})
```

When CoT is enabled:
1. The system instructs the LLM to break down reasoning into explicit steps
2. The output is structured as JSON with separate "reasoning" and "answer" fields
3. This improves answer accuracy for complex questions

### Numerical Answer Extraction

For questions requiring numerical answers, the system includes specialized processing:

```python
def _extract_number_from_text(self, text: str) -> str | None:
    """Extract number from text."""
    patterns = [
        r"answer\s*(?:is|:)\s*(-?\d+(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s*%",
        r"(-?\d+(?:\.\d+)?)\s*(?:million|billion|dollars|USD)",
        r"(?:value|amount|total)\s*(?:of|is|:)\s*(-?\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

    return None
```

## Configuration Parameters

The system can be configured with several parameters:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `search_strategy` | Which search strategy to use ("local" or "global") | "local" |
| `community_level` | Level of community granularity (1-3) | 2 |
| `use_cot` | Whether to use Chain-of-Thought reasoning | False |
| `numerical_answer` | Whether to extract numerical answers | False |
| `llm_model` | LLM model to use | "gpt-4o" |

## Usage Example

Here's an example of how to use the GraphRAG-based KG-RAG system:

```python
from kg_rag.methods.graphrag_based.kg_rag import create_graphrag_system

# Create the GraphRAG system
graphrag = create_graphrag_system(
    artifacts_path="data/graphrag_artifacts.pkl",
    vector_store_dir="vector_stores",
    llm_model="gpt-4o",
    search_strategy="local",
    community_level=2,
    use_cot=True,
    numerical_answer=False,
    verbose=True
)

# Query the system
result = graphrag.query("What was Apple's revenue in Q3 2023?")
print(result["answer"])
```

## Advantages of GraphRAG-based KG-RAG

The GraphRAG-based approach offers several advantages:

1. **Structural Awareness**: By leveraging the graph structure, the system can follow relationship paths to find information that might not be directly matched by embeddings.

2. **Community Context**: The community detection helps to preserve the topical coherence of information, reducing irrelevant context.

3. **Hierarchical Search**: Different levels of community granularity allow for both focused and broad information retrieval.

4. **Dual Search Strategies**: The local and global search strategies provide flexibility in how information is retrieved.

5. **Balance of Precision and Recall**: The approach balances finding specific information (precision) with comprehensive coverage (recall).

## Limitations and Considerations

Despite its advantages, the GraphRAG approach has some limitations:

1. **Indexing Complexity**: The indexing process is more complex and computationally intensive than standard vector store approaches.

2. **Dependency on Quality Graph Construction**: The effectiveness depends on the quality of the knowledge graph construction.

3. **Cold Start Challenge**: Requires an initial indexing phase that may be time-consuming.

4. **Parameter Sensitivity**: Performance can be sensitive to parameters like community level and search strategy.

## References

[- LangChain GraphRAG Documentation](https://github.com/ksachdeva/langchain-graphrag/)

[- Microsoft Graphrag Documentation](https://microsoft.github.io/graphrag/)
