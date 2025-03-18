# KG-RAG: Knowledge Graph-based Retrieval Augmented Generation

This repository contains a collection of implementations for Knowledge Graph-based RAG (Retrieval Augmented Generation) approaches and baseline methods for comparison. The code is structured as a Python package with modular components.

## Overview

The repository implements several RAG approaches:

1. **Baseline approaches**:
   - **Standard RAG**: Traditional retrieval-based approach using vector similarity
   - **Chain-of-Thought RAG**: Enhanced retrieval with explicit reasoning steps

2. **KG-RAG approaches**:
   - **Entity-based approach**: Uses embedding-based entity matching and beam search to find relevant information in the knowledge graph
   - **Cypher-based approach**: Uses Cypher queries to retrieve information from a Neo4j graph database
   - **GraphRAG-based approach**: Implements a community detection and hierarchical search strategy

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kg-rag.git
cd kg-rag

# Install dependencies
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
```

For the Cypher-based approach, also add:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## Usage

### 1. Building Vector Store for Baseline Methods

First, build a vector store for the baseline RAG methods:

```bash
python -m scripts.build_baseline_vectordb \
    --docs-dir data/sec-10-q/docs \
    --collection-name sec_10q \
    --persist-dir chroma_db \
    --verbose
```

### 2. Building Knowledge Graphs

Build a knowledge graph for KG-RAG methods:

```bash
python -m scripts.build_entity_graph \
    --docs-dir data/sec-10-q/docs \
    --output-dir data/graphs \
    --graph-name sec10q_entity_graph \
    --verbose
```

### 3. Running Interactive Query Mode

To interactively query using baseline methods:

```bash
python -m scripts.run_baseline_rag \
    --collection-name sec_10q \
    --persist-dir chroma_db \
    --model gpt-4o \
    --verbose
```

To interactively query using KG-RAG methods:

```bash
python -m scripts.run_entity_rag \
    --graph-path data/graphs/sec10q_entity_graph.pkl \
    --beam-width 10 \
    --max-depth 8 \
    --top-k 100 \
    --verbose
```

### 4. Running Evaluation

To evaluate the performance of various RAG methods on a test dataset:

```bash
python -m kg_rag.evaluation.run_evaluation \
    --data-path data/test_questions.csv \
    --graph-path data/graphs/sec10q_entity_graph.pkl \
    --method all \
    --output-dir evaluation_results \
    --collection-name sec_10q \
    --persist-dir chroma_db \
    --max-samples 50 \
    --verbose
```

To evaluate a specific method:

```bash
python -m kg_rag.evaluation.run_evaluation \
    --data-path data/test_questions.csv \
    --method baseline-cot \
    --output-dir evaluation_results \
    --collection-name sec_10q \
    --persist-dir chroma_db \
    --verbose
```

### 5. Running Hyperparameter Search

To find the optimal hyperparameters for a method:

```bash
python -m kg_rag.evaluation.hyperparameter_search \
    --data-path data/test_questions.csv \
    --graph-path data/graphs/sec10q_entity_graph.pkl \
    --method entity \
    --configs-path kg_rag/evaluation/hyperparameter_configs.json \
    --output-dir hyperparameter_search \
    --max-samples 10 \
    --verbose
```

## Implemented Approaches

### Baseline RAG Methods

#### Standard RAG

The standard baseline RAG approach:

1. Indexes documents in a vector database (ChromaDB)
2. Retrieves relevant chunks based on vector similarity
3. Generates answers based on the retrieved context

#### Chain-of-Thought RAG

The Chain-of-Thought (CoT) baseline RAG approach:

1. Enhances the standard approach with explicit reasoning steps
2. Uses a structured prompt to guide the LLM through systematic reasoning
3. Returns both the final answer and the reasoning process

### Entity-based KG-RAG

The entity-based approach works as follows:

1. Uses an LLM to extract candidate entity-relationship chains from the query
2. Matches entities using embedding similarity
3. Performs beam search to find paths in the knowledge graph matching the extracted pattern
4. Uses the discovered paths to generate an answer

### Cypher-based KG-RAG

The Cypher-based approach:

1. Stores the knowledge graph in Neo4j
2. Uses an LLM to generate Cypher queries from natural language questions
3. Executes the queries against the graph database
4. Formats the results and generates an answer

### GraphRAG-based KG-RAG

The GraphRAG-based approach:

1. Creates a hierarchical community structure from the knowledge graph
2. Uses both local and global search strategies
3. Generates community reports and knowledge summaries
4. Uses the structured information to answer queries

## Programmatic Example

```python
from kg_rag.utils.graph_utils import load_graph
from kg_rag.methods.entity_based.kg_rag import EntityBasedKGRAG
from langchain_openai import ChatOpenAI

# Load pre-built graph
graph = load_graph("data/graphs/sec10q_entity_graph.pkl")

# Initialize LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
llm = llm.bind(response_format={"type": "json_object"})

# Create KG-RAG system
kg_rag = EntityBasedKGRAG(
    graph=graph,
    llm=llm,
    beam_width=10,
    max_depth=8,
    top_k=100,
    verbose=True
)

# Query the system
result = kg_rag.query(
    "What was the Products gross margin percentage for Apple for the quarter ended July 1, 2023?"
)

print(f"Answer: {result['answer']}")
print(f"Reasoning: {result['reasoning']}")
```

## Project Structure

```
kg_rag/
├── utils/                       # Common utilities
│   ├── document_loader.py        # Document loading utilities
│   ├── embedding_handler.py      # Embedding generation and handling
│   ├── graph_utils.py            # Common graph operations
│   └── evaluator.py              # Evaluation framework
│
├── methods/                     # RAG implementations
│   ├── baseline_rag/             # Baseline RAG approaches
│   │   ├── document_processor.py  # Document processing for vector store
│   │   ├── embedder.py            # Embedding generation
│   │   ├── kg_rag.py              # Standard RAG implementation
│   │   ├── kg_rag_cot.py          # Chain-of-Thought RAG implementation
│   │   └── vector_store.py        # Vector store management
│   │
│   ├── entity_based/             # Entity-based approach
│   ├── cypher_based/             # Cypher-based approach
│   └── graphrag_based/           # GraphRAG-based approach
│
├── evaluation/                  # Evaluation tools
│   ├── hyperparameter_configs.json  # Hyperparameter configurations
│   ├── hyperparameter_search.py     # Hyperparameter search
│   └── run_evaluation.py            # Run evaluation across methods
│
├── scripts/                     # Runnable scripts
│   ├── build_baseline_vectordb.py  # Build vector store for baseline RAG
│   ├── build_entity_graph.py       # Build entity-based graph
│   ├── build_cypher_graph.py       # Build Cypher graph
│   ├── build_graphrag.py           # Build GraphRAG artifacts
│   ├── run_baseline_rag.py         # Run baseline RAG
│   ├── run_entity_rag.py           # Run entity-based RAG
│   ├── run_cypher_rag.py           # Run Cypher-based RAG
│   └── generate_synthetic_data.py  # Generate synthetic evaluation data
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.