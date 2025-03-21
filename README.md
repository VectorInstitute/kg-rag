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

### Using uv (Recommended)

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone https://github.com/yourusername/kg-rag.git
cd kg-rag

# Install uv if you don't have it
curl -sSf https://astral.sh/uv/install.sh | bash

uv sync
source .venv/bin/activate
```

For development, you can install the dev dependencies:

```bash
uv sync --dev
source .venv/bin/activate
```


## Environment Variables

Export the following environment variables:

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

## Development

### Pre-commit hooks

This project uses pre-commit hooks to ensure code quality:

```bash
# Run pre-commit hooks on all files
pre-commit run --all-files
```

### Running tests

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=kg_rag tests/
```
