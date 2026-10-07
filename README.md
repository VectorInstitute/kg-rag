# KG-RAG: knowledge graph retrieval augmented generation experiments

> NOTE (2026-10-07): This repository is archived. It is the companion code to the Vector Institute post
> [Enhancing RAG with knowledge graphs](https://vectorinstitute.ai/enhancing-rag-with-knowledge-graphs/)
> and is kept for reference. It is not maintained, and issues and pull requests are closed.

This repository holds exploratory, bare-bones implementations of three knowledge graph RAG methods and
two baselines, written to run the experiments in the post on SEC 10-Q filings. The code is a small
Python package with command-line scripts. It is not a framework: there is no stable API, no test
suite, and no packaging for reuse, and the dependencies are the versions locked in `uv.lock`. Expect to
read and adapt the scripts rather than install the package as a library.

## Methods

Baselines:

- Standard RAG: retrieval by vector similarity over document chunks.
- Chain-of-thought RAG: the same retrieval with explicit reasoning steps in the prompt.

Knowledge graph methods:

- Entity-based: matches query entities to graph entities by embedding, then runs a beam search over
  the graph to collect the chunks to answer from. This is the method the post evaluates.
- Cypher-based: an LLM writes Cypher queries against a Neo4j graph built from the documents.
- GraphRAG-based: community detection over the entity graph with a hierarchical search, after the
  GraphRAG design.

## Contents

- `kg_rag/`: the package, with one module per method under `methods/` and the evaluation scripts
  under `evaluation/`.
- `scripts/`: builds the vector store and the graphs, and runs each method interactively.
- `data/sec-10-q/`: the SEC 10-Q documents and the question sets used in the post.
- `blog/`: the interactive version of the post, served from GitHub Pages at
  [vectorinstitute.github.io/kg-rag](https://vectorinstitute.github.io/kg-rag/).
- `literature_review.md`: the survey of RAG evaluation papers and datasets written during the work.

## Installation

### Using uv

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone https://github.com/VectorInstitute/kg-rag.git
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

The repository uses pre-commit hooks for formatting and linting:

```bash
# Run pre-commit hooks on all files
pre-commit run --all-files
```
