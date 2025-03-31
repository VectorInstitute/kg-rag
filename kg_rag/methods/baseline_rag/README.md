# Baseline RAG Method

<p align="center">
  <img src="../../../assets/baseline.png" />
</p>

## Overview

The Baseline RAG (Retrieval Augmented Generation) method implements a standard vector-based retrieval approach that enhances LLM responses with relevant context from a document collection. This implementation follows a five-stage pipeline:

1. **Query Embedding**: Convert user queries to vector embeddings
2. **Chunk Similarity Matching**: Find relevant document chunks using vector similarity
3. **Chunk Selection**: Choose the most relevant chunks based on similarity scores
4. **Context Assembly**: Format selected chunks into a unified context
5. **LLM Generation**: Generate an answer using the query and assembled context

## Architecture

The baseline RAG system is composed of three main components:

- **Document Processor**: Handles document loading, chunking, and metadata extraction
- **Embedding Handler**: Manages vector embeddings creation and similarity calculations
- **Vector Store**: Stores and retrieves document chunks and their embeddings

## Implementation Details

### 1. Document Processing (`DocumentProcessor` class)

The Document Processor handles the initial ingestion and preparation of documents:

```python
class DocumentProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 24, verbose: bool = False):
        self.text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.verbose = verbose
```

Key functionality:
- Loads PDF documents using PyPDFLoader
- Splits documents into manageable chunks with configurable size and overlap
- Preserves metadata (filename, file path, page numbers) for each chunk
- Filters complex metadata that could cause storage issues

### 2. Embedding Generation (`OpenAIEmbedding` class)

The embedding handler creates and manages vector representations:

```python
class OpenAIEmbedding:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small",
                 batch_size: int = 100, max_tokens_per_batch: int = 8000,
                 rate_limit_pause: float = 60.0, verbose: bool = False):
        # Initialize with OpenAI client and configuration
```

Key functionality:
- Uses OpenAI embedding models (default: text-embedding-3-small)
- Handles batching to optimize API calls and respect rate limits
- Provides efficient token counting via tiktoken
- Includes error handling and retry mechanisms

### 3. Vector Storage (`ChromaDBManager` class)

The vector store manages persistent storage and retrieval:

```python
class ChromaDBManager:
    def __init__(self, collection_name: str, persist_directory: str = "chroma_db",
                 batch_size: int = 100, verbose: bool = False):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        # Additional initialization...
```

Key functionality:
- Uses ChromaDB as the underlying vector database
- Supports persistent storage for embeddings and documents
- Provides batch adding of documents and efficient querying
- Maintains metadata alongside embeddings

### 4. Main RAG Implementation (`BaselineRAG` class)

The main class orchestrates the entire RAG pipeline:

```python
class BaselineRAG:
    def __init__(self, collection_name: str = "document_collection",
                 chroma_persist_dir: str = "chroma_db", model_name: str = "gpt-4o",
                 embedding_model: str = "text-embedding-3-small", top_k: int = 5,
                 use_cot: bool = False, numerical_answer: bool = False,
                 verbose: bool = False):
        # Initialize components and settings
```


## Configuration Options

The baseline RAG system supports several configuration options:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `collection_name` | Name of the ChromaDB collection | "document_collection" |
| `chroma_persist_dir` | Directory to store ChromaDB files | "chroma_db" |
| `model_name` | Name of the OpenAI model for generation | "gpt-4o" |
| `embedding_model` | Name of the OpenAI model for embeddings | "text-embedding-3-small" |
| `top_k` | Number of top results to retrieve | 5 |
| `use_cot` | Whether to use Chain-of-Thought prompting | False |
| `numerical_answer` | Whether to format answers as numerical values | False |
| `verbose` | Whether to print verbose output | False |

## Prompt Templates

The system uses specialized prompts for different answer types:

**Standard Prompt:**
```
You are a helpful assistant that answers queries about SEC 10-Q filings using the blocks of context provided.
Just follow the instructions from the question exactly and use the context to provide accurate information.
The current date is {date}.
```

**Chain-of-Thought Extension:**
```
Follow these steps carefully:
1. PLAN: Break down what information you need to find in the context
2. SEARCH: Locate the relevant information in the provided context only
3. CALCULATE: If needed, perform any calculations step by step
4. VERIFY: Double-check your work and ensure your answer matches the question
5. FORMAT: Format your answer appropriately based on the question

Response should be formatted as valid JSON with the following structure:
{
    "reasoning": "Your detailed step-by-step analysis...",
    "answer": "your final answer"
}
```

## Example Usage

### Building the Vector Store:
```bash
python -m scripts.build_baseline_vectordb \
    --docs-dir data/sec-10-q/docs \
    --collection-name sec_10q \
    --persist-dir chroma_db \
    --verbose
```

### Running Queries:
```bash
python -m scripts.run_baseline_rag \
    --collection-name sec_10q \
    --persist-dir chroma_db \
    --model gpt-4o \
    --top-k 5 \
    --verbose
```

## Evaluation Methods

The baseline RAG system can be evaluated using the included evaluation tools:

```bash
python -m kg_rag.evaluation.run_evaluation \
    --data-path data/test_questions.csv \
    --method baseline \
    --config-path kg_rag/configs/baseline-kgrag.json \
    --output-dir evaluation_results \
    --numerical-answer \
    --use-cot
```

## References

- ChromaDB documentation: [https://docs.trychroma.com/](https://docs.trychroma.com/)
- OpenAI Embeddings API: [https://platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings)
- LangChain Text Splitters: [https://python.langchain.com/docs/how_to/recursive_text_splitter/](https://python.langchain.com/docs/how_to/recursive_text_splitter/)
