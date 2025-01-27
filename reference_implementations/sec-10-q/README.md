# SEC 10-Q Reference Implementation Overview

## Introduction
This folder contains a collection of Jupyter notebooks, Python scripts, and resources designed to demonstrate various approaches to question answering using SEC 10-Q filings. This includes baseline RAG implementations, knowledge graph construction, and document processing pipelines.

## Prerequisites
Before you dive into the materials, ensure you have the following prerequisites:
- Python 3.10 or higher
- Poetry for dependency management
- An OpenAI API key
- A Neo4j database instance (for knowledge graph notebooks)
- Tesseract and Poppler (for PDF processing)

## Notebooks
Here you will find the following Jupyter notebooks:
1.  **`baseline-rag-cot.ipynb`**: This notebook implements a Retrieval-Augmented Generation (RAG) system with chain of thought reasoning for answering questions about SEC 10-Q filings. It includes document loading, chunking, embedding generation, and querying functionalities, as well as evaluation code.
2.  **`baseline-rag.ipynb`**: This notebook implements a basic RAG system for answering questions about SEC 10-Q filings. It includes document loading, chunking, embedding generation, and querying functionalities, as well as evaluation code.
3.  **`kg-rag-cypher.ipynb`**: This notebook demonstrates constructing a knowledge graph from SEC 10-Q filings using LangChain and Neo4j. It extracts entities and relationships using an LLM, stores them in Neo4j, and then uses Cypher queries to answer questions. It also includes evaluation code.
4.  **`kg-rag-entity.ipynb`**: This notebook demonstrates constructing a knowledge graph from SEC 10-Q filings using LangChain and Networkx. It extracts entities and relationships using an LLM and stores them in a Networkx graph. It then uses a GraphQAChain to answer questions. It also includes evaluation code.

## Code
This section includes code files that demonstrate:
-   **`pdf_processor.py`**: This script provides a class `PDFProcessor` that extracts text from PDF files using either `pdftotext` or OCR with `tesseract`. It also includes a function `process_pdf_directory` to process all PDFs in a directory and save the extracted text to text files.
-   **`visualizer.py`**: This script provides a class `GraphVisualizer` that visualizes Neo4j graphs using NetworkX and Matplotlib. It includes methods to get graph statistics and visualize subgraphs.

## Resources
For further reading and additional studies, consider the following resources:
- LangChain documentation: [https://python.langchain.com/docs/](https://python.langchain.com/docs/)
- Neo4j documentation: [https://neo4j.com/docs/](https://neo4j.com/docs/)
- Tesseract documentation: [https://tesseract-ocr.github.io/tessdoc/](https://tesseract-ocr.github.io/tessdoc/)
- Poppler documentation: [https://poppler.freedesktop.org/](https://poppler.freedesktop.org/)

## Getting Started
To get started with the materials in this topic:
1. Ensure you have installed all the required dependencies using `poetry install`.
2. Set up your environment variables, including your OpenAI API key and Neo4j connection details.
3. Explore the notebooks and scripts to understand the different approaches to question answering with SEC 10-Q filings.
4. Run the notebooks and scripts to reproduce the results and experiment with different parameters.
