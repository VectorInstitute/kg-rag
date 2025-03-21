"""Document loading and processing utilities for KG-RAG approaches."""

import os
import pickle
from typing import Any

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter


class DocumentLoader:
    """Loads and processes documents for knowledge graph construction."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 24,
        file_extension: str = ".pdf",
    ):
        """
        Initialize the document loader.

        Args:
            chunk_size: Size of chunks for text splitting
            chunk_overlap: Overlap between chunks
            file_extension: File extension to filter for
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.file_extension = file_extension
        self.text_splitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def load_document(self, file_path: str) -> list[Document]:
        """
        Load and process a single document.

        Args:
            file_path: Path to the document

        Returns
        -------
            List of processed document chunks
        """
        try:
            # Load the document
            raw_documents = PyPDFLoader(file_path=file_path).load()

            # Split the document
            split_documents = self.text_splitter.split_documents(raw_documents)

            # Filter metadata
            return filter_complex_metadata(split_documents)

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []

    def load_directory(
        self, directory_path: str, file_filter: str | None = None
    ) -> list[Document]:
        """
        Load and process all documents in a directory.

        Args:
            directory_path: Path to the directory
            file_filter: Optional string to filter filenames (e.g., "AAPL")

        Returns
        -------
            List of processed document chunks from all files
        """
        documents = []

        # Loop through all files in the directory
        for filename in os.listdir(directory_path):
            # Apply filters
            if not filename.endswith(self.file_extension):
                continue

            if file_filter and file_filter not in filename:
                continue

            # Construct full file path
            file_path = os.path.join(directory_path, filename)

            # Load and process the file
            processed_docs = self.load_document(file_path)
            documents.extend(processed_docs)

            print(f"Processed: {filename} - {len(processed_docs)} chunks")

        print(f"Total documents processed: {len(documents)}")
        return documents


def load_documents(
    directory_path: str,
    file_filter: str | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 24,
    pickle_path: str | None = None,
) -> list[Document]:
    """
    Load documents from a directory or a pickle file.

    Args:
        directory_path: Path to the directory containing documents
        file_filter: Optional string to filter filenames
        chunk_size: Size of chunks for text splitting
        chunk_overlap: Overlap between chunks
        pickle_path: Optional path to a pickle file containing pre-chunked documents

    Returns
    -------
        List of processed document chunks
    """
    # If pickle path is provided and exists, load from pickle
    if pickle_path and os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            documents = pickle.load(f)
            if isinstance(documents, list):
                return documents
            raise ValueError("Invalid pickle file format")
    else:
        # Otherwise load from directory
        loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return loader.load_directory(directory_path, file_filter)


def load_graph_documents(file_path: str) -> Any:
    """
    Load graph documents from a pickle file.

    Args:
        file_path: Path to the graph documents pickle file

    Returns
    -------
        Loaded graph documents
    """
    with open(file_path, "rb") as f:
        graph_documents = pickle.load(f)
    print(f"Graph documents loaded from {file_path}")
    return graph_documents
