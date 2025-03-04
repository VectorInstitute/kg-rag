"""
Document loading and processing utilities for KG-RAG approaches.
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

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
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load and process a single document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of processed document chunks
        """
        try:
            # Load the document
            raw_documents = PyPDFLoader(file_path=file_path).load()
            
            # Split the document
            split_documents = self.text_splitter.split_documents(raw_documents)
            
            # Filter metadata
            processed_documents = filter_complex_metadata(split_documents)
            
            return processed_documents
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []
    
    def load_directory(
        self, 
        directory_path: str,
        file_filter: Optional[str] = None
    ) -> List[Document]:
        """
        Load and process all documents in a directory.
        
        Args:
            directory_path: Path to the directory
            file_filter: Optional string to filter filenames (e.g., "AAPL")
            
        Returns:
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
    file_filter: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 24,
) -> List[Document]:
    """
    Convenience function to load documents from a directory.
    
    Args:
        directory_path: Path to the directory containing documents
        file_filter: Optional string to filter filenames
        chunk_size: Size of chunks for text splitting
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of processed document chunks
    """
    loader = DocumentLoader(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return loader.load_directory(directory_path, file_filter)