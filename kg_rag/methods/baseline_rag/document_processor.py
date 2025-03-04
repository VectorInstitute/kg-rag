"""
Document processing utilities for baseline RAG approaches.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata
from tqdm.auto import tqdm


class DocumentProcessor:
    """Process and chunk documents using LangChain's document loaders and text splitters."""
    
    def __init__(
        self,
        chunk_size: int = 512, 
        chunk_overlap: int = 24,
        verbose: bool = False
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of chunks for text splitting
            chunk_overlap: Overlap between chunks
            verbose: Whether to print verbose output
        """
        self.text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.verbose = verbose
    
    def process_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process a single file using PyPDFLoader for PDFs.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of tuples containing (text_content, metadata)
        """
        metadata = {
            "filename": file_path.name,
            "file_path": str(file_path),
            "source": "document_collection"
        }
        
        if file_path.suffix.lower() == ".pdf":
            if self.verbose:
                print(f"Processing PDF: {file_path.name}")
                
            # Use PyPDFLoader for PDF files
            raw_documents = PyPDFLoader(file_path=str(file_path)).load()
            
            # Split documents using TokenTextSplitter
            documents = self.text_splitter.split_documents(raw_documents)
            
            # Filter complex metadata
            documents = filter_complex_metadata(documents)
            
            # Convert to the format expected by ChromaDB
            chunks = []
            for doc in documents:
                doc_metadata = {**metadata, **doc.metadata}
                chunks.append((doc.page_content, doc_metadata))
                
            if self.verbose:
                print(f"Generated {len(chunks)} chunks from {file_path.name}")
                
            return chunks
        else:
            if self.verbose:
                print(f"Skipping non-PDF file: {file_path.name}")
            return []
            
    def process_directory(self, directory_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to the directory containing PDF files
            
        Returns:
            List of tuples containing (text_content, metadata)
        """
        pdf_files = list(directory_path.glob("*.pdf"))
        if self.verbose:
            print(f"Found {len(pdf_files)} PDF files in {directory_path}")
            
        all_chunks = []
        for file_path in tqdm(pdf_files, desc="Processing documents", disable=not self.verbose):
            chunks = self.process_file(file_path)
            all_chunks.extend(chunks)
            
        if self.verbose:
            print(f"Total chunks generated: {len(all_chunks)}")
            
        return all_chunks