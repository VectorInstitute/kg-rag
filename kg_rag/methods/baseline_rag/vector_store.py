"""Vector store manager for baseline RAG approaches."""

from typing import Any

import chromadb

from .embedder import OpenAIEmbedding


class ChromaDBManager:
    """Manage ChromaDB operations using local persistent client."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "chroma_db",
        batch_size: int = 100,
        verbose: bool = False,
    ):
        """
        Initialize the ChromaDB manager.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to store ChromaDB files
            batch_size: Number of documents to add in a single batch
            verbose: Whether to print verbose output
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.verbose = verbose
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Get an existing collection or create a new one."""
        try:
            collection = self.client.get_collection(self.collection_name)
            if self.verbose:
                print(f"Found existing collection: {self.collection_name}")
        except ValueError:
            if self.verbose:
                print(f"Creating new collection: {self.collection_name}")
            collection = self.client.create_collection(self.collection_name)
        return collection

    def get_collection_stats(self) -> dict[str, Any]:
        """
        Get statistics about the collection.

        Returns
        -------
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {"collection_name": self.collection_name, "document_count": count}

    def add_documents(
        self,
        chunks: list[tuple[str, dict[str, Any]]],
        embedding_function: OpenAIEmbedding,
    ) -> None:
        """
        Add documents to the collection.

        Args:
            chunks: List of tuples containing (text_content, metadata)
            embedding_function: Function to generate embeddings
        """
        if not chunks:
            if self.verbose:
                print("No documents to add")
            return

        if self.verbose:
            print(f"Processing {len(chunks)} chunks...")

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]

            documents = []
            metadatas = []
            ids = []

            for j, (text, metadata) in enumerate(batch):
                documents.append(text)
                metadatas.append(metadata)
                ids.append(f"{metadata['filename']}_{i + j}")

            try:
                if self.verbose:
                    print(
                        f"Generating embeddings for batch {i // self.batch_size + 1}/{(len(chunks) - 1) // self.batch_size + 1}"
                    )
                embeddings = embedding_function.generate(documents)

                if self.verbose:
                    print("Adding batch to collection...")
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings,
                )

            except Exception as e:
                if self.verbose:
                    print(
                        f"Error processing batch {i // self.batch_size + 1}: {str(e)}"
                    )
                raise

        if self.verbose:
            print(f"Successfully added all {len(chunks)} documents to collection")

    def query(
        self, query_text: str, embedding_function: OpenAIEmbedding, n_results: int = 5
    ) -> Any:
        """
        Query the collection for similar documents.

        Args:
            query_text: Query text
            embedding_function: Function to generate embeddings
            n_results: Number of results to return

        Returns
        -------
            Dictionary with query results
        """
        try:
            if self.verbose:
                print(f"Generating embedding for query: {query_text[:100]}...")
            query_embedding = embedding_function.generate([query_text])[0]

            if self.verbose:
                print(f"Querying collection for top {n_results} results...")

            return self.collection.query(
                query_embeddings=[query_embedding], n_results=n_results
            )

        except Exception as e:
            raise Exception(f"Error querying ChromaDB: {str(e)}") from e
