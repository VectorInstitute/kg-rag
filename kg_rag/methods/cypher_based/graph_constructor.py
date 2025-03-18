"""
Graph constructor for Cypher-based KG-RAG approach.
"""

import os
from typing import List, Dict, Any, Optional
import pickle
from tqdm.auto import tqdm

from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document


class CypherGraphConstructor:
    """Constructs a Neo4j graph from documents using LLM-based extraction."""
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        verbose: bool = False
    ):
        """
        Initialize the graph constructor.
        
        Args:
            llm: LLM to use for graph construction (default: ChatOpenAI with gpt-4o)
            neo4j_uri: URI for Neo4j connection
            neo4j_user: Username for Neo4j connection
            neo4j_password: Password for Neo4j connection
            verbose: Whether to print verbose output
        """
        # Set up LLM if not provided
        if llm is None:
            self.llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-4o"
            )
        else:
            self.llm = llm
        
        # Create LLM transformer
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)
        
        # Neo4j connection settings
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        
        # Other settings
        self.verbose = verbose
        
    def convert_documents_to_graph_documents(
        self,
        documents: List[Document],
        graph_documents_path: Optional[str] = None,
        force_rebuild: bool = False
    ) -> List[Any]:
        """
        Convert documents to graph documents using LLM.
        
        Args:
            documents: List of documents to convert
            graph_documents_path: Optional path to save/load graph documents
            force_rebuild: Force rebuilding the graph documents
            
        Returns:
            List of graph documents
        """
        # Check if graph documents can be loaded
        if (graph_documents_path and os.path.exists(graph_documents_path) 
            and not force_rebuild):
            if self.verbose:
                print(f"Loading graph documents from {graph_documents_path}")
            with open(graph_documents_path, "rb") as f:
                return pickle.load(f)
        
        # Convert documents to graph documents
        if self.verbose:
            print("Converting documents to graph documents...")
            graph_documents = self.llm_transformer.convert_to_graph_documents(
                tqdm(documents) if self.verbose else documents
            )
        else:
            graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
        
        # Save graph documents if path is provided
        if graph_documents_path:
            if self.verbose:
                print(f"Saving graph documents to {graph_documents_path}")
            with open(graph_documents_path, "wb") as f:
                pickle.dump(graph_documents, f)
        
        return graph_documents
    
    def build_neo4j_graph(
        self,
        graph_documents: List[Any],
        clear_existing: bool = True
    ) -> Neo4jGraph:
        """
        Build a Neo4j graph from graph documents.
        
        Args:
            graph_documents: List of graph documents
            clear_existing: Whether to clear existing data in Neo4j
            
        Returns:
            Neo4jGraph instance
        """
        # Connect to Neo4j
        if self.verbose:
            print(f"Connecting to Neo4j at {self.neo4j_uri}")
        
        graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password
        )
        
        # Clear existing data if requested
        if clear_existing:
            if self.verbose:
                print("Clearing existing data in Neo4j")
            graph.query("MATCH (n) DETACH DELETE n")
        
        # Add graph documents
        if self.verbose:
            print("Adding graph documents to Neo4j...")
            graph.add_graph_documents(
                tqdm(graph_documents) if self.verbose else graph_documents,
                baseEntityLabel=True,
                include_source=True
            )
        else:
            graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )
        
        if self.verbose:
            print("Graph built successfully")
        
        return graph
    
    def process_documents(
        self,
        documents: List[Document],
        graph_documents_path: Optional[str] = None,
        force_rebuild: bool = False,
        clear_existing: bool = True
    ) -> Neo4jGraph:
        """
        Process documents and build a Neo4j graph.
        
        Args:
            documents: List of documents to process
            graph_documents_path: Optional path to save/load graph documents
            force_rebuild: Force rebuilding the graph documents
            clear_existing: Whether to clear existing data in Neo4j
            
        Returns:
            Neo4jGraph instance
        """
        # Convert documents to graph documents
        graph_documents = self.convert_documents_to_graph_documents(
            documents,
            graph_documents_path,
            force_rebuild
        )
        
        # Build Neo4j graph
        return self.build_neo4j_graph(graph_documents, clear_existing)