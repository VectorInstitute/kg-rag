"""Indexer for GraphRAG-based KG-RAG approach."""

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.cache import SQLiteCache
from langchain_core.documents import Document
from langchain_graphrag.indexing import (
    IndexerArtifacts,
    SimpleIndexer,
    TextUnitExtractor,
)
from langchain_graphrag.indexing.artifacts_generation import (
    CommunitiesReportsArtifactsGenerator,
    EntitiesArtifactsGenerator,
    RelationshipsArtifactsGenerator,
    TextUnitsArtifactsGenerator,
)
from langchain_graphrag.indexing.graph_clustering import (
    HierarchicalLeidenCommunityDetector,
)
from langchain_graphrag.indexing.graph_generation import (
    EntityRelationshipDescriptionSummarizer,
    EntityRelationshipExtractor,
    GraphGenerator,
    GraphsMerger,
)
from langchain_graphrag.indexing.report_generation import (
    CommunityReportGenerator,
    CommunityReportWriter,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter


class GraphRAGIndexer:
    """Indexer for GraphRAG-based KG-RAG approach."""

    def __init__(
        self,
        cache_dir: str | None = None,
        vector_store_dir: str | None = None,
        artifacts_dir: str | None = None,
        llm_model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 512,
        chunk_overlap: int = 24,
        verbose: bool = False,
    ):
        """
        Initialize the GraphRAG indexer.

        Args:
            cache_dir: Directory for caching
            vector_store_dir: Directory for vector stores
            artifacts_dir: Directory for artifacts
            llm_model: LLM model to use
            embedding_model: Embedding model to use
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            verbose: Whether to print verbose output
        """
        # Set up directories
        self.cache_dir = Path(cache_dir or "cache")
        self.vector_store_dir = Path(vector_store_dir or "vector_stores")
        self.artifacts_dir = Path(artifacts_dir or "artifacts")

        # Create directories if they don't exist
        for directory in [self.cache_dir, self.vector_store_dir, self.artifacts_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Set up LLMs
        self.er_llm = ChatOpenAI(
            model=llm_model,
            temperature=0.0,
            cache=SQLiteCache(str(self.cache_dir / "openai_cache.db")),
        )

        self.es_llm = ChatOpenAI(
            model=llm_model,
            temperature=0.0,
            cache=SQLiteCache(str(self.cache_dir / "openai_cache.db")),
        )

        # Set up embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # Set up vector store for entities
        self.entities_vector_store = Chroma(
            collection_name="entities",
            persist_directory=str(self.vector_store_dir),
            embedding_function=self.embeddings,
        )

        # Set up text splitter and extractor
        self.text_splitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.text_unit_extractor = TextUnitExtractor(text_splitter=self.text_splitter)

        # Set up entity relationship extraction and summarization
        self.entity_extractor = EntityRelationshipExtractor.build_default(
            llm=self.er_llm  # type: ignore
        )
        self.entity_summarizer = EntityRelationshipDescriptionSummarizer.build_default(
            llm=self.es_llm  # type: ignore
        )

        # Set up graph generator
        self.graph_generator = GraphGenerator(
            er_extractor=self.entity_extractor,
            graphs_merger=GraphsMerger(),
            er_description_summarizer=self.entity_summarizer,
        )

        # Set up community detector
        self.community_detector = HierarchicalLeidenCommunityDetector()

        # Set up artifact generators
        self.entities_artifacts_generator = EntitiesArtifactsGenerator(
            entities_vector_store=self.entities_vector_store
        )

        self.relationships_artifacts_generator = RelationshipsArtifactsGenerator()

        self.report_generator = CommunityReportGenerator.build_default(llm=self.er_llm)  # type: ignore
        self.report_writer = CommunityReportWriter()

        self.communities_report_artifacts_generator = (
            CommunitiesReportsArtifactsGenerator(
                report_generator=self.report_generator, report_writer=self.report_writer
            )
        )

        self.text_units_artifacts_generator = TextUnitsArtifactsGenerator()

        # Set up indexer
        self.indexer = SimpleIndexer(
            text_unit_extractor=self.text_unit_extractor,
            graph_generator=self.graph_generator,
            community_detector=self.community_detector,
            entities_artifacts_generator=self.entities_artifacts_generator,
            relationships_artifacts_generator=self.relationships_artifacts_generator,
            text_units_artifacts_generator=self.text_units_artifacts_generator,
            communities_report_artifacts_generator=self.communities_report_artifacts_generator,
        )

        # Other settings
        self.verbose = verbose

    def index_documents(self, documents: list[Document]) -> IndexerArtifacts:
        """
        Index documents and generate artifacts.

        Args:
            documents: List of documents to index

        Returns
        -------
            Generated artifacts
        """
        if self.verbose:
            print(f"Indexing {len(documents)} documents...")

        # Run indexing
        artifacts = self.indexer.run(documents)

        if self.verbose:
            print("Indexing complete")

        return artifacts

    def save_artifacts(self, artifacts: dict[str, Any], file_path: str) -> None:
        """
        Save artifacts to a file.

        Args:
            artifacts: Artifacts to save
            file_path: Path to save to
        """
        import pickle

        with open(file_path, "wb") as f:
            pickle.dump(artifacts, f)

        if self.verbose:
            print(f"Artifacts saved to {file_path}")

    def load_artifacts(self, file_path: str) -> Any:
        """
        Load artifacts from a file.

        Args:
            file_path: Path to load from

        Returns
        -------
            Loaded artifacts
        """
        import pickle

        with open(file_path, "rb") as f:
            artifacts = pickle.load(f)

        if self.verbose:
            print(f"Artifacts loaded from {file_path}")

        return artifacts
