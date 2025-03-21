"""Search strategies for GraphRAG-based KG-RAG approach."""

from datetime import datetime
from typing import Any, cast

from langchain_chroma import Chroma
from langchain_graphrag.indexing import IndexerArtifacts
from langchain_graphrag.query.global_search import GlobalSearch
from langchain_graphrag.query.global_search.community_weight_calculator import (
    CommunityWeightCalculator,
)
from langchain_graphrag.query.global_search.key_points_aggregator import (
    KeyPointsAggregator,
    KeyPointsAggregatorPromptBuilder,
    KeyPointsContextBuilder,
)
from langchain_graphrag.query.global_search.key_points_generator import (
    CommunityReportContextBuilder,
    KeyPointsGenerator,
    KeyPointsGeneratorPromptBuilder,
)
from langchain_graphrag.query.local_search import (
    LocalSearch,
    LocalSearchPromptBuilder,
    LocalSearchRetriever,
)
from langchain_graphrag.query.local_search._system_prompt import (
    LOCAL_SEARCH_SYSTEM_PROMPT,
)
from langchain_graphrag.query.local_search.context_builders import ContextBuilder
from langchain_graphrag.query.local_search.context_selectors import ContextSelector
from langchain_graphrag.types.graphs.community import CommunityLevel
from langchain_graphrag.utils import TiktokenCounter
from langchain_openai import ChatOpenAI


class LocalSearchStrategy:
    """Local search strategy for GraphRAG."""

    def __init__(
        self,
        llm: ChatOpenAI,
        entities_vector_store: Chroma,
        artifacts: IndexerArtifacts,
        community_level: int = 2,
        show_references: bool = True,
        verbose: bool = False,
        system_prompt: str | None = None,
        strict_output_format: bool = False,
    ):
        """
        Initialize the local search strategy.

        Args:
            llm: LLM to use for search
            entities_vector_store: Vector store for entities
            artifacts: GraphRAG artifacts
            community_level: Community level to use
            show_references: Whether to show references in response
            verbose: Whether to print verbose output
            system_prompt: Custom system prompt to use
            strict_output_format: Whether to enforce strict output format
        """
        # Use custom system prompt if provided, otherwise use default
        if system_prompt is None:
            current_date = datetime.now().strftime("%B %d, %Y")
            system_prompt = LOCAL_SEARCH_SYSTEM_PROMPT

            if strict_output_format:
                # Add strict formatting rules for JSON output
                system_prompt += f"""
                Important Rules:
                - Base your answer ONLY on the provided context
                - Do not make assumptions or use external knowledge besides the context provided
                - Your entire response must be valid JSON

                The current date is {current_date}.
                """
            else:
                # Add simple date reference
                system_prompt += f"\nThe current date is {current_date}."

        # Create components for local search
        self.context_selector = ContextSelector.build_default(
            entities_vector_store=entities_vector_store,
            entities_top_k=10,
            community_level=cast(CommunityLevel, community_level),
        )

        self.context_builder = ContextBuilder.build_default(
            token_counter=TiktokenCounter(),
        )

        self.retriever = LocalSearchRetriever(
            context_selector=self.context_selector,
            context_builder=self.context_builder,
            artifacts=artifacts,
        )

        self.search_method = LocalSearch(
            prompt_builder=LocalSearchPromptBuilder(
                system_prompt=system_prompt, show_references=show_references
            ),
            llm=llm,  # type: ignore
            retriever=self.retriever,
        )

        # Create the search chain
        self.chain = self.search_method()

        # Other settings
        self.verbose = verbose

    def search(self, query: str) -> Any:
        """
        Perform local search.

        Args:
            query: Query string

        Returns
        -------
            Search result
        """
        if self.verbose:
            print(f"Performing local search for: {query}")

        return self.chain.invoke(query)


class GlobalSearchStrategy:
    """Global search strategy for GraphRAG."""

    def __init__(
        self,
        llm: ChatOpenAI,
        artifacts: IndexerArtifacts,
        community_level: int = 2,
        token_counter: TiktokenCounter | None = None,
        show_references: bool = True,
        verbose: bool = False,
        strict_output_format: bool = False,
    ):
        """
        Initialize the global search strategy.

        Args:
            llm: LLM to use for search
            artifacts: GraphRAG artifacts
            community_level: Community level to use
            token_counter: Token counter to use
            show_references: Whether to show references in response
            verbose: Whether to print verbose output
            strict_output_format: Whether to enforce strict output format
        """
        # Create token counter if not provided
        if token_counter is None:
            token_counter = TiktokenCounter()

        # Modify prompt templates for strict output if needed
        kp_generator_prompt_builder = KeyPointsGeneratorPromptBuilder(
            show_references=show_references
        )

        kp_aggregator_prompt_builder = KeyPointsAggregatorPromptBuilder(
            show_references=show_references
        )

        # Add formatting instructions for strict output
        if strict_output_format:
            # Note: In a real implementation, you would modify the
            # KeyPointsGeneratorPromptBuilder and KeyPointsAggregatorPromptBuilder
            # to accept custom templates. This example approximates that functionality.
            pass

        # Create components for global search
        self.report_context_builder = CommunityReportContextBuilder(
            community_level=cast(CommunityLevel, community_level),
            weight_calculator=CommunityWeightCalculator(),
            artifacts=artifacts,
            token_counter=token_counter,
        )

        self.kp_generator = KeyPointsGenerator(
            llm=llm,  # type: ignore
            prompt_builder=kp_generator_prompt_builder,
            context_builder=self.report_context_builder,
        )

        self.kp_aggregator = KeyPointsAggregator(
            llm=llm,  # type: ignore
            prompt_builder=kp_aggregator_prompt_builder,
            context_builder=KeyPointsContextBuilder(
                token_counter=token_counter,
            ),
        )

        self.search_method = GlobalSearch(
            kp_generator=self.kp_generator, kp_aggregator=self.kp_aggregator
        )

        # Other settings
        self.verbose = verbose

    def search(self, query: str) -> str:
        """
        Perform global search.

        Args:
            query: Query string

        Returns
        -------
            Search result
        """
        if self.verbose:
            print(f"Performing global search for: {query}")

        return self.search_method.invoke(query)
