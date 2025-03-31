"""Graph transformers for converting documents to graph-based documents with metadata and summaries."""

from functools import lru_cache
from typing import Any

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer


class MetadataEnhancedLLMGraphTransformer(LLMGraphTransformer):
    """Transform documents into graph-based documents using a LLM with metadata enhancement.

    Extends the LLMGraphTransformer to preserve hierarchical information by adding
    the file name as a suffix to nodes and relationships. This helps maintain the
    connection between entities and their source documents.

    Args:
        llm (BaseLanguageModel): An instance of a language model supporting structured output.
        allowed_nodes (List[str], optional): Specifies which node types are allowed in the graph.
            Defaults to an empty list, allowing all node types.
        allowed_relationships (List[str], optional): Specifies which relationship types are
            allowed in the graph. Defaults to an empty list, allowing all relationship types.
        prompt (Optional[ChatPromptTemplate], optional): The prompt to pass to the LLM with
            additional instructions.
        strict_mode (bool, optional): Determines whether the transformer should apply filtering
            to strictly adhere to `allowed_nodes` and `allowed_relationships`. Defaults to True.
        node_properties (Union[bool, List[str]]): If True, the LLM can extract any node properties
            from text. Alternatively, a list of valid properties can be provided.
        relationship_properties (Union[bool, List[str]]): If True, the LLM can extract any
            relationship properties from text. Alternatively, a list of valid properties can be
            provided.
        ignore_tool_usage (bool): Indicates whether the transformer should bypass the use of
            structured output functionality of the language model. Defaults to False.
        additional_instructions (str): Allows you to add additional instructions to the prompt.
        metadata_field (str): The metadata field to use for suffixing nodes and relationships.
            Defaults to "source".
        include_metadata_in_id (bool): Whether to include metadata in the node ID.
            Defaults to True.

    Example:
        .. code-block:: python
            from langchain_experimental.graph_transformers import (
                MetadataEnhancedLLMGraphTransformer,
            )
            from langchain_core.documents import Document
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(temperature=0)
            transformer = MetadataEnhancedLLMGraphTransformer(
                llm=llm,
                allowed_nodes=["Person", "Organization"],
                metadata_field="source",
            )

            doc = Document(
                page_content="Elon Musk is suing OpenAI",
                metadata={"source": "news_article.txt"},
            )
            graph_documents = transformer.convert_to_graph_documents([doc])
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        allowed_nodes: list[str] = None,  # type: ignore
        allowed_relationships: list[str] | list[tuple[str, str, str]] = None,  # type: ignore
        prompt: ChatPromptTemplate | None = None,
        strict_mode: bool = True,
        node_properties: bool | list[str] = False,
        relationship_properties: bool | list[str] = False,
        ignore_tool_usage: bool = False,
        metadata_field: str = "source",
        include_metadata_in_id: bool = True,
    ) -> None:
        allowed_nodes = None if allowed_nodes is None else allowed_nodes
        allowed_relationships = (
            None if allowed_relationships is None else allowed_relationships
        )
        super().__init__(
            llm=llm,
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
            prompt=prompt,
            strict_mode=strict_mode,
            node_properties=node_properties,
            relationship_properties=relationship_properties,
            ignore_tool_usage=ignore_tool_usage,
        )
        self.metadata_field = metadata_field
        self.include_metadata_in_id = include_metadata_in_id

    def _add_metadata_to_nodes_and_relationships(
        self, nodes: list[Node], relationships: list[Relationship], document: Document
    ) -> tuple[list[Node], list[Relationship]]:
        """Add metadata as suffix to nodes and relationships."""
        # Get metadata value (typically file name)
        metadata_value = document.metadata.get(self.metadata_field, "unknown")

        # Create new nodes with metadata suffix
        enhanced_nodes = []
        node_id_map = {}  # Maps original IDs to new IDs

        for node in nodes:
            if self.include_metadata_in_id:
                # Create a new ID that includes the metadata
                new_id = f"{node.id}_{metadata_value}"
                node_id_map[node.id] = new_id

                # Create a new node with the modified ID
                enhanced_node = Node(
                    id=new_id,
                    type=node.type,
                    properties={
                        **(node.properties or {}),
                        "original_id": node.id,
                        "source_document": metadata_value,
                    },
                )
            else:
                # Keep the original ID but add metadata to properties
                enhanced_node = Node(
                    id=node.id,
                    type=node.type,
                    properties={
                        **(node.properties or {}),
                        "source_document": metadata_value,
                    },
                )
                if isinstance(node.id, int):
                    node_id_map[str(node.id)] = str(node.id)
                else:
                    node_id_map[node.id] = node.id

            enhanced_nodes.append(enhanced_node)

        # Create new relationships with updated node references
        enhanced_relationships = []
        for rel in relationships:
            source_id = rel.source.id
            target_id = rel.target.id

            # Create new source and target nodes with mapped IDs
            new_source = Node(
                id=node_id_map.get(source_id, source_id),
                type=rel.source.type,
                properties=rel.source.properties,
            )

            new_target = Node(
                id=node_id_map.get(target_id, target_id),
                type=rel.target.type,
                properties=rel.target.properties,
            )

            # Create the enhanced relationship
            enhanced_rel = Relationship(
                source=new_source,
                target=new_target,
                type=rel.type,
                properties={
                    **(rel.properties or {}),
                    "source_document": metadata_value,
                },
            )
            enhanced_relationships.append(enhanced_rel)

        return enhanced_nodes, enhanced_relationships

    def process_response(
        self, document: Document, config: RunnableConfig | None = None
    ) -> GraphDocument:
        """Process a document to create a graph document with metadata enhancement."""
        # First, process using the parent class
        graph_document = super().process_response(document, config)

        # Then enhance nodes and relationships with metadata
        (
            enhanced_nodes,
            enhanced_relationships,
        ) = self._add_metadata_to_nodes_and_relationships(
            graph_document.nodes, graph_document.relationships, document
        )

        # Return a new graph document with enhanced nodes and relationships
        return GraphDocument(
            nodes=enhanced_nodes, relationships=enhanced_relationships, source=document
        )

    async def aprocess_response(
        self, document: Document, config: RunnableConfig | None = None
    ) -> GraphDocument:
        """Asynchronously process a document with metadata enhancement."""
        # First, process using the parent class
        graph_document = await super().aprocess_response(document, config)

        # Then enhance nodes and relationships with metadata
        (
            enhanced_nodes,
            enhanced_relationships,
        ) = self._add_metadata_to_nodes_and_relationships(
            graph_document.nodes, graph_document.relationships, document
        )

        # Return a new graph document with enhanced nodes and relationships
        return GraphDocument(
            nodes=enhanced_nodes, relationships=enhanced_relationships, source=document
        )


class SummaryEnhancedLLMGraphTransformer(LLMGraphTransformer):
    """Transform documents into graph-based documents using a LLM with document summary prefixes.

    Extends the LLMGraphTransformer to generate summaries for documents and add them as
    prefixes to all chunks from that document when converting to nodes and relationships.
    Summaries are cached to avoid regenerating them for the same document.

    Args:
        llm (BaseLanguageModel): An instance of a language model supporting structured output.
        summary_llm (Optional[BaseLanguageModel]): The language model to use for generating
            summaries. If None, the main llm will be used.
        allowed_nodes (List[str], optional): Specifies which node types are allowed in the graph.
            Defaults to an empty list, allowing all node types.
        allowed_relationships (List[str], optional): Specifies which relationship types are
            allowed in the graph. Defaults to an empty list, allowing all relationship types.
        prompt (Optional[ChatPromptTemplate], optional): The prompt to pass to the LLM with
            additional instructions.
        strict_mode (bool, optional): Determines whether the transformer should apply filtering
            to strictly adhere to `allowed_nodes` and `allowed_relationships`. Defaults to True.
        node_properties (Union[bool, List[str]]): If True, the LLM can extract any node properties
            from text. Alternatively, a list of valid properties can be provided.
        relationship_properties (Union[bool, List[str]]): If True, the LLM can extract any
            relationship properties from text. Alternatively, a list of valid properties can be
            provided.
        ignore_tool_usage (bool): Indicates whether the transformer should bypass the use of
            structured output functionality of the language model. Defaults to False.
        additional_instructions (str): Allows you to add additional instructions to the prompt.
        metadata_field (str): The metadata field to use for identifying document sources.
            Defaults to "source".
        summary_cache_size (int): The maximum number of document summaries to cache.
            Defaults to 100.
        summary_prompt_template (str): The prompt template to use for generating document summaries.

    Example:
        .. code-block:: python
            from langchain_experimental.graph_transformers import (
                SummaryEnhancedLLMGraphTransformer,
            )
            from langchain_core.documents import Document
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(temperature=0)
            transformer = SummaryEnhancedLLMGraphTransformer(
                llm=llm, allowed_nodes=["Person", "Organization"]
            )

            doc = Document(
                page_content="Elon Musk is suing OpenAI",
                metadata={"source": "news_article.txt"},
            )
            graph_documents = transformer.convert_to_graph_documents([doc])
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        summary_llm: BaseLanguageModel | None = None,
        allowed_nodes: list[str] = None,  # type: ignore
        allowed_relationships: list[str] | list[tuple[str, str, str]] = None,  # type: ignore
        prompt: ChatPromptTemplate | None = None,
        strict_mode: bool = True,
        node_properties: bool | list[str] = False,
        relationship_properties: bool | list[str] = False,
        ignore_tool_usage: bool = False,
        additional_instructions: str = "",
        metadata_field: str = "source",
        summary_cache_size: int = 100,
        summary_prompt_template: str = "Create a concise summary of the following document, focusing on the key entities and relationships described: {content}",
    ) -> None:
        allowed_nodes = None if allowed_nodes is None else allowed_nodes
        allowed_relationships = (
            None if allowed_relationships is None else allowed_relationships
        )
        super().__init__(
            llm=llm,
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
            prompt=prompt,
            strict_mode=strict_mode,
            node_properties=node_properties,
            relationship_properties=relationship_properties,
            ignore_tool_usage=ignore_tool_usage,
            additional_instructions=additional_instructions,
        )
        self.summary_llm = summary_llm or llm
        self.metadata_field = metadata_field
        self.summary_prompt_template = summary_prompt_template
        self.summary_prompt = ChatPromptTemplate.from_template(summary_prompt_template)

        # Create a cache for document summaries
        self._get_document_summary = lru_cache(maxsize=summary_cache_size)(
            self._generate_document_summary
        )

    def _generate_document_summary(self, doc_id: str, content: str) -> Any:
        """Generate a summary for a document."""
        chain = self.summary_prompt | self.summary_llm
        summary = chain.invoke({"content": content})

        # Handle different return types from LLMs
        if hasattr(summary, "content"):
            return summary.content
        return str(summary)

    def _get_document_content_for_summary(self, document: Document) -> str:
        """Extract the content to be used for summarization."""
        # If available, use the first page or a portion of the document
        if len(document.page_content) > 1500:
            return document.page_content[:1500] + "..."
        return document.page_content

    def _enhance_with_summary(
        self, nodes: list[Node], relationships: list[Relationship], document: Document
    ) -> tuple[list[Node], list[Relationship]]:
        """Enhance nodes and relationships with document summary prefix."""
        # Get document identifier from metadata
        doc_id = document.metadata.get(self.metadata_field, "unknown")

        # Get or generate the document summary
        doc_content = self._get_document_content_for_summary(document)
        summary = self._get_document_summary(doc_id, doc_content)

        # Add summary prefix to nodes
        enhanced_nodes = []
        for node in nodes:
            # Create properties with summary
            properties = {
                **(node.properties or {}),
                "document_summary": summary,
                "source_document": doc_id,
            }

            enhanced_node = Node(id=node.id, type=node.type, properties=properties)
            enhanced_nodes.append(enhanced_node)

        # Add summary to relationships
        enhanced_relationships = []
        for rel in relationships:
            enhanced_rel = Relationship(
                source=rel.source,
                target=rel.target,
                type=rel.type,
                properties={
                    **(rel.properties or {}),
                    "document_summary": summary,
                    "source_document": doc_id,
                },
            )
            enhanced_relationships.append(enhanced_rel)

        return enhanced_nodes, enhanced_relationships

    def process_response(
        self, document: Document, config: RunnableConfig | None = None
    ) -> GraphDocument:
        """Process a document to create a graph document with summary enhancement."""
        # First, process using the parent class
        graph_document = super().process_response(document, config)

        # Then enhance nodes and relationships with document summary
        enhanced_nodes, enhanced_relationships = self._enhance_with_summary(
            graph_document.nodes, graph_document.relationships, document
        )

        # Return a new graph document with enhanced nodes and relationships
        return GraphDocument(
            nodes=enhanced_nodes, relationships=enhanced_relationships, source=document
        )

    async def aprocess_response(
        self, document: Document, config: RunnableConfig | None = None
    ) -> GraphDocument:
        """Asynchronously process a document with summary enhancement."""
        # First, process using the parent class
        graph_document = await super().aprocess_response(document, config)

        # Then enhance nodes and relationships with document summary
        enhanced_nodes, enhanced_relationships = self._enhance_with_summary(
            graph_document.nodes, graph_document.relationships, document
        )

        # Return a new graph document with enhanced nodes and relationships
        return GraphDocument(
            nodes=enhanced_nodes, relationships=enhanced_relationships, source=document
        )


class CombinedEnhancedLLMGraphTransformer(LLMGraphTransformer):
    """Transform documents into graph-based documents with both metadata and summary enhancements.

    Combines the functionality of both MetadataEnhancedLLMGraphTransformer and
    SummaryEnhancedLLMGraphTransformer to provide comprehensive document context
    in the generated graph.

    Args:
        llm (BaseLanguageModel): An instance of a language model supporting structured output.
        summary_llm (Optional[BaseLanguageModel]): The language model to use for generating
            summaries. If None, the main llm will be used.
        allowed_nodes (List[str], optional): Specifies which node types are allowed in the graph.
            Defaults to an empty list, allowing all node types.
        allowed_relationships (List[str], optional): Specifies which relationship types are
            allowed in the graph. Defaults to an empty list, allowing all relationship types.
        prompt (Optional[ChatPromptTemplate], optional): The prompt to pass to the LLM with
            additional instructions.
        strict_mode (bool, optional): Determines whether the transformer should apply filtering
            to strictly adhere to `allowed_nodes` and `allowed_relationships`. Defaults to True.
        node_properties (Union[bool, List[str]]): If True, the LLM can extract any node properties
            from text. Alternatively, a list of valid properties can be provided.
        relationship_properties (Union[bool, List[str]]): If True, the LLM can extract any
            relationship properties from text. Alternatively, a list of valid properties can be
            provided.
        ignore_tool_usage (bool): Indicates whether the transformer should bypass the use of
            structured output functionality of the language model. Defaults to False.
        additional_instructions (str): Allows you to add additional instructions to the prompt.
        metadata_field (str): The metadata field to use for identifying document sources.
            Defaults to "source".
        include_metadata_in_id (bool): Whether to include metadata in the node ID.
            Defaults to True.
        summary_cache_size (int): The maximum number of document summaries to cache.
            Defaults to 100.
        summary_prompt_template (str): The prompt template to use for generating document summaries.

    Example:
        .. code-block:: python
            from langchain_experimental.graph_transformers import (
                CombinedEnhancedLLMGraphTransformer,
            )
            from langchain_core.documents import Document
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(temperature=0)
            transformer = CombinedEnhancedLLMGraphTransformer(
                llm=llm, allowed_nodes=["Person", "Organization"]
            )

            doc = Document(
                page_content="Elon Musk is suing OpenAI",
                metadata={"source": "news_article.txt"},
            )
            graph_documents = transformer.convert_to_graph_documents([doc])
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        summary_llm: BaseLanguageModel | None = None,
        allowed_nodes: list[str] = None,  # type: ignore
        allowed_relationships: list[str] | list[tuple[str, str, str]] = None,  # type: ignore
        prompt: ChatPromptTemplate | None = None,
        strict_mode: bool = True,
        node_properties: bool | list[str] = False,
        relationship_properties: bool | list[str] = False,
        ignore_tool_usage: bool = False,
        additional_instructions: str = "",
        metadata_field: str = "source",
        include_metadata_in_id: bool = True,
        summary_cache_size: int = 100,
        summary_prompt_template: str = "Create a concise summary of the following document, focusing on the key entities and relationships described: {content}",
    ) -> None:
        allowed_nodes = None if allowed_nodes is None else allowed_nodes
        allowed_relationships = (
            None if allowed_relationships is None else allowed_relationships
        )
        super().__init__(
            llm=llm,
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
            prompt=prompt,
            strict_mode=strict_mode,
            node_properties=node_properties,
            relationship_properties=relationship_properties,
            ignore_tool_usage=ignore_tool_usage,
            additional_instructions=additional_instructions,
        )
        self.summary_llm = summary_llm or llm
        self.metadata_field = metadata_field
        self.include_metadata_in_id = include_metadata_in_id
        self.summary_prompt_template = summary_prompt_template
        self.summary_prompt = ChatPromptTemplate.from_template(summary_prompt_template)

        # Create a cache for document summaries
        self._get_document_summary = lru_cache(maxsize=summary_cache_size)(
            self._generate_document_summary
        )

    def _generate_document_summary(self, doc_id: str, content: str) -> Any:
        """Generate a summary for a document."""
        chain = self.summary_prompt | self.summary_llm
        summary = chain.invoke({"content": content})

        # Handle different return types from LLMs
        if hasattr(summary, "content"):
            return summary.content
        return str(summary)

    def _get_document_content_for_summary(self, document: Document) -> str:
        """Extract the content to be used for summarization."""
        # If available, use the first page or a portion of the document
        if len(document.page_content) > 1500:
            return document.page_content[:1500] + "..."
        return document.page_content

    def _enhance_with_metadata_and_summary(
        self, nodes: list[Node], relationships: list[Relationship], document: Document
    ) -> tuple[list[Node], list[Relationship]]:
        """Enhance nodes and relationships with both metadata and document summary."""
        # Get metadata value (typically file name)
        metadata_value = document.metadata.get(self.metadata_field, "unknown")

        # Get or generate the document summary
        doc_content = self._get_document_content_for_summary(document)
        summary = self._get_document_summary(metadata_value, doc_content)

        # Create new nodes with metadata suffix and summary
        enhanced_nodes = []
        node_id_map = {}  # Maps original IDs to new IDs

        for node in nodes:
            if self.include_metadata_in_id:
                # Create a new ID that includes the metadata
                new_id = f"{node.id}_{metadata_value}"
                node_id_map[node.id] = new_id

                # Create a new node with the modified ID and summary
                enhanced_node = Node(
                    id=new_id,
                    type=node.type,
                    properties={
                        **(node.properties or {}),
                        "original_id": node.id,
                        "source_document": metadata_value,
                        "document_summary": summary,
                    },
                )
            else:
                # Keep the original ID but add metadata and summary to properties
                enhanced_node = Node(
                    id=node.id,
                    type=node.type,
                    properties={
                        **(node.properties or {}),
                        "source_document": metadata_value,
                        "document_summary": summary,
                    },
                )
                if isinstance(node.id, int):
                    node_id_map[str(node.id)] = str(node.id)
                else:
                    node_id_map[node.id] = node.id

            enhanced_nodes.append(enhanced_node)

        # Create new relationships with updated node references and summary
        enhanced_relationships = []
        for rel in relationships:
            source_id = rel.source.id
            target_id = rel.target.id

            # Create new source and target nodes with mapped IDs
            new_source = Node(
                id=node_id_map.get(source_id, source_id),
                type=rel.source.type,
                properties=rel.source.properties,
            )

            new_target = Node(
                id=node_id_map.get(target_id, target_id),
                type=rel.target.type,
                properties=rel.target.properties,
            )

            # Create the enhanced relationship with metadata and summary
            enhanced_rel = Relationship(
                source=new_source,
                target=new_target,
                type=rel.type,
                properties={
                    **(rel.properties or {}),
                    "source_document": metadata_value,
                    "document_summary": summary,
                },
            )
            enhanced_relationships.append(enhanced_rel)

        return enhanced_nodes, enhanced_relationships

    def process_response(
        self, document: Document, config: RunnableConfig | None = None
    ) -> GraphDocument:
        """Process a document to create a graph document with metadata and summary enhancements."""
        # First, process using the parent class
        graph_document = super().process_response(document, config)

        # Then enhance nodes and relationships with metadata and document summary
        (
            enhanced_nodes,
            enhanced_relationships,
        ) = self._enhance_with_metadata_and_summary(
            graph_document.nodes, graph_document.relationships, document
        )

        # Return a new graph document with enhanced nodes and relationships
        return GraphDocument(
            nodes=enhanced_nodes, relationships=enhanced_relationships, source=document
        )

    async def aprocess_response(
        self, document: Document, config: RunnableConfig | None = None
    ) -> GraphDocument:
        """Asynchronously process a document with metadata and summary enhancements."""
        # First, process using the parent class
        graph_document = await super().aprocess_response(document, config)

        # Then enhance nodes and relationships with metadata and document summary
        (
            enhanced_nodes,
            enhanced_relationships,
        ) = self._enhance_with_metadata_and_summary(
            graph_document.nodes, graph_document.relationships, document
        )

        # Return a new graph document with enhanced nodes and relationships
        return GraphDocument(
            nodes=enhanced_nodes, relationships=enhanced_relationships, source=document
        )
