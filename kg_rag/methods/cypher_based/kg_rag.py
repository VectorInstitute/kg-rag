"""Main implementation of Cypher-based KG-RAG approach."""

import json
import re
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI

from kg_rag.utils.prompts import create_query_prompt

from .cypher_generator import CYPHER_GENERATION_PROMPT, CypherGenerator


class CypherBasedKGRAG:
    """Main class implementing the Cypher-based KG-RAG system."""

    def __init__(
        self,
        graph: Neo4jGraph,
        llm: ChatOpenAI | None = None,
        cypher_llm: ChatOpenAI | None = None,
        qa_llm: ChatOpenAI | None = None,
        cypher_prompt: PromptTemplate | None = None,
        max_depth: int = 2,
        max_hops: int = 3,
        use_cot: bool = False,
        numerical_answer: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the Cypher-based KG-RAG system.

        Args:
            graph: Neo4jGraph instance
            llm: LLM for both Cypher generation and QA (if cypher_llm and qa_llm not provided)
            cypher_llm: LLM for Cypher generation
            qa_llm: LLM for QA
            cypher_prompt: Prompt template for Cypher generation
            max_depth: Maximum depth for graph exploration
            max_hops: Maximum number of hops for graph exploration
            use_cot: Whether to use Chain-of-Thought prompting
            numerical_answer: Whether to format answers as numerical values only
            verbose: Whether to print verbose output
        """
        # Set up LLMs
        if llm is None:
            llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

        self.cypher_llm = cypher_llm or llm
        self.qa_llm = qa_llm or llm

        # Configure QA LLM with response format if using CoT or numerical answer
        if use_cot or numerical_answer:
            self.qa_llm = self.qa_llm.bind(response_format={"type": "json_object"})  # type: ignore

        # Create Cypher generator
        self.cypher_generator = CypherGenerator(
            llm=self.cypher_llm,
            cypher_prompt=cypher_prompt or CYPHER_GENERATION_PROMPT,
            verbose=verbose,
        )

        # Store graph
        self.graph = graph

        # Get graph schema
        self.schema = self.cypher_generator.get_schema(graph)

        # Create GraphCypherQAChain
        self.chain = GraphCypherQAChain.from_llm(
            cypher_llm=self.cypher_llm,
            qa_llm=self.qa_llm,
            cypher_prompt=cypher_prompt or CYPHER_GENERATION_PROMPT,
            graph=graph,
            verbose=verbose,
            max_depth=max_depth,
            max_hops=max_hops,
            validate_cypher=True,
            allow_dangerous_requests=True,
        )

        # Other settings
        self.use_cot = use_cot
        self.numerical_answer = numerical_answer
        self.verbose = verbose

    def query(self, question: str) -> Any:
        """
        Process query and return structured response.

        Args:
            question: User question to process

        Returns
        -------
            Dictionary with 'answer' and 'reasoning' keys
        """
        if self.verbose:
            print(f"Processing query: {question}")

        # Get Cypher query
        cypher_query = self.cypher_generator.generate_cypher(question, self.schema)

        if self.verbose:
            print(f"Generated Cypher query: {cypher_query}")

        # Execute Cypher query
        results = self.graph.query(cypher_query)

        # Format results as context
        context = json.dumps(results, indent=2)

        # Create the prompt
        messages = create_query_prompt(
            question=question,
            context=context,
            system_type="cypher",
            use_cot=self.use_cot,
            numerical_answer=self.numerical_answer,
        )

        # Generate the answer
        response = self.qa_llm.invoke(messages)
        content = response.content if hasattr(response, "content") else response

        if self.use_cot or self.numerical_answer:
            try:
                if isinstance(response, str):
                    return json.loads(content)
            except json.JSONDecodeError:
                if self.verbose:
                    print(f"Error parsing JSON response: {content}")
                return {
                    "reasoning": "Error parsing the response.",
                    "answer": str(content),
                }
        else:
            # For standard mode, wrap the answer in our consistent format
            return {
                "answer": content,
                "reasoning": "This answer was generated based on data retrieved using Cypher queries against the knowledge graph.",
            }

    def get_explicit_cypher(self, question: str) -> str | list[str | dict]:
        """
        Generate a Cypher query for a question.

        Args:
            question: User question

        Returns
        -------
            Generated Cypher query
        """
        return self.cypher_generator.generate_cypher(question, self.schema)

    def run_cypher(self, cypher: str) -> list[dict[str, Any]]:
        """
        Run a Cypher query directly.

        Args:
            cypher: Cypher query to run

        Returns
        -------
            Query results
        """
        return self.graph.query(cypher)

    def _extract_answer_from_text(self, text: str) -> str:
        """
        Extract the main answer from text response.

        Args:
            text: Response text to extract answer from

        Returns
        -------
            Extracted answer
        """
        if self.numerical_answer:
            # For numerical answers, try to extract a number
            number_patterns = [
                r"answer\s*(?:is|:)\s*(-?\d+(?:\.\d+)?)",  # "answer is 42" or "answer: 42"
                r"(-?\d+(?:\.\d+)?)\s*%",  # "42%"
                r"(-?\d+(?:\.\d+)?)\s*(?:million|billion|dollars|USD)",  # "42 million" or "42 dollars"
                r"(?:value|amount|total)\s*(?:of|is|:)\s*(-?\d+(?:\.\d+)?)",  # "value is 42" or "amount: 42"
                r"(\d+(?:\.\d+)?)",  # Any number as a fallback
            ]

            for pattern in number_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1)

        # For general answers, look for common answer patterns
        answer_patterns = [
            r"(?:answer|conclusion)(?:\s+is|:)\s+(.*?)(?:\.|$)",  # "The answer is..." or "Answer: ..."
            r"(?:in\s+conclusion|therefore)[,:\s]+(.*?)(?:\.|$)",  # "In conclusion..." or "Therefore..."
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no patterns match, return the text as is
        return text
