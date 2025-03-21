"""Cypher query generator for Cypher-based KG-RAG approach."""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


CYPHER_GENERATION_TEMPLATE = """
Task:
Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
If you cannot generate a Cypher statement based on the provided schema, explain the reason to the user.
When you use `MATCH` with `WHERE` clauses, always first check the entities' or relationships; id property rather than name.
Schema:
{schema}
IMPORTANT:
Do not include any explanations or apologies in your response.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples:
Here are a few examples of generated Cypher statements for particular questions:
# Example 1: What entities relates to Nvidia Corporation?
MATCH p=()-[]->(n:Company)-[]->()
where n.id = "Nvidia Corporation"
RETURN P LIMIT 50
# Example 2: What assets do Nvidia Corporation have?
MATCH p=(n:Company)-[r:HAS]->(a)
where n.id = "Nvidia Corporation"
RETURN P LIMIT 25
# Example 3: On April 1, 2023, what was the Amount of BEGINNING_CASH_BALANCE?
MATCH (d:Date)-[r:BEGINNING_CASH_BALANCE]->(a:Amount)
WHERE d.id = "April 1, 2023"
RETURN a.id LIMIT 1
The question is:
{question}
"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE,
)


class CypherGenerator:
    """Generates Cypher queries from natural language questions."""

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        cypher_prompt: PromptTemplate | None = None,
        verbose: bool = False,
    ):
        """
        Initialize the Cypher generator.

        Args:
            llm: LLM to use for query generation (default: ChatOpenAI with gpt-4o)
            cypher_prompt: Prompt template for Cypher generation
            verbose: Whether to print verbose output
        """
        # Set up LLM if not provided
        if llm is None:
            self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
        else:
            self.llm = llm

        # Set up prompt template
        self.prompt = cypher_prompt or CYPHER_GENERATION_PROMPT

        # Other settings
        self.verbose = verbose

    def get_schema(self, graph):
        """
        Get the schema from the graph.

        Args:
            graph: Neo4jGraph instance

        Returns
        -------
            Schema string for use in prompt
        """
        # Get node labels and properties
        node_query = """
        CALL apoc.meta.schema()
        YIELD value
        RETURN value
        """

        schema_result = graph.query(node_query)

        if not schema_result:
            return "No schema information available."

        # Extract node information
        node_info = []
        relation_info = []
        relation_patterns = []

        if "nodes" in schema_result[0]["value"]:
            nodes = schema_result[0]["value"]["nodes"]
            for label, info in nodes.items():
                properties = info.get("properties", {})
                property_str = ", ".join(
                    [f"{prop}: {dtype}" for prop, dtype in properties.items()]
                )
                node_info.append(f"{label} {{{property_str}}}")

        # Extract relationship information
        if "relationships" in schema_result[0]["value"]:
            relationships = schema_result[0]["value"]["relationships"]
            for rel_type, info in relationships.items():
                properties = info.get("properties", {})
                property_str = ", ".join(
                    [f"{prop}: {dtype}" for prop, dtype in properties.items()]
                )
                if property_str:
                    relation_info.append(f"{rel_type} {{{property_str}}}")
                else:
                    relation_info.append(rel_type)

                # Add relationship patterns
                start_labels = info.get("start", [])
                end_labels = info.get("end", [])

                for start in start_labels:
                    for end in end_labels:
                        relation_patterns.append(f"(:{start})-[:{rel_type}]->(:{end})")

        # Construct schema string
        schema = []
        if node_info:
            schema.append("Node properties:")
            schema.append("\n".join(node_info))

        if relation_info:
            schema.append("\nRelationship properties:")
            schema.append("\n".join(relation_info))

        if relation_patterns:
            schema.append("\nThe relationships:")
            schema.append("\n".join(relation_patterns))

        return "\n".join(schema)

    def generate_cypher(self, question: str, schema: str) -> str | list[str | dict]:
        """
        Generate a Cypher query from a natural language question.

        Args:
            question: Natural language question
            schema: Schema of the graph

        Returns
        -------
            Generated Cypher query
        """
        if self.verbose:
            print(f"Generating Cypher query for: {question}")

        # Generate Cypher query
        response = self.llm.invoke(self.prompt.format(schema=schema, question=question))

        query = response.content if hasattr(response, "content") else str(response)

        if self.verbose:
            print(f"Generated Cypher query: {query}")

        return query
