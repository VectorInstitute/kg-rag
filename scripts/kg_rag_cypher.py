import os
from pathlib import Path
from dotenv import load_dotenv
import neo4j
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph

# Load environment variables
load_dotenv()

# Neo4j connection settings
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')

# Initialize OpenAI client
llm = ChatOpenAI(temperature=0, 
                 model_name="gpt-4o-mini", 
                 api_key="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")
llm_transformer = LLMGraphTransformer(llm=llm)

def load_10q_text(file_path):
    """Load SEC 10-Q text file and split into manageable chunks"""
    with open(file_path, 'r') as f:
        text = f.read()
        
    # Split into ~2000 character chunks with overlap
    chunk_size = 2000
    overlap = 200
    chunks = []
    
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(Document(page_content=chunk))
        
    return chunks

def create_graph_from_text(file_path):
    """Load SEC 10-Q text, extract graph info, and store in Neo4j"""
    doc_chunks = load_10q_text(file_path)

    all_nodes = []
    all_relationships = []

    for chunk in doc_chunks:
        graph_documents = llm_transformer.convert_to_graph_documents([chunk])
        for doc in graph_documents:
            all_nodes.extend(doc.nodes)
            all_relationships.extend(doc.relationships)

    unique_nodes = list({node.id: node for node in all_nodes}.values())
    unique_relationships = list({(rel.source.id, rel.target.id): rel for rel in all_relationships}.values())

    print(f"Extracted {len(unique_nodes)} unique nodes and {len(unique_relationships)} unique relationships")

    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD
    )

    # Clear existing data
    graph.query("MATCH (n) DETACH DELETE n")

    # Store nodes
    for node in unique_nodes:
        cypher = f"""
        CREATE (n:{node.type} {{id: $id}})
        """
        graph.query(cypher, {'id': node.id})

    # Store relationships
    for rel in unique_relationships:
        cypher = f"""
        MATCH (source:{rel.source.type} {{id: $source_id}})
        MATCH (target:{rel.target.type} {{id: $target_id}})
        CREATE (source)-[:{rel.type}]->(target)
        """
        graph.query(cypher, {
            'source_id': rel.source.id,
            'target_id': rel.target.id
        })
    
    return graph

from operator import add
from typing import Annotated, List
from typing_extensions import TypedDict
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from typing import List, Optional
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from neo4j.exceptions import CypherSyntaxError
from langgraph.graph import END, START, StateGraph
from IPython.display import Image, display


class InputState(TypedDict):
    question: str

class OverallState(TypedDict):
    question: str
    next_action: str
    cypher_statement: str
    cypher_errors: List[str]
    database_records: List[dict]
    steps: Annotated[List[str], add]

class OutputState(TypedDict):
    answer: str
    steps: List[str]
    cypher_statement: str

guardrails_system = """
As an intelligent assistant, your primary objective is to decide whether a given question is related to the knowledge graph or not. 
If the question is related to the knowledge graph, output "relevant". Otherwise, output "end".
To make this decision, assess the content of the question and determine if it refers to any entities or relationships in the knowledge graph.
Provide only the specified output: "relevant" or "end".
"""

guardrails_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            guardrails_system,
        ),
        (
            "human",
            ("{question}"),
        ),
    ]
)

class GuardrailsOutput(BaseModel):
    decision: Literal["relevant", "end"] = Field(
        description="Decision on whether the question is related to the knowledge graph"
    )

guardrails_chain = guardrails_prompt | llm.with_structured_output(GuardrailsOutput)

def guardrails(state: InputState) -> OverallState:
    """
    Decides if the question is related to the knowledge graph or not.
    """
    guardrails_output = guardrails_chain.invoke({"question": state.get("question")})
    database_records = None
    if guardrails_output.decision == "end":
        database_records = "This question is not related to the knowledge graph. Therefore I cannot answer this question."
    return {
        "next_action": guardrails_output.decision,
        "database_records": database_records,
        "steps": ["guardrail"],
    }

# examples = [
#     {
#         "question": "What are the key risks mentioned?",
#         "query": "MATCH (r:Risk) RETURN r.text",
#     },
#     {
#         "question": "What are the significant events?",
#         "query": "MATCH (e:Event) RETURN e.text",
#     },
#     {
#         "question": "What are the financial highlights?",
#         "query": "MATCH (f:Financial) RETURN f.text",
#     },
#     {
#         "question": "What are the legal proceedings?",
#         "query": "MATCH (l:Legal) RETURN l.text",
#     },
#     {
#         "question": "What are the company's strategies?",
#         "query": "MATCH (s:Strategy) RETURN s.text",
#     },
#     {
#         "question": "What are the company's challenges?",
#         "query": "MATCH (c:Challenge) RETURN c.text",
#     },
#     {
#         "question": "What are the company's opportunities?",
#         "query": "MATCH (o:Opportunity) RETURN o.text",
#     },
#     {
#         "question": "What are the company's products?",
#         "query": "MATCH (p:Product) RETURN p.text",
#     },
#     {
#         "question": "What are the company's services?",
#         "query": "MATCH (s:Service) RETURN s.text",
#     },
#     {
#         "question": "What are the company's competitors?",
#         "query": "MATCH (c:Competitor) RETURN c.text",
#     },
#     {
#         "question": "What are the company's customers?",
#         "query": "MATCH (c:Customer) RETURN c.text",
#     }
# ]

# example_selector = SemanticSimilarityExampleSelector.from_examples(
#     examples, OpenAIEmbeddings(), Neo4jVector, k=5, input_keys=["question"], url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD
# )

text2cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Given an input question, convert it to a Cypher query. No pre-amble."
                "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
            ),
        ),
        (
            "human",
            (
                """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!
Here is the schema information
{schema}

Below are a number of examples of questions and their corresponding Cypher queries.

{fewshot_examples}

User input: {question}
Cypher query:"""
            ),
        ),
    ]
)

text2cypher_chain = text2cypher_prompt | llm | StrOutputParser()

def generate_cypher(state: OverallState) -> OverallState:
    """
    Generates a cypher statement based on the provided schema and user input
    """
    NL = "\n"
    fewshot_examples = (NL * 2).join(
        [
            f"Question: {el['question']}{NL}Cypher:{el['query']}"
            for el in example_selector.select_examples(
                {"question": state.get("question")}
            )
        ]
    )
    
    # Get the graph schema
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD
    )
    enhanced_graph = Neo4jGraph(enhanced_schema=True)
    
    generated_cypher = text2cypher_chain.invoke(
        {
            "question": state.get("question"),
            "fewshot_examples": fewshot_examples,
            "schema": graph.schema,
        }
    )
    return {"cypher_statement": generated_cypher, "steps": ["generate_cypher"]}

validate_cypher_system = """
You are a Cypher expert reviewing a statement written by a junior developer.
"""

validate_cypher_user = """You must check the following:
* Are there any syntax errors in the Cypher statement?
* Are there any missing or undefined variables in the Cypher statement?
* Are any node labels missing from the schema?
* Are any relationship types missing from the schema?
* Are any of the properties not included in the schema?
* Does the Cypher statement include enough information to answer the question?

Examples of good errors:
* Label (:Foo) does not exist, did you mean (:Bar)?
* Property bar does not exist for label Foo, did you mean baz?
* Relationship FOO does not exist, did you mean FOO_BAR?

Schema:
{schema}

The question is:
{question}

The Cypher statement is:
{cypher}

Make sure you don't make any mistakes!"""

validate_cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            validate_cypher_system,
        ),
        (
            "human",
            (validate_cypher_user),
        ),
    ]
)

class Property(BaseModel):
    """
    Represents a filter condition based on a specific node property in a graph in a Cypher statement.
    """
    node_label: str = Field(
        description="The label of the node to which this property belongs."
    )
    property_key: str = Field(description="The key of the property being filtered.")
    property_value: str = Field(
        description="The value that the property is being matched against."
    )

class ValidateCypherOutput(BaseModel):
    """
    Represents the validation result of a Cypher query's output,
    including any errors and applied filters.
    """
    errors: Optional[List[str]] = Field(
        description="A list of syntax or semantical errors in the Cypher statement. Always explain the discrepancy between schema and Cypher statement"
    )
    filters: Optional[List[Property]] = Field(
        description="A list of property-based filters applied in the Cypher statement."
    )

validate_cypher_chain = validate_cypher_prompt | llm.with_structured_output(
    ValidateCypherOutput
)

# Cypher query corrector is experimental
def get_cypher_query_corrector():
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD
    )
    enhanced_graph = Neo4jGraph(enhanced_schema=True)
    corrector_schema = [
        Schema(el["start"], el["type"], el["end"])
        for el in graph.structured_schema.get("relationships")
    ]
    return CypherQueryCorrector(corrector_schema)

def validate_cypher(state: OverallState) -> OverallState:
    """
    Validates the Cypher statements and maps any property values to the database.
    """
    errors = []
    mapping_errors = []
    
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD
    )
    enhanced_graph = Neo4jGraph(enhanced_schema=True)
    
    # Check for syntax errors
    try:
        graph.query(f"EXPLAIN {state.get('cypher_statement')}")
    except CypherSyntaxError as e:
        errors.append(e.message)
    # Experimental feature for correcting relationship directions
    cypher_query_corrector = get_cypher_query_corrector()
    corrected_cypher = cypher_query_corrector(state.get("cypher_statement"))
    if not corrected_cypher:
        errors.append("The generated Cypher statement doesn't fit the graph schema")
    if not corrected_cypher == state.get("cypher_statement"):
        print("Relationship direction was corrected")
    # Use LLM to find additional potential errors and get the mapping for values
    llm_output = validate_cypher_chain.invoke(
        {
            "question": state.get("question"),
            "schema": graph.schema,
            "cypher": state.get("cypher_statement"),
        }
    )
    if llm_output.errors:
        errors.extend(llm_output.errors)
    if llm_output.filters:
        for filter in llm_output.filters:
            # Do mapping only for string values
            if (
                not [
                    prop
                    for prop in enhanced_graph.structured_schema["node_props"][
                        filter.node_label
                    ]
                    if prop["property"] == filter.property_key
                ][0]["type"]
                == "STRING"
            ):
                continue
            mapping = graph.query(
                f"MATCH (n:{filter.node_label}) WHERE toLower(n.`{filter.property_key}`) = toLower($value) RETURN 'yes' LIMIT 1",
                {"value": filter.property_value},
            )
            if not mapping:
                print(
                    f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"
                )
                mapping_errors.append(
                    f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"
                )
    if mapping_errors:
        next_action = "end"
    elif errors:
        next_action = "correct_cypher"
    else:
        next_action = "execute_cypher"

    return {
        "next_action": next_action,
        "cypher_statement": corrected_cypher,
        "cypher_errors": errors,
        "steps": ["validate_cypher"],
    }

correct_cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a Cypher expert reviewing a statement written by a junior developer. "
                "You need to correct the Cypher statement based on the provided errors. No pre-amble."
                "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
            ),
        ),
        (
            "human",
            (
                """Check for invalid syntax or semantics and return a corrected Cypher statement.

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not wrap the response in any backticks or anything else.
Respond with a Cypher statement only!

Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.

The question is:
{question}

The Cypher statement is:
{cypher}

The errors are:
{errors}

Corrected Cypher statement: """
            ),
        ),
    ]
)

correct_cypher_chain = correct_cypher_prompt | llm | StrOutputParser()

def correct_cypher(state: OverallState) -> OverallState:
    """
    Correct the Cypher statement based on the provided errors.
    """
    
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD
    )
    enhanced_graph = Neo4jGraph(enhanced_schema=True)
    
    corrected_cypher = correct_cypher_chain.invoke(
        {
            "question": state.get("question"),
            "errors": state.get("cypher_errors"),
            "cypher": state.get("cypher_statement"),
            "schema": graph.schema,
        }
    )

    return {
        "next_action": "validate_cypher",
        "cypher_statement": corrected_cypher,
        "steps": ["correct_cypher"],
    }

no_results = "I couldn't find any relevant information in the database"

def execute_cypher(state: OverallState) -> OverallState:
    """
    Executes the given Cypher statement.
    """
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD
    )
    records = graph.query(state.get("cypher_statement"))
    return {
        "database_records": records if records else no_results,
        "next_action": "end",
        "steps": ["execute_cypher"],
    }

generate_final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant",
        ),
        (
            "human",
            (
                """Use the following results retrieved from a database to provide
a succinct, definitive answer to the user's question.

Respond as if you are answering the question directly.

Results: {results}
Question: {question}"""
            ),
        ),
    ]
)

generate_final_chain = generate_final_prompt | llm | StrOutputParser()

def generate_final_answer(state: OverallState) -> OutputState:
    """
    Decides if the question is related to the knowledge graph.
    """
    final_answer = generate_final_chain.invoke(
        {"question": state.get("question"), "results": state.get("database_records")}
    )
    return {"answer": final_answer, "steps": ["generate_final_answer"]}

def guardrails_condition(
    state: OverallState,
) -> Literal["generate_cypher", "generate_final_answer"]:
    if state.get("next_action") == "end":
        return "generate_final_answer"
    elif state.get("next_action") == "relevant":
        return "generate_cypher"

def validate_cypher_condition(
    state: OverallState,
) -> Literal["generate_final_answer", "correct_cypher", "execute_cypher"]:
    if state.get("next_action") == "end":
        return "generate_final_answer"
    elif state.get("next_action") == "correct_cypher":
        return "correct_cypher"
    elif state.get("next_action") == "execute_cypher":
        return "execute_cypher"

langgraph = StateGraph(OverallState, input=InputState, output=OutputState)
langgraph.add_node(guardrails)
langgraph.add_node(generate_cypher)
langgraph.add_node(validate_cypher)
langgraph.add_node(correct_cypher)
langgraph.add_node(execute_cypher)
langgraph.add_node(generate_final_answer)

langgraph.add_edge(START, "guardrails")
langgraph.add_conditional_edges(
    "guardrails",
    guardrails_condition,
)
langgraph.add_edge("generate_cypher", "validate_cypher")
langgraph.add_conditional_edges(
    "validate_cypher",
    validate_cypher_condition,
)
langgraph.add_edge("execute_cypher", "generate_final_answer")
langgraph.add_edge("correct_cypher", "validate_cypher")
langgraph.add_edge("generate_final_answer", END)

langgraph = langgraph.compile()

def run_kg_rag_cypher(query, file_path):
    # graph = create_graph_from_text(file_path)
    result = langgraph.invoke({"question": query})
    return result

if __name__ == "__main__":
    query = "What are the key risks mentioned?"
    file_path = "data/sec-10-q/text/2022 Q3 AAPL.txt"
    result = run_kg_rag_cypher(query, file_path)
    print(result)
