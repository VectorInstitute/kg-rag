# Cypher-based KG-RAG Method

<p align="center">
  <img src="../../../assets/cypher.png" />
</p>

## Overview

The Cypher-based Knowledge Graph RAG (KG-RAG) method leverages the power of graph databases, specifically Neo4j, to provide context-rich retrieval for generative AI. Rather than using vector embeddings alone, this approach transforms unstructured text into a structured knowledge graph, and then uses Cypher queries to extract relevant information for answering user questions.

## Architecture

The Cypher-based KG-RAG system consists of the following key components:

### 1. Graph Constructor (`graph_constructor.py`)

The Graph Constructor is responsible for:

- Converting documents into graph-based representations using LLMs
- Extracting entities and relationships from text
- Building and populating a Neo4j graph database

The process uses `LLMGraphTransformer` to analyze documents and identify entities, their types, and relationships between them. This structured information is then loaded into Neo4j, where it can be efficiently queried.

### 2. Cypher Generator (`cypher_generator.py`)

The Cypher Generator transforms natural language questions into Cypher queries that can be executed against the Neo4j database. Key features include:

- A specialized prompt template that helps the LLM understand how to generate valid Cypher
- Schema-awareness that ensures generated queries use the correct entity types and relationships
- Error handling for malformed queries

The Cypher generation process follows these steps:
1. Extract the database schema from Neo4j
2. Interpret the user's natural language question
3. Generate a Cypher query that maps the question to the available entities and relationships
4. Validate the query to ensure it will execute properly

### 3. Main KG-RAG System (`kg_rag.py`)

The main KG-RAG class integrates all components and handles the end-to-end query process:

1. Receive a natural language question
2. Generate appropriate Cypher query
3. Execute the query against the Neo4j database
4. Format the results as context for the LLM
5. Generate a comprehensive answer using the retrieved context

## Workflow

The complete workflow of the Cypher-based method is as follows:

1. **Graph Construction (One-time setup)**
   - Load and chunk documents
   - Use LLM to extract entities and relationships
   - Populate Neo4j database with the structured knowledge graph

2. **Query Processing (For each user question)**
   - Parse the natural language question
   - Generate a Cypher query using the LLM
   - Execute the Cypher query against Neo4j
   - Retrieve relevant graph substructures as context
   - Generate a comprehensive answer using the LLM with the retrieved context

## Implementation Details

### LLM Components

The system uses two main LLM components:
- **Cypher LLM**: Specialized for generating valid Cypher queries from natural language
- **QA LLM**: Optimized for generating comprehensive answers from the retrieved context

Both can be the same model (default: gpt-4o) but are separated for flexibility.

### Neo4j Integration

The system uses the `langchain_neo4j` package to:
- Connect to Neo4j databases
- Execute Cypher queries
- Retrieve and format graph-structured data

### Key Parameters

- `max_depth`: Controls the depth of graph exploration during query execution
- `max_hops`: Limits the number of relationship hops in generated Cypher queries
- `use_cot`: Enables Chain-of-Thought reasoning for more complex questions
- `numerical_answer`: Formats responses as numerical values when appropriate

## Advantages Over Vector-Based RAG

1. **Structural awareness**: Understands relationships between entities, enabling multi-hop reasoning
2. **Precise retrieval**: Can perform targeted queries rather than similarity-based matching
3. **Answer validation**: Can verify answers directly against structured data
4. **Transparency**: Provides explicit query paths that show how information was retrieved
5. **Complex reasoning**: Supports multi-step inference by traversing relationship paths

## Example Cypher Queries

```cypher
// Finding entities related to a specific company
MATCH p=()-[]->(n:Company)-[]->()
WHERE n.id = "Nvidia Corporation"
RETURN p LIMIT 50

// Finding assets of a company
MATCH p=(n:Company)-[r:HAS]->(a)
WHERE n.id = "Nvidia Corporation"
RETURN p LIMIT 25

// Finding specific financial data
MATCH (d:Date)-[r:BEGINNING_CASH_BALANCE]->(a:Amount)
WHERE d.id = "April 1, 2023"
RETURN a.id LIMIT 1
```

## Usage

### Building the Knowledge Graph

```bash
python -m scripts.build_cypher_graph \
    --docs-dir data/sec-10-q/docs \
    --output-dir data/graphs \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --verbose
```

### Running Queries

```bash
python -m scripts.run_cypher_rag \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --max-depth 2 \
    --max-hops 3 \
    --verbose
```

## Limitations and Considerations

1. **Graph quality**: The performance depends heavily on the quality of the constructed knowledge graph
2. **Query complexity**: Some questions may require complex Cypher queries that are difficult to generate
3. **Schema dependency**: The system needs to understand the graph schema to generate valid queries

## Future Improvements

1. Implementing more sophisticated Cypher query templates for complex questions
3. Developing hybrid approaches that combine vector similarity with graph traversal
4. Implementing query optimization in a loop to improve performance.

## References

- Langchain Cypher Documentation: [https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/](https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/)
