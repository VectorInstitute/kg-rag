# %% [markdown]
# # SEC 10-Q Knowledge Graph Construction
# 
# This notebook demonstrates constructing a knowledge graph from SEC 10-Q filings using LangChain. The approach uses LLM-based extraction to identify entities and relationships without pre-defining a schema.

# %%
import os
from pathlib import Path
from dotenv import load_dotenv
import neo4j
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import networkx as nx
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
llm = ChatOpenAI(temperature=0, 
                 model_name="gpt-4o", 
                 api_key=os.environ["OPENAI_API_KEY"])
llm_transformer = LLMGraphTransformer(llm=llm)

# %%
import os
import pickle

# Check if "graph_documents.pkl" exists
if os.path.exists("graph_documents.pkl"):
    # Load graph_documents from the file
    with open("graph_documents.pkl", "rb") as f:
        graph_documents = pickle.load(f)
    print("Loaded graph_documents from graph_documents.pkl")
else:
    # Convert documents to graph documents
    graph_documents = llm_transformer.convert_to_graph_documents(tqdm(documents))
    
    # Save graph_documents to the file
    with open("graph_documents.pkl", "wb") as f:
        pickle.dump(graph_documents, f)
    print("Converted documents to graph_documents and saved to graph_documents.pkl")

# %%
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph

graph = NetworkxEntityGraph()

# Add nodes to the graph
for doc in graph_documents:
    for node in doc.nodes:
        graph.add_node(node.id)

for doc in graph_documents:
    for edge in doc.relationships:
        graph._graph.add_edge(
            edge.source.id,
            edge.target.id,
            relation=edge.type,
        )

# %%
import random

# Define the depth to go down
n_hops = 5

# Select a random node from the graph
random_node = "Apple Inc."

# Function to recursively add a random neighbor up to n_hops
def add_neighbors(node, depth, nodes_to_include):
    if depth > 0:
        neighbors = list(graph._graph.neighbors(node))
        if neighbors:
            random_neighbor = random.choice(neighbors)
            nodes_to_include.add(random_neighbor)
            add_neighbors(random_neighbor, depth - 1, nodes_to_include)

# Create 16 subplots
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
axes = axes.flatten()

for i in range(16):
    # Initialize the list of nodes to include in the subgraph
    nodes_to_include = {random_node}
    
    # Add neighbors starting from the random node
    add_neighbors(random_node, n_hops, nodes_to_include)
    
    # Extract the subgraph containing the selected nodes
    subgraph = graph._graph.subgraph(nodes_to_include)
    
    # Draw the subgraph
    pos = nx.spring_layout(subgraph)
    nx.draw(subgraph, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_weight="bold", ax=axes[i])
    edge_labels = nx.get_edge_attributes(subgraph, 'relation')
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_color='red', ax=axes[i])
    axes[i].set_title(f"Subgraph {i+1}")

plt.tight_layout()
plt.show()

# %% [markdown]
# The knowledge graph has been constructed and stored in Neo4j. You can now query it using Cypher or use it for downstream tasks like question answering.

# %%
from langchain_openai import OpenAIEmbeddings
import numpy as np
from scipy.spatial.distance import cosine

class EntityLinker:
    def __init__(self, graph):
        self.node_ids = list(graph._graph.nodes())
        self.embeddings_model = OpenAIEmbeddings()
        self.node_embeddings = self.embeddings_model.embed_documents(self.node_ids)
    
    def link_entities(self, query, top_n=3):
        # Extract entities using LLM
        entity_prompt = f"""Extract key entities from this query: {query}
        Return as comma-separated list:"""
        entities = llm.invoke(entity_prompt).content.split(',')
        
        # Find closest nodes for each entity
        matched_nodes = []
        for entity in entities:
            query_embed = self.embeddings_model.embed_query(entity.strip())
            similarities = [1 - cosine(query_embed, node_embed) for node_embed in self.node_embeddings]
            top_indices = np.argsort(similarities)[-top_n:]
            matched_nodes.extend([self.node_ids[i] for i in top_indices])
        
        return list(set(matched_nodes))

# %%
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import random

def expand_subgraph(graph, seed_nodes, depth=2):
    """
    Expand subgraph from seed nodes while preserving edge attributes
    """
    subgraph = nx.DiGraph()
    nodes_to_explore = set(seed_nodes)
    explored = set()
    
    for _ in range(depth):
        current_level = nodes_to_explore - explored
        if not current_level:
            break
            
        for node in current_level:
            if node in graph:
                # Add outgoing edges
                for _, target, data in graph.edges(node, data=True):
                    subgraph.add_edge(node, target, **data)
                
                # Add incoming edges
                for source, _, data in graph.in_edges(node, data=True):
                    subgraph.add_edge(source, node, **data)
                
                # Add neighbors to exploration set
                nodes_to_explore.update(graph.successors(node))
                nodes_to_explore.update(graph.predecessors(node))
        
        explored.update(current_level)
    
    return subgraph

class GraphQAChain:
    def __init__(self, llm, nx_graph, entity_linker, verbose=True):
        self.llm = llm
        self.nx_graph = nx_graph
        self.entity_linker = entity_linker
        self.verbose = verbose
        self.qa_prompt = PromptTemplate(
            template="""Based on the following knowledge graph triplets:
            {triplets}
            
            Please answer this question: {question}
            
            If you cannot find the exact information in the triplets, say "I cannot find the specific information in the available data."
            """,
            input_variables=["triplets", "question"]
        )

    def _print_verbose(self, msg, data=None):
        """Helper method for verbose output with optional data"""
        if self.verbose:
            print("\n" + "="*50)
            print(msg)
            if data:
                if isinstance(data, list):
                    for item in data:
                        print(f"  • {item}")
                else:
                    print(data)
            print("="*50 + "\n")

    def invoke(self, query):
        # Step 1: Entity linking
        seed_nodes = self.entity_linker.link_entities(query)
        if self.verbose:
            self._print_verbose("🔍 Detected Entities:", seed_nodes)
        
        # Step 2: Subgraph expansion
        subgraph = expand_subgraph(self.nx_graph, seed_nodes)
        
        # Step 3: Triplet extraction and formatting
        try:
            triplets = [
                f"{u} → {data.get('relation', 'RELATED_TO')} → {v}"
                for u, v, data in subgraph.edges(data=True)
            ]
            
            if not triplets:
                self._print_verbose("⚠️ No relevant triplets found in knowledge graph")
                return self.llm.invoke("I cannot find any relevant connections in the knowledge graph to answer this question.")
            
            # If there are many triplets, show a sample in verbose mode
            if self.verbose:
                sample_size = min(5, len(triplets))
                sample_triplets = random.sample(triplets, sample_size)
                self._print_verbose(
                    f"📊 Found {len(triplets)} relevant connections. Sample of {sample_size}:", 
                    sample_triplets
                )
            
            # Step 4: LLM reasoning
            response = self.llm.invoke(self.qa_prompt.format(
                triplets="\n".join(triplets),
                question=query
            ))
            
            if self.verbose:
                self._print_verbose("🤔 Final Answer:", response.content)
            
            return response

        except Exception as e:
            error_msg = f"Error processing graph: {str(e)}"
            self._print_verbose("❌ Error:", error_msg)
            return self.llm.invoke("There was an error processing the knowledge graph structure.")

# %%
# Initialize linker component
entity_linker = EntityLinker(graph)

# %%
enhanced_chain = GraphQAChain(llm, graph._graph, entity_linker)

# Sample usage
response = enhanced_chain.invoke(
    "What was Apple Inc's Products gross margin percentage for the third quarter of 2022? Provide the percentage rounded to one decimal place."
)

# %%
enhanced_chain.invoke(invoke(input="Where was Apple Inc. Incorporated?"))

# %%
graph_chain.invoke(input=" On April 1, 2023, what was the Amount of CASH_BEGINNING_BALANCE?")

# %%
graph_chain.invoke(input="What assets does Apple Inc. have?")

# %%
graph_chain.invoke(input="Apple inc. What was the amount for Cash Used In Investing Activities in 2023 Q3?")

# %%
graph_chain.invoke(input="What was Apple Inc's Products gross margin percentage for the third quarter of 2022? Provide the percentage rounded to one decimal place.") 

# %%
# Load the CSV file
df = pd.read_csv("../../data/sec-10-q/synthetic_qna_data_7_gpt4o.csv")

# Filter for rows where Source Docs contains only AAPL
apple_df = df[df['Source Docs'].str.contains('AAPL', na=False)]

# Take first 10 samples
apple_df = apple_df.head(10)

# Evaluate the model
correct = 0
for i, row in apple_df.iterrows():
    question = row["New Question"]
    answer = row["New Answer"]
    print(f"\nQuestion: {question}")
    print(f"Expected Answer: {answer}")
    response = graph_chain.invoke(input=question)
    print(f"Model Response: {response}")
    if response == answer:
        correct += 1
        
print(f"\nAccuracy: {correct / 10}")


