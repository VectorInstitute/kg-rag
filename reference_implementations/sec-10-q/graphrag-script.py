# %% [markdown]
# # GraphRAG-based Question Answering
# 
# This notebook demonstrates using the langchain-graphrag library to implement a knowledge graph-based RAG system.
# 
# The approach involves:
# 1. Text extraction and splitting into units
# 2. Graph generation using entity and relationship extraction
# 3. Graph community detection
# 4. Question answering using either local or global search strategies

# %%
import os
from pathlib import Path
from typing import cast
from dotenv import load_dotenv
import pandas as pd

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.cache import SQLiteCache
from langchain_text_splitters import TokenTextSplitter

from langchain_graphrag.indexing import SimpleIndexer, TextUnitExtractor
from langchain_graphrag.indexing.artifacts_generation import (
    CommunitiesReportsArtifactsGenerator,
    EntitiesArtifactsGenerator, 
    RelationshipsArtifactsGenerator,
    TextUnitsArtifactsGenerator
)
from langchain_graphrag.indexing.graph_clustering import HierarchicalLeidenCommunityDetector
from langchain_graphrag.indexing.graph_generation import (
    EntityRelationshipExtractor,
    EntityRelationshipDescriptionSummarizer,
    GraphGenerator,
    GraphsMerger
)
from langchain_graphrag.indexing.report_generation import (
    CommunityReportGenerator,
    CommunityReportWriter
)
from langchain_graphrag.types.graphs.community import CommunityLevel
from langchain_graphrag.utils import TiktokenCounter

# Load environment variables
load_dotenv()

# Setup paths
CACHE_DIR = Path("cache")
VECTOR_STORE_DIR = Path("vector_stores") 
ARTIFACTS_DIR = Path("artifacts")

for p in [CACHE_DIR, VECTOR_STORE_DIR, ARTIFACTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Configure Environment and Models
# 
# Set up the required models and environment variables.

# %%
# Create the LLMs
er_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    api_key=os.environ["OPENAI_API_KEY"],
    cache=SQLiteCache(str(CACHE_DIR / "openai_cache.db")),
)

es_llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.0,
    api_key=os.environ["OPENAI_API_KEY"],
    cache=SQLiteCache(str(CACHE_DIR / "openai_cache.db")),
)

# Create embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"]
)

# Create vector store for entities
entities_vector_store = Chroma(
    collection_name="sec-10q-entities",
    persist_directory=str(VECTOR_STORE_DIR),
    embedding_function=embeddings
)

# Setup text splitter and extractor
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
text_unit_extractor = TextUnitExtractor(text_splitter=text_splitter)

# %% [markdown]
# ## Initialize Graph Components
# 
# Set up the components needed for graph generation and processing.

# %%
# Entity relationship extraction and summarization
entity_extractor = EntityRelationshipExtractor.build_default(llm=er_llm)
entity_summarizer = EntityRelationshipDescriptionSummarizer.build_default(llm=es_llm)

# Graph generator
graph_generator = GraphGenerator(
    er_extractor=entity_extractor,
    graphs_merger=GraphsMerger(),
    er_description_summarizer=entity_summarizer
)

# Community detector
community_detector = HierarchicalLeidenCommunityDetector()

# %% [markdown]
# ## Initialize Artifacts Generators
# 
# Set up components for generating various artifacts from the graph.

# %%
# Create artifacts generators
entities_artifacts_generator = EntitiesArtifactsGenerator(
    entities_vector_store=entities_vector_store
)

relationships_artifacts_generator = RelationshipsArtifactsGenerator()

report_generator = CommunityReportGenerator.build_default(llm=er_llm)
report_writer = CommunityReportWriter()

communities_report_artifacts_generator = CommunitiesReportsArtifactsGenerator(
    report_generator=report_generator,
    report_writer=report_writer
)

text_units_artifacts_generator = TextUnitsArtifactsGenerator()

# %% [markdown]
# ## Load and Process Documents
# 
# Load the input text and split it into manageable units.

# %%
# Load and process the documents
from langchain_community.document_loaders.pdf import PyPDFLoader

documents = []
docs_path = Path("../../data/sec-10-q/docs")

# Load PDF documents
for filename in os.listdir(docs_path):
    if filename.endswith(".pdf"):
        file_path = docs_path / filename
        try:
            docs = PyPDFLoader(str(file_path)).load()
            documents.extend(docs)
            print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

# %% [markdown]
# ## Create Indexer and Generate Artifacts
# 
# Initialize the indexer and process the documents to generate all required artifacts.

# %%
# Create the indexer
indexer = SimpleIndexer(
    text_unit_extractor=text_unit_extractor,
    graph_generator=graph_generator,
    community_detector=community_detector,
    entities_artifacts_generator=entities_artifacts_generator,
    relationships_artifacts_generator=relationships_artifacts_generator,
    text_units_artifacts_generator=text_units_artifacts_generator,
    communities_report_artifacts_generator=communities_report_artifacts_generator
)

# Run indexing
artifacts = indexer.run(documents)

# %%
# save artifacts to .pkl on disk

import pickle

with open('graphrag_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# %% [markdown]
# ## Local Search Example
# 
# Demonstrate using the local search capability for answering specific questions.

# %%
from datetime import datetime

current_date = datetime.now().strftime("%B %d, %Y")

from langchain_graphrag.query.local_search import (
    LocalSearch,
    LocalSearchPromptBuilder,
    LocalSearchRetriever,
)
from langchain_graphrag.query.local_search.context_builders import ContextBuilder
from langchain_graphrag.query.local_search.context_selectors import ContextSelector
from langchain_graphrag.query.local_search._system_prompt import LOCAL_SEARCH_SYSTEM_PROMPT

PROMPT_SUFFIX = f"""
Important Rules:
- Base your answer ONLY on the provided context
- Do not make assumptions or use external knowledge besides the context provided
- Numbers must be whole integers without comma separators, unless specified
- Percentages must be whole numbers without % sign
- The answer field must contain ONLY the numerical value, no text or units
- Your entire response must be valid JSON

The current date is {current_date}.
"""

LOCAL_SEARCH_SYSTEM_PROMPT = LOCAL_SEARCH_SYSTEM_PROMPT + PROMPT_SUFFIX

# Create components for local search
context_selector = ContextSelector.build_default(
    entities_vector_store=entities_vector_store,
    entities_top_k=10,
    community_level=cast(CommunityLevel, 2)
)

context_builder = ContextBuilder.build_default(
    token_counter=TiktokenCounter(),
)

retriever = LocalSearchRetriever(
    context_selector=context_selector,
    context_builder=context_builder,
    artifacts=artifacts,
)

local_search = LocalSearch(
    prompt_builder=LocalSearchPromptBuilder(system_prompt=LOCAL_SEARCH_SYSTEM_PROMPT, show_references=True),
    llm=er_llm,
    retriever=retriever
)

search_chain = local_search()

# %%
print(search_chain.invoke("What was the total net sales for Apple for the quarterly period ended April 1, 2023, as reported in their 2023 Q2 AAPL.pdf? Provide the answer in millions of dollars as a whole number without commas."))

# %%
answer = search_chain.invoke("What was the total net sales for Apple for the quarterly period ended April 1, 2023, as reported in their 2023 Q2 AAPL.pdf? Provide the answer in millions of dollars as a whole number without commas.")

# %%
# Evaluation code
import datetime
import os
from pathlib import Path
import json
import re
from dotenv import load_dotenv
from tqdm.notebook import tqdm

load_dotenv()


def extract_number(text):
    # Find any number (integer or decimal) in the string
    match = re.search(r':\s*(-?\d+(?:\.\d+)?)', text)
    if match:
        return float(match.group(1))
    return None

# Create evaluation results directory if it doesn't exist
eval_dir = Path("evaluation_results_graphrag")
eval_dir.mkdir(exist_ok=True)

# Create timestamp for unique filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = eval_dir / f"evaluation_results_{timestamp}.txt"

# Load the CSV file
df = pd.read_csv("../../data/sec-10-q/synthetic_qna_data_7_gpt4o_v2_mod1.csv")

# Prepare results storage
results = []
correct = 0
total = len(df)

# Open file for writing results
with open(output_file, 'w') as f:
    # Write header information
    f.write("SEC 10-Q RAG System Evaluation Results\n")
    f.write(f"Evaluation Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Questions: {total}\n")
    f.write("=" * 80 + "\n\n")
    
    # Evaluate each question
    for i, row in tqdm(df.iterrows(), total=total, desc="Evaluating questions"):
        question = row["New Question"]
        expected_answer = row["New Answer"]
        
        # Get model response
        try:
            response = search_chain.invoke(question)
            model_answer = extract_number(response)
            is_correct = float(model_answer) == float(expected_answer)
            if is_correct:
                correct += 1
        except Exception as e:
            model_answer = f"ERROR: {str(e)}"
            model_reasoning = "Error occurred during processing"
            is_correct = False
        
        # Write question details
        f.write(f"Question {i+1}/{total}:\n")
        f.write(f"Question: {question}\n")
        f.write(f"Expected Answer: {expected_answer}\n")
        f.write(f"Model Answer: {model_answer}\n")
        f.write(f"Correct: {is_correct}\n")
        f.write("-" * 80 + "\n\n")
        
        # Store result for summary
        results.append({
            'question_id': i+1,
            'question': question,
            'expected': expected_answer,
            'response': model_answer,
            'correct': is_correct
        })
    
    # Calculate and write summary statistics
    accuracy = correct / total
    f.write("\nEvaluation Summary\n")
    f.write("=" * 80 + "\n")
    f.write(f"Total Questions: {total}\n")
    f.write(f"Correct Answers: {correct}\n")
    f.write(f"Accuracy: {accuracy:.2%}\n")

# Create results DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(eval_dir / f"evaluation_detailed_results_{timestamp}.csv", index=False)

print(f"Evaluation complete. Results saved to {output_file}")
print(f"Detailed results saved to {eval_dir}/evaluation_detailed_results_{timestamp}.csv")
print(f"\nFinal Accuracy: {accuracy:.2%}")

# %% [markdown]
# ## Global Search Example
# 
# Demonstrate using the global search capability for broader questions about the document.

# %%
from langchain_graphrag.query.global_search import GlobalSearch
from langchain_graphrag.query.global_search.community_weight_calculator import (
    CommunityWeightCalculator
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

# Create components for global search
report_context_builder = CommunityReportContextBuilder(
    community_level=cast(CommunityLevel, 2),
    weight_calculator=CommunityWeightCalculator(),
    artifacts=artifacts,
    token_counter=TiktokenCounter(),
)

kp_generator = KeyPointsGenerator(
    llm=er_llm,
    prompt_builder=KeyPointsGeneratorPromptBuilder(show_references=True),
    context_builder=report_context_builder,
)

kp_aggregator = KeyPointsAggregator(
    llm=er_llm,
    prompt_builder=KeyPointsAggregatorPromptBuilder(show_references=True),
    context_builder=KeyPointsContextBuilder(
        token_counter=TiktokenCounter(),
    ),
)

global_search = GlobalSearch(
    kp_generator=kp_generator,
    kp_aggregator=kp_aggregator
)


