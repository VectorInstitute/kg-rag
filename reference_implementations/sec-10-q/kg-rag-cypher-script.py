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

# Load environment variables
load_dotenv()

# Neo4j connection settings
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')

# Initialize OpenAI client
llm = ChatOpenAI(temperature=0, 
                 model_name="gpt-4o", 
                 openai_api_key=os.environ['OPENAI_API_KEY'])
llm_transformer = LLMGraphTransformer(llm=llm)

# %%
!docker run --name neo4j  -p7474:7474 -p7687:7687     -e NEO4J_AUTH=neo4j/password     -e NEO4J_PLUGINS='["apoc", "graph-data-science"]'

# %%
from langchain_community.document_loaders.pdf import PyPDFLoader

file_path = "../../data/sec-10-q/docs/2022 Q3 AAPL.pdf"
raw_documents = PyPDFLoader(file_path=file_path).load()

from langchain_text_splitters import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)

documents = text_splitter.split_documents(raw_documents)

from langchain.vectorstores.utils import filter_complex_metadata
documents = filter_complex_metadata(documents)

# %%
import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata

# Initialize empty list to store all documents
documents = []

# Get the docs directory path
docs_path = "../../data/sec-10-q/docs"

# Loop through all files in the docs directory
for filename in os.listdir(docs_path):
    # Check if the file is an AAPL PDF
    if filename.endswith("AAPL.pdf"):
        # Construct full file path
        file_path = os.path.join(docs_path, filename)
        
        # Load and process the PDF
        try:
            raw_documents = PyPDFLoader(file_path=file_path).load()
            
            # Split the documents
            text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
            split_documents = text_splitter.split_documents(raw_documents)
            
            # Filter metadata
            processed_documents = filter_complex_metadata(split_documents)
            
            # Append to our collection
            documents.extend(processed_documents)
            
            print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

# Now all_documents contains the processed documents from all AAPL PDFs
print(f"Total documents processed: {len(documents)}")

# %%
graph_documents = llm_transformer.convert_to_graph_documents(tqdm(documents))

# %%
# Load graph_documents
import pickle

with open("graph_documents_appl.pkl", "rb") as f:
    graph_documents = pickle.load(f)

# %%
# save graph_documents
import pickle

with open("graph_documents_appl.pkl", "wb") as f:
    pickle.dump(graph_documents, f)

# %%
# Connect to Neo4j and store the graph
from langchain_neo4j import Neo4jGraph

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)

# Clear existing data
graph.query("MATCH (n) DETACH DELETE n")

graph.add_graph_documents(
    tqdm(graph_documents),
    baseEntityLabel=True,
    include_source=True
)

# %% [markdown]
# The knowledge graph has been constructed and stored in Neo4j. You can now query it using Cypher or use it for downstream tasks like question answering.

# %%
from langchain_core.prompts import PromptTemplate
# from langchain_neo4j.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT

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
MATCH p=(n:Company)-[r:HAS]->()
where n.id = "Nvidia Corporation"
RETURN P LIMIT 25
The question is:
{question}
"""
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=['schema', 'question'],
    template=CYPHER_GENERATION_TEMPLATE,
)

# %%
print('Node properties:\nOrganization {id: STRING}\nDocument {id: STRING, source: STRING, page: INTEGER, text: STRING}\nLocation {id: STRING}\nFinancial instrument {id: STRING}\nEntity {id: STRING}\nRegulation {id: STRING}\nDate {id: STRING}\nFinancial metric {id: STRING}\nProduct {id: STRING}\nTime period {id: STRING}\nPercentage {id: STRING}\nRisk {id: STRING}\nFinancial asset {id: STRING}\nProgram {id: STRING}\nCurrency {id: STRING}\nFinancial_instrument {id: STRING}\nEvent {id: STRING}\nConcept {id: STRING}\nRegion {id: STRING}\nYear {id: STRING}\nLaw {id: STRING}\nAmount {id: STRING}\nPerson {id: STRING}\nPosition {id: STRING}\nIdentifier {id: STRING}\nContact {id: STRING}\nClassification {id: STRING}\nDocument_section {id: STRING}\nFinancial_metric {id: STRING}\nFinancial_concept {id: STRING}\nService {id: STRING}\nTime_period {id: STRING}\nFinancial statement item {id: STRING}\nFinancial_term {id: STRING}\nLegal_term {id: STRING}\nWebsite {id: STRING}\nPage number {id: STRING}\nFinancial market {id: STRING}\nFinancial obligation {id: STRING}\nFinancial program {id: STRING}\nLegal_concept {id: STRING}\nAuthorization {id: STRING}\nRole {id: STRING}\nRelationship properties:\n\nThe relationships:\n(:Organization)-[:APPEALED_DECISION_IN]->(:Organization)\n(:Organization)-[:FILED_LAWSUIT_AGAINST]->(:Organization)\n(:Organization)-[:FILED_LAWSUIT_IN]->(:Organization)\n(:Organization)-[:RULED_AGAINST]->(:Organization)\n(:Organization)-[:FOUND_VIOLATION_OF]->(:Legal_concept)\n(:Organization)-[:RULED_IN_FAVOR_OF]->(:Organization)\n(:Organization)-[:INCORPORATED_IN]->(:Location)\n(:Organization)-[:FILED_WITH]->(:Organization)\n(:Organization)-[:HAS_IDENTIFIER]->(:Identifier)\n(:Organization)-[:INTRODUCED]->(:Product)\n(:Organization)-[:HAS_ADDRESS]->(:Location)\n(:Organization)-[:ANNOUNCED]->(:Product)\n(:Organization)-[:HAS_CONTACT]->(:Contact)\n(:Organization)-[:FILED]->(:Document)\n(:Organization)-[:REPORTS]->(:Financial statement item)\n(:Organization)-[:REPORTS]->(:Concept)\n(:Organization)-[:REPORT_DATE]->(:Date)\n(:Organization)-[:PUBLISHED]->(:Document)\n(:Organization)-[:FINANCIAL_STATEMENT_DATE]->(:Date)\n(:Organization)-[:POTENTIAL_FUTURE_IMPACT]->(:Event)\n(:Organization)-[:REFERENCE]->(:Document)\n(:Organization)-[:PROVIDES_INFORMATION_ON]->(:Website)\n(:Organization)-[:REPORTED]->(:Concept)\n(:Organization)-[:REPORTED]->(:Financial_metric)\n(:Organization)-[:REPORTED]->(:Financial_concept)\n(:Organization)-[:REPORTED]->(:Financial metric)\n(:Organization)-[:AUTHOR_OF]->(:Document)\n(:Organization)-[:HAS]->(:Financial_concept)\n(:Organization)-[:HAS]->(:Financial_metric)\n(:Organization)-[:HAS]->(:Concept)\n(:Organization)-[:EVALUATION_DATE]->(:Date)\n(:Organization)-[:DOCUMENTED_IN]->(:Document)\n(:Organization)-[:UTILIZED]->(:Authorization)\n(:Organization)-[:CONFORMITY]->(:Concept)\n(:Organization)-[:COMPLIANCE]->(:Regulation)\n(:Organization)-[:COMPLIANCE]->(:Law)\n(:Organization)-[:HAS_DOCUMENT]->(:Document)\n(:Organization)-[:AUTHORIZED]->(:Program)\n(:Organization)-[:AUTHORIZED]->(:Financial program)\n(:Organization)-[:DOCUMENT_OF]->(:Document)\n(:Organization)-[:AUTHORIZED_PURCHASE]->(:Date)\n(:Organization)-[:CLASSIFIED_AS]->(:Classification)\n(:Organization)-[:ELECTED_NOT_TO_USE_EXTENDED_TRANSITION_PERIOD]->(:Entity)\n(:Organization)-[:IS_NOT]->(:Entity)\n(:Organization)-[:SUBJECT_TO]->(:Regulation)\n(:Organization)-[:REFERENCED_IN]->(:Document)\n(:Organization)-[:HAS_NON-TRADE_RECEIVABLES_FROM]->(:Organization)\n(:Organization)-[:SELLS_TO_VENDORS]->(:Product)\n(:Organization)-[:MANAGES_ON_GEOGRAPHIC_BASIS]->(:Region)\n(:Organization)-[:DISCUSSED_IN]->(:Document)\n(:Organization)-[:FILED_COUNTERCLAIM_AGAINST]->(:Organization)\n(:Organization)-[:OPERATES]->(:Product)\n(:Organization)-[:FILED_CROSS-APPEAL_IN]->(:Organization)\n(:Organization)-[:DATE_OF_FINANCIAL_DATA]->(:Date)\n(:Organization)-[:ACCOUNTED_FOR_42%_OF_TRADE_RECEIVABLES]->(:Date)\n(:Organization)-[:ACCOUNTED_FOR_36%_OF_TRADE_RECEIVABLES]->(:Date)\n(:Organization)-[:SUFFICIENT_TO_SATISFY_REQUIREMENTS]->(:Financial asset)\n(:Organization)-[:SUFFICIENT_TO_SATISFY_REQUIREMENTS]->(:Financial instrument)\n(:Organization)-[:SUFFICIENT_TO_SATISFY_REQUIREMENTS]->(:Time period)\n(:Organization)-[:HIGHER_EFFECTIVE_TAX_RATE]->(:Time period)\n(:Organization)-[:CONTINUED_ACCESS]->(:Financial market)\n(:Organization)-[:UTILIZES_OUTSOURCING_PARTNERS]->(:Financial obligation)\n(:Organization)-[:HAD_OBLIGATIONS]->(:Date)\n(:Organization)-[:QUARTERLY_CASH_DIVIDEND]->(:Amount)\n(:Organization)-[:AUTHORIZED_BY]->(:Program)\n(:Organization)-[:AUTHORIZED_BY]->(:Financial program)\n(:Organization)-[:COMPARED_TO]->(:Time period)\n(:Organization)-[:ISSUED_REGULATIONS]->(:Time period)\n(:Document)-[:MENTIONS]->(:Regulation)\n(:Document)-[:MENTIONS]->(:Law)\n(:Document)-[:MENTIONS]->(:Date)\n(:Document)-[:MENTIONS]->(:Document)\n(:Document)-[:MENTIONS]->(:Organization)\n(:Document)-[:MENTIONS]->(:Product)\n(:Document)-[:MENTIONS]->(:Legal_concept)\n(:Document)-[:MENTIONS]->(:Amount)\n(:Document)-[:MENTIONS]->(:Authorization)\n(:Document)-[:MENTIONS]->(:Identifier)\n(:Document)-[:MENTIONS]->(:Financial_instrument)\n(:Document)-[:MENTIONS]->(:Location)\n(:Document)-[:MENTIONS]->(:Contact)\n(:Document)-[:MENTIONS]->(:Entity)\n(:Document)-[:MENTIONS]->(:Classification)\n(:Document)-[:MENTIONS]->(:Document_section)\n(:Document)-[:MENTIONS]->(:Financial_metric)\n(:Document)-[:MENTIONS]->(:Financial metric)\n(:Document)-[:MENTIONS]->(:Concept)\n(:Document)-[:MENTIONS]->(:Financial_concept)\n(:Document)-[:MENTIONS]->(:Service)\n(:Document)-[:MENTIONS]->(:Region)\n(:Document)-[:MENTIONS]->(:Time_period)\n(:Document)-[:MENTIONS]->(:Percentage)\n(:Document)-[:MENTIONS]->(:Financial instrument)\n(:Document)-[:MENTIONS]->(:Financial asset)\n(:Document)-[:MENTIONS]->(:Risk)\n(:Document)-[:MENTIONS]->(:Financial statement item)\n(:Document)-[:MENTIONS]->(:Program)\n(:Document)-[:MENTIONS]->(:Financial program)\n(:Document)-[:MENTIONS]->(:Event)\n(:Document)-[:MENTIONS]->(:Financial_term)\n(:Document)-[:MENTIONS]->(:Legal_term)\n(:Document)-[:MENTIONS]->(:Website)\n(:Document)-[:MENTIONS]->(:Time period)\n(:Document)-[:MENTIONS]->(:Year)\n(:Document)-[:MENTIONS]->(:Person)\n(:Document)-[:MENTIONS]->(:Position)\n(:Document)-[:MENTIONS]->(:Role)\n(:Document)-[:MENTIONS]->(:Currency)\n(:Document)-[:MENTIONS]->(:Page number)\n(:Document)-[:MENTIONS]->(:Financial obligation)\n(:Document)-[:MENTIONS]->(:Financial market)\n(:Document)-[:CONTAINS]->(:Document_section)\n(:Document)-[:CONTAINS]->(:Page number)\n(:Document)-[:REPORT_OF]->(:Organization)\n(:Document)-[:FILED_WITH]->(:Organization)\n(:Document)-[:BELONGS_TO]->(:Organization)\n(:Document)-[:COMPLIES_WITH]->(:Regulation)\n(:Document)-[:COMPLIES_WITH]->(:Law)\n(:Location)-[:UNFAVORABLE_IMPACT]->(:Currency)\n(:Location)-[:UNFAVORABLE_IMPACT]->(:Concept)\n(:Location)-[:FAVORABLE_IMPACT]->(:Currency)\n(:Location)-[:HAS_VALUE]->(:Financial_metric)\n(:Location)-[:LOWER_NET_SALES]->(:Product)\n(:Location)-[:HIGHER_NET_SALES]->(:Product)\n(:Location)-[:HIGHER_NET_SALES]->(:Concept)\n(:Location)-[:HIGHER_NET_SALES]->(:Service)\n(:Financial instrument)-[:DATE]->(:Date)\n(:Financial instrument)-[:ISSUED_BY]->(:Organization)\n(:Entity)-[:CLASSIFIED_AS]->(:Classification)\n(:Entity)-[:ELECTED_NOT_TO_USE_EXTENDED_TRANSITION_PERIOD]->(:Entity)\n(:Entity)-[:IS_NOT]->(:Entity)\n(:Entity)-[:SUBJECT_TO]->(:Regulation)\n(:Entity)-[:ISSUED_AND_OUTSTANDING_ON]->(:Date)\n(:Entity)-[:AUTHORIZED]->(:Program)\n(:Entity)-[:AUTHORIZED]->(:Financial program)\n(:Entity)-[:PART_OF]->(:Organization)\n(:Entity)-[:PART_OF]->(:Entity)\n(:Date)-[:BEGINNING_CASH_BALANCE]->(:Amount)\n(:Date)-[:NET_INCOME]->(:Amount)\n(:Date)-[:DEPRECIATION_AMORTIZATION]->(:Amount)\n(:Date)-[:SHARE_BASED_COMPENSATION]->(:Amount)\n(:Date)-[:DEFERRED_INCOME_TAX]->(:Amount)\n(:Date)-[:OTHER_ADJUSTMENTS]->(:Amount)\n(:Date)-[:ACCOUNTS_RECEIVABLE_CHANGE]->(:Amount)\n(:Date)-[:INVENTORIES_CHANGE]->(:Amount)\n(:Date)-[:VENDOR_NON_TRADE_RECEIVABLES_CHANGE]->(:Amount)\n(:Date)-[:OTHER_ASSETS_CHANGE]->(:Amount)\n(:Date)-[:ACCOUNTS_PAYABLE_CHANGE]->(:Amount)\n(:Date)-[:DEFERRED_REVENUE_CHANGE]->(:Amount)\n(:Date)-[:OTHER_LIABILITIES_CHANGE]->(:Amount)\n(:Date)-[:CASH_GENERATED_OPERATING_ACTIVITIES]->(:Amount)\n(:Date)-[:PURCHASES_MARKETABLE_SECURITIES]->(:Amount)\n(:Date)-[:PROCEEDS_MATURITIES_MARKETABLE_SECURITIES]->(:Amount)\n(:Date)-[:PROCEEDS_SALES_MARKETABLE_SECURITIES]->(:Amount)\n(:Date)-[:PAYMENTS_ACQUISITION_PROPERTY]->(:Amount)\n(:Date)-[:PAYMENTS_BUSINESS_ACQUISITIONS]->(:Amount)\n(:Date)-[:OTHER_INVESTING_ACTIVITIES]->(:Amount)\n(:Date)-[:CASH_USED_INVESTING_ACTIVITIES]->(:Amount)\n(:Date)-[:PAYMENTS_TAXES_EQUITY_AWARDS]->(:Amount)\n(:Date)-[:PAYMENTS_DIVIDENDS]->(:Amount)\n(:Date)-[:REPURCHASES_COMMON_STOCK]->(:Amount)\n(:Date)-[:REPAYMENTS_TERM_DEBT]->(:Amount)\n(:Date)-[:PROCEEDS_COMMERCIAL_PAPER]->(:Amount)\n(:Date)-[:PROCEEDS_ISSUANCE_TERM_DEBT]->(:Amount)\n(:Product)-[:CATEGORY]->(:Financial_metric)\n(:Product)-[:COMPARISON_YEAR]->(:Date)\n(:Product)-[:COMPARISON_YEAR]->(:Time period)\n(:Product)-[:COMPARISON_YEAR]->(:Year)\n(:Product)-[:NET_SALES_INCREASED]->(:Date)\n(:Product)-[:NET_SALES_INCREASED]->(:Year)\n(:Product)-[:NET_SALES_INCREASED]->(:Time period)\n(:Product)-[:HIGHER_PROPORTION_OF_SALES]->(:Location)\n(:Product)-[:HIGHER_PROPORTION_OF_SALES]->(:Region)\n(:Product)-[:CONTRIBUTES_TO]->(:Financial_metric)\n(:Product)-[:NET_SALES_DECREASED]->(:Time period)\n(:Product)-[:OFFSET_BY]->(:Concept)\n(:Product)-[:INCREASED]->(:Financial metric)\n(:Product)-[:INCREASED]->(:Financial_metric)\n(:Product)-[:COMPARED_TO]->(:Time_period)\n(:Product)-[:DUE_TO_HIGHER_NET_SALES_FROM]->(:Concept)\n(:Product)-[:INCREASED_DURING]->(:Time_period)\n(:Product)-[:LOWER_NET_SALES]->(:Product)\n(:Product)-[:POWERED_BY]->(:Product)\n(:Product)-[:HIGHER_NET_SALES]->(:Product)\n(:Product)-[:DECREASED]->(:Financial metric)\n(:Product)-[:DECREASED]->(:Financial_metric)\n(:Percentage)-[:TIME_FRAME]->(:Time_period)\n(:Risk)-[:EXPOSED_TO]->(:Organization)\n(:Financial asset)-[:DATE]->(:Date)\n(:Program)-[:AUTHORIZED_BY]->(:Organization)\n(:Program)-[:AUTHORIZED_BY]->(:Entity)\n(:Financial_instrument)-[:REGISTERED_ON]->(:Organization)\n(:Financial_instrument)-[:ISSUED_BY]->(:Organization)\n(:Financial_instrument)-[:DATE]->(:Date)\n(:Financial_instrument)-[:OWNED_BY]->(:Organization)\n(:Financial_instrument)-[:USED_BY]->(:Organization)\n(:Financial_instrument)-[:VALUE_14,250_MILLION]->(:Date)\n(:Financial_instrument)-[:VALUE_15,954_MILLION]->(:Date)\n(:Financial_instrument)-[:VALUE_(19,281)_MILLION]->(:Date)\n(:Financial_instrument)-[:VALUE_(17,857)_MILLION]->(:Date)\n(:Financial_instrument)-[:HELD_BY]->(:Organization)\n(:Financial_instrument)-[:COMPONENT_OF]->(:Financial_instrument)\n(:Event)-[:IMPACT]->(:Organization)\n(:Concept)-[:INCLUDES]->(:Concept)\n(:Concept)-[:INCLUDES]->(:Financial_metric)\n(:Concept)-[:INCLUDES]->(:Financial_concept)\n(:Concept)-[:INCLUDES]->(:Financial instrument)\n(:Concept)-[:INCLUDES]->(:Financial_instrument)\n(:Concept)-[:COMPONENT_OF]->(:Financial_concept)\n(:Concept)-[:DATE]->(:Date)\n(:Concept)-[:OFFSET_BY]->(:Concept)\n(:Concept)-[:CONTRIBUTES_TO]->(:Financial_metric)\n(:Concept)-[:CATEGORY]->(:Financial_metric)\n(:Concept)-[:INCREASED]->(:Financial metric)\n(:Concept)-[:INCREASED]->(:Financial_metric)\n(:Concept)-[:COMPARED_TO]->(:Time_period)\n(:Concept)-[:COMPARED_TO]->(:Date)\n(:Concept)-[:COMPARED_TO]->(:Time period)\n(:Concept)-[:COMPARED_TO]->(:Year)\n(:Concept)-[:DUE_TO_HIGHER_NET_SALES_FROM]->(:Concept)\n(:Concept)-[:INCREASED_DURING]->(:Time_period)\n(:Concept)-[:INCREASED_DURING]->(:Time period)\n(:Concept)-[:ISSUED_BY]->(:Organization)\n(:Concept)-[:VALUE_126,918_MILLION]->(:Date)\n(:Concept)-[:VALUE_76,234_MILLION]->(:Date)\n(:Concept)-[:VALUE_76,475_MILLION]->(:Date)\n(:Concept)-[:VALUE_84,506_MILLION]->(:Date)\n(:Concept)-[:VALUE_16,875_MILLION]->(:Date)\n(:Concept)-[:VALUE_20,775_MILLION]->(:Date)\n(:Concept)-[:ACTIVITY]->(:Event)\n(:Concept)-[:EXPENSE]->(:Date)\n(:Concept)-[:COST]->(:Date)\n(:Concept)-[:RELATED_TO]->(:Organization)\n(:Concept)-[:AFFECTS]->(:Concept)\n(:Concept)-[:INVOLVES]->(:Entity)\n(:Concept)-[:INVOLVES]->(:Concept)\n(:Concept)-[:RELATIVE_TO]->(:Currency)\n(:Concept)-[:RELATIVE_TO]->(:Concept)\n(:Concept)-[:DECREASED]->(:Date)\n(:Concept)-[:DECREASED]->(:Year)\n(:Region)-[:UNFAVORABLE_IMPACT]->(:Currency)\n(:Region)-[:UNFAVORABLE_IMPACT]->(:Concept)\n(:Region)-[:FAVORABLE_IMPACT]->(:Currency)\n(:Region)-[:HAS_VALUE]->(:Financial_metric)\n(:Region)-[:LOWER_NET_SALES]->(:Product)\n(:Region)-[:HIGHER_NET_SALES]->(:Product)\n(:Region)-[:HIGHER_NET_SALES]->(:Concept)\n(:Region)-[:HIGHER_NET_SALES]->(:Service)\n(:Region)-[:INCREASED_NET_SALES]->(:Product)\n(:Region)-[:INCREASED_NET_SALES]->(:Concept)\n(:Region)-[:INCREASED_NET_SALES]->(:Service)\n(:Person)-[:HAS]->(:Concept)\n(:Person)-[:CERTIFY]->(:Document)\n(:Person)-[:REVIEWED]->(:Document)\n(:Person)-[:HOLDS_POSITION]->(:Role)\n(:Person)-[:SIGNED_ON]->(:Date)\n(:Person)-[:CERTIFYING_OFFICER]->(:Organization)\n(:Person)-[:CERTIFYING_OFFICER]->(:Entity)\n(:Person)-[:AFFILIATION]->(:Organization)\n(:Person)-[:HOLDS]->(:Position)\n(:Person)-[:DISCLOSED_TO]->(:Entity)\n(:Person)-[:DISCLOSED_TO]->(:Organization)\n(:Classification)-[:DEFINED_IN]->(:Regulation)\n(:Financial_metric)-[:INCLUDES]->(:Concept)\n(:Financial_metric)-[:INCLUDES]->(:Financial_metric)\n(:Financial_metric)-[:INCLUDES]->(:Financial_concept)\n(:Financial_metric)-[:COMPONENT_OF]->(:Financial_concept)\n(:Financial_metric)-[:DATE]->(:Date)\n(:Financial_metric)-[:REPORTED_ON]->(:Date)\n(:Financial_metric)-[:PROPORTION_OF]->(:Financial_metric)\n(:Financial_metric)-[:HIGHER_PROPORTION_IN]->(:Location)\n(:Financial_metric)-[:HIGHER_PROPORTION_IN]->(:Region)\n(:Financial_metric)-[:VALUE_AS_OF]->(:Amount)\n(:Financial_metric)-[:REALIZED_IN]->(:Percentage)\n(:Financial_metric)-[:CONTRIBUTES_TO]->(:Financial_metric)\n(:Financial_metric)-[:REDUCES]->(:Financial_metric)\n(:Financial_concept)-[:INCLUDES]->(:Concept)\n(:Financial_concept)-[:INCLUDES]->(:Financial_concept)\n(:Financial_concept)-[:COMPONENT_OF]->(:Financial_concept)\n(:Financial_concept)-[:RELATED_TO]->(:Organization)\n(:Service)-[:OFFSET_BY]->(:Concept)\n(:Service)-[:CONTRIBUTES_TO]->(:Financial_metric)\n(:Service)-[:CATEGORY]->(:Financial_metric)\n(:Service)-[:INCREASED]->(:Financial metric)\n(:Service)-[:INCREASED]->(:Financial_metric)\n(:Service)-[:COMPARED_TO]->(:Time_period)\n(:Service)-[:DUE_TO_HIGHER_NET_SALES_FROM]->(:Concept)\n(:Service)-[:INCREASED_DURING]->(:Time_period)\n(:Financial statement item)-[:AS_OF]->(:Date)\n(:Financial statement item)-[:COMPONENT_OF]->(:Financial statement item)\n(:Financial_term)-[:RECORDED_ON]->(:Date)\n(:Legal_term)-[:ASSOCIATED_WITH]->(:Organization)\n(:Financial obligation)-[:PAYABLE_WITHIN_12_MONTHS]->(:Amount)\n(:Financial obligation)-[:TOTAL_AMOUNT]->(:Amount)\n(:Financial program)-[:AUTHORIZED_BY]->(:Organization)\n(:Financial program)-[:AUTHORIZED_BY]->(:Entity)\n(:Authorization)-[:UTILIZATION]->(:Amount)\n(:Authorization)-[:TOTAL_AUTHORIZATION]->(:Amount)')

# %%
from langchain_neo4j import GraphCypherQAChain
# from langchain_neo4j.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT

graph_chain = GraphCypherQAChain.from_llm(
    # llm,
    cypher_llm=llm,
    qa_llm=llm,
    # qa_prompt=CYPHER_QA_PROMPT,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    graph=graph,
    verbose=True,
    max_depth=2,
    max_hops=3,
    # return_intermediate_steps=True,
    # return_direct=True,
    validate_cypher=True,
    allow_dangerous_requests=True,
)

# %%
# Load the CSV file
import pandas as pd
import time
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
    time.sleep(1)
        
print(f"\nAccuracy: {correct / 10}")

# %%
graph_chain.invoke(input="Where was Apple Inc. Incorporated?")

# %%
graph_documents[10].relationships

# %%
graph_chain.invoke(input=" On April 1, 2023, what was the Amount of CASH_BEGINNING_BALANCE?")

# %%
graph_chain.invoke(input="What assets does Apple Inc. have?")

# %%
graph_chain.invoke(input="Apple inc. What was the amount for Cash Used In Investing Activities in 2023 Q3?")

# %%
graph_chain.invoke(input="What was Apple Inc's Products gross margin percentage for the third quarter of 2022 as reported in their most recent 10-Q? Provide the percentage rounded to one decimal place.")


