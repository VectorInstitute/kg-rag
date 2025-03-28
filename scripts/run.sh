#!/bin/bash


python -m scripts.run_entity_rag \
    --graph-documents-pkl-path data/sec-10-q/graphs/sec10q_entity_metadata_graph_documents_formatted.pkl \
    --documents-pkl-path data/sec-10-q/graphs/documents.pkl \
    --documents-path data/sec-10-q/docs \
    --query "What was the Products gross margin percentage for Apple for the quarter ended July 1, 2023, as reported in their Q3 2023 10-Q? Provide the answer rounded to one decimal place." \
    --top-k-nodes 10 \
    --top-k-chunks 5 \
    --similarity-threshold 0.5 \
    --node-freq-weight 0.4 \
    --node-sim-weight 0.6 \
    --numerical-answer \
    --verbose

# python -m scripts.run_baseline_rag \
#     --query "What was the Products gross margin percentage for Apple for the quarter ended July 1, 2023, as reported in their Q3 2023 10-Q? Provide the answer rounded to one decimal place." \
#     --numerical-answer \
#     --verbose

# =================================================================================================

# python -m kg_rag.evaluation.run_evaluation \
#     --data-path data/sec-10-q/synthetic_qna_data_7_gpt4o_v2_mod1_50.csv \
#     --config-path kg_rag/configs/baseline-kgrag.json \
#     --method baseline \
#     --output-dir evaluation_results \
#     --numerical-answer \
#     --use-cot \
#     # --max-samples 1 \
#     # --verbose

# python -m kg_rag.evaluation.run_evaluation \
#     --data-path data/sec-10-q/synthetic_qna_data_7_gpt4o_v2_mod1_50.csv \
#     --config-path kg_rag/configs/entity-based-kgrag.json \
#     --method entity \
#     --output-dir evaluation_results \
#     --numerical-answer \
#     --use-cot \
#     # --max-samples 1 \
#     # --verbose
