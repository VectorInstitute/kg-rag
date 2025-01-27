# Bootcamp Repository Template

This repository serves as the template sharing materials related bootcamps, workshops, or labs.

## Datasets

## Question-Answering Based
**Natural Questions (NQ)**
   - Size: 307,373 examples with 1-way annotations, 7,830 examples with 5-way annotations for dev data and 7,842 with 5-way annotations for test data
   - Ground-truth: Human-verified 
   - Tags: wikipedia, factual QA
   - Reference: [Kwiatkowski et al., 2019](https://doi.org/10.1162/tacl_a_00276)

**HotpotQA**
   - Size: 113,000 examples
   - Ground-truth: Human-verified
   - Tags: multi-hop reasoning, wikipedia
   - Reference: [Yang et al., 2018](https://arxiv.org/abs/1809.09600)

**FEVER**
   - Size: 185,445 examples
   - Ground-truth: Human-verified
   - Tags: fact verification, wikipedia
   - Reference: [Thorne et al., 2018](https://aclanthology.org/N18-1074/)

## Domain-Specific
**MIRAGE (MedRAG)**
   - Size: 7,663 questions across 5 datasets
   - Ground-truth: Human-verified
   - Tags: biomedical, healthcare, zero-shot evaluation, question-only retrieval
   - Reference: [Xiong et al., 2024](https://arxiv.org/abs/2402.13178)

**DomainRAG**
   - Size: 36,166 examples, focused on 6 specific domain capabilities:  
     - Extractive, Conversational, Structural, Time-sensitive, Multi-document, Faithfulness
   - Ground-truth: Machine-generated
   - Tags: college admissions, temporal data
   - Reference: [Wang et al., 2024](https://arxiv.org/abs/2406.05654)

## Dynamic/Time-Sensitive
**RealTimeQA**
   - Size: Continuously updated
   - Ground-truth: Human-verified
   - Tags: current events, temporal
   - Reference: [Kasai et al., 2022](https://arxiv.org/abs/2207.13332)

## Synthetic/Generated
**RGB Dataset**
    - Size: 300 base questions, 200 for information integration and 200 for counterfactual robustness.
   - Source: News articles, tests four fundamental capabilities:
     - noise robustness, negative rejection, information integration, and counterfactual robust-ness.
   - Ground-truth: Machine-generated
   - Tags: news, robustness testing
   - Reference: [Chen et al., 2023](https://arxiv.org/abs/2309.01431)

**CRUD-RAG**
   - Size: Total 36,166 examples, 10,728 text continuation, 9,580 question answering, 5,130 hallucination modification, 10,728 multi-doc summarization
   - Source: News articles
   - Knowledge base: 86,834 documents
   - Ground-truth: Machine-generated
   - Tags: create, read, update, delete operations
   - Reference: [Lyu et al., 2024](https://arxiv.org/abs/2401.17043)

**MultiHop-RAG**
   - Size: 2,556 queries, 31.92% inference queries, 33.49% comparison queries, 22.81% temporal queries, 11.78% null queries
   - Knowledge base: 609 news articles
   - Source: News articles
   - Ground-truth: Machine-generated
   - Tags: multi-hop reasoning
   - Reference: [Tang & Yang, 2024](https://arxiv.org/abs/2401.15391)

## About [Bootcamp Name]

*Add info on the bootcamp.*

## Repository Structure

- **docs/**: Contains detailed documentation, additional resources, installation guides, and setup instructions that are not covered in this README.
- **reference_implementations/**: Reference Implementations are organized by topics. Each topic has its own directory containing codes, notebooks, and a README for guidance.
- **utils/**: Add any installable library components here. It contains replicated codes or class definitions that need not be included directly in the reference implementations.
- **data/**: Includes sample datasets or links to datasets used in the bootcamp, along with usage instructions. It also contains the implementation of dataset modules, or anything related to that.
- **scripts/**: Utility scripts for setup, data processing, submiting jobs, or other repetitive tasks.
- **pyproject.toml**: The `pyproject.toml` file in this repository configures various build system requirements and dependencies, centralizing project settings in a standardized format.


### Reference Implementations Directory

Each topic within the [choice of bootcamp/lab/workshop] has a dedicated directory in the `reference_implementations/` directory. In each directory, there is a README file that provides an overview of the topic, prerequisites, and notebook descriptions.

Here is the list of the covered topics:
- [Reference Implementation 1]
- [Reference Implementation 2]

## Getting Started

To get started with this bootcamp (*Change or modify the following steps based your needs.*):
1. Clone this repository to your machine.
2. *Include setup and installation instructions here. For additional documentation, refer to the `docs/` directory.*
3. Begin with each topic in the `reference_implementations/` directory, as guided by the README files.

## License
*Add appropriate LICENSE for this bootcamp in the main directory.*
This project is licensed under the terms of the [LICENSE] file located in the root directory of this repository.

## Contribution
*Add appropriate CONTRIBUTING.md for this bootcamp in the main directory.*
To get started with contributing to our project, please read our [CONTRIBUTING.md] guide. 

## Contact Information

For more information or help with navigating this repository, please contact [Vector AI ENG Team/Individual] at [Contact Email].