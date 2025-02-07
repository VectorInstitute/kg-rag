# %%
# Import required libraries
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata
import os
from pathlib import Path
import chromadb
import openai
from tqdm.auto import tqdm
import pandas as pd
from chromadb.errors import InvalidCollectionException
import time
import tiktoken
from typing import List, Dict, Any
import json

# %%
# Configuration
class Config:
    COLLECTION_NAME = "sec_10q"
    CHROMA_PERSIST_DIR = "chroma_db"  # Directory to store ChromaDB files
    OPENAI_MODEL = "gpt-4o"
    OPENAI_EMBEDDING = "text-embedding-3-small"
    BATCH_SIZE = 100
    MAX_TOKENS_PER_BATCH = 8000
    RATE_LIMIT_PAUSE = 60.0

def setup_directories():
    """Create necessary directories."""
    dirs = {
        "DATA_DIR": Path("data/sec-10-q"),
        "TEXT_DIR": Path("data/sec-10-q/text")
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(exist_ok=True)
        
    return dirs

# %%
class OpenAIEmbedding:
    """Generate embeddings using OpenAI's API with rate limiting and batching."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", 
                 batch_size: int = 100, max_tokens_per_batch: int = 8000,
                 rate_limit_pause: float = 60.0):
        self.client = openai.Client(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.rate_limit_pause = rate_limit_pause
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def create_batches(self, texts: List[str]) -> List[List[str]]:
        batches = []
        current_batch = []
        current_tokens = 0
        
        for text in texts:
            tokens = self.count_tokens(text)
            
            if tokens > self.max_tokens_per_batch:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                
                words = text.split()
                chunk = []
                chunk_tokens = 0
                
                for word in words:
                    word_tokens = self.count_tokens(word + ' ')
                    if chunk_tokens + word_tokens > self.max_tokens_per_batch:
                        batches.append([' '.join(chunk)])
                        chunk = [word]
                        chunk_tokens = word_tokens
                    else:
                        chunk.append(word)
                        chunk_tokens += word_tokens
                
                if chunk:
                    current_batch = [' '.join(chunk)]
                    current_tokens = chunk_tokens
                
            elif current_tokens + tokens > self.max_tokens_per_batch or len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = [text]
                current_tokens = tokens
            else:
                current_batch.append(text)
                current_tokens += tokens
        
        if current_batch:
            batches.append(current_batch)
            
        return batches

    def generate(self, texts: List[str]) -> List[List[float]]:
        batches = self.create_batches(texts)
        all_embeddings = []
        
        print(f"Processing {len(texts)} texts in {len(batches)} batches...")
        
        for i, batch in enumerate(tqdm(batches, desc="Generating embeddings")):
            while True:
                try:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model
                    )
                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break
                except openai.RateLimitError as e:
                    print(f"Rate limit hit on batch {i+1}/{len(batches)}. Pausing for {self.rate_limit_pause} seconds...")
                    time.sleep(self.rate_limit_pause)
                except Exception as e:
                    print(f"Error in batch {i+1}/{len(batches)}: {str(e)}")
                    raise
                    
            time.sleep(0.5)
            
        return all_embeddings

# %%

class DocumentProcessor:
    """Process and chunk SEC-10Q documents using LangChain's document loaders and text splitters."""
    
    def __init__(self):
        self.text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    
    def process_file(self, file_path: Path) -> list:
        """Process a single file using PyPDFLoader for PDFs or direct text reading."""
        metadata = {
            "filename": file_path.name,
            "file_path": str(file_path),
            "source": "sec_10q"
        }
        
        if file_path.suffix.lower() == ".pdf":
            # Use PyPDFLoader for PDF files
            raw_documents = PyPDFLoader(file_path=str(file_path)).load()
            # Split documents using TokenTextSplitter
            documents = self.text_splitter.split_documents(raw_documents)
            # Filter complex metadata
            documents = filter_complex_metadata(documents)
            # Convert to the format expected by ChromaDB
            chunks = []
            for doc in documents:
                doc_metadata = {**metadata, **doc.metadata}
                chunks.append((doc.page_content, doc_metadata))
            return chunks
        else:
            # For text files, read directly and split
            text = file_path.read_text()
            texts = self.text_splitter.split_text(text)
            return [(text, metadata) for text in texts]

# %%
class ChromaDBManager:
    """Manage ChromaDB operations using local persistent client."""
    
    def __init__(self, collection_name: str, persist_directory: str = "chroma_db",
                 batch_size: int = 100):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.collection = self._get_or_create_collection()
        
    def _get_or_create_collection(self):
        try:
            collection = self.client.get_collection(self.collection_name)
            print(f"Found existing collection: {self.collection_name}")
        except InvalidCollectionException:
            print(f"Creating new collection: {self.collection_name}")
            collection = self.client.create_collection(self.collection_name)
        return collection
            
    def add_documents(self, chunks: List[tuple], embedding_function):
        if not chunks:
            print("No documents to add")
            return
            
        print(f"Processing {len(chunks)} chunks...")
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            
            documents = []
            metadatas = []
            ids = []
            
            for j, (text, metadata) in enumerate(batch):
                documents.append(text)
                metadatas.append(metadata)
                ids.append(f"{metadata['filename']}_{i+j}")
            
            try:
                print(f"Generating embeddings for batch {i//self.batch_size + 1}/{(len(chunks)-1)//self.batch_size + 1}")
                embeddings = embedding_function.generate(documents)
                
                print(f"Adding batch to collection...")
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
                
            except Exception as e:
                print(f"Error processing batch {i//self.batch_size + 1}: {str(e)}")
                raise

        print(f"Successfully added all {len(chunks)} documents to collection")

    def query(self, query_text: str, embedding_function, n_results: int = 5):
        try:
            print(f"Generating embedding for query: {query_text[:100]}...")
            query_embedding = embedding_function.generate([query_text])[0]
            
            print(f"Querying collection for top {n_results} results...")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            return results
        except Exception as e:
            raise Exception(f"Error querying ChromaDB: {str(e)}")

# %%
class RAGSystem:
    """Main RAG system implementation with enhanced chain of thought reasoning."""
    def __init__(self, openai_api_key: str):
        self.embedder = OpenAIEmbedding(openai_api_key)
        self.processor = DocumentProcessor()
        self.db_manager = ChromaDBManager(
            collection_name=Config.COLLECTION_NAME,
            persist_directory=Config.CHROMA_PERSIST_DIR
        )
        self.client = openai.Client(api_key=openai_api_key)

    def load_documents(self, text_dir: Path):
        all_chunks = []
        for file_path in tqdm(list(text_dir.glob("*.txt"))):
            chunks = self.processor.process_file(file_path)
            all_chunks.extend(chunks)
        self.db_manager.add_documents(all_chunks, self.embedder)

    def generate_answer(self, query: str, context: str) -> dict:
        """Generate answer with enhanced chain of thought reasoning and structured response."""
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that answers queries about SEC 10-Q filings.
                You must provide your response as a JSON object.

                Follow these steps carefully:
                1. PLAN: Break down what information you need to find in the context
                2. SEARCH: Locate the relevant information in the provided context only
                3. CALCULATE: If needed, perform any calculations step by step
                4. VERIFY: Double-check your work and ensure your answer matches the question
                5. FORMAT: Format whole numbers without commas or units unless specified
                    - For percentages, use whole numbers without the % sign (e.g., 25 instead of 25%)

                Response Format:
                {
                    "reasoning": "Your detailed step-by-step analysis showing:
                        1. What specific information you're looking for
                        2. Where you found it in the context
                        3. Any calculations performed
                        4. How you verified the answer",
                    "answer": "final numerical value only, properly formatted with commas and no units unless the question specifies otherwise"
                }

                Important Rules:
                - Base your answer ONLY on the provided context
                - Do not make assumptions or use external knowledge besides the context provided
                - Numbers must be whole integers without comma separators, unless specified
                - Percentages must be whole numbers without % sign
                - The answer field must contain ONLY the numerical value, no text or units
                - Your entire response must be valid JSON

                The current date is January 8, 2025."""
            },
            {
                "role": "user",
                "content": f"Using the following context, answer this question: {query}\n\nContext: {context}"
            }
        ]
        response = self.client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=messages,
            temperature=0,
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)

    def query(self, question: str, n_results: int = 5) -> dict:
        """Process query and return structured response with reasoning."""
        results = self.db_manager.query(question, self.embedder, n_results)
        context = "\n\n".join([
            f"[From {meta['filename']}]:\n{doc}"
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ])
        return self.generate_answer(question, context)

# %%
# Evaluation code
import datetime
import os
from pathlib import Path
import json
from dotenv import load_dotenv

load_dotenv()

# Create evaluation results directory if it doesn't exist
eval_dir = Path("evaluation_results")
eval_dir.mkdir(exist_ok=True)

# Create timestamp for unique filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = eval_dir / f"evaluation_results_{timestamp}.txt"

# Load the CSV file
df = pd.read_csv("../../data/sec-10-q/synthetic_qna_data_7_gpt4o_v2_mod1.csv")

# get first 50 questions
df = df.head(50)

# Initialize RAG system
rag_system = RAGSystem(openai_api_key=os.environ["OPENAI_API_KEY"])

# Uncomment for first time loading of documents
# rag_system.load_documents(text_dir=Path("../../data/sec-10-q/text"))

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
            response = rag_system.query(question, n_results=5)
            model_answer = response["answer"]
            model_reasoning = response["reasoning"]
            is_correct = model_answer.strip() == expected_answer.strip()
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
        f.write(f"Reasoning:\n{model_reasoning}\n")
        f.write(f"Correct: {is_correct}\n")
        f.write("-" * 80 + "\n\n")
        
        # Store result for summary
        results.append({
            'question_id': i+1,
            'question': question,
            'expected': expected_answer,
            'response': model_answer,
            'reasoning': model_reasoning,
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


