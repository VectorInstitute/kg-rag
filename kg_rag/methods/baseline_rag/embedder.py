"""
OpenAI embedding handler for baseline RAG approaches.
"""

import time
from typing import List
from tqdm.auto import tqdm
import tiktoken
import openai


class OpenAIEmbedding:
    """Generate embeddings using OpenAI's API with rate limiting and batching."""
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "text-embedding-3-small", 
        batch_size: int = 100, 
        max_tokens_per_batch: int = 8000,
        rate_limit_pause: float = 60.0,
        verbose: bool = False
    ):
        """
        Initialize the OpenAI embedding handler.
        
        Args:
            api_key: OpenAI API key
            model: Name of the embedding model to use
            batch_size: Number of texts to embed in a single batch
            max_tokens_per_batch: Maximum number of tokens per batch
            rate_limit_pause: Number of seconds to pause when rate limited
            verbose: Whether to print verbose output
        """
        self.client = openai.Client(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.rate_limit_pause = rate_limit_pause
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.verbose = verbose
        
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.tokenizer.encode(text))
    
    def create_batches(self, texts: List[str]) -> List[List[str]]:
        """
        Create batches of texts for embedding.
        
        Args:
            texts: List of texts to batch
            
        Returns:
            List of batches, where each batch is a list of texts
        """
        batches = []
        current_batch = []
        current_tokens = 0
        
        for text in texts:
            tokens = self.count_tokens(text)
            
            if tokens > self.max_tokens_per_batch:
                # Handle texts that exceed max_tokens_per_batch
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                
                # Split large text into smaller chunks
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
                # Start a new batch if the current one is full
                batches.append(current_batch)
                current_batch = [text]
                current_tokens = tokens
            else:
                # Add to the current batch
                current_batch.append(text)
                current_tokens += tokens
        
        # Add the final batch if not empty
        if current_batch:
            batches.append(current_batch)
            
        return batches

    def generate(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings, where each embedding is a list of floats
        """
        batches = self.create_batches(texts)
        all_embeddings = []
        
        if self.verbose:
            print(f"Processing {len(texts)} texts in {len(batches)} batches...")
        
        for i, batch in enumerate(tqdm(batches, desc="Generating embeddings", disable=not self.verbose)):
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
                    if self.verbose:
                        print(f"Rate limit hit on batch {i+1}/{len(batches)}. Pausing for {self.rate_limit_pause} seconds...")
                    time.sleep(self.rate_limit_pause)
                except Exception as e:
                    if self.verbose:
                        print(f"Error in batch {i+1}/{len(batches)}: {str(e)}")
                    raise
                    
            time.sleep(0.5)  # Small pause between batches
            
        return all_embeddings