from google import genai
from pathlib import Path
from typing import List
from src.utils.config import Config

class EmbeddingService:
    def __init__(self):
            
        llm_config = Config.data.get("llm", {})
        memory_config = Config.data.get("memory", {})
        
        self.api_key = llm_config.get("api_key")
        self.model_name = memory_config.get("embedding_model", "text-embedding-004")
        
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            print("Warning: API Key not found for EmbeddingService")

    def embed(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Generate an embedding for the given text."""
        if not self.client:
            # MVP Fallback: return dummy zeros
            return [0.0] * 768

        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                task_type=task_type
            )
            return response.embeddings[0].values
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * 768

    def batchEmbedContents(self, texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        """Generate embeddings for a batch of texts, handling up to 250 strings per request."""
        if not self.client:
            # MVP Fallback: return dummy zeros
            return [[0.0] * 768 for _ in texts]

        try:
            # Split into batches of 250 as per API limits
            batch_size = 250
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch_texts,
                    task_type=task_type
                )
                all_embeddings.extend([emb.values for emb in response.embeddings])

            return all_embeddings
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return [[0.0] * 768 for _ in texts]