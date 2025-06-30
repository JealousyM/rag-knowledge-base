import requests
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class OllamaEmbeddings:
    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama embeddings.
        
        Args:
            model_name: Name of the Ollama model to use for embeddings
            base_url: Base URL of the Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.headers = {"Content-Type": "application/json"}
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using Ollama.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embeddings, one for each document
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            if embedding is not None:
                embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> Optional[List[float]]:
        """
        Embed a single query using Ollama.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if embedding fails
        """
        try:
            response = requests.post(
                f"{self.api_url}/embeddings",
                headers=self.headers,
                json={
                    "model": self.model_name,
                    "prompt": text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("embedding")
            else:
                logger.error(f"Error getting embeddings: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Exception while getting embeddings: {str(e)}")
            return None

# Example usage:
if __name__ == "__main__":
    embeddings = OllamaEmbeddings()
    text = "This is a test document."
    embedding = embeddings.embed_query(text)
    print(f"Embedding length: {len(embedding) if embedding else 0}")
