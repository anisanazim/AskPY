import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger()

class EmbeddingService:
    def __init__(self):
        self.model_name = settings.embedding_model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for document chunks"""
        try:
            texts = [doc['content'] for doc in documents]
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # Add embeddings to documents
            for i, doc in enumerate(documents):
                doc['embedding'] = embeddings[i]
            
            logger.info(f"Generated embeddings for {len(documents)} document chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query"""
        try:
            embedding = self.model.encode([query])
            return embedding[0]
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.model.get_sentence_embedding_dimension()