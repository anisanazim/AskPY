import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger()

class VectorStore:
    def __init__(self):
        self.index = None
        self.documents = []
        self.db_path = settings.vector_db_path
        self.index_file = os.path.join(self.db_path, "faiss_index.bin")
        self.docs_file = os.path.join(self.db_path, "documents.pkl")
        
        # Create vector db directory
        os.makedirs(self.db_path, exist_ok=True)
    
    def create_index(self, documents: List[Dict[str, Any]], embedding_dim: int):
        """Create FAISS index from documents"""
        try:
            # Create FAISS index
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for similarity
            
            # Extract embeddings
            embeddings = np.array([doc['embedding'] for doc in documents]).astype('float32')
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings)
            
            # Store documents
            self.documents = documents
            
            logger.info(f"Created FAISS index with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            raise
    
    def save_index(self):
        """Save FAISS index and documents to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)
            
            # Save documents
            with open(self.docs_file, 'wb') as f:
                pickle.dump(self.documents, f)
            
            logger.info("FAISS index and documents saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            raise
    
    def load_index(self) -> bool:
        """Load FAISS index and documents from disk"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.docs_file):
                # Load FAISS index
                self.index = faiss.read_index(self.index_file)
                
                # Load documents
                with open(self.docs_file, 'rb') as f:
                    self.documents = pickle.load(f)
                
                logger.info(f"Loaded FAISS index with {len(self.documents)} documents")
                return True
            else:
                logger.info("No existing FAISS index found")
                return False
                
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            return False
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = None) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents"""
        if k is None:
            k = settings.top_k_results
            
        try:
            # Normalize query embedding
            query_embedding = query_embedding.astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, k)
            
            # Get results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0:  # Valid index
                    results.append((self.documents[idx], float(score)))
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    # New method to add
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get vector store statistics without calculating disk size.
        """
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': self.index.d if self.index else 0
        }

