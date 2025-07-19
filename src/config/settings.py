import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    groq_api_key: str = ""
    
    # Model Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "mixtral-8x7b-32768"
    
    # Vector Database Configuration
    vector_db_path: str = "./vector_db"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k_results: int = 5
    
    # Enhanced Web Search Configuration
    enable_web_search: bool = True
    web_search_results: int = 3
    relevance_threshold: float = 0.65
    web_search_threshold: float = 0.6
    
    # Quality Control
    internal_confidence_threshold: float = 0.65
    python_only_mode: bool = True
    
    # Data paths
    data_source_path: str = "./data_source"
    logs_path: str = "./logs"
    
    class Config:
        # This tells Pydantic to automatically load from a .env file
        env_file = ".env"
        # This makes Pydantic case-insensitive for environment variables
        case_sensitive = False

settings = Settings()