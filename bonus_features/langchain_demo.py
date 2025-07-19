import sys
import os

# Add warning filters to suppress NumPy warnings
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

import json
import time
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime

# Updated LangChain imports for version 0.3.x
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.language_models.llms import LLM
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.memory import ConversationBufferWindowMemory
from pydantic import Field

# Standard imports
import requests
from groq import Groq
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Settings class to handle configuration
class Settings:
    def __init__(self):
        self.llm_model = os.getenv('LLM_MODEL', 'llama3-8b-8192')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '800'))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '150'))
        self.top_k_results = int(os.getenv('TOP_K_RESULTS', '5'))
        # Vector store persistence settings
        self.vector_db_path = os.getenv('VECTOR_DB_PATH', './vector_db')

settings = Settings()

class GroqLLM(LLM):
    """Custom LangChain wrapper for Groq API with proper Pydantic field handling"""
    
    # Define fields properly with Pydantic Field
    groq_api_key: str = Field(...)
    model_name: str = Field(default="llama3-8b-8192")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=1500)
    client: Optional[Any] = Field(default=None, exclude=True)
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True
        
    def __init__(self, groq_api_key: str, **kwargs):
        super().__init__(
            groq_api_key=groq_api_key,
            model_name=settings.llm_model,
            temperature=0.1,
            max_tokens=1500,
            **kwargs
        )
        # Initialize client after Pydantic initialization
        self.client = Groq(api_key=groq_api_key)
        logger.info(f"Initializing GroqLLM with model: {self.model_name}")
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            if self.client is None:
                self.client = Groq(api_key=self.groq_api_key)
                
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API error with model {self.model_name}: {e}")
            return f"Error: Unable to generate response - {str(e)}"

class WebSearchTool:
    """Web search tool using duckduckgo_search 8.1.1"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def search(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Search web for Python-related content"""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            search_query = f"{query} python programming tutorial"
            
            # Updated syntax for duckduckgo_search 8.x
            ddgs = DDGS()
            search_results = ddgs.text(search_query, max_results=max_results)
            
            for result in search_results:
                if self._is_python_relevant(result):
                    results.append({
                        'title': result.get('title', 'No title'),
                        'url': result.get('href', '#'),
                        'content': result.get('body', 'No content')
                    })
            
            return results
                
        except ImportError:
            # Fallback mock results for demo
            logger.warning("duckduckgo_search not available, using mock results")
            return [
                {
                    'title': f"Python Documentation: {query}",
                    'url': 'https://docs.python.org/',
                    'content': f"Official Python documentation for {query} with examples and best practices."
                }
            ]
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
    
    def _is_python_relevant(self, result: Dict) -> bool:
        """Check if result is Python-relevant"""
        text = f"{result.get('title', '')} {result.get('body', '')}".lower()
        python_keywords = ['python', 'programming', 'code', 'tutorial', 'function', 'class']
        return any(keyword in text for keyword in python_keywords)

class LangChainRAGPipeline:
    """LangChain-based RAG Pipeline with persistent vector store"""
    
    def __init__(self, groq_api_key: str, data_source_path: str = "./data_source"):
        self.data_source_path = data_source_path
        self.groq_api_key = groq_api_key
        
        # Vector store persistence
        self.vector_db_path = settings.vector_db_path
        self.metadata_file = os.path.join(self.vector_db_path, "metadata.json")
        
        # Initialize components
        self.llm = GroqLLM(groq_api_key=groq_api_key)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.web_search = WebSearchTool()
        
        # Initialize vector store
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        
        # Performance metrics
        self.metrics = {
            'total_queries': 0,
            'avg_response_time': 0.0,
            'total_tokens_used': 0,
            'web_searches_performed': 0,
            'internal_docs_retrieved': 0,
            'vector_store_loaded_from_cache': False,
            'documents_processed': 0
        }
        
        # Memory for conversation
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Remember last 5 exchanges
            memory_key="chat_history",
            return_messages=True
        )
        
        self._initialize_pipeline()
    
    def _get_data_hash(self, documents: List[Document]) -> str:
        """Create a hash of document contents to detect changes"""
        content_hash = hashlib.md5()
        for doc in documents:
            content_hash.update(doc.page_content.encode('utf-8'))
            content_hash.update(str(doc.metadata).encode('utf-8'))
        return content_hash.hexdigest()
    
    def _should_rebuild_vector_store(self, documents: List[Document]) -> bool:
        """Check if vector store needs to be rebuilt"""
        if not os.path.exists(self.vector_db_path):
            logger.info("Vector store directory doesn't exist - will create new one")
            return True
        
        if not os.path.exists(self.metadata_file):
            logger.info("Metadata file doesn't exist - will rebuild vector store")
            return True
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            current_hash = self._get_data_hash(documents)
            stored_hash = metadata.get('data_hash', '')
            
            if current_hash != stored_hash:
                logger.info("Document content has changed - will rebuild vector store")
                return True
            
            logger.info("Documents unchanged - will load existing vector store")
            return False
            
        except Exception as e:
            logger.warning(f"Error reading metadata: {e} - will rebuild vector store")
            return True
    
    def _save_metadata(self, documents: List[Document], chunks_count: int):
        """Save metadata about the vector store"""
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        metadata = {
            'data_hash': self._get_data_hash(documents),
            'documents_count': len(documents),
            'chunks_count': chunks_count,
            'embedding_model': settings.embedding_model,
            'chunk_size': settings.chunk_size,
            'chunk_overlap': settings.chunk_overlap,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata: {chunks_count} chunks from {len(documents)} documents")
    
    def _initialize_pipeline(self):
        """Initialize the LangChain RAG pipeline with persistence"""
        try:
            logger.info("Initializing LangChain RAG pipeline with persistence...")
            
            # Load and process documents
            documents = self._load_documents()
            if not documents:
                logger.warning("No documents found to process")
                # Create a comprehensive dummy document
                documents = [Document(
                    page_content="""
Python Programming Comprehensive Guide

# Object-Oriented Programming

## Classes and Objects
A class in Python is defined using the 'class' keyword. Classes serve as blueprints for creating objects.

Basic syntax:
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old"

# Creating an instance
person = Person("Alice", 30)
print(person.introduce())
```

### Inheritance
Python supports inheritance, allowing classes to inherit from parent classes:

```python
class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id
    
    def study(self, subject):
        return f"{self.name} is studying {subject}"
```

# Functions and Methods

## Function Definition
Functions are defined using the 'def' keyword:

```python
def calculate_area(length, width):
    '''Calculate the area of a rectangle'''
    return length * width

def greet(name, greeting="Hello"):
    '''Greet someone with a customizable greeting'''
    return f"{greeting}, {name}!"
```

## Lambda Functions
Lambda functions are anonymous functions:

```python
square = lambda x: x ** 2
add = lambda x, y: x + y

numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(square, numbers))
```

# Data Structures

## Lists
Lists are ordered, mutable collections:

```python
fruits = ["apple", "banana", "cherry"]
fruits.append("date")
fruits.remove("banana")

# List comprehension
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]
```

## Dictionaries
Dictionaries store key-value pairs:

```python
student = {
    "name": "John",
    "age": 20,
    "courses": ["Math", "Physics", "Chemistry"]
}

# Accessing values
print(student["name"])
print(student.get("grade", "Not available"))

# Dictionary comprehension
word_lengths = {word: len(word) for word in ["hello", "world", "python"]}
```

## Tuples
Tuples are ordered, immutable sequences:

```python
coordinates = (10, 20)
x, y = coordinates  # Unpacking
colors = ("red", "green", "blue")
```

# Exception Handling

## Try-Except Blocks
Handle errors gracefully with try-except blocks:

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("This always executes")
```

## Custom Exceptions
Create your own exception classes:

```python
class CustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

# Raising custom exceptions
if value < 0:
    raise CustomError("Value cannot be negative")
```

# File Operations

## Reading Files
```python
# Method 1: Using with statement (recommended)
with open("file.txt", "r") as f:
    content = f.read()
    print(content)

# Method 2: Reading lines
with open("file.txt", "r") as f:
    for line in f:
        print(line.strip())
```

## Writing Files
```python
# Writing to a file
with open("output.txt", "w") as f:
    f.write("Hello, World!")

# Appending to a file
with open("output.txt", "a") as f:
    f.write("\\nThis is a new line")
```

# Decorators

## Basic Decorators
Decorators modify or enhance functions:

```python
def my_decorator(func):
    def wrapper():
        print("Before function execution")
        func()
        print("After function execution")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

# When called, this will print:
# Before function execution
# Hello!
# After function execution
```

## Decorators with Arguments
```python
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")
```
""",
                    metadata={'source': 'python_comprehensive_guide', 'type': 'internal', 'filename': 'python_guide.txt'}
                )]
            
            self.metrics['documents_processed'] = len(documents)
            
            # Check if we should rebuild the vector store
            if self._should_rebuild_vector_store(documents):
                logger.info("Building new vector store...")
                
                # Split documents into chunks
                split_docs = self.text_splitter.split_documents(documents)
                logger.info(f"Split documents into {len(split_docs)} chunks")
                
                # Create vector store
                self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
                
                # Save to disk
                self.vectorstore.save_local(self.vector_db_path)
                self._save_metadata(documents, len(split_docs))
                
                logger.info(f"Vector store saved to {self.vector_db_path}")
                self.metrics['vector_store_loaded_from_cache'] = False
                
            else:
                logger.info("Loading existing vector store from cache...")
                
                # Load existing vector store
                self.vectorstore = FAISS.load_local(
                    self.vector_db_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                logger.info("Vector store loaded from cache successfully")
                self.metrics['vector_store_loaded_from_cache'] = True
            
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.top_k_results}
            )
            
            # Create custom prompt template
            prompt_template = self._create_prompt_template()
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": prompt_template},
                return_source_documents=True
            )
            
            cache_status = "from cache" if self.metrics['vector_store_loaded_from_cache'] else "rebuilt"
            logger.info(f"LangChain RAG pipeline initialized successfully ({cache_status})")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise
    
    def _load_documents(self) -> List[Document]:
        """Load documents using LangChain loaders"""
        documents = []
        
        # Check if data source path exists
        if not os.path.exists(self.data_source_path):
            logger.warning(f"Data source path does not exist: {self.data_source_path}")
            return documents
        
        # Load PDF documents
        pdf_dir = Path(self.data_source_path) / "pdf"
        if pdf_dir.exists():
            for pdf_file in pdf_dir.glob("*.pdf"):
                try:
                    loader = PyPDFLoader(str(pdf_file))
                    docs = loader.load()
                    # Add metadata
                    for doc in docs:
                        doc.metadata.update({
                            'source_type': 'internal',
                            'file_type': 'pdf',
                            'filename': pdf_file.name
                        })
                    documents.extend(docs)
                    logger.info(f"Loaded PDF: {pdf_file.name}")
                except Exception as e:
                    logger.error(f"Error loading PDF {pdf_file}: {e}")
        
        # Load TXT documents
        txt_dir = Path(self.data_source_path) / "txt"
        if txt_dir.exists():
            for txt_file in txt_dir.glob("*.txt"):
                try:
                    loader = TextLoader(str(txt_file), encoding='utf-8')
                    docs = loader.load()
                    # Add metadata
                    for doc in docs:
                        doc.metadata.update({
                            'source_type': 'internal',
                            'file_type': 'txt',
                            'filename': txt_file.name
                        })
                    documents.extend(docs)
                    logger.info(f"Loaded TXT: {txt_file.name}")
                except Exception as e:
                    logger.error(f"Error loading TXT {txt_file}: {e}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create enhanced prompt template for dual-source integration"""
        template = """You are PythonDocBot, an advanced AI assistant specialized in Python programming.

INSTRUCTIONS:
1. Answer the question using the provided context from internal Python documentation
2. If the context doesn't fully answer the question, clearly state what's missing
3. Provide practical Python code examples when relevant
4. Be specific and cite the source material when possible

CONTEXT FROM INTERNAL DOCUMENTATION:
{context}

QUESTION: {question}

COMPREHENSIVE ANSWER:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search and return formatted results"""
        web_results = self.web_search.search(query)
        self.metrics['web_searches_performed'] += 1
        
        formatted_results = []
        for result in web_results:
            formatted_results.append({
                'title': result['title'],
                'source': result['url'],
                'content': result['content'],
                'type': 'external',
                'relevance_score': 0.8  # Default score for web results
            })
        
        return formatted_results
    
    def _safety_check(self, query: str, response: str) -> Dict[str, Any]:
        """Basic safety guardrails and hallucination detection"""
        safety_results = {
            'is_safe': True,
            'confidence': 'high',
            'warnings': [],
            'fallback_triggered': False
        }
        
        # Check for potentially harmful queries
        harmful_patterns = ['hack', 'malicious', 'attack', 'exploit', 'crack']
        if any(pattern in query.lower() for pattern in harmful_patterns):
            safety_results['is_safe'] = False
            safety_results['warnings'].append("Potentially harmful query detected")
        
        # Basic hallucination detection
        if len(response) < 50:
            safety_results['confidence'] = 'low'
            safety_results['warnings'].append("Response too short, may be incomplete")
        
        contradiction_indicators = ['however', 'but', 'although', 'contradicts']
        contradiction_count = sum(1 for indicator in contradiction_indicators if indicator in response.lower())
        if contradiction_count > 2:
            safety_results['confidence'] = 'medium'
            safety_results['warnings'].append("Multiple contradictory statements detected")
        
        return safety_results
    
    def process_query(self, query: str, use_web_search: bool = True) -> Dict[str, Any]:
        """Process query with LangChain pipeline and safety guardrails"""
        start_time = time.time()
        self.metrics['total_queries'] += 1
        
        try:
            # Step 1: Get internal documentation response
            internal_result = self.qa_chain.invoke({"query": query})
            internal_response = internal_result['result']
            source_docs = internal_result['source_documents']
            
            self.metrics['internal_docs_retrieved'] += len(source_docs)
            
            # Step 2: Format internal sources
            internal_sources = []
            for doc in source_docs:
                internal_sources.append({
                    'title': doc.metadata.get('filename', 'Unknown'),
                    'source': doc.metadata.get('source', 'Internal Documentation'),
                    'type': 'internal',
                    'content': doc.page_content[:200] + "...",
                    'score': 0.8  # LangChain doesn't provide similarity scores by default
                })
            
            # Step 3: Perform web search if needed
            web_sources = []
            if use_web_search:
                web_sources = self._perform_web_search(query)
            
            # Step 4: Combine responses if web search was performed
            if web_sources:
                # Create enhanced response combining both sources
                combined_context = f"""
INTERNAL DOCUMENTATION RESPONSE:
{internal_response}

EXTERNAL WEB SOURCES:
{'; '.join([f"- {ws['title']}: {ws['content'][:100]}..." for ws in web_sources])}
"""
                
                enhancement_prompt = f"""Based on the internal documentation and external web sources, provide a comprehensive answer to: {query}

{combined_context}

Provide a response that:
1. Integrates both internal and external information
2. Clearly indicates source attribution
3. Highlights any complementary information
4. Provides practical examples

Enhanced Answer:"""
                
                final_response = self.llm.invoke(enhancement_prompt)
            else:
                final_response = internal_response
            
            # Step 5: Safety check
            safety_check = self._safety_check(query, final_response)
            
            # Step 6: Calculate metrics
            end_time = time.time()
            response_time = end_time - start_time
            self._update_metrics(response_time)
            
            # Step 7: Update memory
            self.memory.save_context(
                {"input": query},
                {"output": final_response}
            )
            
            return {
                'query': query,
                'response': final_response,
                'sources': internal_sources + web_sources,
                'internal_sources_count': len(internal_sources),
                'external_sources_count': len(web_sources),
                'response_time': response_time,
                'safety_check': safety_check,
                'framework': 'langchain',
                'dual_source_integration': len(internal_sources) > 0 and len(web_sources) > 0
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'query': query,
                'response': f"Error processing query with LangChain: {str(e)}",
                'sources': [],
                'internal_sources_count': 0,
                'external_sources_count': 0,
                'response_time': time.time() - start_time,
                'safety_check': {'is_safe': False, 'warnings': ['Processing error']},
                'framework': 'langchain',
                'dual_source_integration': False
            }
    
    def _update_metrics(self, response_time: float):
        """Update performance metrics"""
        total_queries = self.metrics['total_queries']
        current_avg = self.metrics['avg_response_time']
        
        # Calculate new average response time
        self.metrics['avg_response_time'] = (
            (current_avg * (total_queries - 1) + response_time) / total_queries
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'framework': 'LangChain',
            'total_queries': self.metrics['total_queries'],
            'avg_response_time': round(self.metrics['avg_response_time'], 3),
            'total_docs_processed': self.metrics['documents_processed'],
            'web_searches_performed': self.metrics['web_searches_performed'],
            'internal_docs_retrieved': self.metrics['internal_docs_retrieved'],
            'memory_size': len(self.memory.buffer) if self.memory else 0,
            'vector_store_cached': self.metrics['vector_store_loaded_from_cache'],
            'vector_store_path': self.vector_db_path
        }
    
    def clear_vector_cache(self):
        """Clear the vector store cache to force rebuild"""
        import shutil
        if os.path.exists(self.vector_db_path):
            shutil.rmtree(self.vector_db_path)
            logger.info(f"Cleared vector store cache at {self.vector_db_path}")
            return True
        return False
    
    def export_conversation(self) -> str:
        """Export conversation history"""
        conversation = []
        if self.memory and hasattr(self.memory, 'buffer'):
            for message in self.memory.buffer:
                if hasattr(message, 'content'):
                    conversation.append({
                        'type': message.__class__.__name__,
                        'content': message.content
                    })
        
        return json.dumps({
            'framework': 'LangChain',
            'conversation': conversation,
            'metrics': self.get_metrics(),
            'export_time': datetime.now().isoformat()
        }, indent=2)