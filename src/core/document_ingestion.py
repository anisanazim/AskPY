import os
import PyPDF2
from typing import List, Dict, Any
from pathlib import Path
from src.utils.logger import get_logger
from src.config.settings import settings
import re

logger = get_logger()

class DocumentIngestion:
    def __init__(self):
        self.data_source_path = settings.data_source_path
        
    def load_pdf_documents(self, pdf_dir: str) -> List[Dict[str, Any]]:
        """Load and extract text from PDF files with enhanced metadata"""
        documents = []
        pdf_path = Path(pdf_dir)
        
        if not pdf_path.exists():
            logger.warning(f"PDF directory not found: {pdf_dir}")
            return documents
            
        for pdf_file in pdf_path.glob("*.pdf"):
            try:
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        text += f"[Page {page_num + 1}] {page_text}\\n"
                    
                    # Enhanced metadata extraction
                    title = self._extract_title_from_filename(pdf_file.stem)
                    category = self._categorize_document(text, pdf_file.stem)
                    
                    documents.append({
                        'content': text,
                        'source': str(pdf_file),
                        'type': 'pdf',
                        'title': title,
                        'category': category,
                        'filename': pdf_file.name,
                        'page_count': len(pdf_reader.pages)
                    })
                    
                logger.info(f"Successfully loaded PDF: {pdf_file.name} ({len(pdf_reader.pages)} pages)")
                
            except Exception as e:
                logger.error(f"Error loading PDF {pdf_file}: {str(e)}")
                
        return documents
    
    def load_text_documents(self, txt_dir: str) -> List[Dict[str, Any]]:
        """Load and extract text from TXT files with enhanced metadata"""
        documents = []
        txt_path = Path(txt_dir)
        
        if not txt_path.exists():
            logger.warning(f"Text directory not found: {txt_dir}")
            return documents
            
        for txt_file in txt_path.glob("*.txt"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as file:
                    text = file.read()
                    
                    # Enhanced metadata extraction
                    title = self._extract_title_from_content(text) or self._extract_title_from_filename(txt_file.stem)
                    category = self._categorize_document(text, txt_file.stem)
                    
                    documents.append({
                        'content': text,
                        'source': str(txt_file),
                        'type': 'txt',
                        'title': title,
                        'category': category,
                        'filename': txt_file.name,
                        'word_count': len(text.split())
                    })
                    
                logger.info(f"Successfully loaded TXT: {txt_file.name} ({len(text.split())} words)")
                
            except Exception as e:
                logger.error(f"Error loading TXT {txt_file}: {str(e)}")
                
        return documents
    
    def _extract_title_from_filename(self, filename: str) -> str:
        """Extract a readable title from filename"""
        # Remove common prefixes and clean up
        title = filename.replace('_', ' ').replace('-', ' ')
        
        # Capitalize words properly
        title = ' '.join(word.capitalize() for word in title.split())
        
        # Handle common Python documentation patterns
        title_mappings = {
            'Datastructures': 'Python Data Structures',
            'Controlflow': 'Python Control Flow',
            'Functional': 'Python Functional Programming',
            'Classes': 'Python Classes and Objects',
            'Modules': 'Python Modules and Packages',
            'Errors': 'Python Error Handling',
            'Stdlib': 'Python Standard Library',
            'Tutorial': 'Python Tutorial'
        }
        
        for key, value in title_mappings.items():
            if key.lower() in title.lower():
                return value
        
        return f"Python {title}" if 'python' not in title.lower() else title
    
    def _extract_title_from_content(self, text: str) -> str:
        """Extract title from document content"""
        lines = text.split('\\n')[:10]  # Check first 10 lines
        
        for line in lines:
            line = line.strip()
            # Look for markdown headers
            if line.startswith('# '):
                return line[2:].strip()
            # Look for title patterns
            if 'title:' in line.lower():
                return line.split(':', 1)[1].strip()
        
        return None
    
    def _categorize_document(self, text: str, filename: str) -> str:
        """Categorize document based on content and filename"""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        categories = {
            'basics': ['variable', 'type', 'string', 'number', 'basic', 'introduction', 'getting started'],
            'data_structures': ['list', 'dict', 'tuple', 'set', 'array', 'datastructure'],
            'control_flow': ['if', 'else', 'while', 'for', 'loop', 'condition', 'controlflow'],
            'functions': ['function', 'def', 'return', 'parameter', 'argument', 'lambda'],
            'classes': ['class', 'object', 'method', 'inheritance', 'oop', 'self'],
            'modules': ['import', 'module', 'package', 'library', 'from'],
            'errors': ['error', 'exception', 'try', 'except', 'finally', 'raise'],
            'advanced': ['decorator', 'generator', 'iterator', 'metaclass', 'context', 'advanced'],
            'libraries': ['numpy', 'pandas', 'matplotlib', 'requests', 'django', 'flask'],
            'web': ['web', 'http', 'api', 'flask', 'django', 'fastapi', 'requests'],
            'data_science': ['numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn', 'data']
        }
        
        # Check filename first
        for category, keywords in categories.items():
            if any(keyword in filename_lower for keyword in keywords):
                return category
        
        # Check content
        for category, keywords in categories.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            if keyword_count >= 2:  # At least 2 keywords found
                return category
        
        return 'general'
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks with enhanced metadata preservation"""
        chunks = []
        
        for doc in documents:
            text = doc['content']
            
            # Improved chunking: try to break at sentence boundaries
            sentences = self._split_into_sentences(text)
            
            current_chunk = ""
            current_chunk_sentences = []
            
            for sentence in sentences:
                # Check if adding this sentence would exceed chunk size
                potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
                
                if len(potential_chunk.split()) <= settings.chunk_size:
                    current_chunk = potential_chunk
                    current_chunk_sentences.append(sentence)
                else:
                    # Save current chunk if it has content
                    if current_chunk.strip():
                        chunks.append(self._create_chunk(
                            current_chunk, doc, len(chunks), current_chunk_sentences
                        ))
                    
                    # Start new chunk with current sentence
                    current_chunk = sentence
                    current_chunk_sentences = [sentence]
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append(self._create_chunk(
                    current_chunk, doc, len(chunks), current_chunk_sentences
                ))
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences more intelligently"""
        # Simple sentence splitting that handles code blocks
        sentences = []
        current_sentence = ""
        in_code_block = False
        
        for line in text.split('\\n'):
            line = line.strip()
            
            # Check for code block markers
            if line.startswith('```') or line.startswith('    ') and len(line) > 4:
                in_code_block = not in_code_block
            
            if in_code_block:
                current_sentence += line + "\\n"
            else:
                # Split on sentence boundaries
                line_sentences = re.split(r'(?<=[.!?])\\s+', line)
                for i, sentence in enumerate(line_sentences):
                    if i == 0:
                        current_sentence += " " + sentence if current_sentence else sentence
                    else:
                        if current_sentence.strip():
                            sentences.append(current_sentence.strip())
                        current_sentence = sentence
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return [s for s in sentences if len(s.strip()) > 10]  # Filter very short sentences
    
    def _create_chunk(self, content: str, doc: Dict[str, Any], chunk_index: int, sentences: List[str]) -> Dict[str, Any]:
        """Create a chunk with enhanced metadata"""
        return {
            'content': content,
            'source': doc['source'],
            'type': doc['type'],
            'title': doc['title'],
            'category': doc['category'],
            'filename': doc['filename'],
            'chunk_index': chunk_index,
            'sentence_count': len(sentences),
            'word_count': len(content.split()),
            'has_code': bool(re.search(r'(def |class |import |from |```)', content)),
            'chunk_summary': self._generate_chunk_summary(content)
        }
    
    def _generate_chunk_summary(self, content: str) -> str:
        """Generate a brief summary of chunk content"""
        # Extract first meaningful sentence
        sentences = content.split('.')[:2]
        summary = '. '.join(sentences).strip()
        
        # Limit length
        if len(summary) > 100:
            summary = summary[:97] + "..."
        
        return summary
    
    def load_all_documents(self) -> List[Dict[str, Any]]:
        """Load all documents from data source directory with enhanced processing"""
        all_documents = []
        
        # Load PDF documents
        pdf_dir = os.path.join(self.data_source_path, 'pdf')
        all_documents.extend(self.load_pdf_documents(pdf_dir))
        
        # Load text documents
        txt_dir = os.path.join(self.data_source_path, 'txt')
        all_documents.extend(self.load_text_documents(txt_dir))
        
        # Chunk all documents
        chunks = self.chunk_documents(all_documents)
        
        logger.info(f"Loaded {len(all_documents)} documents with {len(chunks)} chunks")
        logger.info(f"Categories found: {set(doc['category'] for doc in all_documents)}")
        
        return chunks