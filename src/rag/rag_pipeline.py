# Cleaned src/rag/rag_pipeline.py - Using existing files only

import json
from typing import List, Dict, Any, Tuple
from groq import Groq
from src.core.document_ingestion import DocumentIngestion
from src.core.embedding_service import EmbeddingService
from src.core.vector_store import VectorStore
from src.core.web_search import WebSearchService
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger()

class RAGPipeline:
    def __init__(self):
        self.groq_client = Groq(api_key=settings.groq_api_key)
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.web_search = WebSearchService()
        self.conversation_history = []

        # Dual-source strategy settings
        self.always_check_both = True
        self.min_internal_sources = 1
        self.min_web_sources = 1

        self.internal_quality_threshold = 0.4   # Minimum score for internal docs
        self.web_quality_threshold = 0.6        # Minimum score for web sources  
        self.max_internal_sources = 4           # Maximum internal sources
        self.max_web_sources = 3                # Maximum web sources

        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize vector store with existing documents"""
        if not self.vector_store.load_index():
            logger.info("Creating vector store from existing documents...")
            self._create_vector_store()
        else:
            logger.info("Loaded existing vector store")

    def _create_vector_store(self):
        """Create vector store from existing documents in data_source/"""
        doc_ingestion = DocumentIngestion()
        documents = doc_ingestion.load_all_documents()

        if documents:
            documents_with_embeddings = self.embedding_service.embed_documents(documents)
            embedding_dim = self.embedding_service.get_embedding_dimension()
            self.vector_store.create_index(documents_with_embeddings, embedding_dim)
            self.vector_store.save_index()
            logger.info(f"Vector store created with {len(documents)} documents")
        else:
            logger.warning("No documents found in data_source/ directory")

    def rebuild_vector_database(self):
        """Rebuild the vector database with current documents"""
        try:
            logger.info("Rebuilding vector database from existing documents...")
            
            # Load documents from data_source/
            doc_ingestion = DocumentIngestion()
            documents = doc_ingestion.load_all_documents()
            
            if documents:
                # Generate embeddings
                documents_with_embeddings = self.embedding_service.embed_documents(documents)
                
                # Create and save new index
                embedding_dim = self.embedding_service.get_embedding_dimension()
                self.vector_store.create_index(documents_with_embeddings, embedding_dim)
                self.vector_store.save_index()
                
                logger.info(f"Vector database rebuilt with {len(documents)} documents")
                return True
            else:
                logger.warning("No documents found to rebuild vector database")
                return False
                
        except Exception as e:
            logger.error(f"Error rebuilding vector database: {e}")
            return False

    def _retrieve_internal_sources(self, query_embedding, query: str) -> List[Tuple[Dict[str, Any], float]]:
        """Retrieve from internal Python documentation"""
        try:
            results = self.vector_store.similarity_search(query_embedding, k=max(self.min_internal_sources, 3))
            logger.info(f"Retrieved {len(results)} internal sources")
            return results
        except Exception as e:
            logger.error(f"Error retrieving internal sources: {e}")
            return []

    def _retrieve_web_sources(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve from external web sources"""
        try:
            if self.always_check_both:
                results = self.web_search.search_web(query)
                logger.info(f"Retrieved {len(results)} web sources")
                return results[:self.min_web_sources + 1]
            return []
        except Exception as e:
            logger.error(f"Error retrieving web sources: {e}")
            return []

    def _create_dual_source_context(self, internal_results: List[Tuple[Dict[str, Any], float]], 
                              web_results: List[Dict[str, Any]], query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Create context that shows both internal Python docs and external sources with dynamic quality filtering"""
        context_parts = []
        sources = []

        # Filter high-quality internal sources dynamically
        good_internal = [(doc, score) for doc, score in internal_results 
                        if score >= self.internal_quality_threshold]
        selected_internal = good_internal[:self.max_internal_sources]

        # Internal Python documentation
        if selected_internal:
            context_parts.append("## INTERNAL PYTHON DOCUMENTATION:")
            context_parts.append("(From your uploaded PDF and TXT Python documentation files)")
            context_parts.append("")

            for doc, score in selected_internal:
                context_parts.append(f"**Internal Source: {doc['title']}**")
                context_parts.append(doc['content'][:700] + "...")
                context_parts.append("---")

                sources.append({
                    'title': doc['title'],
                    'source': doc['source'],
                    'type': 'internal',
                    'score': score,
                    'category': doc.get('category', 'python_docs'),
                    'filename': doc.get('filename', 'N/A')
                })

        # Filter high-quality web sources dynamically
        good_web = [result for result in web_results 
                    if result.get('relevance_score', 0.8) >= self.web_quality_threshold]
        selected_web = good_web[:self.max_web_sources]

        # External web sources
        if selected_web:
            context_parts.append("\n## EXTERNAL WEB SOURCES:")
            context_parts.append("(From external Python resources and tutorials)")
            context_parts.append("")

            for result in selected_web:
                relevance = result.get('relevance_score', 0.8)
                context_parts.append(f"**External Source: {result['title']}**")
                context_parts.append(result['content'][:700] + "...")
                context_parts.append("---")

                sources.append({
                    'title': result['title'],
                    'source': result['url'],
                    'type': 'external',
                    'score': relevance,
                    'category': 'web_resource'
                })

        # Dynamic response instructions based on available sources
        context_parts.append("\n## RESPONSE INSTRUCTIONS:")
        if selected_internal and selected_web:
            context_parts.append("Provide a comprehensive answer that:")
            context_parts.append("1. Integrates information from BOTH internal Python documentation AND external web sources")
            context_parts.append("2. Clearly indicates when information comes from internal docs vs external sources")
            context_parts.append("3. Shows how internal documentation complements external resources")
            context_parts.append("4. Provides practical Python examples from both perspectives")
        elif selected_internal:
            context_parts.append("Provide a comprehensive answer based on internal Python documentation:")
            context_parts.append("1. Use the high-quality internal documentation provided")
            context_parts.append("2. Include practical Python examples from the documentation")
        elif selected_web:
            context_parts.append("Provide a comprehensive answer based on external web sources:")
            context_parts.append("1. Use the high-quality external sources provided") 
            context_parts.append("2. Include practical Python examples from web resources")

        return "\n".join(context_parts), sources

    def _create_enhanced_prompt(self, query: str, context: str, conversation_history: List[Dict]) -> str:
        """Create prompt that emphasizes dual-source integration"""
        
        history_text = ""
        if conversation_history:
            history_text = "\nRecent Conversation:\n"
            for msg in conversation_history[-2:]:
                history_text += f"User: {msg['user']}\n"
                history_text += f"Assistant: {msg['assistant'][:150]}...\n"

        prompt = f"""You are PythonDocBot, an advanced RAG system that integrates internal Python documentation with external web resources.

CRITICAL INSTRUCTIONS:
1. You have access to BOTH internal Python documentation (PDFs, TXT files) AND external web sources
2. Always mention and integrate information from BOTH source types when available
3. Clearly distinguish between internal documentation and external web resources
4. Show how internal docs complement external resources for comprehensive understanding

AVAILABLE CONTEXT:
{context}

{history_text}

USER QUESTION: {query}

RESPONSE FORMAT:
1. Start with a clear and concise answer that integrates insights from both internal and external sources
2. Attribute each insight: use phrases like "According to the internal documentation..." or "Based on an external source such as [website or tool]..."
3. Include practical, runnable Python examples — cite the source of each example
4. End with a summary showing how combining internal and external knowledge gives a more complete, trustworthy, or nuanced answer

ANSWER:"""

        return prompt

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query with dual-source strategy"""
        try:
            logger.info(f"Processing query with dual-source strategy: {query}")
            
            # Always retrieve from both sources
            query_embedding = self.embedding_service.embed_query(query)
            internal_results = self._retrieve_internal_sources(query_embedding, query)
            web_results = self._retrieve_web_sources(query)
            
            # Create dual-source context
            context, sources = self._create_dual_source_context(internal_results, web_results, query)
            
            # Generate response
            prompt = self._create_enhanced_prompt(query, context, self.conversation_history)

            response = self.groq_client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": "You are PythonDocBot, an advanced dual-source RAG system for Python programming."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.1
            )

            assistant_response = response.choices[0].message.content

            # Update conversation history
            self.conversation_history.append({
                'user': query,
                'assistant': assistant_response,
                'sources': sources,
                'dual_source_demo': True
            })

            return {
                'query': query,
                'response': assistant_response,
                'sources': sources,
                'internal_sources_count': len([s for s in sources if s['type'] == 'internal']),
                'external_sources_count': len([s for s in sources if s['type'] == 'external']),
                'dual_source_integration': True,
                'demonstration_mode': True,
                'web_search_used': len([s for s in sources if s['type'] == 'external']) > 0,
                'context_used': len(sources) > 0
            }

        except Exception as e:
            logger.error(f"Error in dual-source processing: {str(e)}")
            return {
                'query': query,
                'response': f"Error processing query: {str(e)}",
                'sources': [],
                'dual_source_integration': False,
                'demonstration_mode': False,
                'web_search_used': False,
                'context_used': False
            }

    def get_demo_queries(self) -> List[str]:
        """Get demo queries tailored for Python documentation"""
        return [
            "What are Python functions and how do I define them?",
            "How do I handle errors in Python?",
            "What are Python classes and objects?",
            "How do I work with lists in Python?",
            "What are Python modules and how do I import them?",
            "How do I format strings in Python?",
            "What are Python dictionaries?",
            "How do I use control flow statements in Python?"
        ]

    def get_conversation_history(self) -> List[Dict]:
        return self.conversation_history

    def clear_conversation_history(self):
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def export_conversation(self) -> str:
        return json.dumps(self.conversation_history, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            total_conversations = len(self.conversation_history)
            dual_source_conversations = sum(1 for conv in self.conversation_history if conv.get('dual_source_demo', False))
            
            return {
                'vector_store': {
                    'total_documents': len(self.vector_store.documents) if hasattr(self.vector_store, 'documents') and self.vector_store.documents else 0,
                    'index_size': self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') and self.vector_store.index else 0,
                    'dimension': self.vector_store.index.d if hasattr(self.vector_store, 'index') and self.vector_store.index else 0
                },
                'conversation_length': total_conversations,
                'web_search_enabled': self.web_search.enabled if hasattr(self.web_search, 'enabled') else True,
                'dual_source_demonstrations': dual_source_conversations,
                'demonstration_mode': True,
                'system_type': 'Dual-Source RAG Pipeline'
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                'vector_store': {'total_documents': 0, 'index_size': 0, 'dimension': 0},
                'conversation_length': 0,
                'web_search_enabled': True,
                'dual_source_demonstrations': 0,
                'demonstration_mode': True,
                'system_type': 'Dual-Source RAG Pipeline'
            }

    def get_system_stats(self) -> Dict[str, Any]:
        """Alternative method name for statistics"""
        return self.get_statistics()