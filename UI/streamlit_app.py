import streamlit as st
import json
from datetime import datetime
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rag.rag_pipeline import RAGPipeline
from src.utils.logger import get_logger

logger = get_logger()

st.set_page_config(
    page_title="AskPy - RAG Chatbot",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/186px-Python-logo-notext.svg.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with dual-source styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .demo-subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left-color: #4caf50;
    }
    .source-type-internal {
        border-left: 4px solid #4caf50;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        background-color: #e8f5e9;
        border-radius: 5px;
    }
    .source-type-web {
        border-left: 4px solid #2196f3;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        background-color: #e3f2fd;
        border-radius: 5px;
    }
    .dual-source-badge {
        background: linear-gradient(45deg, #4caf50, #2196f3);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .stats-container {
        background-color: #e8f4f8;
        padding: 0.8rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .integration-highlight {
        background: linear-gradient(90deg, #e8f5e9, #e3f2fd);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 2px solid #9c27b0;
    }
    .file-info {
        background-color: #e8f4f8;
        padding: 0.8rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_pipeline():
    return RAGPipeline()

def count_existing_files():
    """Count existing files in data_source directory"""
    pdf_count = 0
    txt_count = 0
    
    # Count PDF files
    pdf_dir = "data_source/pdf"
    if os.path.exists(pdf_dir):
        pdf_count = len([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])
    
    # Count TXT files
    txt_dir = "data_source/txt"
    if os.path.exists(txt_dir):
        txt_count = len([f for f in os.listdir(txt_dir) if f.endswith('.txt')])
    
    return pdf_count, txt_count

def main():
    # Header
    st.markdown('<h1 class="main-header">AskPy</h1>', unsafe_allow_html=True)
    st.markdown('<div class="demo-subtitle">RAG Chatbot with Python Documentation + Web Search Integration</div>', unsafe_allow_html=True)
    
    # Initialize RAG pipeline
    if 'rag_pipeline' not in st.session_state:
        with st.spinner("Initializing AskPy..."):
            st.session_state.rag_pipeline = initialize_rag_pipeline()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        
        # System stats with error handling
        if st.button("📊 System Info"):
            try:
                stats = st.session_state.rag_pipeline.get_statistics()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Docs", value=stats.get('vector_store', {}).get('total_documents', 'N/A'))
                    st.metric(label="Web Search", value="On" if stats.get('web_search_enabled', False) else "Off")
                    
                with col2:
                    # st.metric(label="Index Size (MB)", value=f"{stats.get('vector_store', {}).get('index_size_mb', 0):.2f}")
                    st.metric(label="Chats", value=stats.get('conversation_length', 0))

            except Exception as e:
                st.error(f"Error getting stats: {e}")
                logger.error(f"Stats error: {e}")
        
        # Simplified setup section
        st.markdown("---")
        st.subheader("🛠️ Database Management")
        
        # Show current file status
        pdf_count, txt_count = count_existing_files()
        st.markdown(f'''
        <div class="file-info">
            <strong>📁 Current Files:</strong><br>
            • {pdf_count} PDF files<br>
            • {txt_count} TXT files<br>
            • Total: {pdf_count + txt_count} documents
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("---")
        # Rebuild vector database button
        if st.button("🔄 Rebuild Vector Database", help="Process all documents and rebuild vector database"):
            with st.spinner("Processing documents and rebuilding vector database..."):
                try:
                    if st.session_state.rag_pipeline.rebuild_vector_database():
                        st.success("✅ Vector database rebuilt successfully!")
                        st.info(f"🎯 Processed {pdf_count + txt_count} documents - ready for dual-source demonstrations!")
                        # Clear cache to reload pipeline
                        st.cache_resource.clear()
                    else:
                        st.error("❌ Failed to rebuild vector database")
                        st.warning("Make sure you have documents in data_source/pdf/ or data_source/txt/")
                except Exception as e:
                    st.error(f"Error rebuilding: {e}")
                    logger.error(f"Rebuild error: {e}")
        
        st.caption("💡 Use this if you've added new documents or if internal sources aren't working")
        
        st.markdown("---")
        st.subheader("💬 Chat Controls")
        if st.button("🗑️ Clear Chat"):
            st.session_state.rag_pipeline.clear_conversation_history()
            st.session_state.messages = []
            st.success("Chat cleared!")
        
        if st.button("💾 Export Chat"):
            if st.session_state.messages:
                try:
                    export_data = st.session_state.rag_pipeline.export_conversation()
                    st.download_button(
                        label="Download Demo Results",
                        data=export_data,
                        file_name=f"pythondocbot_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Export error: {e}")
            else:
                st.warning("No conversation to export")
        
        # Demo queries section
        st.markdown("---")
        st.subheader("🎯 Demo Queries")
        st.caption("Click to test dual-source integration:")
        
        demo_queries = [
            "What are Python classes?",
            "How do I import modules in Python?",
            "What are Python functions?",
            "Explain Python error handling",
            "What are list comprehensions?",
            "How do I format strings in Python?",
            "What are Python dictionaries?",
            "How do I create a list in Python?",
            
            
        ]
        
        for i, query in enumerate(demo_queries):
            if st.button(f"📝 {query}", key=f"demo_q_{i}", help="Click to ask this question"):
                st.session_state.demo_query = query
        
        # Information section
        st.markdown("---")
        st.subheader("💡 What to Expect")
        st.info(
        """
        **Internal Sources:**
        * 📄 Python PDFs (tutorial, library, reference)
        * 📝 Python TXT files (classes, controlflow, etc.)
        * 📋 Official Python documentation

        **External Sources:**
        * 🌐 Python.org documentation
        * 📚 Tutorial websites
        * 🔎 Latest Python practices

        **Integration:**
        * Both source types in responses
        * Clear source differentiation
        * Comprehensive answers
        """
    )
        
        st.markdown("---")
        st.caption("🚀 Using your existing Python documentation + web search")
    
    # Main chat interface
    st.subheader("💬 Chat Interface")
    
    # Display conversation history with enhanced source visualization
    for message in st.session_state.messages:
        with st.container():
            if message["role"] == "user":
                st.markdown(f'''
                <div class="chat-message user-message">
                    <strong>🧑‍💻 You:</strong> {message["content"]}
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="chat-message assistant-message">
                    <strong>AskPy:</strong><br>{message["content"]}
                </div>
                ''', unsafe_allow_html=True)
                
                # Enhanced sources display with safe access
                sources = message.get("sources", [])
                if sources:
                    # Categorize sources safely
                    internal_sources = [s for s in sources if s.get('type', '') == 'internal']
                    web_sources = [s for s in sources if s.get('type', '') == 'external']
                    
                    # Show integration badge if both types present
                    if internal_sources and web_sources:
                        st.markdown(f'''
                        <div class="integration-highlight">
                            <span class="dual-source-badge">🎯 DUAL-SOURCE INTEGRATION</span>
                            <strong>Successfully combined {len(internal_sources)} internal + {len(web_sources)} external sources!</strong>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with st.expander(f"📚 Sources ({len(sources)} found)", expanded=False):
                        # Internal sources section
                        if internal_sources:
                            st.markdown("### 📄 Internal Python Documentation")
                            for source in internal_sources:
                                title = source.get('title', 'Unknown Document')
                                filename = source.get('filename', source.get('source', 'N/A'))
                                score = source.get('score', 0)
                                
                                st.markdown(f'''
                                <div class="source-type-internal">
                                    <strong>📖 {title}</strong><br>
                                    <small>📁 {filename}</small>
                                </div>
                                ''', unsafe_allow_html=True)
                        
                        # Web sources section
                        if web_sources:
                            if internal_sources:
                                st.markdown("---")
                            st.markdown("### 🌐 External Web Sources")
                            for source in web_sources:
                                title = source.get('title', 'Unknown Source')
                                url = source.get('source', '#')
                                
                                st.markdown(f'''
                                <div class="source-type-web">
                                    <strong>🔗 {title}</strong><br>
                                    <small><a href="{url}" target="_blank">View Source</a></small>
                                </div>
                                ''', unsafe_allow_html=True)
                        
                        # Summary section
                        if internal_sources and web_sources:
                            st.markdown("---")
                            st.markdown("**📊 Integration Summary:**")
                            st.markdown(f"• Internal Python docs: {len(internal_sources)}")
                            st.markdown(f"• External web sources: {len(web_sources)}")
                            st.markdown("• ✅ Successfully demonstrated dual-source RAG capability")
    
    # Handle demo query selection
    user_input = None
    if 'demo_query' in st.session_state:
        user_input = st.session_state.demo_query
        del st.session_state.demo_query
    
    # Chat input
    if not user_input:
        user_input = st.chat_input("Ask me anything about Python programming to see dual-source integration...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Process query with comprehensive error handling
        with st.spinner("🔍 Searching Python docs + web sources..."):
            try:
                result = st.session_state.rag_pipeline.process_query(user_input)
                
                # Safe extraction with defaults
                response = result.get('response', 'Sorry, I could not generate a response.')
                sources = result.get('sources', [])
                
                # Add assistant response
                assistant_message = {
                    "role": "assistant", 
                    "content": response,
                    "sources": sources
                }
                st.session_state.messages.append(assistant_message)
                
                # Enhanced status display with safe access
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    source_count = len(sources)
                    internal_count = len([s for s in sources if s.get('type', '') == 'internal'])
                    web_count = len([s for s in sources if s.get('type', '') == 'external'])
                    st.info(f"📊 {source_count} sources ({internal_count} internal, {web_count} web)")
                
                with col2:
                    # Safe check for web search usage
                    web_used = result.get('web_search_used', web_count > 0)
                    if web_used:
                        st.info("🌐 Enhanced with web search")
                    else:
                        st.info("📄 From Python documentation")
                
                with col3:
                    # Safe check for context usage
                    context_used = result.get('context_used', len(sources) > 0)
                    if context_used:
                        st.success("✅ Context found")
                    else:
                        st.warning("⚠️ Limited context")
                
                # Show dual-source success
                if internal_count > 0 and web_count > 0:
                    st.success("🎯 **Dual-Source Integration Successful!** Combined Python documentation with external web sources.")
                elif internal_count > 0:
                    st.info("📄 Used Python documentation only")
                elif web_count > 0:
                    st.info("🌐 Used web sources only")
                else:
                    st.warning("⚠️ No sources retrieved - try rebuilding vector database")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                logger.error(f"Query processing error: {e}")
                
                # Add error message to conversation
                error_message = {
                    "role": "assistant",
                    "content": f"I apologize, but I encountered an error processing your question: {str(e)}",
                    "sources": []
                }
                st.session_state.messages.append(error_message)
        
        st.rerun()

if __name__ == "__main__":
    main()