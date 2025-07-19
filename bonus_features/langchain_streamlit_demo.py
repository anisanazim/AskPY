import streamlit as st
import sys
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

st.set_page_config(
    page_title="AskPy – Supercharged with LangChain & Smart Features",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/186px-Python-logo-notext.svg.png",
    layout="wide"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF9800;
        text-align: center;
        margin-bottom: 1rem;
    }
    .langchain-badge {
        background: linear-gradient(45deg, #FF9800, #F57C00);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .metric-card {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF9800;
        margin: 0.5rem 0;
    }
    .error-card {
        background: #ffebee;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f44336;
        margin: 0.5rem 0;
    }
    .success-card {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
    }
    .demo-subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">AskPy</h1>', unsafe_allow_html=True)

st.markdown('<div class="demo-subtitle">RAG Chatbot – Supercharged with LangChain & Smart Features</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing LangChain RAG Pipeline...")
def get_langchain_pipeline():
    """Initialize and cache the LangChain pipeline"""
    
    # Get the project root (current working directory when running streamlit)
    project_root = os.getcwd()
    
    # Add project root to Python path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Load environment variables
    env_path = os.path.join(project_root, '.env')
    if os.path.exists(env_path):
        from dotenv import load_dotenv
        load_dotenv(env_path)
    
    # Check API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        st.error("❌ GROQ_API_KEY not set properly in .env file")
        return None
    
    try:
        # Import here to avoid issues with caching
        from bonus_features.langchain_demo import LangChainRAGPipeline
        
        data_source_path = os.path.join(project_root, "data_source")
        pipeline = LangChainRAGPipeline(groq_api_key, data_source_path)
        
        return pipeline
        
    except Exception as e:
        st.error(f"❌ Error initializing LangChain pipeline: {str(e)}")
        return None

# Initialize pipeline ONCE using caching
pipeline = None
try:
    # Get the project root
    project_root = os.getcwd()
    
    # Add project root to Python path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    st.markdown(f'<div class="success-card">✅ Project root: {project_root}</div>', unsafe_allow_html=True)
    
    # Load environment variables
    env_path = os.path.join(project_root, '.env')
    if os.path.exists(env_path):
        from dotenv import load_dotenv
        load_dotenv(env_path)
        st.markdown('<div class="success-card">✅ Environment variables loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="error-card">❌ .env file not found at: {env_path}</div>', unsafe_allow_html=True)
    
    # Check imports first
    try:
        from bonus_features.langchain_demo import LangChainRAGPipeline
        st.markdown('<div class="success-card">✅ LangChain demo imported successfully</div>', unsafe_allow_html=True)
        
        # Check API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key or groq_api_key == "your_groq_api_key_here":
            st.markdown('<div class="error-card">❌ GROQ_API_KEY not set properly in .env file</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-card">✅ GROQ_API_KEY found</div>', unsafe_allow_html=True)
            
            # Initialize pipeline using cache
            pipeline = get_langchain_pipeline()
            
            if pipeline:
                st.markdown('<div class="success-card">✅ LangChain pipeline initialized successfully!</div>', unsafe_allow_html=True)
                
                # Show cache status
                metrics = pipeline.get_metrics()
                if metrics.get('vector_store_cached', False):
                    st.markdown('<div class="success-card">⚡ Vector store loaded from cache - Lightning fast!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-card">🔧 Vector store built from scratch - Next time will be faster!</div>', unsafe_allow_html=True)
        
    except ImportError as e:
        st.markdown(f'<div class="error-card">❌ Import error: {str(e)}</div>', unsafe_allow_html=True)

except Exception as e:
    st.markdown(f'<div class="error-card">❌ Setup error: {str(e)}</div>', unsafe_allow_html=True)

# Sidebar with LangChain features
with st.sidebar:
    st.header("🦜 LangChain Features")
    
    st.markdown("**Framework Benefits:**")
    features = [
        "✅ Built-in document loaders",
        "✅ Conversation memory", 
        "✅ Safety guardrails",
        "✅ Performance metrics",
        "✅ Extensible architecture",
        "✅ Industry standard"
    ]
    for feature in features:
        st.text(feature)
    
    if pipeline:
        st.markdown("---")
        st.subheader("📊 System Metrics")
        
        if st.button("🔄 Refresh Metrics"):
            try:
                metrics = pipeline.get_metrics()
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        st.metric(key.replace('_', ' ').title(), value)
                    else:
                        st.text(f"{key.replace('_', ' ').title()}: {value}")
            except Exception as e:
                st.error(f"Error getting metrics: {e}")
        
        st.markdown("---")
        
        # Add cache management
        st.subheader("🗃️ Cache Management")
        if st.button("🗑️ Clear Vector Cache"):
            try:
                if pipeline.clear_vector_cache():
                    st.success("✅ Vector cache cleared! Next restart will rebuild.")
                    # Clear Streamlit cache too
                    st.cache_resource.clear()
                else:
                    st.info("ℹ️ No cache to clear.")
            except Exception as e:
                st.error(f"Error clearing cache: {e}")
        
        if st.button("💾 Export Conversation"):
            try:
                conversation_data = pipeline.export_conversation()
                st.download_button(
                    label="Download LangChain Results",
                    data=conversation_data,
                    file_name=f"langchain_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Export error: {e}")

# Main interface
if pipeline:
    st.subheader("💬 LangChain RAG Chat")
    
    # Demo queries
    st.markdown("**Quick Test Queries:**")
    demo_queries = [
        "How do I create a Python class?",
        "What are Python decorators?",
        "How do I handle exceptions in Python?",
        "What is the difference between lists and tuples?",
        "How do I work with files in Python?"
    ]
    
    cols = st.columns(len(demo_queries))
    for i, demo_query in enumerate(demo_queries):
        with cols[i]:
            if st.button(f"📝 {demo_query[:20]}...", key=f"demo_{i}"):
                st.session_state.selected_query = demo_query
    
    # Query input
    if 'selected_query' in st.session_state:
        query = st.text_input("Question:", value=st.session_state.selected_query)
        # Don't delete immediately to prevent re-initialization
        if query != st.session_state.selected_query:
            del st.session_state.selected_query
    else:
        query = st.text_input("Question:", placeholder="Ask any Python programming question...")
    
    # Options
    col1, col2 = st.columns(2)
    with col1:
        use_web_search = st.checkbox("🌐 Include web search", value=True)
    with col2:
        show_sources = st.checkbox("📚 Show detailed sources", value=True)
    
    # Process query
    if st.button("🚀 Process with LangChain", type="primary") and query:
        try:
            with st.spinner("🦜 Processing with LangChain framework..."):
                result = pipeline.process_query(query, use_web_search)
            
            # Display response
            st.markdown("### 🤖 LangChain Response")
            st.markdown(f'<div class="metric-card">{result["response"]}</div>', unsafe_allow_html=True)
            
            # Performance metrics
            st.markdown("### 📊 Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("⚡ Response Time", f"{result['response_time']:.2f}s")
            with col2:
                st.metric("📄 Internal Sources", result['internal_sources_count'])
            with col3:
                st.metric("🌐 External Sources", result['external_sources_count'])
            with col4:
                dual_source = "✅" if result.get('dual_source_integration', False) else "❌"
                st.metric("🎯 Dual-Source", dual_source)
            
            # Safety check results
            safety_check = result.get('safety_check', {})
            if safety_check.get('warnings'):
                st.markdown(f'''
                <div class="warning-card">
                    <strong>🛡️ Safety Warnings:</strong><br>
                    {', '.join(safety_check['warnings'])}
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.success("🛡️ Safety check passed - no issues detected")
            
            # Sources section
            if show_sources and result.get('sources'):
                st.markdown("### 📚 Sources Used")
                
                sources = result['sources']
                internal_sources = [s for s in sources if s.get('type') == 'internal']
                external_sources = [s for s in sources if s.get('type') == 'external']
                
                if internal_sources:
                    st.markdown("**📄 Internal Documentation (LangChain Retrieved):**")
                    for source in internal_sources:
                        with st.expander(f"📖 {source.get('title', 'Unknown')}"):
                            st.markdown(f"**Source:** {source.get('source', 'Unknown')}")
                            st.markdown(f"**Content Preview:** {source.get('content', 'No preview')}")
                
                if external_sources:
                    st.markdown("**🌐 External Web Sources:**")
                    for source in external_sources:
                        with st.expander(f"🔗 {source.get('title', 'Unknown')}"):
                            st.markdown(f"**URL:** [{source.get('source', '#')}]({source.get('source', '#')})")
                            st.markdown(f"**Content:** {source.get('content', 'No preview')}")
            
            # Framework comparison note
            st.markdown("---")
            st.info("""
            **🦜 LangChain Framework Benefits Demonstrated:**
            - **Document Loading**: Automatic PDF/TXT processing with metadata
            - **Memory Management**: Conversation context preservation  
            - **Safety Guardrails**: Built-in content validation and filtering
            - **Performance Tracking**: Comprehensive metrics and monitoring
            - **Extensibility**: Modular architecture for easy enhancement
            - **Persistent Caching**: Lightning-fast startup after first run
            """)
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            with st.expander("🔍 Error Details"):
                import traceback
                st.code(traceback.format_exc())

else:
    # Show troubleshooting information
    st.markdown("---")
    st.markdown("## 🔧 Setup Status")
    
    # Show directory info
    project_root = os.getcwd()
    st.markdown(f"**Project root:** `{project_root}`")
    
    # Check files
    st.markdown("### 📁 File Check:")
    files_to_check = [
        (os.path.join(project_root, '.env'), 'Environment variables'),
        (os.path.join(project_root, 'bonus_features', 'langchain_demo.py'), 'LangChain demo'),
        (os.path.join(project_root, 'data_source'), 'Data source directory'),
        (os.path.join(project_root, 'vector_db'), 'Vector cache directory')
    ]
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            st.markdown(f'✅ **{description}**: Found')
        else:
            st.markdown(f'❌ **{description}**: Missing at `{file_path}`')
    
    st.markdown("""
    ### 🚀 Quick Setup Commands:
    
    **1. Make sure .env file exists with:**
    ```
    GROQ_API_KEY=your_actual_api_key_here
    VECTOR_DB_PATH=./vector_db
    ```
    
    **2. Create data source directory:**
    ```bash
    mkdir -p data_source/pdf data_source/txt
    ```
    """)
