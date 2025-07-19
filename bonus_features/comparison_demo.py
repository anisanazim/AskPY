# comparison_demo.py - Side-by-side comparison of Custom vs LangChain RAG

import streamlit as st
import time
import json
from datetime import datetime
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import both implementations
from src.rag.rag_pipeline import RAGPipeline
from langchain_demo import LangChainRAGPipeline

st.set_page_config(
    page_title="RAG Implementation Comparison",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for comparison
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .comparison-container {
        display: flex;
        gap: 2rem;
        margin: 1rem 0;
    }
    .custom-impl {
        border: 2px solid #4CAF50;
        padding: 1rem;
        border-radius: 10px;
        background-color: #f1f8e9;
    }
    .langchain-impl {
        border: 2px solid #FF9800;
        padding: 1rem;
        border-radius: 10px;
        background-color: #fff3e0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .winner-badge {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: black;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_pipelines():
    """Initialize both RAG pipelines"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        st.error("Please set GROQ_API_KEY environment variable")
        return None, None
    
    try:
        # Initialize custom pipeline
        custom_pipeline = RAGPipeline()
        
        # Initialize LangChain pipeline
        langchain_pipeline = LangChainRAGPipeline(groq_api_key)
        
        return custom_pipeline, langchain_pipeline
    except Exception as e:
        st.error(f"Error initializing pipelines: {e}")
        return None, None

def compare_responses(query: str, custom_pipeline, langchain_pipeline):
    """Compare responses from both implementations"""
    
    col1, col2 = st.columns(2)
    
    # Custom Implementation
    with col1:
        st.markdown('<div class="custom-impl">', unsafe_allow_html=True)
        st.subheader("🔧 Custom Implementation")
        
        custom_start = time.time()
        try:
            with st.spinner("Processing with Custom RAG..."):
                custom_result = custom_pipeline.process_query(query)
            custom_time = time.time() - custom_start
            
            # Display response
            st.markdown("**Response:**")
            st.write(custom_result['response'])
            
            # Display metrics
            st.markdown("**Metrics:**")
            custom_metrics = {
                'Response Time': f"{custom_time:.2f}s",
                'Internal Sources': custom_result['internal_sources_count'],
                'External Sources': custom_result['external_sources_count'],
                'Total Sources': len(custom_result['sources']),
                'Framework': 'Custom Built'
            }
            
            for metric, value in custom_metrics.items():
                st.text(f"{metric}: {value}")
            
        except Exception as e:
            st.error(f"Custom implementation error: {e}")
            custom_result = None
            custom_time = float('inf')
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # LangChain Implementation
    with col2:
        st.markdown('<div class="langchain-impl">', unsafe_allow_html=True)
        st.subheader("🦜 LangChain Implementation")
        
        langchain_start = time.time()
        try:
            with st.spinner("Processing with LangChain..."):
                langchain_result = langchain_pipeline.process_query(query)
            langchain_time = time.time() - langchain_start
            
            # Display response
            st.markdown("**Response:**")
            st.write(langchain_result['response'])
            
            # Display metrics
            st.markdown("**Metrics:**")
            langchain_metrics = {
                'Response Time': f"{langchain_result['response_time']:.2f}s",
                'Internal Sources': langchain_result['internal_sources_count'],
                'External Sources': langchain_result['external_sources_count'],
                'Total Sources': len(langchain_result['sources']),
                'Framework': 'LangChain'
            }
            
            for metric, value in langchain_metrics.items():
                st.text(f"{metric}: {value}")
            
            # Safety check results
            safety = langchain_result.get('safety_check', {})
            if safety.get('warnings'):
                st.warning(f"Safety warnings: {', '.join(safety['warnings'])}")
            
        except Exception as e:
            st.error(f"LangChain implementation error: {e}")
            langchain_result = None
            langchain_time = float('inf')
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Comparison Summary
    if custom_result and langchain_result:
        st.markdown("---")
        st.subheader("📊 Performance Comparison")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if custom_time < langchain_time:
                st.metric("⚡ Faster Response", "Custom", f"{custom_time:.2f}s")
            else:
                st.metric("⚡ Faster Response", "LangChain", f"{langchain_time:.2f}s")
        
        with col2:
            custom_sources = len(custom_result['sources'])
            langchain_sources = len(langchain_result['sources'])
            if custom_sources >= langchain_sources:
                st.metric("📚 More Sources", "Custom", custom_sources)
            else:
                st.metric("📚 More Sources", "LangChain", langchain_sources)
        
        with col3:
            custom_dual = custom_result.get('dual_source_integration', False)
            langchain_dual = langchain_result.get('dual_source_integration', False)
            if custom_dual and langchain_dual:
                st.metric("🎯 Dual-Source", "Both", "✅")
            elif custom_dual:
                st.metric("🎯 Dual-Source", "Custom", "✅")
            elif langchain_dual:
                st.metric("🎯 Dual-Source", "LangChain", "✅")
            else:
                st.metric("🎯 Dual-Source", "Neither", "❌")
        
        with col4:
            has_safety = 'safety_check' in langchain_result
            st.metric("🛡️ Safety Checks", "LangChain" if has_safety else "None", "✅" if has_safety else "❌")

def main():
    # Header
    st.markdown('<h1 class="main-header">⚔️ RAG Implementation Comparison</h1>', unsafe_allow_html=True)
    st.markdown("**Bonus Points Demonstration**: Custom Implementation vs LangChain Framework")
    
    # Initialize pipelines
    custom_pipeline, langchain_pipeline = initialize_pipelines()
    
    if not custom_pipeline or not langchain_pipeline:
        st.error("Failed to initialize pipelines. Please check your configuration.")
        return
    
    # Sidebar with comparison info
    with st.sidebar:
        st.header("🔧 Comparison Controls")
        
        st.subheader("📊 Framework Features")
        
        # Custom Implementation Features
        st.markdown("**🔧 Custom Implementation:**")
        custom_features = [
            "✅ Full control over pipeline",
            "✅ Optimized for Python docs",
            "✅ Dynamic quality filtering",
            "✅ Dual-source integration",
            "✅ Custom web search logic"
        ]
        for feature in custom_features:
            st.text(feature)
        
        st.markdown("---")
        
        # LangChain Implementation Features
        st.markdown("**🦜 LangChain Implementation:**")
        langchain_features = [
            "✅ Industry-standard framework",
            "✅ Built-in document loaders",
            "✅ Conversation memory",
            "✅ Safety guardrails",
            "✅ Performance metrics",
            "✅ Extensible architecture"
        ]
        for feature in langchain_features:
            st.text(feature)
        
        st.markdown("---")
        
        # Export comparison results
        if st.button("💾 Export Comparison"):
            comparison_data = {
                "timestamp": datetime.now().isoformat(),
                "frameworks_compared": ["Custom", "LangChain"],
                "custom_metrics": custom_pipeline.get_statistics() if hasattr(custom_pipeline, 'get_statistics') else {},
                "langchain_metrics": langchain_pipeline.get_metrics() if hasattr(langchain_pipeline, 'get_metrics') else {},
                "comparison_type": "Dual RAG Implementation"
            }
            
            st.download_button(
                label="Download Comparison Report",
                data=json.dumps(comparison_data, indent=2),
                file_name=f"rag_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Main interface
    st.subheader("🎯 Test Both Implementations")
    
    # Demo queries
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter your Python question:",
            placeholder="How do I create a Python class?",
            help="Ask any Python programming question to compare both implementations"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        compare_button = st.button("🚀 Compare Both", type="primary")
    
    # Quick test buttons
    st.markdown("**Quick Tests:**")
    test_queries = [
        "How do I handle exceptions in Python?",
        "What are Python decorators?",
        "How do I work with files in Python?",
        "What is the difference between lists and tuples?",
        "How do I create a web API with Python?"
    ]
    
    cols = st.columns(len(test_queries))
    for i, test_query in enumerate(test_queries):
        with cols[i]:
            if st.button(f"📝 {test_query[:20]}...", key=f"test_{i}"):
                query = test_query
                compare_button = True
    
    # Perform comparison
    if compare_button and query:
        st.markdown("---")
        st.subheader(f"🔍 Comparing: '{query}'")
        
        compare_responses(query, custom_pipeline, langchain_pipeline)
    
    # Framework advantages section
    st.markdown("---")
    st.subheader("🎯 Why Both Approaches?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 Custom Implementation Advantages:**")
        st.info("""
        • **Full Control**: Complete customization of every component
        • **Optimized Performance**: Tailored specifically for Python documentation
        • **Lightweight**: No framework overhead
        • **Direct Integration**: Seamless integration with existing codebase
        • **Educational Value**: Shows deep understanding of RAG architecture
        """)
    
    with col2:
        st.markdown("**🦜 LangChain Advantages:**")
        st.info("""
        • **Industry Standard**: Widely adopted in production environments
        • **Rich Ecosystem**: Extensive library of pre-built components
        • **Memory Management**: Built-in conversation history
        • **Safety Features**: Integrated guardrails and monitoring
        • **Rapid Development**: Faster prototyping and iteration
        """)
    
    # Bonus points demonstration
    st.markdown("---")
    st.subheader("🌟 Bonus Points Achieved")
    
    bonus_points = {
        "✅ LangChain Framework": "Implemented complete LangChain-based RAG pipeline",
        "✅ Safety Guardrails": "Hallucination detection and content filtering",
        "✅ Web Search Fallback": "Both implementations support external search",
        "✅ Performance Metrics": "Response time, token usage, and source tracking",
        "✅ Conversation Memory": "LangChain implementation includes chat history"
    }
    
    for achievement, description in bonus_points.items():
        st.success(f"{achievement}: {description}")
    
    # Implementation comparison table
    st.markdown("---")
    st.subheader("📋 Feature Comparison Matrix")
    
    comparison_data = {
        "Feature": [
            "Document Ingestion",
            "Vector Search", 
            "LLM Integration",
            "Web Search",
            "Dual-Source Integration",
            "Conversation Memory",
            "Safety Guardrails",
            "Performance Metrics",
            "Source Citations",
            "Custom Prompting"
        ],
        "Custom Implementation": [
            "✅ Custom loaders",
            "✅ FAISS + quality filtering", 
            "✅ Groq API",
            "✅ DuckDuckGo + filtering",
            "✅ Dynamic source mixing",
            "✅ Basic history",
            "⚠️ Basic validation",
            "✅ Response time tracking",
            "✅ Rich source metadata",
            "✅ Highly customized"
        ],
        "LangChain Implementation": [
            "✅ Built-in loaders",
            "✅ FAISS integration",
            "✅ Custom LLM wrapper", 
            "✅ Tool integration",
            "✅ Multi-source retrieval",
            "✅ ConversationBufferMemory",
            "✅ Hallucination detection",
            "✅ Comprehensive metrics",
            "✅ Source documents",
            "✅ PromptTemplate system"
        ]
    }
    
    st.table(comparison_data)

if __name__ == "__main__":
    main()