# 🐍 PythonDocBot

A Retrieval-Augmented Generation (RAG) chatbot specialized in Python programming documentation. Built with FAISS vector search, Groq LLM API, and Streamlit UI.

## 🚀 Features

- **Document Ingestion**: Loads PDF and TXT files from local directories
- **Vector Search**: Uses FAISS for efficient similarity search
- **Web Search Fallback**: Automatically searches web when internal docs are insufficient
- **Chat Interface**: Clean Streamlit UI with conversation history
- **Source Citations**: Shows relevant sources for each answer
- **Export Functionality**: Download conversation history as JSON

## 📋 Requirements

- Python 3.8+
- Groq API key (free at console.groq.com)
- 2GB+ RAM for embedding models

## 🛠️ Installation

### 1. Clone and Setup
```bash
git clone <your-repo>
cd pythondocbot
python setup.py
```

### 2. Configure API Key
Edit `.env` file and add your Groq API key:
```
GROQ_API_KEY=your_actual_api_key_here
```

### 3. Add Documentation
Place your Python documentation files in:
- `data_source/pdf/` - PDF files
- `data_source/txt/` - Text files

Or create sample data:
```bash
python create_sample_data.py
```

### 4. Run the Application
```bash
streamlit run ui/streamlit_app.py
```

## 📁 Project Structure

```
pythondocbot/
├── README.md
├── requirements.txt
├── .env
├── src/
│   ├── config/settings.py      # Configuration management
│   ├── core/
│   │   ├── document_ingestion.py  # Document loading and chunking
│   │   ├── embedding_service.py   # Text embeddings
│   │   ├── vector_store.py        # FAISS vector database
│   │   └── web_search.py          # Web search fallback
│   ├── rag/rag_pipeline.py     # Main RAG orchestration
│   └── utils/logger.py         # Logging utilities
├── ui/streamlit_app.py         # Web interface
├── data_source/                # Your documentation files
├── vector_db/                  # FAISS index storage
└── logs/                       # Application logs
```

## 🔧 Configuration

Key settings in `.env`:

- `GROQ_API_KEY`: Your Groq API key
- `EMBEDDING_MODEL`: Sentence transformer model (default: all-MiniLM-L6-v2)
- `LLM_MODEL`: Groq model (default: mixtral-8x7b-32768)
- `CHUNK_SIZE`: Document chunk size (default: 1000)
- `TOP_K_RESULTS`: Number of similar chunks to retrieve (default: 5)
- `RELEVANCE_THRESHOLD`: Minimum similarity score for using internal docs (default: 0.6)

## 💡 Usage

1. **Ask Questions**: Type Python programming questions in the chat
2. **View Sources**: Expand the "Sources" section to see referenced documents
3. **Export Chat**: Use sidebar to download conversation history
4. **Clear History**: Reset conversation using sidebar button

### Example Queries:
- "How do I create a list in Python?"
- "Explain Python decorators with examples"
- "What's the difference between lists and tuples?"
- "How to handle exceptions in Python?"

## 🔍 How It Works

1. **Document Processing**: PDFs and TXT files are loaded and split into chunks
2. **Vectorization**: Text chunks are converted to embeddings using sentence-transformers
3. **Storage**: Embeddings stored in FAISS index for fast similarity search
4. **Query Processing**: User questions are embedded and matched against document chunks
5. **Fallback Search**: If no relevant internal docs found, searches web for additional context
6. **Response Generation**: Groq LLM generates answers using retrieved context
7. **Source Citation**: Relevant sources displayed with confidence scores

## 🎯 System Architecture

```
User Query → Embedding → Vector Search → Context Retrieval
                ↓
Web Search (if needed) → LLM Generation → Response + Sources
                ↓
Conversation History → Streamlit UI
```

## 🚨 Troubleshooting

### Common Issues:

1. **"No documents found"**
   - Ensure files are in `data_source/pdf/` or `data_source/txt/`
   - Run `python create_sample_data.py` for test data

2. **"API key error"**
   - Verify GROQ_API_KEY in `.env` file
   - Get free API key from console.groq.com

3. **Slow responses**
   - Reduce CHUNK_SIZE in `.env`
   - Use smaller embedding model
   - Reduce TOP_K_RESULTS

4. **Memory issues**
   - Use sentence-transformers/all-MiniLM-L6-v2 (smaller model)
   - Reduce document count
   - Increase CHUNK_SIZE to reduce total chunks

## 📊 Performance

- **Embedding**: ~1-2 seconds for query embedding
- **Search**: <100ms for FAISS similarity search
- **Generation**: 2-5 seconds for LLM response
- **Memory**: ~500MB base + ~50MB per 1000 document chunks

## 🔮 Future Enhancements

- [ ] Multi-language documentation support
- [ ] Advanced chunking strategies (semantic, hierarchical)
- [ ] Chat memory and context persistence
- [ ] Document upload via UI
- [ ] Hybrid search (vector + keyword)
- [ ] Response quality scoring
- [ ] Docker containerization
- [ ] Authentication and user management

## 📄 License

MIT License - feel free to modify and distribute!

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Submit pull request

## 📞 Support

For issues and questions:
- Check logs in `logs/pythondocbot.log`
- Review configuration in `.env`
- Ensure all requirements are installed

---

**Built with ❤️ using Python, FAISS, Groq, and Streamlit**