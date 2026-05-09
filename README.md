# EcoGuide Chatbot - Deloitte Tech Experience Bootcamp Challenge

A RAG-based (Retrieval-Augmented Generation) chatbot that answers questions about climate change and recommends sustainable products.

## 📋 Project Overview

**Goal**: Develop an AI-powered chatbot (EcoGuide) that provides climate change information and sustainable product recommendations.

### Architecture

- **RAG Pattern**: Retrieval-Augmented Generation for accurate, context-based answers
- **Vector Database**: FAISS for semantic search and document retrieval
- **LLM**: Azure OpenAI GPT-4o for response generation
- **Embeddings**: Azure OpenAI text-embedding-3-large (3072 dimensions)
- **Document Processing**: Support for PDF, HTML, DOCX, PPTX files
- **Chunking**: Token-based chunking with llama-index alternatives documented

## 🚀 Quick Start

### Prerequisites

- Python 3.12.4 (required for match-case statements and modern type hints)
- Azure OpenAI account with API access and deployed models
- Basic understanding of RAG architecture and vector databases

### Installation

1. **Extract the project files** to your local machine
2. **Create and activate virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # OR
   .venv\Scripts\activate  # Windows
   ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   This installs:

   - Core packages: faiss-cpu, openai, llama-index, python-dotenv
   - Document loaders: PyPDF2, python-docx, python-pptx, html2text
   - Optional: gradio (for web interface)
4. **Configure Azure OpenAI credentials** - The `.env` file is already included:

   ```env
   # Azure OpenAI LLM Configuration (GPT-4o)
   AZURE_LLM_ENDPOINT="https://your-resource.openai.azure.com/"
   AZURE_LLM_API_KEY="your-api-key"
   AZURE_LLM_DEPLOYMENT_NAME="gpt-4o"
   AZURE_LLM_MODEL_NAME="gpt-4o"
   AZURE_LLM_API_VERSION="2024-12-01-preview"

   # Azure OpenAI Embeddings Configuration (text-embedding-3-large)
   AZURE_EMBEDDINGS_ENDPOINT="https://your-resource.openai.azure.com/"
   AZURE_EMBEDDINGS_API_KEY="your-api-key"
   AZURE_EMBEDDINGS_DEPLOYMENT_NAME="text-embedding-3-large"
   AZURE_EMBEDDINGS_MODEL_NAME="text-embedding-3-large"
   AZURE_EMBEDDINGS_API_VERSION="2024-02-01"
   ```

   **Note**: The provided `.env` file has working credentials pre-configured.
5. **Run the chatbot**:

   ```bash
   python main.py
   ```

   On first run, the application will:

   - Process all documents in the `data/` folder
   - Generate embeddings for each text chunk
   - Build and save the FAISS index (takes 2-3 minutes)
   - Start the interactive chat interface

   Subsequent runs load the saved index instantly.

## ✅ What's Working

### Core Features (Implemented)

- ✅ **RAG Architecture**: Complete retrieval-augmented generation pipeline
- ✅ **Azure OpenAI Integration**: GPT-4o LLM + text-embedding-3-large embeddings
- ✅ **Document Loaders**: PDF, HTML, DOCX, PPTX support with robust text extraction
- ✅ **FAISS Vector Database**: Semantic search with L2 distance similarity
- ✅ **Token-based Chunking**: Intelligent text splitting with overlap
- ✅ **Conversation History**: Multi-turn conversations with context preservation
- ✅ **Terminal Interface**: Functional CLI for testing and development
- ✅ **Index Persistence**: Save/load FAISS index to avoid re-processing
- ✅ **Comprehensive Documentation**: All files thoroughly commented with examples

### Data Sources (Included)

- 📄 GHG Protocol documents (PDF) - Corporate carbon accounting standards
- 📄 Climate Action Tracker reports (PDF) - Paris Agreement benchmarks
- 📄 Climate change overview (DOCX) - General information
- 🌐 Greenhouse gas protocol FAQ (HTML) - Common questions
- 📊 Sustainable products database (CSV) - 50+ eco-friendly products

## 🚧 TODO: Missing Features for Candidates

The following features are **intentionally incomplete** and should be implemented as part of the challenge:

### Priority 1: Essential Features

1. **❌ Gradio Web Interface** (`gradio_app.py`)

   - **Current State**: Template file with detailed TODOs
   - **What to do**: Implement complete web UI to replace terminal interface
   - **Requirements**:
     - Chat interface with message history
     - Source citation display for each response
     - File upload for new documents
     - Conversation export functionality
     - Settings panel for RAG parameters
   - **Resources**: `gradio_app.py` has complete implementation guide
   - **Estimated Time**: 2-3 hours
2. **❌ Source Citation** (Multiple files)

   - **Current State**: Chunks retrieved but sources not tracked
   - **What to do**: Track and display which documents were used
   - **Files to modify**:
     - `src/services/vectorial_db/faiss_index.py` - Add metadata storage
     - `main.py` - Format and display sources
     - `gradio_app.py` - Show sources in UI
   - **Requirements**:
     - Store document name, page number with each chunk
     - Return metadata alongside retrieved chunks
     - Format as citations (e.g., "Source: climate_report.pdf, page 5")
   - **Estimated Time**: 1-2 hours
3. **❌ CSV Product Recommendations** (New file + integration)

   - **Current State**: `sustainable_products.csv` exists but not integrated
   - **What to do**: Load CSV and recommend products based on queries
   - **Requirements**:
     - Create `src/ingestion/loaders/loaderCSV.py`
     - Parse `sustainable_products.csv` structure
     - Implement product search/recommendation logic
     - Format product results as cards in responses
     - Detect when user asks for product recommendations
   - **Estimated Time**: 2-3 hours

### Priority 2: Enhancements

4. **⚠️ Improved System Prompt** (`src/services/models/llm.py`)

   - **Current State**: Basic system prompt
   - **What to do**: Create comprehensive climate expert persona
   - **Suggestions**:
     - Define AI as "EcoGuide, a climate change expert"
     - Add instructions to cite sources
     - Include tone guidelines (informative, empathetic, actionable)
     - Add rules for product recommendations
   - **Estimated Time**: 30 minutes
5. **⚠️ Advanced Chunking Strategies** (`src/ingestion/chunking/`)

   - **Current State**: Token-based chunking works, alternatives documented
   - **What to do**: Implement better chunking for improved retrieval
   - **Options** (choose one or more):
     - **SentenceSplitter**: Respects sentence boundaries (recommended)
     - **SemanticSplitterNodeParser**: Groups by topic similarity
     - **SentenceWindowNodeParser**: Sliding window with metadata
     - **HierarchicalNodeParser**: Multi-level document structure
   - **Resources**:
     - `src/ingestion/chunking/chunking_base.py` - Detailed examples
     - `src/ingestion/chunking/token_chunking.py` - Implementation guide
   - **Estimated Time**: 1-2 hours per strategy
6. **❌ Multimodal RAG** (Advanced challenge)

   - **Current State**: Not implemented
   - **What to do**: Extract information from images/charts in PDFs
   - **Approach**:
     - Use GPT-4o vision to describe charts/graphs
     - Convert visual information to text chunks
     - Add to FAISS index alongside text
   - **Estimated Time**: 3-4 hours

## 📚 Code Structure & Documentation

All files include comprehensive documentation:

```
gen-ai-business-case-recruiting/
├── main.py                          # Main application entry point (180 lines)
├── gradio_app.py                    # Web interface template with TODOs (547 lines)
├── requirements.txt                 # Python dependencies with version pins
├── .env                             # Azure OpenAI credentials (pre-configured)
├── README.md                        # This file
│
├── data/                            # Source documents (DO NOT MODIFY)
│   ├── ghg-protocol-revised*.pdf    # GHG Protocol standards
│   ├── climate_change_overview_en.docx
│   ├── Greenhouse_ga_protocol*.html
│   ├── CAT_2024-10-29*.pdf
│   └── sustainable_products.csv     # Products for recommendations
│
├── src/
│   ├── services/
│   │   ├── models/
│   │   │   ├── llm.py              # GPT-4o integration (145 lines)
│   │   │   └── embeddings.py       # Embedding generation (145 lines)
│   │   └── vectorial_db/
│   │       └── faiss_index.py      # Vector database manager (337 lines)
│   │
│   └── ingestion/
│       ├── ingest_files.py         # Document processing pipeline
│       ├── loaders/
│       │   ├── loader.py           # Loader factory pattern
│       │   ├── loaderPDF.py        # PDF text extraction
│       │   ├── loaderHTML.py       # HTML to markdown
│       │   ├── loaderDOCX.py       # Word document loader
│       │   ├── loaderPPTX.py       # PowerPoint loader
│       │   └── loaderCSV.py        # ❌ TODO: Implement CSV loader
       │
       └── chunking/
           ├── chunking_base.py    # Abstract base class (191 lines)
           └── token_chunking.py   # Token-based implementation (462 lines)
```

### Documentation Highlights

- **Module-level docstrings**: Architecture overview, purpose, TODO lists
- **Function/method docstrings**: Parameters, returns, examples, implementation hints
- **Inline comments**: Explain logic, highlight important details
- **TODO markers**: Clear instructions for missing features
- **Code examples**: Copy-paste ready snippets for common tasks

## 🔍 Key Technical Details

### Embedding Configuration

- **Model**: text-embedding-3-large
- **Dimensions**: 3072 (CRITICAL: must match FAISS index)
- **API Version**: 2024-02-01
- **Use case**: Convert text to semantic vectors for similarity search

### LLM Configuration

- **Model**: GPT-4o
- **API Version**: 2024-12-01-preview
- **Temperature**: 0.7 (configurable in `llm.py`)
- **Max tokens**: 1000 (configurable)
- **Use case**: Generate responses based on retrieved context

### FAISS Index

- **Type**: IndexFlatL2 (exact L2 distance search)
- **Distance metric**: Euclidean (L2)
- **Persistence**: Binary format (index.faiss) + pickle (chunks.pkl)
- **Performance**: Fast for <100K vectors, scales well for this use case

### Chunking Strategy

- **Current**: Token-based with overlap
- **Chunk size**: ~500 tokens
- **Overlap**: 50 tokens (configurable)
  - **Alternatives**: Sentence, semantic, window-based (see `chunking_base.py`)

## Implementation Tips For Gradio Interface

1. Start with `gr.ChatInterface` for quick MVP
2. Use `gr.Blocks` for advanced features
3. Initialize RAG components globally (not per request)
4. Handle errors gracefully with try-except
5. Test locally before deploying

### For Source Citation

1. Modify `FAISSIndex.ingest_text()` to store metadata
2. Use a parallel list or dict: `{index: {"text": "...", "source": "...", "page": N}}`
3. Return metadata in `retrieve_chunks()`
4. Format citations in chatbot response

### For Product Recommendations

1. Load CSV with pandas: `df = pd.read_csv('data/sustainable_products.csv')`
2. Detect product queries with keywords or LLM classification
3. Filter products by category, sustainability score
4. Format as markdown table or HTML cards
5. Append to chatbot response

### For Advanced Chunking

1. Review examples in `chunking_base.py`
2. Choose strategy based on document type (semantic for long articles, sentence for mixed content)
3. Update `ingest_files.py` to use new chunking
4. Re-build FAISS index after changing chunking strategy
5. Compare retrieval quality before/after

## 💡 Resources

### Common Issues

**"ModuleNotFoundError"**: Install dependencies with `pip install -r requirements.txt`

**"AssertionError" in FAISS**: Embedding dimension mismatch. Ensure `FAISSIndex(dimension=3072)`

**"FileNotFoundError: Index not found"**: Normal on first run. Index will be created automatically.

**"API Error 401"**: Check `.env` file has correct API keys

**"API Error 404"**: Verify deployment names match Azure OpenAI deployments

**Empty/poor responses**: Check if documents were ingested (look for `faiss_index/` folder)

**Import errors in gradio_app.py**: Gradio is optional. Install with `pip install gradio`

## 📚 Resources

### Documentation

- **FAISS**: https://github.com/facebookresearch/faiss/wiki
- **llama-index**: https://docs.llamaindex.ai/
- **Azure OpenAI**: https://learn.microsoft.com/en-us/azure/ai-services/openai/
- **Gradio**: https://www.gradio.app/docs

### Chunking Strategies

- `src/ingestion/chunking/chunking_base.py` - Detailed strategy comparison
- llama-index docs: https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/

---

**Good luck with the challenge! 🚀**
