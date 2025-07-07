# RAG (Retrieval-Augmented Generation) Knowledge Base

This project implements a Retrieval-Augmented Generation (RAG) system that allows you to process documents, store them in a vector database, and query them using natural language. The system uses Qdrant for vector storage, Hugging Face's RoSBERTa for embeddings, and a local GGUF model (T-lite-it-1.0) for generating responses. It also features LangGraph workflows for structured RAG pipelines and LangSmith tracing for monitoring and debugging.

## Features

- Document processing for various file formats (PDF, DOCX, HTML, plain text)
- Vector embeddings using Hugging Face's RoSBERTa model
- Vector storage and retrieval with Qdrant
- Interactive chat interface with Gradio
- Local LLM inference with llama-cpp-python
- Hybrid search combining semantic and keyword-based retrieval
- LangGraph integration for structured RAG workflows
- LangSmith tracing for monitoring and debugging performance
- Support for multiple document formats in vector storage (flexible field mapping)

## Prerequisites

- Python 3.8+
- [Qdrant](https://qdrant.tech/) vector database (running on localhost:6333)
- [GGUF Model](https://huggingface.co/models?search=gguf) (T-lite-it-1.0-Q4_K_M-GGUF included)
- [Poetry](https://python-poetry.org/) (recommended) or pip
- LangChain and LangGraph (included in requirements.txt)
- LangSmith API key (optional, for tracing and monitoring)

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd rag
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   # For Ubuntu/Debian:
   sudo apt-get update
   sudo apt-get install -y build-essential cmake python3-dev python3-venv
   
   # For CentOS/RHEL:
   sudo yum groupinstall -y "Development Tools"
   sudo yum install -y cmake python3-devel
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```

4. Download and run Qdrant (using Docker):
   ```bash
   docker pull qdrant/qdrant
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

5. Download the GGUF model (if not included):
   - The project includes a pre-configured model in `model/T-lite-it-1.0-Q4_K_M-GGUF/`
   - To use a different model, update the path in `gradio_app.py`

## Project Structure

- `data/` - Directory for storing documents to be processed
- `model/` - Contains the GGUF model files
- `data_processing.py` - Script for processing and indexing documents
- `rag_app.py` - Main RAG application with LangGraph workflows and Gradio interface
- `prompts.py` - System prompts and templates
- `requirements.txt` - Python dependencies

## Usage

### 1. Prepare Your Documents

Place the documents you want to process in the `data/` directory. Supported formats include:
- Text files (.txt)
- PDF documents (.pdf)
- Word documents (.docx)
- HTML files (.html, .htm)

### 2. Process Documents

Run the document processing script to create vector embeddings and store them in Qdrant:

```bash
python data_processing.py
```

This will:
1. Load documents from the `data/` directory
2. Split them into chunks
3. Generate embeddings using RoSBERTa
4. Store them in the Qdrant database

### 3. Start the Chat Interface

Start the Gradio chat interface:

```bash
python rag_app.py
```

This will start a local web server (usually at http://localhost:7860) with the chat interface where you can ask questions about your documents.

## Configuration

You can customize the following aspects of the system:

- **Model Parameters**: Adjust `n_ctx`, `n_threads`, and other parameters in the `LangChainAssistant` class
- **Search Settings**: Modify the hybrid search parameters in `HybridSearch` class
- **UI Settings**: Customize the Gradio interface in `create_demo()`
- **LangGraph Workflow**: Modify the RAG workflow in `RAGAssistant._create_rag_graph()`
- **LangSmith Tracing**: Configure with environment variables `LANGCHAIN_API_KEY` and `LANGCHAIN_PROJECT`
- **Chunk size and overlap**: Adjust in `data_processing.py`
- **Qdrant connection**: Modify the connection parameters in `Config` class

## Troubleshooting

### Common Issues

1. **Qdrant connection issues**:
   - Make sure Qdrant is running (`docker ps` should show the container)
   - Check if ports 6333 and 6334 are available

2. **OCR issues**:
   - Verify Tesseract is installed and in your PATH
   - For non-English text, you might need to install additional language packs

3. **Vector dimension mismatch**:
   - If you see "Vector dimension error" from Qdrant, check the embedding dimensions in `SimpleEmbeddings`
   - The system now automatically pads vectors to the expected 1024 dimensions

4. **LangGraph and LangSmith integration**:
   - Do not use `@trace` decorators on functions passed to LangGraph's `add_node`
   - Instead, use `with trace(name="function_name"):` inside functions
   - This prevents the "trace object not callable" error

### Memory Management

For large document collections, you might need to increase the available memory for Python and Ollama.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [LangChain](https://python.langchain.com/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangSmith](https://smith.langchain.com/)
- [Qdrant](https://qdrant.tech/)
- [Hugging Face](https://huggingface.co/)

---

For any questions or issues, please open an issue in the repository.
