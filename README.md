# RAG (Retrieval-Augmented Generation) Knowledge Base

This project implements a Retrieval-Augmented Generation (RAG) system that allows you to process documents, store them in a vector database, and query them using natural language. The system uses Qdrant for vector storage, Hugging Face for embeddings, and Ollama's Mistral model for generating responses.

## Features

- Document processing for various file formats (PDF, DOCX, HTML, plain text, images with OCR)
- Vector embeddings using Hugging Face models
- Vector storage and retrieval with Qdrant
- Chat interface with Streamlit
- Local LLM support via Ollama

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running with Mistral model
- [Qdrant](https://qdrant.tech/) vector database
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (for image processing)
- [Poetry](https://python-poetry.org/) (recommended) or pip

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd rag
   ```

2. Copy the environment file and set up your Hugging Face token:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file and add your Hugging Face API token.

3. Install dependencies:
   Using Poetry:
   ```bash
   poetry install
   ```
   
   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR:
   - **Windows**: Download and install from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr` (Ubuntu/Debian)

5. Download and run Qdrant (using Docker):
   ```bash
   docker pull qdrant/qdrant
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

6. Download the Mistral model with Ollama:
   ```bash
   ollama pull mistral
   ```

## Project Structure

- `data/` - Directory for storing documents to be processed
- `data_processing.py` - Script for processing and indexing documents
- `chat_app.py` - Streamlit-based chat interface
- `requirements.txt` - Python dependencies
- `.env` - Environment variables

## Usage

### 1. Prepare Your Documents

Place the documents you want to process in the `data/` directory. Supported formats include:
- Text files (.txt)
- PDF documents (.pdf)
- Word documents (.docx)
- HTML files (.html, .htm)
- Images with text (.jpg, .png, etc.)

### 2. Process Documents

Run the document processing script to create vector embeddings and store them in Qdrant:

```bash
python data_processing.py
```

This will:
1. Load documents from the `data/` directory
2. Split them into chunks
3. Generate embeddings
4. Store them in the Qdrant database

### 3. Start the Chat Interface

Make sure Ollama is running in the background:

```bash
ollama serve
```

In a new terminal, start the Streamlit application:

```bash
streamlit run chat_app.py
```

This will open a web browser with the chat interface where you can ask questions about your documents.

## Configuration

You can modify the following settings in the code:

- Embedding model: Edit `data_processing.py` to change the Hugging Face model
- Chunk size and overlap: Adjust in `data_processing.py`
- Qdrant connection: Modify the connection parameters in both `data_processing.py` and `chat_app.py`

## Troubleshooting

### Common Issues

1. **Ollama connection issues**:
   - Ensure Ollama is running (`ollama serve`)
   - Check if the model is downloaded (`ollama list`)
   - Verify the API is accessible at `http://localhost:11434`

2. **Qdrant connection issues**:
   - Make sure Qdrant is running (`docker ps` should show the container)
   - Check if ports 6333 and 6334 are available

3. **OCR issues**:
   - Verify Tesseract is installed and in your PATH
   - For non-English text, you might need to install additional language packs

### Memory Management

For large document collections, you might need to increase the available memory for Python and Ollama.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [LangChain](https://python.langchain.com/)
- [Qdrant](https://qdrant.tech/)
- [Hugging Face](https://huggingface.co/)
- [Ollama](https://ollama.ai/)
- [Streamlit](https://streamlit.io/)

---

For any questions or issues, please open an issue in the repository.
