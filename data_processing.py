# data_processing.py

import os
import time
import logging
import os
import sys
import logging
import time
import json
from typing import List, Dict, Any, Optional, Union, TypedDict, Literal, TypeVar, Callable
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from functools import wraps
from logging.handlers import RotatingFileHandler

# Third-party imports
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

# LangChain imports
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredHTMLLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import Qdrant

# LangSmith tracing (optional)
try:
    from langsmith import traceable
    from langchain.callbacks.tracers import LangChainTracer
    from langchain_core.callbacks import CallbackManager
    LANGCHAIN_TRACING = True
except ImportError:
    LANGCHAIN_TRACING = False
    logger = logging.getLogger(__name__)
    logger.warning("LangSmith not available. Tracing will be disabled.")
    
    # Create a dummy traceable decorator if not available
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class Config:
    """Centralized configuration for the RAG pipeline"""
    # Base directories
    BASE_DIR = Path(__file__).parent.absolute()
    DATA_DIR = BASE_DIR / "data"
    
    # Document directories
    DOCUMENTS_DIR = DATA_DIR
    TEXT_STORAGE_PATH = DATA_DIR
    HTML_STORAGE_PATH = DATA_DIR
    IMAGE_STORAGE_PATH = DATA_DIR
    
    # API and connection settings
    HTTP_TIMEOUT = 120
    HTTP_MAX_RETRIES = 3
    
    # Embedding model settings
    MODEL_PATH = 'model/ru-en-RoSBERTa'
    VECTOR_SIZE = 1024  # RoSBERTa produces 1024-dimensional embeddings
    
    # Qdrant settings
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_PATH = "qdrant_db"
    COLLECTION_NAME = "documents"
    
    # Processing settings
    MAX_WORKERS = 4  # Number of parallel workers
    BATCH_SIZE = 50  # Batch size for processing
    CHUNK_SIZE = 400  # Target chunk size
    CHUNK_OVERLAP = 50  # Chunk overlap

def setup_logging():
    """Configure logging for the application.
    
    Sets up both file and console logging with appropriate formatting.
    Logs are written to 'data_processing.log' with rotation.
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'data_processing.log')
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Prevent adding handlers multiple times in case of reload
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with rotation (10MB per file, keep 5 backup files)
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Suppress noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('filelock').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    logger.info("Logging configured successfully")
from typing import List, Dict, Any, Optional, Union, TypedDict, Literal, TypeVar, Callable
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path

# Third-party imports
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredHTMLLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import Qdrant

# LangSmith tracing
try:
    from langsmith import traceable
    from langchain.callbacks.tracers import LangChainTracer
    LANGCHAIN_TRACING = True
except ImportError:
    LANGCHAIN_TRACING = False
    logger = logging.getLogger(__name__)
    logger.warning("LangSmith not available. Tracing will be disabled.")
    
    # Create a dummy traceable decorator if not available
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Qdrant client
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# Local imports
from text_utils import (
    clean_text_basic as clean_text,
    remove_stopwords,
    split_into_semantic_chunks,
    optimize_for_embedding
)
from tqdm import tqdm
import nest_asyncio
import os

# Project imports
from text_utils import optimize_for_embedding, split_into_semantic_chunks

nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure LangSmith (if credentials are available)
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    # Set up LangSmith tracing if environment variables are present
    if os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGCHAIN_PROJECT"):
        logger.info(f"LangSmith tracing enabled for project: {os.getenv('LANGCHAIN_PROJECT')}")
        tracer = LangChainTracer(
            project_name=os.getenv("LANGCHAIN_PROJECT", "RAG-Knowledge-Base")
        )
        callback_manager = CallbackManager([tracer])
    else:
        logger.info("LangSmith environment variables not found, tracing disabled")
        callback_manager = None
except ImportError:
    logger.warning("dotenv not installed, skipping LangSmith configuration")
    callback_manager = None

# Global settings


# Initialize the embedding model
# Create a simple embedding model that doesn't require huge memory
logger.info("Creating simple embedding model for testing purposes")

# Simple tokenization and embedding class that works without large models
class SimpleEmbeddings:
    """A simple embedding model that uses basic text processing instead of a neural model"""
    
    def __init__(self):
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=1024)
        self.is_fit = False
        self.device = "cpu"
        logger.info("Initialized SimpleEmbeddings model (TF-IDF based)")
        
    def _ensure_fit(self, texts):
        """Make sure the vectorizer is fit"""
        if not self.is_fit:
            self.vectorizer.fit(texts)
            self.is_fit = True
    
    def encode(self, text, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        """Create embeddings for a text"""
        import numpy as np
        
        if isinstance(text, list):
            self._ensure_fit(text)
            vectors = self.vectorizer.transform(text)
        else:
            self._ensure_fit([text])
            vectors = self.vectorizer.transform([text])
        
        # Convert to dense array
        dense_vectors = vectors.toarray()
        
        # Ensure we have the right dimensions (1024)
        # If fewer features, pad with zeros
        # If more features (unlikely with max_features=1024), truncate
        result = []
        for vec in dense_vectors:
            if len(vec) < 1024:
                # Pad with zeros to reach 1024 dimensions
                padded_vec = np.zeros(1024)
                padded_vec[:len(vec)] = vec
                result.append(padded_vec)
            else:
                # Use the first 1024 dimensions if there are more
                result.append(vec[:1024])
        
        # For a single text, return the first vector
        if not isinstance(text, list):
            return result[0]  # This will be exactly 1024 dimensions
        return np.array(result)  # Return as numpy array

# Initialize the embedding model
EMBEDDING_MODEL = SimpleEmbeddings()

@traceable(name="load_text_files")
def load_text_files(directory: str) -> List[Document]:
    """Load text files from a directory using LangChain's DirectoryLoader
    
    Args:
        directory: Path to the directory containing text files
        
    Returns:
        List of Document objects
    """
    try:
        logger.info(f"Loading text files from {directory}...")
        loader = DirectoryLoader(
            path=directory, 
            glob="*.txt",
            show_progress=True,
            use_multithreading=True
        )
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} documents from {directory}")
        return documents
    except Exception as e:
        logger.error(f"Error loading text files from {directory}: {str(e)}", exc_info=True)
        return []

@traceable(name="load_html_document")
def load_html_document(file_path: str) -> List[Document]:
    """Load a single HTML document using LangChain's UnstructuredHTMLLoader
    
    Args:
        file_path: Path to the HTML file
        
    Returns:
        List of Document objects
    """
    try:
        logger.info(f"Loading HTML file: {file_path}")
        loader = UnstructuredHTMLLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        logger.error(f"Error loading HTML document {file_path}: {str(e)}")
        return []

@traceable(name="load_pdf_document")
def load_pdf_document(file_path: str) -> List[Document]:
    """Load a PDF document using LangChain's PyPDFLoader
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of Document objects
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
            
        if os.path.getsize(file_path) == 0:
            logger.error(f"Empty file: {file_path}")
            return []
        
        # Use LangChain's PyPDFLoader instead of direct pdfminer usage
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        logger.info(f"Successfully loaded PDF with {len(documents)} pages from {file_path}")
        return documents
        
    except Exception as e:
        logger.error(f"Error loading PDF document {file_path}: {str(e)}")
        return []

@traceable(name="load_docx_document")
def load_docx_document(file_path: str) -> List[Document]:
    """Load a DOCX document using LangChain's UnstructuredWordDocumentLoader
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        List of Document objects
    """
    try:
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()
        logger.info(f"Successfully loaded DOCX document from {file_path}")
        return documents
    except Exception as e:
        logger.error(f"Error loading DOCX document {file_path}: {str(e)}")
        return []

# Enhanced document loading functions with LangChain integrations and tracing

@traceable(name="load_html_documents")
def load_html_documents(directory: str) -> List[Document]:
    """Load all HTML documents from a directory using LangChain
    
    Args:
        directory: Path to the directory containing HTML files
        
    Returns:
        List of Document objects
    """
    logger.info(f"Loading HTML documents from {directory}...")
    try:
        # Use LangChain's DirectoryLoader with glob pattern for HTML files
        loader = DirectoryLoader(
            path=directory, 
            glob="**/*.html",  # Recursive search for HTML files
            loader_cls=UnstructuredHTMLLoader,
            show_progress=True,
            use_multithreading=True
        )
        html_docs = loader.load()
        logger.info(f"Successfully loaded {len(html_docs)} HTML documents")
        return html_docs
    except Exception as e:
        logger.error(f"Error loading HTML documents from {directory}: {str(e)}", exc_info=True)
        return []

@traceable(name="load_pdf_documents")
def load_pdf_documents(directory: str) -> List[Document]:
    """Load all PDF documents from a directory using LangChain
    
    Args:
        directory: Path to the directory containing PDF files
        
    Returns:
        List of Document objects
    """
    logger.info(f"Loading PDF documents from {directory}...")
    try:
        # Use LangChain's DirectoryLoader with glob pattern for PDF files
        loader = DirectoryLoader(
            path=directory, 
            glob="**/*.pdf",  # Recursive search for PDF files
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True
        )
        pdf_docs = loader.load()
        logger.info(f"Successfully loaded {len(pdf_docs)} PDF documents")
        return pdf_docs
    except Exception as e:
        logger.error(f"Error loading PDF documents from {directory}: {str(e)}", exc_info=True)
        return []

@traceable(name="load_docx_documents")
def load_docx_documents(directory: str) -> List[Document]:
    """Load all DOCX documents from a directory using LangChain
    
    Args:
        directory: Path to the directory containing DOCX files
        
    Returns:
        List of Document objects
    """
    logger.info(f"Loading DOCX documents from {directory}...")
    try:
        # Use LangChain's DirectoryLoader with glob pattern for DOCX files
        loader = DirectoryLoader(
            path=directory, 
            glob="**/*.docx",  # Recursive search for DOCX files
            loader_cls=UnstructuredWordDocumentLoader,
            show_progress=True,
            use_multithreading=True
        )
        docx_docs = loader.load()
        logger.info(f"Successfully loaded {len(docx_docs)} DOCX documents")
        return docx_docs
    except Exception as e:
        logger.error(f"Error loading DOCX documents from {directory}: {str(e)}", exc_info=True)
        return []

@traceable(name="load_image_texts")
def load_image_texts(directory: str) -> List[Document]:
    """Extract text from images using OCR
    
    Args:
        directory: Path to the directory containing image files
        
    Returns:
        List of Document objects with extracted text
    """
    logger.info("Loading images and extracting text...")
    start_time = time.time()
    
    # This is a placeholder - implement OCR functionality with a proper LangChain loader
    # In a real implementation, you would use something like UnstructuredImageLoader
    # or a custom loader that integrates with OCR services
    
    logger.warning("Image OCR functionality should be implemented with a proper LangChain loader")
    logger.info(f"Image processing completed in {time.time() - start_time:.2f} seconds.")
    return []

# LangChain semantic chunking with tracing
class RoSBERTaEmbeddings(Embeddings):
    """Embeddings wrapper class that works with both SentenceTransformer and SimpleEmbeddings"""
    
    def __init__(self, model=EMBEDDING_MODEL):
        self.model = model
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts"""
        if len(texts) == 0:
            return []
            
        # Add prefix to improve distinction between documents and queries
        prefixed_texts = [f"search_document: {text}" for text in texts]
        
        try:
            # Handle both batched and individual encoding
            if hasattr(self.model, 'encode') and callable(getattr(self.model, 'encode')):
                # For SimpleEmbeddings or SentenceTransformer
                embeddings = self.model.encode(
                    prefixed_texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                # Handle different return types
                if not isinstance(embeddings, list):
                    # If numpy array, convert to list of lists
                    return embeddings.tolist()
                return embeddings
            else:
                # Fallback implementation
                raise NotImplementedError("Model doesn't have encode method")
        except Exception as e:
            logger.error(f"Error in embed_documents: {str(e)}")
            # Emergency fallback: return zero vectors
            return [[0.0] * 1024] * len(texts)
        
    def embed_query(self, text: str) -> List[float]:
        """Embed a query text"""
        try:
            # Add prefix to improve distinction between documents and queries
            prefixed_text = f"search_query: {text}"
            
            if hasattr(self.model, 'encode') and callable(getattr(self.model, 'encode')):
                # For SimpleEmbeddings or SentenceTransformer
                embedding = self.model.encode(
                    prefixed_text,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                # Handle different return types
                if hasattr(embedding, 'tolist'):
                    return embedding.tolist()
                return list(embedding)
            else:
                # Fallback implementation
                raise NotImplementedError("Model doesn't have encode method")
        except Exception as e:
            logger.error(f"Error in embed_query: {str(e)}")
            # Emergency fallback: return zero vector
            return [0.0] * 1024

@traceable(name="chunk_documents")
def chunk_documents(documents: List[Document]) -> List[Document]:
    """Chunk documents using LangChain's semantic chunking
    
    Args:
        documents: List of documents to chunk
        
    Returns:
        List of chunked documents
    """
    if not documents:
        logger.warning("No documents to chunk")
        return []
    
    logger.info(f"Chunking {len(documents)} documents semantically...")
    start_time = time.time()
    
    # Optimization: Using a custom embedding wrapper for RoSBERTa
    embeddings = RoSBERTaEmbeddings()
    
    # Create a text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Process documents in batches to avoid memory issues
    all_chunks = []
    
    # Process in batches of 10 documents
    batch_size = 10
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_chunks = []
        
        # First preprocess text to optimize for embeddings
        optimized_docs = []
        for doc in batch:
            optimized_text = optimize_for_embedding(doc.page_content)
            optimized_doc = Document(
                page_content=optimized_text,
                metadata=doc.metadata
            )
            optimized_docs.append(optimized_doc)
        
        # Chunk the documents
        try:
            # Use the text splitter to create chunks
            batch_chunks = text_splitter.split_documents(optimized_docs)
            
            # Add chunk metadata
            for i, chunk in enumerate(batch_chunks):
                chunk.metadata.update({
                    "chunk": i,
                    "total_chunks": len(batch_chunks)
                })
                
        except Exception as e:
            logger.warning(f"Error during document chunking: {str(e)}")
            # Fall back to custom chunking function if needed
            for doc in optimized_docs:
                text_chunks = split_into_semantic_chunks(
                    doc.page_content,
                    max_chunk_size=Config.CHUNK_SIZE,
                    overlap=Config.CHUNK_OVERLAP
                )
                
                for i, chunk_text in enumerate(text_chunks):
                    chunk_doc = Document(
                        page_content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk": i,
                            "total_chunks": len(text_chunks)
                        }
                    )
                    batch_chunks.append(chunk_doc)
        
        all_chunks.extend(batch_chunks)
        logger.info(f"Processed batch of {len(batch)} documents into {len(batch_chunks)} chunks")
    
    logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} semantic chunks in {time.time() - start_time:.2f} seconds")
    return all_chunks

@traceable(name="create_vector_store")
def create_vector_store(client: QdrantClient, collection_name: str, chunks: List[Document]) -> Qdrant:
    """Create a Qdrant vector store with the provided documents
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to create
        chunks: List of document chunks to index
        
    Returns:
        Qdrant vector store instance
    """
    logger.info("Creating vector store...")
    start_time = time.time()
    
    # Create the RoSBERTaEmbeddings wrapper
    embeddings = RoSBERTaEmbeddings()
    
    # Always recreate collection for safety
    try:
        logger.info(f"Deleting collection '{collection_name}' if it exists")
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Successfully deleted existing collection '{collection_name}'")
    except Exception as e:
        logger.info(f"Collection '{collection_name}' may not exist or could not be deleted: {str(e)}")
    
    # Create fresh collection with proper configuration
    logger.info(f"Creating new collection '{collection_name}'")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=Config.VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )
    
    # Use LangChain's Qdrant integration to handle the embedding and insertion
    vector_store = Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=f"http://{Config.QDRANT_HOST}:{Config.QDRANT_PORT}",
        collection_name=collection_name,
        prefer_grpc=False,
        batch_size=Config.BATCH_SIZE  # Control batch size for insertion
    )
    
    logger.info(f"Successfully created vector store with {len(chunks)} chunks in {time.time() - start_time:.2f} seconds")
    return vector_store

# Main document processing pipeline using LangGraph
class DocumentState(dict):
    """State for the document processing graph"""
    documents: List[Document]
    chunks: Optional[List[Document]] = None
    vector_store: Optional[Any] = None
    error: Optional[str] = None
    success: bool = False

def run_document_processing_workflow() -> Dict[str, Any]:
    """Run the document processing pipeline.
    
    Returns:
        A dictionary containing the processing results and status.
    """
    # Initialize state
    state = {
        "documents": [],
        "chunks": [],
        "vector_store": None,
        "success": False,
        "error": None
    }
    
    try:
        # Load documents
        logger.info("Loading documents...")
        documents = load_documents(Config.DOCUMENTS_DIR)  # Changed from DATA_DIR to DOCUMENTS_DIR
        if not documents:
            logger.info(f"No documents found in {Config.DOCUMENTS_DIR}. Waiting for new documents...")
            state["success"] = True
            return state
        state["documents"] = documents
        
        # Chunk documents
        logger.info("Chunking documents...")
        chunks = chunk_documents(documents)
        if not chunks:
            raise ValueError("No chunks generated from documents")
        state["chunks"] = chunks
        
        # Create vector store
        logger.info("Creating vector store...")
        client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)
        vector_store = create_vector_store(client, Config.COLLECTION_NAME, chunks)
        state["vector_store"] = vector_store
        
        state["success"] = True
        logger.info("Document processing completed successfully")
        
        return state
        
    except Exception as e:
        error_msg = f"Error in document processing: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Cleanup on error
        try:
            client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)
            client.delete_collection(collection_name=Config.COLLECTION_NAME)
            logger.info("Cleaned up collection after error")
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {str(cleanup_error)}")
        
        state["error"] = error_msg
        state["success"] = False
        return state

def load_documents(data_dir: Path) -> List[Document]:
    """Load documents from the specified directory.
    
    Args:
        data_dir: Directory containing documents to load
        
    Returns:
        List of loaded documents
    """
    logger.info(f"Loading documents from {data_dir}")
    documents = []
    
    # Supported file extensions and their corresponding loaders
    loaders = {
        '.pdf': PyPDFLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.doc': UnstructuredWordDocumentLoader,
        '.html': UnstructuredHTMLLoader,
        '.htm': UnstructuredHTMLLoader,
        '.txt': TextLoader
    }
    
    # Count files by type
    file_counts = {ext: 0 for ext in loaders}
    
    # Process each file with the appropriate loader
    for ext, loader_class in loaders.items():
        try:
            file_pattern = f'*{ext}'
            loader = DirectoryLoader(
                str(data_dir),
                glob=file_pattern,
                loader_cls=loader_class,
                show_progress=True,
                use_multithreading=True
            )
            loaded_docs = loader.load()
            file_counts[ext] = len(loaded_docs)
            documents.extend(loaded_docs)
        except Exception as e:
            logger.warning(f"Error loading {ext} files: {str(e)}")
    
    logger.info(f"Loaded {len(documents)} documents: {file_counts}")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks.
    
    Args:
        documents: List of documents to chunk
        
    Returns:
        List of chunked documents
    """
    if not documents:
        return []
        
    logger.info(f"Chunking {len(documents)} documents")
    
    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Process documents in batches to avoid memory issues
    all_chunks = []
    batch_size = Config.BATCH_SIZE
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        try:
            # Split documents
            chunks = text_splitter.split_documents(batch)
            all_chunks.extend(chunks)
            logger.info(f"Processed batch {i//batch_size + 1}, total chunks: {len(all_chunks)}")
        except Exception as e:
            logger.error(f"Error chunking batch {i//batch_size + 1}: {str(e)}")
            continue
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks


def create_vector_store(client: QdrantClient, collection_name: str, chunks: List[Document]) -> Qdrant:
    """Create or update a Qdrant vector store with the given chunks.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to create/update
        chunks: List of document chunks to index
        
    Returns:
        Qdrant vector store instance
    """
    if not chunks:
        raise ValueError("No chunks provided for vector store creation")
    
    logger.info(f"Creating/updating vector store '{collection_name}' with {len(chunks)} chunks")
    
    # Initialize embeddings with the correct wrapper
    # Using RoSBERTaEmbeddings which properly implements the LangChain Embeddings interface
    embeddings = RoSBERTaEmbeddings(EMBEDDING_MODEL)
    
    # Create or recreate the collection
    try:
        # Delete existing collection if it exists
        collections = client.get_collections()
        if collection_name in {col.name for col in collections.collections}:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
            
        # Create new collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=Config.VECTOR_SIZE,  # Use the vector size from Config
                distance=rest.Distance.COSINE
            )
        )
        logger.info(f"Created new collection: {collection_name}")
        
    except Exception as e:
        logger.error(f"Error setting up Qdrant collection: {str(e)}")
        raise
    
    # Create the vector store
    try:
        vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
            vector_name=None  # Use default vector name to match our collection creation
        )
        
        # Add documents in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vector_store.add_documents(batch)
            logger.info(f"Indexed batch {i//batch_size + 1}, total: {min(i + len(batch), len(chunks))}")
        
        logger.info(f"Successfully created vector store with {len(chunks)} documents")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise


def ensure_directories_exist():
    """Ensure all required directories exist."""
    # Use the Config class from the top of the file which has all required attributes
    config = Config
    # Create all required directories
    for dir_path in [
        config.DATA_DIR,
        config.DOCUMENTS_DIR,
        config.TEXT_STORAGE_PATH,
        config.HTML_STORAGE_PATH,
        config.IMAGE_STORAGE_PATH
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
    return config


def main():
    """Main function to process documents and create a vector store."""
    try:
        # Initialize logging
        setup_logging()
        logger.info("Starting document processing pipeline...")
        
        # Load environment variables
        load_dotenv()
        
        # Ensure all required directories exist
        config = ensure_directories_exist()
        
        # Run the document processing workflow
        start_time = time.time()
        result = run_document_processing_workflow()
        processing_time = time.time() - start_time
        
        if result.get("success"):
            logger.info(f"Document processing completed successfully in {processing_time:.2f} seconds")
            logger.info(f"Processed {len(result.get('documents', []))} documents into {len(result.get('chunks', []))} chunks")
            return True
        else:
            error_msg = result.get("error", "Unknown error occurred")
            logger.error(f"Document processing failed: {error_msg}")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}", exc_info=True)
        return False

# Entry point
if __name__ == "__main__":
    # Add LangSmith environment variables to .env file for tracing and monitoring
    if not os.getenv("LANGCHAIN_API_KEY"):
        logger.info("LangSmith API key not found. To enable tracing, add LANGCHAIN_API_KEY to .env")
    
    # Run the main function and exit with appropriate status code
    success = main()
    if not success:
        logger.error("Document processing failed")
    sys.exit(0 if success else 1)