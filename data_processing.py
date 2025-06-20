# data_processing.py

import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader, 
    UnstructuredFileLoader,
    UnstructuredHTMLLoader
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models
from PIL import Image
import pytesseract
from tqdm import tqdm
import uuid
import base64
import nest_asyncio

nest_asyncio.apply()

# install requirements
# pip install -r requirements.txt
# Настройка логов
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HTML_STORAGE_PATH = 'data/'
IMAGE_STORAGE_PATH = 'data/'
TEXT_STORAGE_PATH = 'data/'
# Using ai-forever/ru-en-RoSBERTa from Hugging Face
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="ai-forever/ru-en-RoSBERTa")
QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "documents"
MAX_WORKERS = 4  # Number of parallel workers

def load_text_files(directory):
    try:
        logger.info(f"Loading text files from {directory}...")
        loader = DirectoryLoader(directory, glob="*.txt")
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} documents from {directory}")
        return documents
    except Exception as e:
        logger.error(f"Error loading text files from {directory}: {str(e)}", exc_info=True)
        return []

def load_html_document(file_path):
    try:
        logger.info(f"Loading HTML file: {file_path}")
        loader = UnstructuredHTMLLoader(file_path)
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} documents from {file_path}")
        return documents
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}", exc_info=True)
        return []

def load_image_text(file_path):
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logger.error(f"Error processing image {file_path}: {str(e)}")
        return ""

def load_html_documents(directory):
    logger.info(f"Loading HTML documents from {directory}...")
    start_time = time.time()
    documents = []
    
    try:
        # Check if directory exists
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return []
            
        # Get all .htm files
        html_files = [f for f in os.listdir(directory) if f.endswith(".htm")]
        logger.info(f"Found {len(html_files)} HTML files to process")
        
        if not html_files:
            logger.warning(f"No HTML files found in {directory}")
            return []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(load_html_document, os.path.join(directory, filename)): filename 
                for filename in html_files
            }
            
            for future in tqdm(as_completed(futures), total=len(html_files), desc="Loading HTML files"):
                filename = futures[future]
                try:
                    docs = future.result()
                    if docs:
                        documents.extend(docs)
                        logger.debug(f"Added {len(docs)} documents from {filename}")
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error in load_html_documents: {str(e)}", exc_info=True)
    finally:
        logger.info(f"Loaded {len(documents)} documents in {time.time() - start_time:.2f} seconds")
    
    return documents

def load_image_texts(directory):
    logger.info("Loading images and extracting text...")
    start_time = time.time()
    texts = []
    image_files = [f for f in os.listdir(directory) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(load_image_text, os.path.join(directory, filename)): filename 
            for filename in image_files
        }
        
        for future in tqdm(as_completed(futures), total=len(image_files), desc="Processing images"):
            text = future.result()
            if text:
                texts.append(text)
    
    logger.info(f"Extracted text from {len(texts)} images in {time.time() - start_time:.2f} seconds.")
    return texts

def chunk_documents(raw_documents):
    logger.info("Chunking documents...")
    start_time = time.time()
    # Adjusted chunk size for better performance with large documents
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Reduced chunk size for better processing
        chunk_overlap=100,
        add_start_index=True
    )
    chunks = text_processor.split_documents(raw_documents)
    logger.info(f"Chunked {len(raw_documents)} documents into {len(chunks)} chunks in {time.time() - start_time:.2f} seconds.")
    return chunks

def process_documents():
    # Загрузка данных
    start_time = time.time()
    logger.info("Starting document processing...")
    
    try:
        # Load and process documents
        html_docs = load_html_documents(HTML_STORAGE_PATH)
        image_texts = load_image_texts(IMAGE_STORAGE_PATH)
        texts = load_text_files(TEXT_STORAGE_PATH)
        
        # Convert image texts to Document objects
        image_docs = [
            Document(
                page_content=text,
                metadata={"source": f"image_{i}", "type": "image"}
            )
            for i, text in enumerate(image_texts) if text.strip()
        ]
        
        # Combine all documents
        all_documents = html_docs + image_docs + texts
        
        if not all_documents:
            logger.error("No valid documents found to process")
            return None
        
        logger.info(f"Processing {len(all_documents)} total documents ({len(html_docs)} HTML, {len(image_docs)} images)")
        
        # Chunk documents
        logger.info("Chunking documents...")
        processed_chunks = chunk_documents(all_documents)
        
        # Initialize Qdrant client
        client = QdrantClient(host="localhost", port=6333)
        
        # Check if collection exists
        try:
            collections = client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            
            if COLLECTION_NAME in collection_names:
                logger.info(f"Collection '{COLLECTION_NAME}' already exists. Dropping and recreating...")
                client.delete_collection(collection_name=COLLECTION_NAME)
        except Exception as e:
            logger.warning(f"Error checking collections: {str(e)}")
        
        # Create collection configuration with correct dimensions for Ru-en RoBERTa
        VECTOR_SIZE = 1024  # Updated dimension size for ai-forever/ru-en-RoSBERTa
        vectors_config = models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        )
        
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=vectors_config,
        )
        logger.info(f"Created new collection '{COLLECTION_NAME}' with vector size {VECTOR_SIZE}")
        
        # Add documents to the collection using a simpler approach
        from qdrant_client.http import models as rest
        
        # Convert documents to Qdrant points format
        texts = [doc.page_content for doc in processed_chunks]
        metadatas = [doc.metadata for doc in processed_chunks]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = EMBEDDING_MODEL.embed_documents(texts)
        
        # Prepare points for upload
        points = []
        for idx, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
            points.append(
                rest.PointStruct(
                    id=idx,
                    payload={
                        "text": text,
                        "metadata": metadata
                    },
                    vector=embedding,
                )
            )
        
        # Upload points in batches
        logger.info(f"Uploading {len(points)} points to Qdrant...")
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )
        
        # Create and return the vector store
        from langchain_community.vectorstores.qdrant import Qdrant
        vector_store = Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=EMBEDDING_MODEL,
        )
        
        logger.info("Documents indexed successfully")
        logger.info(f"Qdrant index created in {time.time() - start_time:.2f} seconds")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    vector_store = process_documents()