# data_processing.py

import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader, 
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader
)
from ollama_embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
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
DOCUMENTS_DIR = 'data/'
# Using Ollama for embeddings
EMBEDDING_MODEL = OllamaEmbeddings(model_name="nomic-embed-text")
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
    except Exception as e:
        logging.error(f"Error loading HTML document {file_path}: {str(e)}")
        return []

def load_pdf_document(file_path):
    """Load a single PDF document using pdfminer.six."""
    try:
        # Проверяем, что файл существует и имеет правильный размер
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return []
            
        if os.path.getsize(file_path) == 0:
            logging.error(f"Empty file: {file_path}")
            return []
            
        from pdfminer.high_level import extract_text
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdfdocument import PDFDocument
        from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
        from pdfminer.converter import PDFPageAggregator
        from pdfminer.layout import LAParams, LTTextBox, LTTextLine
        
        # Инициализируем ресурсы
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        
        # Открываем PDF файл
        with open(file_path, 'rb') as fp:
            parser = PDFParser(fp)
            doc = PDFDocument(parser)
            
            # Извлекаем текст
            text = extract_text(file_path)
            
            # Создаем документ
            return [Document(
                page_content=text,
                metadata={
                    'source': file_path,
                    'type': 'pdf'
                }
            )]
    except ModuleNotFoundError as e:
        if 'pdfminer' in str(e):
            logging.error(f"Module pdfminer not found. Please install it with 'pip install pdfminer.six'")
        else:
            logging.error(f"Error loading PDF document {file_path}: {str(e)}")
    except Exception as e:
        logging.error(f"Error loading PDF document {file_path}: {str(e)}")
    return []

def load_docx_document(file_path):
    """Load a single DOCX document."""
    try:
        loader = UnstructuredWordDocumentLoader(file_path)
        return loader.load()
    except Exception as e:
        logging.error(f"Error loading DOCX document {file_path}: {str(e)}")
        return []

def load_html_documents(directory):
    logger.info(f"Loading HTML documents from {directory}...")
    html_docs = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.html'):
                file_path = os.path.join(root, file)
                docs = load_html_document(file_path)
                if docs:
                    html_docs.extend(docs)
    logger.info(f"Successfully loaded {len(html_docs)} HTML documents")
    return html_docs

def load_pdf_documents(directory):
    logger.info(f"Loading PDF documents from {directory}...")
    pdf_docs = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                docs = load_pdf_document(file_path)
                if docs:
                    pdf_docs.extend(docs)
    logger.info(f"Successfully loaded {len(pdf_docs)} PDF documents")
    return pdf_docs

def load_docx_documents(directory):
    logger.info(f"Loading DOCX documents from {directory}...")
    docx_docs = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.docx'):
                file_path = os.path.join(root, file)
                docs = load_docx_document(file_path)
                if docs:
                    docx_docs.extend(docs)
    logger.info(f"Successfully loaded {len(docx_docs)} DOCX documents")
    return docx_docs

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
    start_time = time.time()
    logger.info("Starting document processing...")
    
    # Initialize Qdrant client
    client = QdrantClient(host="localhost", port=6333)
    
    # Define vector size for Ollama's nomic-embed-text model
    VECTOR_SIZE = 768
    
    # Check if collection exists, if yes, delete it to recreate with correct dimensions
    try:
        collections = client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if COLLECTION_NAME in collection_names:
            logger.info(f"Collection '{COLLECTION_NAME}' exists. Deleting to recreate with correct dimensions...")
            client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception as e:
        logger.warning(f"Error checking collections: {str(e)}")
    
    # Create collection with correct vector size
    logger.info(f"Creating collection '{COLLECTION_NAME}' with vector size {VECTOR_SIZE}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "text": models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        }
    )
    
    # Load documents from different sources
    documents = []
    try:
        logger.info("Loading text files...")
        documents.extend(load_text_files(TEXT_STORAGE_PATH))
        logger.info(f"Loaded {len(documents)} text documents")
        
        logger.info("Loading HTML files...")
        html_docs = load_html_documents(HTML_STORAGE_PATH)
        documents.extend(html_docs)
        logger.info(f"Loaded {len(html_docs)} HTML documents")
        
        logger.info("Loading PDF files...")
        pdf_docs = load_pdf_documents(DOCUMENTS_DIR)
        documents.extend(pdf_docs)
        logger.info(f"Loaded {len(pdf_docs)} PDF documents")
        
        logger.info("Loading DOCX files...")
        docx_docs = load_docx_documents(DOCUMENTS_DIR)
        documents.extend(docx_docs)
        logger.info(f"Loaded {len(docx_docs)} DOCX documents")
        
        logger.info("Loading images for OCR...")
        image_texts = load_image_texts(IMAGE_STORAGE_PATH)
        image_docs = [
            Document(
                page_content=text,
                metadata={"source": f"image_{i}", "type": "image"}
            )
            for i, text in enumerate(image_texts) if text.strip()
        ]
        documents.extend(image_docs)
        logger.info(f"Processed {len(image_docs)} images")
        
        if not documents:
            logger.error("No documents were loaded. Check your data directory.")
            return None
            
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}", exc_info=True)
        raise

    # Split documents into chunks
    logger.info("Chunking documents...")
    chunks = chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    if not chunks:
        logger.warning("No document chunks were created. Check your input files.")
        return None
    
    # Process chunks in batches to avoid memory issues
    BATCH_SIZE = 50
    total_chunks = len(chunks)
    
    logger.info(f"Processing {total_chunks} chunks in batches of {BATCH_SIZE}...")
    
    try:
        # Process chunks in batches
        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(total_chunks-1)//BATCH_SIZE + 1}...")
            
            # Extract texts and metadatas for the current batch
            batch_texts = [chunk.page_content for chunk in batch]
            batch_metadatas = [chunk.metadata for chunk in batch]
            
            # Generate embeddings for the batch
            logger.info(f"Generating embeddings for batch {i//BATCH_SIZE + 1}...")
            batch_embeddings = EMBEDDING_MODEL.embed_documents(batch_texts)
            
            # Prepare points for Qdrant
            points = []
            for idx, (text, metadata, embedding) in enumerate(zip(batch_texts, batch_metadatas, batch_embeddings)):
                points.append(
                    models.PointStruct(
                        id=i + idx,  # Use a unique ID for each point
                        payload={
                            "text": text,
                            "metadata": metadata
                        },
                        vector={"text": embedding},  # Specify the vector name 'text' as defined in the collection
                    )
                )
            
            # Upload batch to Qdrant
            logger.info(f"Uploading batch {i//BATCH_SIZE + 1} to Qdrant...")
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True
            )
        
        # Create and return the vector store
        vector_store = Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=EMBEDDING_MODEL,
        )
        
        logger.info(f"Successfully processed and indexed {total_chunks} document chunks")
        logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        # Try to clean up the collection if there was an error
        try:
            client.delete_collection(collection_name=COLLECTION_NAME)
            logger.info("Cleaned up collection after error")
        except Exception as cleanup_error:
            logger.warning(f"Error cleaning up collection: {str(cleanup_error)}")
        raise

if __name__ == "__main__":
    vector_store = process_documents()