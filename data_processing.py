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
    UnstructuredWordDocumentLoader,
    PyPDFLoader,
    UnstructuredFileLoader
)
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import nest_asyncio
from sentence_transformers import SentenceTransformer
from text_utils import optimize_for_embedding, split_into_semantic_chunks

nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global settings
HTTP_TIMEOUT = 120
HTTP_MAX_RETRIES = 3
HTML_STORAGE_PATH = 'data/'
IMAGE_STORAGE_PATH = 'data/'
TEXT_STORAGE_PATH = 'data/'
DOCUMENTS_DIR = 'data/'
MODEL_PATH = 'model/ru-en-RoSBERTa'
EMBEDDING_MODEL = SentenceTransformer(MODEL_PATH)

VECTOR_SIZE = 1024  # RoSBERTa produces 1024-dimensional embeddings
QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "documents"
MAX_WORKERS = 4  # Number of parallel workers
BATCH_SIZE = 50   # Batch size for processing

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
    try:
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
        
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        
        with open(file_path, 'rb') as fp:
            parser = PDFParser(fp)
            doc = PDFDocument(parser)
            
            text = extract_text(file_path)
            
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
    logger.info(f"Chunking {len(raw_documents)} documents semantically...")
    start_time = time.time()
    
    semantic_chunks = []
    
    for doc in raw_documents:
        optimized_text = optimize_for_embedding(doc.page_content)
        
        text_chunks = split_into_semantic_chunks(
            optimized_text,
            max_chunk_size=400,  
            overlap=50  
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
            semantic_chunks.append(chunk_doc)
    
    logger.info(f"Chunked {len(raw_documents)} documents into {len(semantic_chunks)} semantic chunks in {time.time() - start_time:.2f} seconds")
    return semantic_chunks

def create_vector_store(client, collection_name, chunks):
    logger.info("Creating vector store...")
    start_time = time.time()
    
    # Always recreate collection for safety
    try:
        logger.info(f"Deleting collection '{collection_name}' if it exists")
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Successfully deleted existing collection '{collection_name}'")
    except Exception as e:
        logger.info(f"Collection '{collection_name}' may not exist or could not be deleted: {str(e)}")
    
    # Create fresh collection
    logger.info(f"Creating new collection '{collection_name}'")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,  
            distance=models.Distance.COSINE,
        ),
    )
    
    BATCH_SIZE = 100  
    total_chunks = len(chunks)
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_ids = [i + idx for idx in range(len(batch))]
        batch_texts = [chunk.page_content for chunk in batch]
        batch_metadatas = [chunk.metadata for chunk in batch]
        
        # Generate embeddings with proper task prefix for documents
        batch_embeddings = []
        for text in batch_texts:
            batch_embeddings.append(get_embeddings(text, task="search_document"))
        
        points = []
        for idx, (doc_id, vector, metadata, text) in enumerate(zip(batch_ids, batch_embeddings, batch_metadatas, batch_texts)):
            full_metadata = {**metadata, "text": text}
            
            points.append(
                models.PointStruct(
                    id=doc_id,  # Using integer ID
                    vector=vector.tolist(),  
                    payload=full_metadata
                )
            )
        
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
    
    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=EMBEDDING_MODEL,
    )
    
    logger.info(f"Successfully created vector store in {time.time() - start_time:.2f} seconds")
    return vector_store

def process_documents():
    start_time = time.time()
    logger.info("Starting document processing...")
    
    client = QdrantClient(host="localhost", port=6333)

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
        
        logger.info("Loading image texts...")
        image_texts = load_image_texts(IMAGE_STORAGE_PATH)
        documents.extend(image_texts)
        logger.info(f"Loaded {len(image_texts)} image texts")
        
        if not documents:
            logger.warning("No documents found.")
            return None
        
        logger.info(f"Loaded {len(documents)} documents in total")
        
        chunks = chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        if not chunks:
            logger.warning("No document chunks were created. Check your input files.")
            return None
        
        vector_store = create_vector_store(client, COLLECTION_NAME, chunks)
        
        logger.info(f"Successfully processed and indexed {len(chunks)} document chunks")
        logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
        return vector_store
    
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        try:
            client.delete_collection(collection_name=COLLECTION_NAME)
            logger.info("Cleaned up collection after error")
        except Exception as cleanup_error:
            logger.warning(f"Error cleaning up collection: {str(cleanup_error)}")
        raise

def get_embeddings(text, task="search_document"):
    """Generate embeddings with proper task prefix as per RoSBERTa model requirements"""
    prefixed_text = f"{task}: {text}"
    
    embedding = EMBEDDING_MODEL.encode(
        prefixed_text,
        normalize_embeddings=True,
        convert_to_numpy=False,
        show_progress_bar=False
    )
    return embedding

if __name__ == "__main__":
    vector_store = process_documents()