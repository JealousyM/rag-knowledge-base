# data_processing.py

import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from PIL import Image
import pytesseract
from tqdm import tqdm

# install requirements
# pip install -r requirements.txt
# Настройка логов
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HTML_STORAGE_PATH = 'HTML/'
IMAGE_STORAGE_PATH = 'HTML/'
# Using a more efficient embedding model optimized for speed
EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")
FAISS_INDEX_PATH = "faiss_index"
MAX_WORKERS = 4  # Number of parallel workers

def load_html_document(file_path):
    try:
        loader = UnstructuredHTMLLoader(file_path)
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
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
    logger.info("Loading HTML documents...")
    start_time = time.time()
    documents = []
    html_files = [f for f in os.listdir(directory) if f.endswith(".htm")]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(load_html_document, os.path.join(directory, filename)): filename 
            for filename in html_files
        }
        
        for future in tqdm(as_completed(futures), total=len(html_files), desc="Loading HTML files"):
            docs = future.result()
            documents.extend(docs)
    
    logger.info(f"Loaded {len(documents)} HTML documents in {time.time() - start_time:.2f} seconds.")
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
    html_docs = load_html_documents(HTML_STORAGE_PATH)
    image_texts = load_image_texts(IMAGE_STORAGE_PATH)
    image_docs = [Document(page_content=text) for text in image_texts]
    all_docs = html_docs + image_docs
    
    # Обработка и индексация данных
    processed_chunks = chunk_documents(all_docs)
    logger.info(f"Processed {len(processed_chunks)} document chunks")
    
    # Process chunks in batches to avoid memory issues
    batch_size = 100
    vector_store = None
    
    for i in tqdm(range(0, len(processed_chunks), batch_size), desc="Indexing documents"):
        batch = processed_chunks[i:i + batch_size]
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, EMBEDDING_MODEL)
        else:
            vector_store.add_documents(batch)
    
    logger.info("Documents indexed successfully")
    
    # Сохранение индекса на диск
    start_time = time.time()
    vector_store.save_local(FAISS_INDEX_PATH)
    logger.info(f"FAISS index saved to {FAISS_INDEX_PATH} in {time.time() - start_time:.2f} seconds")
    
    return vector_store

if __name__ == "__main__":
    vector_store = process_documents()