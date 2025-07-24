"""
Test script for the reranker functionality.
This script tests both the standalone reranker and the integrated hybrid search with reranking.
"""

import sys
import logging
from typing import List
from langchain.schema import Document

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_document_reranker():
    """Test the standalone DocumentReranker."""
    try:
        from reranker import DocumentReranker
        
        logger.info("Testing DocumentReranker...")
        
        # Create test documents
        test_docs = [
            Document(page_content="Python is a programming language used for web development.", metadata={"source": "doc1"}),
            Document(page_content="Machine learning algorithms can be implemented in Python.", metadata={"source": "doc2"}),
            Document(page_content="JavaScript is used for frontend web development.", metadata={"source": "doc3"}),
            Document(page_content="Deep learning models require large datasets for training.", metadata={"source": "doc4"}),
            Document(page_content="Python libraries like scikit-learn make machine learning accessible.", metadata={"source": "doc5"})
        ]
        
        # Test query
        query = "Python machine learning"
        
        # Initialize reranker
        reranker = DocumentReranker()
        
        if reranker.model is None:
            logger.warning("Cross-encoder model not available, skipping reranker test")
            return False
        
        # Test reranking
        logger.info(f"Original document order:")
        for i, doc in enumerate(test_docs):
            logger.info(f"  {i+1}. {doc.page_content[:50]}...")
        
        reranked_docs = reranker.rerank_documents(query, test_docs, top_k=3)
        
        logger.info(f"Reranked document order (top 3):")
        for i, doc in enumerate(reranked_docs):
            score = doc.metadata.get('rerank_score', 'N/A')
            logger.info(f"  {i+1}. [Score: {score}] {doc.page_content[:50]}...")
        
        logger.info("DocumentReranker test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing DocumentReranker: {str(e)}")
        return False

def test_hybrid_reranker():
    """Test the HybridReranker."""
    try:
        from reranker import HybridReranker
        
        logger.info("Testing HybridReranker...")
        
        # Create test documents
        test_docs = [
            Document(page_content="Python programming language for data science and machine learning.", metadata={"source": "doc1"}),
            Document(page_content="Web development with JavaScript and React framework.", metadata={"source": "doc2"}),
            Document(page_content="Machine learning algorithms in Python using scikit-learn.", metadata={"source": "doc3"}),
            Document(page_content="Database management with SQL and PostgreSQL.", metadata={"source": "doc4"}),
            Document(page_content="Python data analysis with pandas and numpy libraries.", metadata={"source": "doc5"})
        ]
        
        # Test query
        query = "Python data science machine learning"
        
        # Initialize hybrid reranker
        reranker = HybridReranker()
        
        # Test reranking
        logger.info(f"Original document order:")
        for i, doc in enumerate(test_docs):
            logger.info(f"  {i+1}. {doc.page_content[:50]}...")
        
        reranked_docs = reranker.rerank_documents(query, test_docs, top_k=3)
        
        logger.info(f"Hybrid reranked document order (top 3):")
        for i, doc in enumerate(reranked_docs):
            score = doc.metadata.get('hybrid_rerank_score', 'N/A')
            logger.info(f"  {i+1}. [Score: {score}] {doc.page_content[:50]}...")
        
        logger.info("HybridReranker test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing HybridReranker: {str(e)}")
        return False

def test_hybrid_search_with_reranking():
    """Test the HybridSearch with reranking enabled."""
    try:
        from hybrid_search import HybridSearch
        from constants import COLLECTION_NAME
        
        logger.info("Testing HybridSearch with reranking...")
        
        # Initialize hybrid search with reranking enabled
        search_engine = HybridSearch(
            collection_name=COLLECTION_NAME,
            enable_reranking=True,
            reranker_type="cross_encoder"
        )
        
        # Test query
        query = "machine learning algorithms"
        
        # Perform search
        results = search_engine.search(query, k=5, rerank_top_k=10)
        
        logger.info(f"Search results with reranking:")
        for i, result in enumerate(results):
            score = result.score
            content = result.payload.get('text', result.payload.get('page_content', 'No content'))[:100]
            logger.info(f"  {i+1}. [Score: {score:.4f}] {content}...")
        
        logger.info("HybridSearch with reranking test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing HybridSearch with reranking: {str(e)}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting reranker functionality tests...")
    
    tests = [
        ("DocumentReranker", test_document_reranker),
        ("HybridReranker", test_hybrid_reranker),
        ("HybridSearch with Reranking", test_hybrid_search_with_reranking)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    logger.info(f"\nTotal tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        logger.info("All tests passed! ✅")
        return True
    else:
        logger.info("Some tests failed! ❌")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
