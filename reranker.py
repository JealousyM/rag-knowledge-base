"""
Reranker module for improving document relevance in RAG systems.
Supports multiple reranking strategies including cross-encoder models.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import CrossEncoder
import numpy as np
from langchain.schema import Document

logger = logging.getLogger(__name__)

class DocumentReranker:
    """
    Document reranker using cross-encoder models for improved relevance scoring.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker with a cross-encoder model.
        
        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the cross-encoder model."""
        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {str(e)}")
            self.model = None
    
    def rerank_documents(self, query: str, documents: List[Any], top_k: Optional[int] = None) -> List[Any]:
        """
        Rerank documents based on their relevance to the query using cross-encoder.
        
        Args:
            query: The search query
            documents: List of documents (can be Document objects or scored points)
            top_k: Number of top documents to return (if None, returns all)
            
        Returns:
            List of reranked documents with updated scores
        """
        if not self.model:
            logger.warning("Cross-encoder model not available, returning original documents")
            return documents[:top_k] if top_k else documents
        
        if not documents:
            return documents
        
        try:
            # Extract text content from documents
            doc_texts = []
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    # LangChain Document
                    doc_texts.append(doc.page_content)
                elif hasattr(doc, 'payload') and 'text' in doc.payload:
                    # Qdrant scored point with text
                    doc_texts.append(doc.payload['text'])
                elif hasattr(doc, 'payload') and 'page_content' in doc.payload:
                    # Qdrant scored point with page_content
                    doc_texts.append(doc.payload['page_content'])
                else:
                    # Fallback - try to convert to string
                    doc_texts.append(str(doc))
            
            # Create query-document pairs for cross-encoder
            query_doc_pairs = [[query, doc_text] for doc_text in doc_texts]
            
            # Get relevance scores from cross-encoder
            logger.info(f"Reranking {len(documents)} documents with cross-encoder")
            scores = self.model.predict(query_doc_pairs)
            
            # Create list of (document, score) pairs
            doc_score_pairs = list(zip(documents, scores))
            
            # Sort by relevance score (descending)
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Update document scores and return top_k
            reranked_docs = []
            for i, (doc, new_score) in enumerate(doc_score_pairs):
                if top_k and i >= top_k:
                    break
                
                # Update the score in the document
                if hasattr(doc, 'score'):
                    # For scored points, update the score
                    doc.score = float(new_score)
                elif hasattr(doc, 'metadata'):
                    # For LangChain documents, add score to metadata
                    if not doc.metadata:
                        doc.metadata = {}
                    doc.metadata['rerank_score'] = float(new_score)
                
                reranked_docs.append(doc)
            
            logger.info(f"Reranked {len(reranked_docs)} documents")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            # Return original documents if reranking fails
            return documents[:top_k] if top_k else documents
    
    def rerank_with_scores(self, query: str, documents: List[Any]) -> List[Tuple[Any, float]]:
        """
        Rerank documents and return them with their relevance scores.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            
        Returns:
            List of (document, relevance_score) tuples sorted by relevance
        """
        if not self.model or not documents:
            # Return with dummy scores if model not available
            return [(doc, 0.5) for doc in documents]
        
        try:
            # Extract text content
            doc_texts = []
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    doc_texts.append(doc.page_content)
                elif hasattr(doc, 'payload') and 'text' in doc.payload:
                    doc_texts.append(doc.payload['text'])
                elif hasattr(doc, 'payload') and 'page_content' in doc.payload:
                    doc_texts.append(doc.payload['page_content'])
                else:
                    doc_texts.append(str(doc))
            
            # Get cross-encoder scores
            query_doc_pairs = [[query, doc_text] for doc_text in doc_texts]
            scores = self.model.predict(query_doc_pairs)
            
            # Create and sort (document, score) pairs
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return [(doc, float(score)) for doc, score in doc_score_pairs]
            
        except Exception as e:
            logger.error(f"Error during reranking with scores: {str(e)}")
            return [(doc, 0.5) for doc in documents]


class HybridReranker:
    """
    Hybrid reranker that combines multiple reranking strategies.
    """
    
    def __init__(self, 
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 use_semantic_similarity: bool = True,
                 use_keyword_matching: bool = True):
        """
        Initialize hybrid reranker.
        
        Args:
            cross_encoder_model: Cross-encoder model name
            use_semantic_similarity: Whether to use semantic similarity scoring
            use_keyword_matching: Whether to use keyword matching scoring
        """
        self.cross_encoder = DocumentReranker(cross_encoder_model)
        self.use_semantic_similarity = use_semantic_similarity
        self.use_keyword_matching = use_keyword_matching
    
    def _calculate_keyword_score(self, query: str, text: str) -> float:
        """Calculate keyword matching score."""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = query_words.intersection(text_words)
        union = query_words.union(text_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def rerank_documents(self, query: str, documents: List[Any], 
                        cross_encoder_weight: float = 0.7,
                        keyword_weight: float = 0.2,
                        original_weight: float = 0.1,
                        top_k: Optional[int] = None) -> List[Any]:
        """
        Rerank documents using hybrid approach.
        
        Args:
            query: Search query
            documents: Documents to rerank
            cross_encoder_weight: Weight for cross-encoder scores
            keyword_weight: Weight for keyword matching scores
            original_weight: Weight for original retrieval scores
            top_k: Number of top documents to return
            
        Returns:
            Reranked documents
        """
        if not documents:
            return documents
        
        try:
            # Get cross-encoder scores
            cross_encoder_scores = []
            if self.cross_encoder.model:
                doc_score_pairs = self.cross_encoder.rerank_with_scores(query, documents)
                cross_encoder_scores = [score for _, score in doc_score_pairs]
            else:
                cross_encoder_scores = [0.5] * len(documents)
            
            # Calculate keyword scores
            keyword_scores = []
            if self.use_keyword_matching:
                for doc in documents:
                    text = ""
                    if hasattr(doc, 'page_content'):
                        text = doc.page_content
                    elif hasattr(doc, 'payload') and 'text' in doc.payload:
                        text = doc.payload['text']
                    elif hasattr(doc, 'payload') and 'page_content' in doc.payload:
                        text = doc.payload['page_content']
                    
                    keyword_scores.append(self._calculate_keyword_score(query, text))
            else:
                keyword_scores = [0.0] * len(documents)
            
            # Get original scores
            original_scores = []
            for doc in documents:
                if hasattr(doc, 'score'):
                    original_scores.append(float(doc.score))
                elif hasattr(doc, 'metadata') and 'score' in doc.metadata:
                    original_scores.append(float(doc.metadata['score']))
                else:
                    original_scores.append(0.5)
            
            # Normalize scores to [0, 1] range
            def normalize_scores(scores):
                if not scores or max(scores) == min(scores):
                    return [0.5] * len(scores)
                min_score, max_score = min(scores), max(scores)
                return [(s - min_score) / (max_score - min_score) for s in scores]
            
            cross_encoder_scores = normalize_scores(cross_encoder_scores)
            keyword_scores = normalize_scores(keyword_scores)
            original_scores = normalize_scores(original_scores)
            
            # Calculate combined scores
            combined_scores = []
            for i in range(len(documents)):
                combined_score = (
                    cross_encoder_weight * cross_encoder_scores[i] +
                    keyword_weight * keyword_scores[i] +
                    original_weight * original_scores[i]
                )
                combined_scores.append(combined_score)
            
            # Create (document, score) pairs and sort
            doc_score_pairs = list(zip(documents, combined_scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Update scores and return top_k documents
            reranked_docs = []
            for i, (doc, score) in enumerate(doc_score_pairs):
                if top_k and i >= top_k:
                    break
                
                # Update document score
                if hasattr(doc, 'score'):
                    doc.score = float(score)
                elif hasattr(doc, 'metadata'):
                    if not doc.metadata:
                        doc.metadata = {}
                    doc.metadata['hybrid_rerank_score'] = float(score)
                
                reranked_docs.append(doc)
            
            logger.info(f"Hybrid reranking completed: {len(reranked_docs)} documents")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in hybrid reranking: {str(e)}")
            return documents[:top_k] if top_k else documents


# Factory function for easy instantiation
def create_reranker(reranker_type: str = "cross_encoder", **kwargs) -> Any:
    """
    Factory function to create different types of rerankers.
    
    Args:
        reranker_type: Type of reranker ("cross_encoder" or "hybrid")
        **kwargs: Additional arguments for reranker initialization
        
    Returns:
        Reranker instance
    """
    if reranker_type == "cross_encoder":
        return DocumentReranker(**kwargs)
    elif reranker_type == "hybrid":
        return HybridReranker(**kwargs)
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")
