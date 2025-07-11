from typing import TypedDict, List, Optional, Dict
from langchain.schema import Document

# Определяем тип состояния для LangGraph
class RAGState(TypedDict):
    """State for the RAG workflow"""
    question: str
    chat_history: List[List[str]]
    context: List[Document]
    formatted_context: str
    answer: Optional[str] = None
    sources: List[Dict] = []