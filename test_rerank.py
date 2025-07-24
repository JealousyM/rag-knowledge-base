#!/usr/bin/env python3
"""
Тестовый скрипт для проверки корректной работы rerank
"""

import sys
import os
import json
import time
from typing import List, Dict, Any

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reranker import DocumentReranker, HybridReranker
from hybrid_search import HybridSearch
from memory_storage import MemoryStorage
from constants import MODEL_PATH

def create_test_data():
    """Создаем тестовые данные для проверки rerank"""
    
    # Тестовые документы с разной релевантностью
    test_documents = [
        {
            "content": "Python это высокоуровневый язык программирования с динамической типизацией",
            "metadata": {"source": "python_docs", "score": 0.8}
        },
        {
            "content": "JavaScript используется для веб-разработки и работает в браузере",
            "metadata": {"source": "js_docs", "score": 0.7}
        },
        {
            "content": "RS.SCM это система управления поставками в ритейле",
            "metadata": {"source": "retail_docs", "score": 0.9}
        },
        {
            "content": "База данных это организованный набор структурированной информации",
            "metadata": {"source": "db_docs", "score": 0.6}
        }
    ]
    
    return test_documents

def test_document_reranker():
    """Тестируем DocumentReranker"""
    print("=== Тест DocumentReranker ===")
    
    reranker = DocumentReranker()
    test_query = "что такое RS.SCM"
    documents = create_test_data()
    
    print(f"Запрос: {test_query}")
    print(f"Исходные документы: {len(documents)}")
    
    # Показываем исходный порядок
    for i, doc in enumerate(documents):
        print(f"  {i+1}. {doc['content'][:50]}... (score: {doc['metadata']['score']})")
    
    # Применяем rerank
    start_time = time.time()
    reranked_docs = reranker.rerank_documents(test_query, documents, top_k=3)
    rerank_time = time.time() - start_time
    
    print(f"\nПосле rerank ({rerank_time:.3f}s):")
    for i, doc in enumerate(reranked_docs):
        print(f"  {i+1}. {doc['content'][:50]}... (new_score: {doc.get('rerank_score', 'N/A')})")
    
    return reranked_docs

def test_hybrid_reranker():
    """Тестируем HybridReranker"""
    print("\n=== Тест HybridReranker ===")
    
    reranker = HybridReranker()
    test_query = "что такое RS.SCM"
    documents = create_test_data()
    
    reranked_docs = reranker.rerank_documents(test_query, documents, top_k=3)
    
    print("Результаты HybridReranker:")
    for i, doc in enumerate(reranked_docs):
        print(f"  {i+1}. {doc['content'][:50]}...")
        print(f"     Cross-encoder score: {doc.get('cross_encoder_score', 'N/A')}")
        print(f"     BM25 score: {doc.get('bm25_score', 'N/A')}")
        print(f"     Final score: {doc.get('final_score', 'N/A')}")
    
    return reranked_docs

def test_integration_with_search():
    """Тестируем интеграцию rerank с полным поисковым pipeline"""
    print("\n=== Тест интеграции ===")
    
    # Создаем временное хранилище
    storage = MemoryStorage()
    
    # Добавляем тестовые документы
    test_docs = create_test_data()
    for doc in test_docs:
        storage.add_document(doc['content'], doc['metadata'])
    
    # Создаем гибридный поиск с rerank
    search = HybridSearch(
        vector_store=storage,
        enable_reranking=True,
        reranker_type="hybrid",
        rerank_top_k=3
    )
    
    # Выполняем поиск
    test_query = "RS SCM система управления поставками"
    results = search.search(test_query, k=4)
    
    print(f"Результаты поиска с rerank для: '{test_query}'")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['content'][:60]}...")
        print(f"     Score: {result.get('score', 'N/A')}")
        print(f"     Rerank Score: {result.get('rerank_score', 'N/A')}")

def test_performance():
    """Тестируем производительность rerank"""
    print("\n=== Тест производительности ===")
    
    reranker = DocumentReranker()
    
    # Создаем больше документов для теста
    large_test_data = create_test_data() * 5  # 20 документов
    
    start_time = time.time()
    reranked = reranker.rerank_documents("тестовый запрос", large_test_data, top_k=10)
    elapsed = time.time() - start_time
    
    print(f"Rerank 20 документов заняло: {elapsed:.3f} секунд")
    print(f"Средняя latency на документ: {elapsed/len(large_test_data)*1000:.1f} мс")

def main():
    """Главная функция тестирования"""
    print("🧪 Тестирование rerank функциональности\n")
    
    try:
        # Тестируем каждый компонент
        test_document_reranker()
        test_hybrid_reranker()
        test_integration_with_search()
        test_performance()
        
        print("\n✅ Все тесты пройдены успешно!")
        print("\n📊 Проверка корректности:")
        print("   - Убедитесь, что наиболее релевантные документы поднимаются вверх")
        print("   - Проверьте, что RS.SCM документ имеет высокий rerank_score")
        print("   - Время rerank должно быть в пределах 200-500ms для 10 документов")
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
