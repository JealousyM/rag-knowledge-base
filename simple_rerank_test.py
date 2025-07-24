#!/usr/bin/env python3
"""
Простой тест для проверки работы rerank
"""

import sys
import os
import time

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reranker import DocumentReranker, HybridReranker

def test_rerank_working():
    """Простой тест показывающий как работает rerank"""
    
    print("🧪 Тестирование rerank механизма\n")
    
    # Создаем reranker
    reranker = DocumentReranker()
    
    # Тестовые документы
    documents = [
        {
            "content": "Python это высокоуровневый язык программирования",
            "metadata": {"source": "python_docs", "score": 0.8}
        },
        {
            "content": "RS.SCM это система управления поставками в ритейле",
            "metadata": {"source": "retail_docs", "score": 0.9}
        },
        {
            "content": "JavaScript используется для веб-разработки",
            "metadata": {"source": "js_docs", "score": 0.7}
        },
        {
            "content": "База данных это организованный набор информации",
            "metadata": {"source": "db_docs", "score": 0.6}
        }
    ]
    
    # Тестовый запрос
    query = "что такое RS.SCM"
    
    print(f"Запрос: {query}")
    print(f"Исходные документы ({len(documents)}):")
    
    for i, doc in enumerate(documents):
        print(f"  {i+1}. {doc['content'][:50]}... (score: {doc['metadata']['score']})")
    
    print("\nПрименяем rerank...")
    
    # Применяем rerank
    start_time = time.time()
    reranked = reranker.rerank_documents(query, documents, top_k=4)
    elapsed = time.time() - start_time
    
    print(f"Результат rerank ({elapsed:.3f}s):")
    
    for i, doc in enumerate(reranked):
        # Получаем новый скор из метаданных
        new_score = doc.get('metadata', {}).get('cross_encoder_score', 'N/A')
        print(f"  {i+1}. {doc['content'][:50]}... (new_score: {new_score})")
    
    # Проверяем, поднялся ли RS.SCM документ
    rs_scm_doc = next((doc for doc in reranked if "RS.SCM" in doc['content']), None)
    if rs_scm_doc:
        rs_index = next(i for i, doc in enumerate(reranked) if "RS.SCM" in doc['content'])
        print(f"\n✅ Документ про RS.SCM находится на позиции {rs_index + 1}")
        if rs_index == 0:
            print("   Это означает, что rerank корректно определил релевантность!")
    
    return reranked

def test_hybrid_rerank():
    """Тестируем HybridReranker"""
    
    print("\n" + "="*50)
    print("Тест HybridReranker")
    
    reranker = HybridReranker()
    
    documents = [
        {"page_content": "Python это язык программирования", "metadata": {"score": 0.8}},
        {"page_content": "RS SCM система управления поставками ритейл", "metadata": {"score": 0.9}},
        {"page_content": "JavaScript для веб разработки", "metadata": {"score": 0.7}},
    ]
    
    query = "RS SCM"
    
    reranked = reranker.rerank_documents(query, documents, top_k=3)
    
    print(f"Результаты после hybrid rerank:")
    for i, doc in enumerate(reranked):
        print(f"  {i+1}. {doc['page_content']}")
    
    return reranked

if __name__ == "__main__":
    try:
        reranked_docs = test_rerank_working()
        test_hybrid_rerank()
        
        print("\n" + "="*50)
        print("Как работает rerank:")
        print("1. Cross-encoder модель анализирует пару 'запрос-документ'")
        print("2. Каждый документ получает новую оценку релевантности")
        print("3. Документы сортируются по новым оценкам")
        print("4. Наиболее релевантные документы поднимаются вверх")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
