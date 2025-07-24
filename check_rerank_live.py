#!/usr/bin/env python3
"""
Проверка работы rerank в реальном времени
"""

import sys
import os

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hybrid_search import HybridSearch
from reranker import DocumentReranker

def test_rerank_flow():
    """Тест показывает полный поток работы rerank"""
    
    print("🔍 Проверка работы rerank в реальном времени\n")
    
    # Создаем search с включенным rerank
    search = HybridSearch(
        collection_name="documents",
        enable_reranking=True,
        reranker_type="cross_encoder"
    )
    
    # Проверяем, что reranker инициализирован
    if search.enable_reranking and search.reranker:
        print("✅ Reranker успешно инициализирован")
        print(f"   Тип реранкера: {type(search.reranker).__name__}")
    else:
        print("❌ Reranker не инициализирован")
        return
    
    # Тестовый запрос
    test_query = "RS SCM система управления поставками"
    
    print(f"\n📋 Тестовый запрос: '{test_query}'")
    print("🔄 Выполняется поиск с rerank...")
    
    try:
        # Выполняем поиск - rerank будет применен автоматически
        results = search.search(test_query, k=5)
        
        print(f"✅ Найдено результатов: {len(results)}")
        
        if results:
            print("\n📊 Результаты после rerank:")
            for i, result in enumerate(results[:3]):
                content = getattr(result, 'page_content', str(result))
                score = getattr(result, 'score', 'N/A')
                
                print(f"  {i+1}. {content[:80]}...")
                print(f"     Score: {score}")
                
                # Проверяем наличие RS/SCM
                if any(keyword in content.upper() for keyword in ['RS', 'SCM', 'СИСТЕМА', 'ПОСТАВКИ']):
                    print("     🎯 Ключевые слова найдены!")
        
    except Exception as e:
        print(f"❌ Ошибка при поиске: {e}")
        import traceback
        traceback.print_exc()

def test_rerank_vs_no_rerank():
    """Сравнение результатов с rerank и без rerank"""
    
    print("\n" + "="*60)
    print("Сравнение: с rerank vs без rerank")
    
    query = "RS SCM система управления поставками"
    
    # Поиск без rerank
    search_no_rerank = HybridSearch(
        collection_name="documents",
        enable_reranking=False
    )
    
    # Поиск с rerank
    search_with_rerank = HybridSearch(
        collection_name="documents",
        enable_reranking=True,
        reranker_type="cross_encoder"
    )
    
    print(f"\nЗапрос: '{query}'")
    
    try:
        results_no_rerank = search_no_rerank.search(query, k=3)
        results_with_rerank = search_with_rerank.search(query, k=3)
        
        print(f"\nБез rerank ({len(results_no_rerank)} результатов):")
        for i, result in enumerate(results_no_rerank):
            content = getattr(result, 'page_content', str(result))
            print(f"  {i+1}. {content[:60]}...")
        
        print(f"\nС rerank ({len(results_with_rerank)} результатов):")
        for i, result in enumerate(results_with_rerank):
            content = getattr(result, 'page_content', str(result))
            score = getattr(result, 'score', 'N/A')
            print(f"  {i+1}. {content[:60]}... (score: {score})")
            
    except Exception as e:
        print(f"Ошибка при сравнении: {e}")

if __name__ == "__main__":
    test_rerank_flow()
    test_rerank_vs_no_rerank()
    
    print("\n" + "="*60)
    print("📋 Краткое объяснение:")
    print("1. Rerank создается при старте приложения")
    print("2. Применяется каждый раз при поиске")
    print("3. Не сохраняет результаты - пересчитывает каждый раз")
    print("4. Работает в реальном времени")
