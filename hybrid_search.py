from constants import *
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import Qdrant
from data_processing import RoSBERTaEmbeddings, Config, EMBEDDING_MODEL
import logging
import time

# LangSmith tracing
from langsmith import trace

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridSearch:
    """Класс для выполнения гибридного поиска с использованием LangChain EnsembleRetriever"""
    
    def __init__(self, collection_name: str = COLLECTION_NAME):
        """
        Инициализация гибридного поиска с использованием LangChain
        
        Args:
            collection_name: Имя коллекции в Qdrant
        """
        self.client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)
        
        # Инициализируем обертку эмбеддингов для LangChain
        self.embeddings = RoSBERTaEmbeddings(EMBEDDING_MODEL)
        
        # Создаем векторный поиск на основе Qdrant
        try:
            # Используем векторное хранилище через LangChain
            self.vector_store = Qdrant(
                client=self.client,
                collection_name=collection_name,
                embeddings=self.embeddings,
                vector_name=None
            )
            
            # Создаем векторный ретривер
            self.vector_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": MAX_CONTEXT_CHUNKS, "score_threshold": 0.5}
            )
            
            # Получаем все документы для построения BM25 индекса
            all_docs = []
            try:
                # Проверяем информацию о коллекции
                try:
                    collection_info = self.client.get_collection(collection_name=collection_name)
                    logger.info(f"Информация о коллекции {collection_name}: размер = {collection_info.vectors_count}")
                except Exception as e:
                    logger.error(f"Ошибка при получении информации о коллекции: {str(e)}")

                # Получаем до 10000 документов для построения индекса
                logger.info(f"Попытка загрузки документов из коллекции {collection_name}")
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    limit=10000,
                    with_payload=True,
                    with_vectors=False
                )
                
                # Проверяем результат
                logger.info(f"Результат scroll: тип = {type(scroll_result)}, длина = {len(scroll_result)}")
                
                if len(scroll_result) > 0 and len(scroll_result[0]) > 0:
                    # Проверяем первый документ
                    first_point = scroll_result[0][0]
                    logger.info(f"Первый документ: id={first_point.id}, ключи в payload: {list(first_point.payload.keys() if first_point.payload else [])}")
                
                points = scroll_result[0]
                
                # Конвертируем в формат документов LangChain
                for point in points:
                    try:
                        # Поддержка обоих форматов документов
                        content = None
                        metadata = {}
                        
                        if not point.payload:
                            logger.warning(f"Point {point.id} не имеет payload")
                            continue
                            
                        # Вариант 1: контент в 'text'
                        if 'text' in point.payload:
                            content = point.payload['text']
                            metadata = {k: v for k, v in point.payload.items() if k != 'text'}
                            
                        # Вариант 2: контент в 'page_content'
                        elif 'page_content' in point.payload:
                            content = point.payload['page_content']
                            # Проверяем метаданные в поле 'metadata'
                            if 'metadata' in point.payload and isinstance(point.payload['metadata'], dict):
                                metadata = point.payload['metadata']
                            else:
                                # Все остальные поля как метаданные
                                metadata = {k: v for k, v in point.payload.items() if k != 'page_content'}
                            
                        # Создаем документ, если нашли контент
                        if content:
                            doc = Document(
                                page_content=content,
                                metadata=metadata
                            )
                            all_docs.append(doc)
                        else:
                            logger.warning(f"Не найден контент в документе {point.id}. Ключи: {list(point.payload.keys())}")
                    except Exception as doc_error:
                        logger.error(f"Ошибка при обработке документа: {str(doc_error)}")
                        continue
                
                logger.info(f"Загружено {len(all_docs)} документов для BM25 индекса")
                
                # Отладка - показываем первый документ, если есть
                if all_docs and len(all_docs) > 0:
                    first_doc = all_docs[0]
                    logger.info(f"Первый документ для BM25: \nсодержимое: '{first_doc.page_content[:100]}...'\nметаданные: {first_doc.metadata}")
                    logger.debug("Найдены следующие первые 5 документов:")
                    for i, doc in enumerate(all_docs[:5]):
                        logger.debug(f"Doc {i+1}: {doc.page_content[:50]}...")
            except Exception as e:
                logger.error(f"Ошибка при загрузке документов для BM25: {str(e)}")
                all_docs = []
            
            # Создаем BM25 ретривер
            if all_docs:
                self.bm25_retriever = BM25Retriever.from_documents(all_docs)
                self.bm25_retriever.k = 8
                
                # Создаем ансамбль для гибридного поиска
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.vector_retriever, self.bm25_retriever],
                    weights=[0.7, 0.3]
                )
                logger.info("Инициализирован гибридный поиск (Qdrant + BM25)")
            else:
                # Если не удалось загрузить документы, используем только векторный поиск
                self.ensemble_retriever = self.vector_retriever
                logger.info("Инициализирован только векторный поиск (без BM25)")
                
        except Exception as e:
            logger.error(f"Ошибка при инициализации поиска: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Выполняет гибридный поиск, используя LangChain EnsembleRetriever
        
        Args:
            query: Поисковый запрос
            k: Количество результатов
            
        Returns:
            Список релевантных документов с метаданными и оценками
        """
        start_time = time.time()
        try:
            # Проверяем наличие запроса
            if not query or query.strip() == "":
                logger.warning("Пустой запрос, поиск не выполняется")
                return []
                
            logger.info(f"Выполняем поиск для запроса: '{query}'")
            
            # Выполняем поиск через LangChain retriever
            # Используем trace внутри функции вместо декоратора
            with trace(name="search"):
                try:
                    results = self.ensemble_retriever.invoke(query)
                    logger.info(f"Получено {len(results)} документов через ретривер")
                    
                    # Проверка и отладка - показываем первый документ, если есть
                    if results and len(results) > 0:
                        first_doc = results[0]
                        # Адаптация к новой структуре документов
                        if hasattr(first_doc, 'page_content'):
                            content = first_doc.page_content
                            metadata = first_doc.metadata if hasattr(first_doc, 'metadata') else {}
                        elif hasattr(first_doc, 'text'):
                            content = first_doc.text
                            metadata = first_doc.metadata if hasattr(first_doc, 'metadata') else {}
                        else:
                            content = str(first_doc)
                            metadata = getattr(first_doc, 'metadata', {})
                        logger.info(f"Первый результат: \nсодержимое: '{content[:100]}...'\nметаданные: {metadata}")
                except Exception as search_error:
                    logger.error(f"Ошибка при поиске: {str(search_error)}")
                    
                    # Пробуем прямой поиск через Qdrant
                    logger.info("Попытка прямого поиска через Qdrant...")
                    try:
                        # Преобразуем запрос в эмбеддинг
                        query_vector = self.embeddings.embed_query(query)
                        
                        # Выполняем прямой поиск через Qdrant
                        search_result = self.client.search(
                            collection_name=COLLECTION_NAME,
                            query_vector=query_vector,
                            limit=k,
                            with_payload=True
                        )
                        logger.info(f"Прямой поиск через Qdrant вернул {len(search_result)} результатов")
                        
                        # Проверяем формат результатов
                        if search_result and len(search_result) > 0:
                            first = search_result[0]
                            if hasattr(first, 'payload'):
                                logger.info(f"Ключи в payload первого результата: {list(first.payload.keys())}")
                        
                        # Преобразуем в формат LangChain Document
                        results = []
                        for point in search_result:
                            if point.payload:
                                content = None
                                metadata = {}
                                
                                # Поддержка обоих форматов
                                if 'text' in point.payload:
                                    content = point.payload['text']
                                    metadata = {k: v for k, v in point.payload.items() if k != 'text'}
                                elif 'page_content' in point.payload:
                                    content = point.payload['page_content']
                                    if 'metadata' in point.payload and isinstance(point.payload['metadata'], dict):
                                        metadata = point.payload['metadata']
                                    else:
                                        metadata = {k: v for k, v in point.payload.items() if k != 'page_content'}
                                
                                if content:
                                    doc = Document(
                                        page_content=content,
                                        metadata=metadata
                                    )
                                    results.append(doc)
                    except Exception as qdrant_error:
                        logger.error(f"Ошибка при прямом поиске через Qdrant: {str(qdrant_error)}")
                        results = []
            
            # Преобразуем результаты в формат, совместимый с предыдущей версией
            processed_results = []
            for i, doc in enumerate(results[:k]):
                # Создаем объект, имитирующий ScoredPoint из Qdrant
                # Сохраняем формат оригинального документа
                payload = {
                    'page_content': doc.page_content  # Используем оригинальное поле
                }
                
                # Метаданные могут быть как отдельным полем, так и на уровне payload
                if hasattr(doc, 'metadata') and doc.metadata:
                    payload['metadata'] = doc.metadata
                
                # Дублируем и в text для обратной совместимости
                payload['text'] = doc.page_content
                
                result = type('ScoredPoint', (), {
                    'score': 1.0 - (i * 0.1),  # Имитация оценки от 1.0 до 0.5
                    'payload': payload,
                    'id': f"result_{i}"  # Добавляем ID для удобства отладки
                })
                processed_results.append(result)
            
            # Показываем первый результат для отладки
            if processed_results and len(processed_results) > 0:
                first_result = processed_results[0]
                logger.info(f"Первый обработанный результат: id={first_result.id}, ключи в payload: {list(first_result.payload.keys())}")
                
                # Показываем содержимое
                if 'page_content' in first_result.payload:
                    logger.info(f"page_content: {first_result.payload['page_content'][:100]}...")
                elif 'text' in first_result.payload:
                    logger.info(f"text: {first_result.payload['text'][:100]}...")
            
            logger.info(f"Найдено {len(processed_results)} релевантных документов за {time.time() - start_time:.2f} сек")
            return processed_results
        except Exception as e:
            logger.error(f"Ошибка при поиске: {str(e)}")
            return []