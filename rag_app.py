import sys
import logging
import json
import time
import os
from typing import List, Dict, Any, Tuple, Optional, Annotated, TypedDict

import gradio as gr
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# LangChain imports
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.schema.runnable import RunnableConfig, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# LangGraph imports
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# LangSmith tracing
from langsmith import Client, trace

# Local imports
from prompts import SYSTEM_PROMPT
from data_processing import RoSBERTaEmbeddings, Config

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Константы
COLLECTION_NAME = "documents"
MAX_CONTEXT_CHUNKS = 8  # Максимальное количество чанков для контекста
MAX_HISTORY_LENGTH = 10  # Максимальное количество сообщений в истории
RETRIEVAL_ERROR_MESSAGE = "Извините, эта информация временно недоступна. Уточните детали у менеджера"
MODEL_PATH = 'model/ru-en-RoSBERTa'

# Настройка LangSmith трейсинга
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "rag_assistant")

# Инициализация компонентов
# Загружаем модель для эмбеддингов из data_processing.py
# Так как мы уже загрузили модель в data_processing.py, мы используем ее напрямую
from data_processing import EMBEDDING_MODEL

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
                    results = self.ensemble_retriever.get_relevant_documents(query)
                    logger.info(f"Получено {len(results)} документов через ретривер")
                    
                    # Проверка и отладка - показываем первый документ, если есть
                    if results and len(results) > 0:
                        first_doc = results[0]
                        logger.info(f"Первый результат: \nсодержимое: '{first_doc.page_content[:100]}...'\nметаданные: {first_doc.metadata}")
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

class LangChainAssistant:
    """Класс для работы с LLM моделью через LangChain"""
    
    def __init__(self, model_path: str = "model/T-lite-it-1.0-Q4_K_M-GGUF/t-lite-it-1.0-q4_k_m.gguf"):
        self.model_path = model_path
        self.llm = None
        self._load_model()
        logger.info(f"Инициализирован LangChain ассистент с моделью: {model_path}")
    
    def _load_model(self):
        """Загружает модель через LangChain LLAMACPP интеграцию"""
        try:
            # Инициализируем модель через LangChain
            self.llm = LlamaCpp(
                model_path=self.model_path,
                temperature=0.4,
                max_tokens=768,
                top_p=0.9,
                stop=["</s>", "<|im_end|>"],
                verbose=False,
                n_ctx=2048,
                n_threads=8,
                n_gpu_layers=0
            )
            logger.info("Модель успешно загружена через LangChain")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели через LangChain: {str(e)}")
            raise RuntimeError(f"Не удалось загрузить модель: {str(e)}")
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Генерирует ответ от языковой модели с использованием LangChain
        
        Args:
            messages: Список сообщений в формате {"role": "...", "content": "..."}
            
        Returns:
            Сгенерированный ответ модели
        """
        try:
            # Используем trace внутри функции вместо декоратора
            with trace(name="generate_response"):
                if not self.llm:
                    logger.error("Попытка генерации ответа при незагруженной модели")
                    raise RuntimeError("Модель не загружена")
                
                logger.info("Получен запрос на генерацию ответа")
                logger.debug(f"Входные сообщения: {messages}")
                
                # Форматируем сообщения в промпт
                prompt = self._format_messages(messages)
                logger.debug(f"Сформированный промпт: {prompt[:200]}...")  # Логируем начало промпта
                
                # Генерируем ответ через LangChain
                logger.info("Запуск генерации ответа через LangChain...")
                start_time = time.time()
                
                # Используем вызов LangChain
                # Создаем конфигурацию прямо, а не как контекстный менеджер
                config = {
                    "callbacks": None,
                    "run_name": "generate_response"
                }
                generated_text = self.llm.invoke(prompt, config=config)
                
                end_time = time.time()
                logger.info(f"Ответ сгенерирован за {end_time - start_time:.2f} секунд")
                logger.debug(f"Сгенерированный ответ: {generated_text[:200]}...")  # Логируем начало ответа
                
                return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {str(e)}", exc_info=True)
            return f"Произошла ошибка при генерации ответа: {str(e)}"
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Форматирует историю сообщений в промпт для модели"""
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                formatted.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                formatted.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        # Добавляем приглашение для модели
        formatted.append("<|im_start|>assistant\n")
        return "\n".join(formatted)

# Определяем тип состояния для LangGraph
class RAGState(TypedDict):
    """State for the RAG workflow"""
    question: str
    chat_history: List[List[str]]
    context: List[Document]
    formatted_context: str
    answer: Optional[str] = None
    sources: List[Dict] = []

class RAGAssistant:
    """
    Основной класс для системы RAG с использованием LangChain и LangGraph
    Интегрирует гибридный поиск и LLM для ответов на вопросы
    """
    
    def __init__(self):
        logger.info("Инициализация RAGAssistant с LangGraph...")
        try:
            # Инициализируем компоненты
            self.search_engine = HybridSearch()
            logger.info("Инициализирован поисковый движок LangChain")
            
            logger.info("Загрузка LLM модели через LangChain...")
            self.assistant = LangChainAssistant()
            logger.info("LLM модель успешно загружена")
            
            # Создаем LangGraph для процесса RAG
            self.graph = self._create_rag_graph()
            
            self.feedback_data = []  # Для хранения обратной связи
            logger.info("RAGAssistant успешно инициализирован с LangChain и LangGraph")
        except Exception as e:
            logger.error(f"Ошибка при инициализации RAGAssistant: {str(e)}", exc_info=True)
            raise
    
    def _create_rag_graph(self):
        """
        Создает граф LangGraph для процесса Retrieval-Augmented Generation
        """
        # Создаем новый граф
        workflow = StateGraph(RAGState)
        
        # Определяем узлы графа
        
        # 1. Поиск релевантных документов
        def retrieve_documents(state: RAGState) -> RAGState:
            try:
                logger.info(f"Выполняется поиск по запросу: {state['question']}")
                # Используем trace внутри функции
                with trace(name="retrieve_documents"):
                    search_results = self.search_engine.search(state['question'])
                
                if not search_results:
                    logger.warning("Не найдены релевантные документы")
                    # Преобразуем в формат Document для совместимости
                    state['context'] = []
                    state['formatted_context'] = ""
                    state['sources'] = []
                    return state
                
                # Форматируем результаты поиска
                formatted_context, sources = self.format_search_results(search_results)
                
                # Преобразуем в формат Document для LangChain
                documents = []
                for source in sources:
                    doc = Document(
                        page_content=source['text'],
                        metadata=source['metadata']
                    )
                    documents.append(doc)
                
                state['context'] = documents
                state['formatted_context'] = formatted_context
                state['sources'] = sources
                
                logger.info(f"Найдено {len(documents)} релевантных документов")
                return state
            except Exception as e:
                logger.error(f"Ошибка при поиске документов: {str(e)}")
                # В случае ошибки продолжаем с пустым контекстом
                state['context'] = []
                state['formatted_context'] = ""
                state['sources'] = []
                return state
        
        # 2. Генерация ответа на основе контекста
        def generate_answer(state: RAGState) -> RAGState:
            try:
                # Проверяем, есть ли контекст
                if not state['context']:
                    state['answer'] = RETRIEVAL_ERROR_MESSAGE
                    return state
                    
                # Создаем системное сообщение с контекстом и вопросом
                system_message = {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(
                        context=state['formatted_context'], 
                        question=state['question']
                    )
                }
                messages = [system_message]
                
                # Добавляем историю диалога (не больше MAX_HISTORY_LENGTH сообщений)
                for user_msg, assistant_msg in state['chat_history'][-MAX_HISTORY_LENGTH:]:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})
                
                # Добавляем текущий запрос
                messages.append({"role": "user", "content": state['question']})
                
                # Генерируем ответ
                logger.info("Генерация ответа...")
                # Используем trace внутри функции
                with trace(name="generate_answer"):
                    response = self.assistant.generate_response(messages)
                
                state['answer'] = response
                return state
                
            except Exception as e:
                logger.error(f"Ошибка при генерации ответа: {str(e)}")
                state['answer'] = f"Произошла ошибка при обработке запроса: {str(e)}"
                return state
                
        # Добавляем узлы в граф
        workflow.add_node("retrieve", retrieve_documents)
        workflow.add_node("generate", generate_answer)
        
        # Определяем порядок выполнения
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # Создаем и компилируем граф
        graph = workflow.compile()
        logger.info("Создан и скомпилирован LangGraph для RAG")
        return graph
    
    def format_search_results(self, results: List) -> Tuple[str, List[Dict]]:
        """
        Форматирует результаты поиска в контекст для модели
        
        Args:
            results: Список объектов ScoredPoint из Qdrant
            
        Returns:
            Кортеж (форматированный контекст, список чанков с метаданными)
        """
        context_chunks = []
        formatted_context = ""
        
        for i, result in enumerate(results[:MAX_CONTEXT_CHUNKS]):
            try:
                # Получаем атрибуты из ScoredPoint
                score = getattr(result, 'score', 0.0)
                payload = getattr(result, 'payload', {})
                
                # Извлекаем текст и метаданные
                text = payload.get('text', '')
                metadata = {k: v for k, v in payload.items() if k != 'text'}
                
                # Добавляем в контекст для модели
                if text:  # Добавляем только если есть текст
                    chunk_context = f"[Документ {i+1}] {text}\n"
                    formatted_context += chunk_context
                
                # Сохраняем для отображения источников
                context_chunks.append({
                    'text': text,
                    'metadata': metadata,
                    'score': float(score),  # Преобразуем в стандартный float
                    'source_type': 'hybrid'  # Используем гибридный поиск
                })
                
            except Exception as e:
                logger.error(f"Ошибка при обработке результата поиска: {str(e)}", exc_info=True)
                continue
        
        return formatted_context, context_chunks

    def answer_query(self, query: str, history: List[List[str]]) -> Tuple[str, List[Dict]]:
        """
        Отвечает на вопрос с использованием LangGraph RAG процесса
        
        Args:
            query: Текущий запрос пользователя
            history: История диалога в формате [[user_msg1, assistant_msg1], [user_msg2, assistant_msg2], ...]
            
        Returns:
            Кортеж (ответ, список источников)
        """
        try:
            logger.info(f"Обработка запроса: {query}")
            
            # Создаем начальное состояние для LangGraph
            initial_state = {
                "question": query,
                "chat_history": history,
                "context": [],
                "formatted_context": "",
                "answer": None,
                "sources": []
            }
            
            # Создаем конфигурацию для трейсинга в LangSmith
            config = {}
            if LANGCHAIN_API_KEY:
                config = {
                    "configurable": {
                        "project_name": LANGCHAIN_PROJECT,
                        "tags": ["rag", "production"]
                    }
                }
            
            # Запускаем LangGraph
            logger.info("Запуск LangGraph RAG процесса...")
            # Используем trace внутри функции вместо декоратора
            with trace(name="answer_query"):
                final_state = self.graph.invoke(initial_state, config=config)
            
            # Получаем результаты
            answer = final_state.get("answer", RETRIEVAL_ERROR_MESSAGE)
            sources = final_state.get("sources", [])
            
            return answer, sources
        
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса через LangGraph: {str(e)}", exc_info=True)
            return f"Произошла ошибка при обработке запроса: {str(e)}", []
    
    def save_feedback(self, query: str, response: str, rating: int, comments: str = ""):
        """
        Сохраняет обратную связь пользователя
        """
        logger.info(f"Получен отзыв: оценка={rating}, комментарий={comments}")
        
        feedback = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "response": response,
            "rating": rating,
            "comments": comments
        }
        
        self.feedback_data.append(feedback)
        
        # Сохраняем в файл
        try:
            with open("feedback_data.json", "w", encoding="utf-8") as f:
                json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Отзыв успешно сохранен. Рейтинг: {rating}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении отзыва: {str(e)}")
            return False

def chat_with_feedback(message, history):
    """
    Обрабатывает взаимодействие пользователя с чатом
    и возвращает ответ с информацией об источниках
    """
    logger.info(f"Получен запрос от пользователя: {message[:100]}...")
    logger.debug(f"Текущая история чата (до обработки): {history}")
    
    if not message or message.strip() == "":
        logger.warning("Получен пустой запрос от пользователя")
        return "", history, "Пожалуйста, введите ваш вопрос."
    
    try:
        # Convert history to the old format for the assistant
        old_format_history = []
        for i in range(0, len(history), 2):
            if i + 1 < len(history):
                old_format_history.append([history[i]["content"], history[i+1]["content"]])
        
        # Get response and sources
        response, sources = rag_assistant.answer_query(message, old_format_history)
        
        # Format sources for display
        sources_html = format_source_display(sources)
        
        # Add user message and assistant response to history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        logger.debug("Запрос пользователя и ответ ассистента добавлены в историю")
        
        # Ограничиваем длину истории
        if len(history) > MAX_HISTORY_LENGTH * 2:  # Умножаем на 2, так как каждая пара вопрос-ответ - это 2 элемента
            removed_count = len(history) - (MAX_HISTORY_LENGTH * 2)
            history = history[-MAX_HISTORY_LENGTH * 2:]
            logger.debug(f"История чата обрезана. Удалено {removed_count} старых сообщений")
        
        logger.info("Запрос успешно обработан")
        return "", history, sources_html
    
    except Exception as e:
        error_msg = f"Ошибка при обработке запроса: {str(e)}"
        logger.error(error_msg, exc_info=True)
        history.append({"role": "assistant", "content": "Извините, произошла ошибка при обработке вашего запроса."})
        return "", history, ""

def format_source_display(sources: List[Dict[str, Any]]) -> str:
    """Форматирует список источников в HTML для отображения
    
    Args:
        sources: Список словарей с информацией об источниках
        
    Returns:
        HTML-строка с отформатированными источниками
    """
    if not sources:
        return "<div>Источники не найдены</div>"
    
    html_parts = ["<div style='margin-top: 20px;'><h4>Источники:</h4><div style='padding-left: 20px;'>"]
    logger.info(f"Количество источников: {len(sources)}")
    logger.info(f"Источники: {sources}")
    for i, source in enumerate(sources[:5]):  # Ограничиваем количество отображаемых источников
        # Извлекаем метаданные и текст, если они есть
        metadata = source.get('metadata', {})
        source_name = metadata.get('source', 'Неизвестный источник')
        page = metadata.get('page', '')
        text = source.get('text', '')[:100]  # Берем первые 100 символов текста
        
        # Формируем отображаемое имя источника
        display_name = f"{source_name}"
        if page:
            display_name += f" (стр. {page})"
        
        # Добавляем ссылку, если есть
        link_start = f"<a href='{metadata['url']}' target='_blank'>" if 'url' in metadata else ""
        link_end = "</a>" if 'url' in metadata else ""
        
        # Формируем элемент списка с текстом
        html_parts.append(f"""
        <div>
            <div>{link_start}{display_name}{link_end}</div>
            <div style='color: #666; font-size: 0.9em; margin-top: 3px;'>{text}...</div>
        </div>
        """)
    
    html_parts.append("</div>")
    return "\n".join(html_parts)

def clear_chat():
    """Очищает историю чата и возвращает пустые значения для всех элементов"""
    logger.info("Очистка истории чата")
    return [], "", ""  # Пустые значения для chat_history, sources_display, feedback_status

def submit_feedback(rating, comments, history):
    """Отправляет обратную связь о последнем ответе"""
    logger.info(f"Получен отзыв: оценка={rating}, комментарий={comments}")
    
    if not history or len(history) < 2:
        logger.warning("Не удалось сохранить отзыв: история сообщений пуста или недостаточно сообщений")
        return "Не удалось сохранить отзыв: история сообщений пуста"
    
    try:
        # Get the last user message and assistant response
        last_query = history[-2]["content"]  # Second to last is user message
        last_response = history[-1]["content"]  # Last is assistant response
        success = rag_assistant.save_feedback(last_query, last_response, rating, comments)
        
        if success:
            logger.info("Отзыв успешно сохранен")
            return f"Спасибо за вашу оценку: {rating}! Ваш отзыв поможет улучшить систему."
        else:
            logger.error("Не удалось сохранить отзыв")
            return "Не удалось сохранить отзыв. Пожалуйста, попробуйте позже."
    
    except Exception as e:
        logger.error(f"Ошибка при сохранении отзыва: {str(e)}", exc_info=True)
        return "Произошла ошибка при сохранении отзыва. Пожалуйста, попробуйте позже."

def create_demo(rag_assistant):
    """Создает демонстрационный интерфейс Gradio"""
    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        gr.Markdown("# 🔍 Система вопросов и ответов с гибридным поиском")        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500, 
                    label="Диалог",
                    type="messages"  # Using the new messages format
                )
                msg = gr.Textbox(
                    placeholder="Введите ваш вопрос...",
                    label="Вопрос",
                    lines=2
                )
                with gr.Row():
                    submit_btn = gr.Button("Отправить", variant="primary")
                    clear_btn = gr.Button("Очистить")
            
            with gr.Column(scale=2):
                sources_display = gr.HTML(label="Источники")
                with gr.Group():  # Replaced Box with Group
                    gr.Markdown("### Оцените ответ")
                    with gr.Row():
                        rating = gr.Slider(
                            minimum=1, 
                            maximum=5, 
                            step=1, 
                            value=3, 
                            label="Оценка"
                        )
                    comments = gr.Textbox(
                        placeholder="Дополнительные комментарии...", 
                        label="Комментарии"
                    )
                    feedback_btn = gr.Button("Отправить отзыв")
                    feedback_status = gr.Markdown()
        
        # Обработчики событий
        submit_btn.click(
            chat_with_feedback, 
            inputs=[msg, chatbot], 
            outputs=[msg, chatbot, sources_display]
        )
        
        msg.submit(
            chat_with_feedback, 
            inputs=[msg, chatbot], 
            outputs=[msg, chatbot, sources_display]
        )
        
        clear_btn.click(
            clear_chat, 
            outputs=[chatbot, sources_display, feedback_status]
        )
        
        feedback_btn.click(
            submit_feedback,
            inputs=[rating, comments, chatbot],
            outputs=[feedback_status]
        )
        
    return demo

def get_embeddings_with_prefix(texts, task="search_query"):
    """
    Generate embeddings with proper task prefix for RoSBERTa
    
    Args:
        texts: Text or list of texts to embed
        task: Task type ('search_query', 'search_document', etc.)
        
    Returns:
        List of embeddings
    """
    if isinstance(texts, str):
        texts = [texts]
    
    # Add appropriate prefix based on task
    prefixed_texts = [f"{task}: {text}" for text in texts]
    
    # Log first few characters of the first text for debugging
    sample = prefixed_texts[0][:50] + '...' if len(prefixed_texts[0]) > 50 else prefixed_texts[0]
    logger.debug(f"Generating embeddings for text: {sample}")
    
    try:
        # Get embeddings with proper configuration
        embeddings = EMBEDDING_MODEL.encode(
            prefixed_texts,
            normalize_embeddings=True,
            convert_to_numpy=False,
            show_progress_bar=False,
            batch_size=32
        )
        
        # Convert to list if it's a single embedding
        if len(embeddings) == 1 and isinstance(texts, str):
            return embeddings[0]
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def get_hybrid_search(query, k=5):
    """
    Perform hybrid search using Qdrant's built-in hybrid search capabilities.
    Combines vector similarity with keyword matching in a single query.
    Performs case-insensitive search by converting both query and text to lowercase.
    
    Args:
        query: Search query string
        k: Number of results to return
        
    Returns:
        List of search results with combined scores
    """
    try:
        client = QdrantClient(host="localhost", port=6333)
        logger.info(f"Performing hybrid search for query: {query}")
        
        # Generate query embedding
        try:
            query_embedding = get_embeddings_with_prefix(query, task="search_query")[0]
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return []
        
        try:
            # Convert query to lowercase for case-insensitive search
            query_lower = query.lower()
            
            # Split query into words for better matching
            query_terms = [term for term in query_lower.split() if len(term) > 2]  # Only terms longer than 2 chars
            
            # If no valid query terms, just do vector search
            if not query_terms:
                search_results = client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_embedding,
                    limit=k,
                    with_payload=True,
                    score_threshold=0.3
                )
                return search_results
            
            # Get more results than needed to account for filtering
            search_results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=k*3,  # Get more results initially to filter
                with_payload=True,
                score_threshold=0.3  # Minimum similarity score
            )
            
            # Filter results case-insensitively
            filtered_results = []
            for result in search_results:
                text = result.payload.get('text', '').lower()
                # Check if all query terms are in the text (case-insensitive)
                if all(term in text for term in query_terms):
                    filtered_results.append(result)
                    if len(filtered_results) >= k:
                        break
                        
            # If no results after filtering, return the top vector results
            if not filtered_results and search_results:
                return search_results[:k]
                
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error during hybrid search: {str(e)}", exc_info=True)
            # Fallback to simple vector search if hybrid search fails
            try:
                search_results = client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_embedding,
                    limit=k,
                    with_payload=True,
                    score_threshold=0.3
                )
                logger.info(f"Falling back to vector search, found {len(search_results)} results")
                return search_results
            except Exception as e2:
                logger.error(f"Error in fallback vector search: {str(e2)}")
                return []
                
    except Exception as e:
        logger.error(f"Error initializing Qdrant client: {str(e)}")
        return []

# Запуск приложения
if __name__ == "__main__":
    try:
        # Проверка соединения с Qdrant
        client = QdrantClient(host="localhost", port=6333)
        client.get_collection(COLLECTION_NAME)
        logger.info(f"Успешное подключение к коллекции {COLLECTION_NAME}")
        
        # Создаем экземпляр ассистента
        rag_assistant = RAGAssistant()
        
        # Запуск интерфейса Gradio
        demo = create_demo(rag_assistant)
        demo.launch(server_name="0.0.0.0", share=False)
    
    except Exception as e:
        logger.error(f"Ошибка при запуске приложения: {str(e)}")
        print(f"Ошибка при запуске приложения: {str(e)}")
        sys.exit(1)
