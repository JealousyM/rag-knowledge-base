import sys
import logging
import json
import os
import time
from typing import List, Dict, Union, Any, Optional, Tuple, Annotated, TypedDict
import uuid
import re
import numpy as np
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# LangChain imports
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, AIMessage
from langchain.schema.runnable import RunnableConfig, RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.agents import AgentType, initialize_agent, create_react_agent, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages, format_to_openai_functions
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.tools.convert_to_openai import format_tool_to_openai_function

# LangGraph imports
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# LangSmith tracing
from langsmith import Client, trace

# Local imports
from prompts import SYSTEM_PROMPT
from data_processing import RoSBERTaEmbeddings, Config

# Импорт класса для работы с Oracle Text2SQL
from oracle_text2sql import OracleText2SQL
from sql_tool import set_oracle_tool, get_oracle_tool

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Константы
COLLECTION_NAME = "documents"
MAX_CONTEXT_CHUNKS = 8  # Максимальное количество чанков для контекста
LOADING_WEIGHTS_STR = "⚙️ Загружаю модель... это может занять несколько минут"
MAX_HISTORY_LENGTH = 10  # Максимальное количество сообщений в истории
LAST_RUN_ID = None  # Идентификатор последнего запуска в LangSmith для отзывов
RETRIEVAL_ERROR_MESSAGE = "Извините, эта информация временно недоступна. Уточните детали у менеджера"
MODEL_PATH = 'model/ru-en-RoSBERTa'

# Для управления инструментом Oracle Text2SQL используется модуль sql_tool

# Индикатор для отслеживания попыток доступа к БД
DATABASE_KEYWORDS = ['база данных', 'бд', 'запрос', 'sql', 'oracle', 'таблиц', 'запис', 'столбц', 'выбери', 'найди в', 'покажи из']

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

class LangChainAssistant:
    """Класс для работы с LLM моделью через LangChain"""
    
    # Доступные типы моделей
    MODEL_TYPE_OPENAI = "openai"
    MODEL_TYPE_LOCAL = "local"
    
    # Доступные модели
    AVAILABLE_MODELS = {
        MODEL_TYPE_OPENAI: [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo"
        ],
        MODEL_TYPE_LOCAL: [
            "model/T-lite-it-1.0-Q4_K_M-GGUF/t-lite-it-1.0-q4_k_m.gguf"
        ],
    }
    
    # Параметры модели по умолчанию
    model_params = {
        "temperature": 0.4,
        "max_tokens": 768,
        "top_p": 0.9,
        "verbose": True,
        "n_ctx": 2048,
        "n_threads": 8,
        "n_gpu_layers": 0
    }
    
    def __init__(self, model_type: str = MODEL_TYPE_OPENAI, model_name: str = "gpt-3.5-turbo"):
        # Устанавливаем тип модели (локальная или OpenAI)
        self.model_type = model_type
        
        # По умолчанию теперь используется OpenAI gpt-3.5-turbo
        self.model_name = model_name
        
        self.llm = None
        self._load_model()
        logger.info(f"Инициализирован LangChain ассистент с моделью: {self.model_name}, тип: {self.model_type}")
    
    def _load_model(self):
        """Загружает модель через LangChain в зависимости от выбранного типа"""
        try:
            # Загружаем модель в зависимости от ее типа
            if self.model_type == self.MODEL_TYPE_LOCAL:
                # Локальная модель LLamaCpp
                try:
                    from langchain_community.llms import LlamaCpp
                    
                    self.llm = LlamaCpp(
                        model_path=self.model_name,
                        temperature=self.model_params["temperature"],
                        max_tokens=self.model_params["max_tokens"],
                        top_p=self.model_params["top_p"],
                        stop=["</s>", "<|im_end|>"],
                        verbose=self.model_params["verbose"],
                        n_ctx=self.model_params["n_ctx"],
                        n_threads=self.model_params["n_threads"],
                        n_gpu_layers=self.model_params["n_gpu_layers"]
                    )
                    logger.info(f"Локальная модель {self.model_name} успешно загружена через LlamaCpp")
                except Exception as local_err:
                    logger.error(f"Ошибка при загрузке локальной модели: {str(local_err)}")
                    raise
                
            elif self.model_type == self.MODEL_TYPE_OPENAI:
                # OpenAI API модель
                try:
                    from langchain_openai import ChatOpenAI
                    
                    # Проверка, есть ли ключ API
                    api_key = os.environ.get("OPENAI_API_KEY")
                    if not api_key:
                        raise ValueError("Требуется OPENAI_API_KEY в переменных окружения")
                    
                    self.llm = ChatOpenAI(
                        model=self.model_name,
                        temperature=self.model_params["temperature"],
                        max_tokens=self.model_params["max_tokens"]
                        # Примечание: некоторые параметры неприменимы к OpenAI (n_ctx, n_threads, n_gpu_layers)
                    )
                    logger.info(f"OpenAI модель {self.model_name} успешно инициализирована")
                except ImportError:
                    logger.error("Не удалось импортировать langchain_openai. Установите пакет: pip install langchain-openai")
                    raise
                except Exception as openai_err:
                    logger.error(f"Ошибка при инициализации OpenAI API: {str(openai_err)}")
                    raise
            else:
                raise ValueError(f"Неизвестный тип модели: {self.model_type}")
                
            logger.info(f"Параметры модели: {self.model_params}")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {str(e)}")
            raise RuntimeError(f"Не удалось загрузить модель {self.model_name} типа {self.model_type}: {str(e)}")
    
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
                result = self.llm.invoke(prompt, config=config)
                
                # Обработка разных форматов возврата (для LlamaCpp и ChatOpenAI)
                if hasattr(result, 'content'):  # Это AIMessage из ChatOpenAI
                    generated_text = result.content
                else:  # Это строка из LlamaCpp
                    generated_text = result
                
                end_time = time.time()
                logger.info(f"Ответ сгенерирован за {end_time - start_time:.2f} секунд")
                logger.debug(f"Сгенерированный ответ: {str(generated_text)[:200]}...")  # Логируем начало ответа
                
                return generated_text.strip() if isinstance(generated_text, str) else generated_text
            
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
        
    def update_model_params(self, **kwargs):
        """Обновляет параметры модели и перезагружает её
        
        Args:
            model_type: Тип модели ('local' или 'openai')
            model_name: Название модели
            **kwargs: Другие именованные аргументы с новыми значениями параметров
            
        Returns:
            bool: True если модель была успешно перезагружена, False в случае ошибки
        """
        try:
            # Проверяем наличие указания новой модели
            model_changed = False
            
            # Проверяем тип модели
            if "model_type" in kwargs and kwargs["model_type"] in [self.MODEL_TYPE_LOCAL, self.MODEL_TYPE_OPENAI]:
                new_model_type = kwargs.pop("model_type")
                if new_model_type != self.model_type:
                    self.model_type = new_model_type
                    model_changed = True
                    logger.info(f"Тип модели изменен на: {self.model_type}")
                    
            # Проверяем название модели
            if "model_name" in kwargs:
                new_model_name = kwargs.pop("model_name")
                # Проверяем, есть ли такая модель в списке доступных
                if new_model_name in self.AVAILABLE_MODELS.get(self.model_type, []):
                    if new_model_name != self.model_name:
                        self.model_name = new_model_name
                        model_changed = True
                        logger.info(f"Название модели изменено на: {self.model_name}")
                else:
                    logger.warning(f"Модель {new_model_name} не найдена в списке доступных моделей типа {self.model_type}")
            
            # Обновляем другие параметры
            for param, value in kwargs.items():
                if param in self.model_params:
                    # Преобразуем типы для числовых параметров
                    if param in ["temperature", "top_p"]:
                        value = float(value)
                    elif param in ["max_tokens", "n_ctx", "n_threads", "n_gpu_layers"]:
                        value = int(value)
                    elif param == "verbose":
                        value = bool(value)
                    
                    if self.model_params[param] != value:
                        self.model_params[param] = value
                        model_changed = True
                        logger.info(f"Параметр {param} изменен на: {value}")
            
            # Перезагружаем модель только если были изменения
            if model_changed:
                logger.info(f"Перезагрузка модели с новыми параметрами")
                self._load_model()
                return True
            else:
                logger.info("Нет изменений в параметрах модели")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка при обновлении параметров модели: {str(e)}")
            return False

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
    
    def __init__(self, load_model=True):
        logger.info("Инициализация RAGAssistant с ReAct агентами...")
        self.assistant = None
        self.llm = None
        self.oracle_tool = None
        self.tools = []
        self.graph = None
        self.router_chain = None
        self.sql_agent_executor = None
        self.rag_agent_executor = None
        self.general_agent_executor = None
        self.supports_functions = False
        self.feedback_data = []
        
        try:
            # Инициализируем компоненты для поиска
            self.search_engine = HybridSearch()
            logger.info("Инициализирован поисковый движок для RAG")
            
            # Загружаем основную LLM модель
            logger.info("Загрузка LLM модели...")
            self.assistant = LangChainAssistant()
            self.llm = self.assistant.llm  # получаем ChatOpenAI или LlamaCpp
            model_class_name = type(self.llm).__name__
            logger.info(f"LLM успешно загружен: {model_class_name}")
            
            # Определяем, поддерживает ли модель function calling
            is_openai_model = model_class_name == 'ChatOpenAI'
            logger.info(f"Поддержка OpenAI function calling: {is_openai_model}")
            self.supports_functions = is_openai_model
            
            # Проверяем настройки модели OpenAI
            if is_openai_model:
                model_name = getattr(self.llm, 'model_name', 'gpt-3.5-turbo')
                logger.info(f"Используется OpenAI модель: {model_name}")
            
            # Инициализируем Oracle Text2SQL с основной моделью LLM
            logger.info("Инициализация Oracle Text2SQL...")
            try:
                # Создаем инструмент Oracle с нашей основной моделью LLM
                oracle_tool_instance = OracleText2SQL(
                    llm=self.llm,  # Передаем существующую модель
                    temperature=0.0
                )
                logger.info("Oracle Text2SQL инструмент создан, подключение к БД...")
                connected = oracle_tool_instance.connect()
                if connected:
                    logger.info("Oracle Text2SQL успешно подключен к БД")
                    # Регистрируем инструмент в модуле sql_tool
                    set_oracle_tool(oracle_tool_instance)
                else:
                    logger.warning("Oracle Text2SQL создан, но не удалось подключиться к БД")
            except Exception as db_err:
                logger.error(f"Ошибка при инициализации Oracle Text2SQL: {str(db_err)}", exc_info=True)
                oracle_tool_instance = None
            
            # Сохраняем инструмент
            self.oracle_tool = oracle_tool_instance
            
            # Создаем инструменты для агентов
            self.tools = self._create_tools()
            logger.info(f"Создано {len(self.tools)} инструментов для агентов")
            
            # Инициализируем роутер и ReAct агентов
            self._initialize_agents()
            
            # Создаем LangGraph для процесса RAG
            self.graph = self._create_rag_graph()
            
            self.feedback_data = []  # Для хранения обратной связи
            logger.info("RAGAssistant успешно инициализирован с ReAct агентами")
            
        except Exception as e:
            logger.error(f"Ошибка при инициализации RAGAssistant: {str(e)}", exc_info=True)
            raise
            
    def _convert_tools_to_openai_functions(self, tools):
        """Преобразует инструменты в формат функций OpenAI"""
        from langchain.tools.convert_to_openai import convert_to_openai_function
        return [format_tool_to_openai_function(tool) for tool in tools]
        
    def _create_tools(self):
        """Создает инструменты для использования в ReAct агентах"""
        tools = []
        
        # Создаем инструмент для запросов к базе данных
        if self.oracle_tool:
            @tool
            def database_tool(question: str) -> str:
                """Выполняет запросы к базе данных Oracle. Используй этот инструмент, когда вопрос касается 
                базы данных, требует доступа к хранимым данным, или когда упоминаются таблицы, записи, SQL или коды из систем учета."""
                logger.info(f"Вызов инструмента database_tool с вопросом: {question}")
                try:
                    # Используем блок with trace вместо декоратора @trace
                    with trace("database_tool_execution"):
                        oracle_instance = get_oracle_tool()
                        if not oracle_instance:
                            return "Oracle Text2SQL не инициализирован, не могу выполнить запрос к БД."
                        
                        schema_info = oracle_instance.get_schema_info()
                        sql_query = oracle_instance.generate_sql(question, schema_info)
                        sql_query = sql_query.strip().rstrip(";")  # убираем ; в конце
                        logger.info(f"Сгенерирован SQL запрос: {sql_query}")
                        
                        rows, error = oracle_instance.execute_sql(sql_query)
                        if error:
                            return f"Ошибка при выполнении запроса к БД: {error}"
                        
                        # Форматируем ответ в читаемом виде
                        result_text = f"SQL запрос: ```sql\n{sql_query}\n```\n\n"
                        if not rows:
                            return result_text + "Запрос выполнен успешно, но данные не найдены."
                        
                        # Добавляем таблицу с результатами
                        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                            result_text += "Результаты:\n\n"
                            # Формируем таблицу в markdown
                            headers = rows[0].keys()
                            result_text += "| " + " | ".join(headers) + " |\n"
                            result_text += "| " + " | ".join(["---" for _ in headers]) + " |\n"
                            
                            # Добавляем строки
                            for row in rows[:20]:  # ограничиваем вывод
                                result_text += "| " + " | ".join([str(row.get(h, "")) for h in headers]) + " |\n"
                            
                            if len(rows) > 20:
                                result_text += f"\n*Показано 20 записей из {len(rows)}*\n"
                        else:
                            result_text += f"Результат: {rows}"
                        
                        return result_text
                except Exception as ex:
                    logger.error(f"Ошибка в database_tool: {str(ex)}", exc_info=True)
                    return f"Произошла ошибка при обработке запроса к БД: {str(ex)}"
            
            tools.append(database_tool)
            logger.info("Инструмент database_tool создан и добавлен")
        
        # Создаем инструмент для RAG поиска по документам
        @tool
        def rag_tool(question: str) -> str:
            """Ищет информацию в документах и возвращает релевантные фрагменты. Используй этот инструмент, когда 
            вопрос касается общих знаний, определений, процессов или политик, описанных в документах."""
            logger.info(f"Вызов инструмента rag_tool с вопросом: {question}")
            try:
                # Используем блок with trace вместо декоратора @trace
                with trace("rag_search"):
                    # Получаем документы через поисковый движок
                    docs = self.search_engine.search(question)
                    if not docs:
                        return "Не найдено релевантной информации в документах."
                    
                    # Детальная отладка первого документа для понимания структуры ScoredPoint
                    if docs and len(docs) > 0:
                        first_doc = docs[0]
                        logger.info(f"DEBUG: Тип первого документа: {type(first_doc)}")
                        logger.info(f"DEBUG: Все атрибуты: {dir(first_doc)}")
                        
                        # Попробуем получить все возможные атрибуты
                        try_attributes = ['page_content', 'text', 'content', 'payload', 'metadata', 'id', 'score', 'vector']
                        for attr in try_attributes:
                            if hasattr(first_doc, attr):
                                try:
                                    value = getattr(first_doc, attr)
                                    if attr == 'vector' and value is not None:
                                        logger.info(f"DEBUG: {attr} = [vector с длиной {len(value)}]")
                                    else:
                                        logger.info(f"DEBUG: {attr} = {value}")
                                except Exception as e:
                                    logger.info(f"DEBUG: Ошибка при получении {attr}: {str(e)}")
                    
                    # Форматируем результаты поиска
                    context_text = "Найденная информация из документов:\n\n"
                    for i, doc in enumerate(docs[:5], 1):
                        try:
                            # Обработка ScoredPoint объектов на основе отладочных данных
                            # ScoredPoint объекты должны иметь поле payload с текстовым содержимым
                            if hasattr(doc, 'payload') and isinstance(doc.payload, dict):
                                # В Qdrant содержимое обычно хранится в payload под ключами page_content или text
                                if 'page_content' in doc.payload:
                                    content = doc.payload['page_content']
                                elif 'text' in doc.payload:
                                    content = doc.payload['text']
                                elif 'content' in doc.payload:
                                    content = doc.payload['content']
                                elif '_content' in doc.payload:
                                    content = doc.payload['_content']
                                else:
                                    # Если нет стандартных ключей, ищем первое текстовое поле
                                    text_field = None
                                    for key, value in doc.payload.items():
                                        if isinstance(value, str) and len(value) > 50:  # Достаточно длинное поле скорее всего содержит текст
                                            text_field = value
                                            break
                                    if text_field:
                                        content = text_field
                                    else:
                                        # Если нет длинных текстовых полей, возвращаем весь payload
                                        content = str(doc.payload)
                            # Обработка стандартных документов LangChain
                            elif hasattr(doc, 'page_content'):
                                content = doc.page_content
                            elif hasattr(doc, 'text'):
                                content = doc.text
                            elif hasattr(doc, 'content'):
                                content = doc.content
                            else:
                                # Последняя попытка - пробуем сериализовать документ целиком
                                try:
                                    import json
                                    if hasattr(doc, '__dict__'):
                                        content = json.dumps(doc.__dict__, default=str)
                                    elif hasattr(doc, 'to_dict'):
                                        content = json.dumps(doc.to_dict(), default=str)
                                    else:
                                        content = f"[Не удалось извлечь содержимое из {type(doc).__name__}]"
                                except:
                                    content = f"[Не удалось извлечь содержимое из {type(doc).__name__}]"
                                    
                        except Exception as e:
                            logger.error(f"Ошибка при извлечении содержимого документа: {str(e)}")
                            content = f"[Ошибка при извлечении текста: {str(e)}]"
                            
                        context_text += f"**Источник {i}**\n{content}\n\n"
                    
                    return context_text
            except Exception as ex:
                logger.error(f"Ошибка в rag_tool: {str(ex)}", exc_info=True)
                return f"Произошла ошибка при поиске в документах: {str(ex)}"
        
        tools.append(rag_tool)
        logger.info("Инструмент rag_tool создан и добавлен")
        
        # Инструмент для прямых ответов без использования внешних источников
        @tool
        def direct_answer(question: str) -> str:
            """Отвечает на вопросы напрямую без поиска в документах или БД. Используй этот инструмент для простых 
            вопросов, приветствий, или когда нужны только логические рассуждения/вычисления."""
            logger.info(f"Вызов инструмента direct_answer с вопросом: {question}")
            # Этот инструмент просто сигнализирует роутеру, что нужно отвечать напрямую
            return "Отвечу на этот вопрос напрямую, используя общие знания, логику или математические вычисления."
        
        tools.append(direct_answer)
        logger.info("Инструмент direct_answer создан и добавлен")
        
        return tools
    
    def _initialize_agents(self):
        """Инициализирует роутер и специализированные ReAct агенты"""
        try:
            logger.info("Инициализация агентов...")
            
            # 1. Создаем роутер-агент для определения типа запроса
            if self.supports_functions:
                logger.info("Создание роутера с поддержкой function-calling")
                # Используем OPENAI_FUNCTIONS формат для роутера
                from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
                from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
                from prompts import ROUTER_AGENT_PROMPT
                # Создаем промпт для роутера
                router_prompt = ChatPromptTemplate.from_messages([
                    ("system", ROUTER_AGENT_PROMPT),
                    ("human", "{question}"),
                    ("ai", "{agent_scratchpad}")
                ])
                
                # Определяем функции для роутера
                router_functions = [
                    {
                        "name": "route_query",
                        "description": "Маршрутизирует запрос к наиболее подходящему инструменту",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "route": {
                                    "type": "string",
                                    "enum": ["database", "documents", "direct"],
                                    "description": "Выбранный маршрут для запроса"
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Объяснение, почему был выбран этот маршрут"
                                }
                            },
                            "required": ["route", "reasoning"]
                        }
                    }
                ]
                
                # Создаем цепочку роутера с парсером JSON вывода
                self.llm_with_routing = self.llm.bind(functions=router_functions)
                self.router_chain = (
                    {"question": RunnablePassthrough(), "agent_scratchpad": lambda x: ""} 
                    | router_prompt 
                    | self.llm_with_routing 
                    | JsonOutputFunctionsParser()
                )
                
                logger.info("Роутер с function-calling создан")
                
                # 2. Создаем SQL ReAct агент
                if self.supports_functions and self.oracle_tool:
                    logger.info("Создание SQL ReAct агента с function-calling")
                    from prompts import SQL_AGENT_PROMPT
                    # Создаем промпт для SQL агента
                    sql_prompt = ChatPromptTemplate.from_messages([
                        ("system", SQL_AGENT_PROMPT),
                        ("human", "{input}"),
                        ("ai", "{agent_scratchpad}")
                    ])
                    
                    # Определяем инструменты для SQL агента
                    sql_tools = [tool for tool in self.tools if tool.name == "database_tool"]
                    
                    # Форматируем скретчпад (мыслительный процесс) агента
                    def _format_sql_scratchpad(steps):
                        return format_to_openai_function_messages(steps)
                    
                    # Создаем и собираем агента
                    sql_agent = (
                        {
                            "input": lambda x: x["input"],
                            "agent_scratchpad": lambda x: _format_sql_scratchpad(x["intermediate_steps"]),
                            "tools": lambda x: sql_tools
                        }
                        | sql_prompt
                        | self.llm.bind(functions=[format_tool_to_openai_function(tool) for tool in sql_tools])
                        | OpenAIFunctionsAgentOutputParser()
                    )
                    
                    # Создаем исполнитель агента
                    self.sql_agent_executor = AgentExecutor(agent=sql_agent, tools=sql_tools, verbose=True)
                    logger.info("SQL ReAct агент успешно создан")
                else:
                    self.sql_agent_executor = None
                    logger.info("SQL ReAct агент не создан (нет Oracle)")
                
                # 3. Создаем RAG ReAct агент для поиска в документах
                if self.supports_functions:
                    logger.info("Создание RAG ReAct агента с function-calling")
                    from prompts import GENERAL_AGENT_PROMPT
                    # Создаем промпт для RAG агента
                    rag_prompt = ChatPromptTemplate.from_messages([
                        ("system", GENERAL_AGENT_PROMPT),
                        ("human", "{input}"),
                        ("ai", "{agent_scratchpad}")
                    ])
                    
                    # Определяем инструменты для RAG агента
                    rag_tools = [tool for tool in self.tools if tool.name == "rag_tool"]
                    
                    # Форматируем скретчпад агента
                    def _format_rag_scratchpad(steps):
                        return format_to_openai_function_messages(steps)
                    
                    # Создаем и собираем агента
                    rag_agent = (
                        {
                            "input": lambda x: x["input"],
                            "agent_scratchpad": lambda x: _format_rag_scratchpad(x["intermediate_steps"]),
                            "tools": lambda x: rag_tools
                        }
                        | rag_prompt
                        | self.llm.bind(functions=[format_tool_to_openai_function(tool) for tool in rag_tools])
                        | OpenAIFunctionsAgentOutputParser()
                    )
                    
                    # Создаем исполнитель агента
                    self.rag_agent_executor = AgentExecutor(agent=rag_agent, tools=rag_tools, verbose=True)
                    logger.info("RAG ReAct агент успешно создан")
                else:
                    self.rag_agent_executor = None
                    logger.info("RAG ReAct агент не создан (нет поддержки function-calling)")
                
                # 4. Создаем общий агент для прямых ответов
                if self.supports_functions:
                    logger.info("Создание общего ReAct агента с function-calling")
                    from prompts import GENERAL_AGENT_PROMPT
                    # Создаем промпт для общего агента
                    general_prompt = ChatPromptTemplate.from_messages([
                        ("system", GENERAL_AGENT_PROMPT),
                        ("human", "{input}"),
                        ("ai", "{agent_scratchpad}")
                    ])
                    
                    # Определяем инструменты для общего агента (все инструменты)
                    general_tools = self.tools
                    
                    # Форматируем скретчпад агента
                    def _format_general_scratchpad(steps):
                        return format_to_openai_function_messages(steps)
                    
                    # Создаем и собираем агента
                    general_agent = (
                        {
                            "input": lambda x: x["input"],
                            "agent_scratchpad": lambda x: _format_general_scratchpad(x["intermediate_steps"]),
                            "tools": lambda x: general_tools
                        }
                        | general_prompt
                        | self.llm.bind(functions=[format_tool_to_openai_function(tool) for tool in general_tools])
                        | OpenAIFunctionsAgentOutputParser()
                    )
                    
                    # Создаем исполнитель агента
                    self.general_agent_executor = AgentExecutor(agent=general_agent, tools=general_tools, verbose=True)
                    logger.info("Общий ReAct агент успешно создан")
                else:
                    # Для моделей без поддержки функций используем fallback на LangGraph
                    self.general_agent_executor = None
                    logger.info("Общий ReAct агент не создан (нет поддержки function-calling)")
                
                # Определяем инструменты для RAG агента
                rag_tools = [tool for tool in self.tools if tool.name == "rag_tool"]
                
                # Форматируем скретчпад агента
                def _format_rag_scratchpad(steps):
                    return format_to_openai_function_messages(steps)
                
                # Создаем и собираем агента
                rag_agent = (
                    {
                        "input": lambda x: x["input"],
                        "agent_scratchpad": lambda x: _format_rag_scratchpad(x["intermediate_steps"]),
                        "tools": lambda x: rag_tools
                    }
                    | rag_prompt
                    | self.llm.bind(functions=[format_tool_to_openai_function(tool) for tool in rag_tools])
                    | OpenAIFunctionsAgentOutputParser()
                )
                
                # Создаем исполнитель агента
                self.rag_agent_executor = AgentExecutor(agent=rag_agent, tools=rag_tools, verbose=True)
                logger.info("RAG ReAct агент успешно создан")
            else:
                self.rag_agent_executor = None
                logger.info("RAG ReAct агент не создан (нет поддержки function-calling)")
            
            # 4. Создаем общий агент для прямых ответов
            if self.supports_functions:
                logger.info("Создание общего ReAct агента с function-calling")
                from prompts import GENERAL_AGENT_PROMPT
                # Создаем промпт для общего агента
                general_prompt = ChatPromptTemplate.from_messages([
                    ("system", GENERAL_AGENT_PROMPT),
                    ("human", "{input}"),
                    ("ai", "{agent_scratchpad}")
                ])
                
                # Определяем инструменты для общего агента (все инструменты)
                general_tools = self.tools
                
                # Форматируем скретчпад агента
                def _format_general_scratchpad(steps):
                    return format_to_openai_function_messages(steps)
                
                # Создаем и собираем агента
                general_agent = (
                    {
                        "input": lambda x: x["input"],
                        "agent_scratchpad": lambda x: _format_general_scratchpad(x["intermediate_steps"]),
                        "tools": lambda x: general_tools
                    }
                    | general_prompt
                    | self.llm.bind(functions=[format_tool_to_openai_function(tool) for tool in general_tools])
                    | OpenAIFunctionsAgentOutputParser()
                )
                
                # Создаем исполнитель агента
                self.general_agent_executor = AgentExecutor(agent=general_agent, tools=general_tools, verbose=True)
                logger.info("Общий ReAct агент успешно создан")
            else:
                # Для моделей без поддержки функций используем fallback на LangGraph
                self.general_agent_executor = None
                logger.info("Общий ReAct агент не создан (нет поддержки function-calling)")
            
            logger.info("Роутер без function-calling создан")
        
            # 2. Создаем SQL ReAct агент
            if self.supports_functions and self.oracle_tool:
                logger.info("Создание SQL ReAct агента с function-calling")
                from prompts import SQL_AGENT_PROMPT
                # Создаем промпт для SQL агента
                sql_prompt = ChatPromptTemplate.from_messages([
                    ("system", SQL_AGENT_PROMPT),
                    ("human", "{input}"),
                    ("ai", "{agent_scratchpad}")
                ])
                
                # Определяем инструменты для SQL агента
                sql_tools = [tool for tool in self.tools if tool.name == "database_tool"]
                
                # Форматируем скретчпад (мыслительный процесс) агента
                def _format_sql_scratchpad(steps):
                    return format_to_openai_function_messages(steps)
                
                # Создаем и собираем агента
                sql_agent = (
                    {
                        "input": lambda x: x["input"],
                        "agent_scratchpad": lambda x: _format_sql_scratchpad(x["intermediate_steps"]),
                        "tools": lambda x: sql_tools
                    }
                    | sql_prompt
                    | self.llm.bind(functions=[format_tool_to_openai_function(tool) for tool in sql_tools])
                    | OpenAIFunctionsAgentOutputParser()
                )
                
                # Создаем исполнитель агента
                self.sql_agent_executor = AgentExecutor(agent=sql_agent, tools=sql_tools, verbose=True)
                logger.info("SQL ReAct агент успешно создан")
            else:
                self.sql_agent_executor = None
                logger.info("SQL ReAct агент не создан (нет поддержки function-calling или Oracle)")
            
            # 3. Создаем RAG ReAct агент для поиска в документах
            if self.supports_functions:
                logger.info("Создание RAG ReAct агента с function-calling")
                from prompts import GENERAL_AGENT_PROMPT                
                # Создаем промпт для RAG агента
                rag_prompt = ChatPromptTemplate.from_messages([
                    ("system", GENERAL_AGENT_PROMPT),
                    ("human", "{input}"),
                    ("ai", "{agent_scratchpad}")
                ])
                
                # Определяем инструменты для RAG агента
                rag_tools = [tool for tool in self.tools if tool.name == "rag_tool"]
                
                # Форматируем скретчпад агента
                def _format_rag_scratchpad(steps):
                    return format_to_openai_function_messages(steps)
                
                # Создаем и собираем агента
                rag_agent = (
                    {
                        "input": lambda x: x["input"],
                        "agent_scratchpad": lambda x: _format_rag_scratchpad(x["intermediate_steps"]),
                        "tools": lambda x: rag_tools
                    }
                    | rag_prompt
                    | self.llm.bind(functions=[format_tool_to_openai_function(tool) for tool in rag_tools])
                    | OpenAIFunctionsAgentOutputParser()
                )
                
                # Создаем исполнитель агента
                self.rag_agent_executor = AgentExecutor(agent=rag_agent, tools=rag_tools, verbose=True)
                logger.info("RAG ReAct агент успешно создан")
            else:
                self.rag_agent_executor = None
                logger.info("RAG ReAct агент не создан (нет поддержки function-calling)")
            
            # 4. Создаем общий агент для прямых ответов
            if self.supports_functions:
                logger.info("Создание общего ReAct агента с function-calling")
                from prompts import GENERAL_AGENT_PROMPT
                # Создаем промпт для общего агента
                general_prompt = ChatPromptTemplate.from_messages([
                    ("system", GENERAL_AGENT_PROMPT),
                    ("human", "{input}"),
                    ("ai", "{agent_scratchpad}")
                ])
                
                # Определяем инструменты для общего агента (все инструменты)
                general_tools = self.tools
                
                # Форматируем скретчпад агента
                def _format_general_scratchpad(steps):
                    return format_to_openai_function_messages(steps)
                
                # Создаем и собираем агента
                general_agent = (
                    {
                        "input": lambda x: x["input"],
                        "agent_scratchpad": lambda x: _format_general_scratchpad(x["intermediate_steps"]),
                        "tools": lambda x: general_tools
                    }
                    | general_prompt
                    | self.llm.bind(functions=[format_tool_to_openai_function(tool) for tool in general_tools])
                    | OpenAIFunctionsAgentOutputParser()
                )
                
                # Создаем исполнитель агента
                self.general_agent_executor = AgentExecutor(agent=general_agent, tools=general_tools, verbose=True)
                logger.info("Общий ReAct агент успешно создан")
            else:
                # Для моделей без поддержки функций используем fallback на LangGraph
                self.general_agent_executor = None
                logger.info("Общий ReAct агент не создан (нет поддержки function-calling)")
        except Exception as e:
            logger.error(f"Ошибка при инициализации агентов: {str(e)}", exc_info=True)
            # Устанавливаем исполнителей в None при ошибке
            self.sql_agent_executor = None
            self.rag_agent_executor = None
            self.general_agent_executor = None
    
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

    def process_query(self, query: str, history: List[List[str]] = None) -> Tuple[str, List[Dict], Optional[str]]:
        """
        Обрабатывает запрос с использованием роутера и соответствующего ReAct агента
        
        Args:
            query: Запрос пользователя
            history: История диалога (опционально)
            
        Returns:
            Кортеж (ответ, список источников, run_id для LangSmith)
        """
        global LAST_RUN_ID
        sources = []
        run_id = None
        
        try:
            logger.info(f"Обработка запроса через ReAct архитектуру: {query[:100]}...")
            
            # Шаг 1: Маршрутизация запроса через router_chain
            if self.supports_functions:
                # Используем роутер с function-calling
                try:
                    logger.info("Определение маршрута с помощью function-calling роутера")
                    with trace(name="router_decision") as router_run:
                        routing_result = self.router_chain.invoke({"question": query})
                        route = routing_result.get("route", "direct")
                        logger.info(f"Роутер определил маршрут: {route}")
                        if router_run:
                            run_id = router_run.id
                except Exception as router_err:
                    logger.error(f"Ошибка при маршрутизации запроса: {router_err}", exc_info=True)
                    route = "direct"  # Используем прямой ответ как fallback
                    logger.info(f"Установлен fallback маршрут: {route}")
            else:
                # Роутер без function-calling на основе регулярных выражений
                try:
                    logger.info("Определение маршрута с помощью regex-роутера")
                    with trace(name="router_decision") as router_run:
                        response = self.router_chain.invoke({"question": query})
                        # Извлекаем маршрут из ответа регулярным выражением
                        route_match = re.search(r"Route:\s*(database|documents|direct)", response, re.IGNORECASE)
                        route = route_match.group(1).lower() if route_match else "direct"
                        logger.info(f"Роутер определил маршрут: {route}")
                        if router_run:
                            run_id = router_run.id
                except Exception as router_err:
                    logger.error(f"Ошибка при маршрутизации запроса: {router_err}", exc_info=True)
                    route = "direct"  # Используем прямой ответ как fallback
                    logger.info(f"Установлен fallback маршрут: {route}")
            
            # Шаг 2: Выполнение запроса с соответствующим агентом
            if route == "database" and self.sql_agent_executor:
                # Запрос к базе данных через SQL агент
                logger.info("Выполнение SQL запроса через специализированный агент")
                with trace(name="sql_agent_execution") as sql_run:
                    response = self.sql_agent_executor.invoke({"input": query})
                    answer = response.get("output", "")
                    if sql_run:
                        run_id = sql_run.id
                        
            elif route == "documents" and self.rag_agent_executor:
                # Поиск в документах через RAG агент
                logger.info("Выполнение поиска в документах через специализированный агент")
                with trace(name="rag_agent_execution") as rag_run:
                    response = self.rag_agent_executor.invoke({"input": query})
                    answer = response.get("output", "")
                    # Проверяем, есть ли информация об источниках в ответе
                    if "sources" in response:
                        sources = response["sources"]
                    if rag_run:
                        run_id = rag_run.id
                        
            elif self.general_agent_executor:
                # Прямой ответ через общий агент
                logger.info("Выполнение запроса через общий агент")
                with trace(name="general_agent_execution") as general_run:
                    response = self.general_agent_executor.invoke({"input": query})
                    answer = response.get("output", "")
                    if "sources" in response:
                        sources = response["sources"]
                    if general_run:
                        run_id = general_run.id
                        
            else:
                # Fallback к LangGraph если агенты не доступны
                logger.info("Fallback к LangGraph RAG")
                return self._fallback_to_langgraph(query, history)
            
            # Сохраняем run_id для последующего доступа
            if run_id:
                LAST_RUN_ID = run_id
                logger.info(f"LangSmith run_id: {run_id} сохранен в LAST_RUN_ID")
                
            # Форматируем источники для отображения, если есть
            sources_html = format_source_display(sources) if sources else None
            
            return answer, sources, sources_html
            
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса через ReAct агенты: {str(e)}", exc_info=True)
            # Fallback к LangGraph при ошибке
            logger.info("Fallback к LangGraph RAG из-за ошибки")
            return self._fallback_to_langgraph(query, history)
    
    def _fallback_to_langgraph(self, query: str, history: List[List[str]] = None) -> Tuple[str, List[Dict], Optional[str]]:
        """
        Fallback метод для использования LangGraph RAG процесса
        
        Args:
            query: Запрос пользователя
            history: История диалога
            
        Returns:
            Кортеж (ответ, список источников, run_id для LangSmith)
        """
        global LAST_RUN_ID
        run_id = None
        
        try:
            logger.info(f"Переход на fallback через LangGraph для запроса: {query[:100]}...")
            
            # Создаем правильный формат входных данных для LangGraph
            initial_state = {
                "question": query,
                "chat_history": history if history else [],  # Ключ chat_history вместо history для совместимости
                "context": [],
                "formatted_context": "",
                "answer": None,
                "sources": []
            }
            
            # Конфигурация для LangSmith
            config = {}
            if LANGCHAIN_API_KEY:
                config = {
                    "configurable": {
                        "project_name": LANGCHAIN_PROJECT,
                        "tags": ["rag", "fallback", "production"]
                    }
                }
            
            # Используем trace внутри функции вместо декоратора
            with trace(name="fallback_rag") as run:
                final_state = self.graph.invoke(initial_state, config=config)
                # Получаем run_id для LangSmith
                if run is not None:
                    run_id = run.id
                    # Сохраняем в глобальной переменной для последующего доступа
                    LAST_RUN_ID = run_id
                    logger.info(f"LangSmith run_id: {run_id} сохранен в LAST_RUN_ID")
            
            # Получаем результаты
            answer = final_state.get("answer", RETRIEVAL_ERROR_MESSAGE)
            sources = final_state.get("sources", [])
            # Форматируем источники для отображения
            sources_html = format_source_display(sources)
            
            return answer, sources, sources_html
            
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса через LangGraph: {str(e)}", exc_info=True)
            return f"Произошла ошибка при обработке запроса: {str(e)}", [], None
            
    def answer_query(self, query: str, history: List[List[str]]) -> Tuple[str, List[Dict], Optional[str]]:
        """
        Отвечает на вопрос с использованием ReAct агентов и роутера, с fallback на LangGraph RAG
        
        Args:
            query: Текущий запрос пользователя
            history: История диалога в формате [[user_msg1, assistant_msg1], [user_msg2, assistant_msg2], ...]
            
        Returns:
            Кортеж (ответ, список источников, run_id для LangSmith)
        """
        # Используем новый метод process_query с ReAct агентами и роутером
        return self.process_query(query, history)
    
    def save_feedback(self, query: str, response: str, rating: int, comments: str = "", run_id: Optional[str] = None):
        """
        Сохраняет обратную связь пользователя локально и в LangSmith (если доступен)
        
        Args:
            query: Запрос пользователя
            response: Ответ системы
            rating: Оценка (обычно 1-5)
            comments: Комментарии пользователя
            run_id: Идентификатор запуска в LangSmith
            
        Returns:
            bool: Успешно ли сохранен отзыв
        """
        logger.info(f"Получен отзыв: оценка={rating}, комментарий={comments}")
        
        feedback = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "response": response,
            "rating": rating,
            "comments": comments,
            "run_id": run_id
        }
        
        self.feedback_data.append(feedback)
        success = True
        
        # Отправляем отзыв в LangSmith, если доступен run_id и API ключ
        if run_id and LANGCHAIN_API_KEY:
            try:
                from langsmith.client import Client
                
                client = Client()
                # Преобразуем рейтинг от 1-5 к формату LangSmith (от 1 до 10 или строка)
                langsmith_score = None
                if isinstance(rating, int):
                    # Приводим рейтинг от 1-5 к шкале 1-10
                    langsmith_score = min(10, rating * 2)
                
                # Отправляем отзыв в LangSmith
                client.create_feedback(
                    run_id=run_id,
                    key="user_rating",
                    score=langsmith_score,
                    comment=comments,
                    value=rating
                )
                logger.info(f"Отзыв успешно отправлен в LangSmith для run_id={run_id}")
            except Exception as e:
                logger.error(f"Ошибка при отправке отзыва в LangSmith: {str(e)}", exc_info=True)
                success = False
        
        # Класс для сериализации UUID в JSON
        class UUIDEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, uuid.UUID):
                    # Convert UUID to string
                    return str(obj)
                return json.JSONEncoder.default(self, obj)
        
        # Сохраняем в локальный файл
        try:
            with open("feedback_data.json", "w", encoding="utf-8") as f:
                json.dump(self.feedback_data, f, ensure_ascii=False, indent=2, cls=UUIDEncoder)
            logger.info(f"Отзыв успешно сохранен локально. Рейтинг: {rating}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении отзыва в локальный файл: {str(e)}", exc_info=True)
            success = False
            
        return success

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
        # Преобразуем историю в формат для ассистента
        # Ожидаемый формат: [[user_msg1, assistant_msg1], [user_msg2, assistant_msg2], ...]
        chat_history = []
        if history:
            for i, item in enumerate(history):
                # Если это словари, используем их content
                if isinstance(item, dict) and "role" in item and "content" in item:
                    if item["role"] == "user":
                        # Добавляем только если это пользовательское сообщение
                        # и следующее сообщение от ассистента
                        if i + 1 < len(history) and isinstance(history[i+1], dict) and history[i+1]["role"] == "assistant":
                            chat_history.append([item["content"], history[i+1]["content"]])
                # Если это кортежи, используем их напрямую
                elif isinstance(item, tuple) and len(item) == 2:
                    chat_history.append([item[0], item[1]])
        
        # Генерируем ответ, используя метод answer_query
        response, sources, run_id = rag_assistant.answer_query(message, chat_history)
        
        # Запоминаем ID запроса для отзывов
        global LAST_RUN_ID
        LAST_RUN_ID = run_id
        
        # Форматируем источники для отображения
        sources_html = format_source_display(sources)
        
        # Добавляем вопрос пользователя и ответ ассистента в историю
        # Используем формат, совместимый с Gradio Chatbot
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        # Ограничиваем длину истории
        if len(history) > MAX_HISTORY_LENGTH * 2:  # Умножаем на 2, так как каждая пара вопрос-ответ - это 2 элемента
            removed_count = len(history) - MAX_HISTORY_LENGTH * 2
            history = history[-MAX_HISTORY_LENGTH * 2:]
            logger.debug(f"История чата обрезана. Удалено {removed_count} старых сообщений")
        
        logger.info("Запрос успешно обработан")
        return "", history, sources_html
    
    except Exception as e:
        error_msg = f"Ошибка при обработке запроса: {str(e)}"
        logger.error(error_msg, exc_info=True)
        history.append({"role": "assistant", "content": "Извините, произошла ошибка при обработке вашего запроса."})
        return "", history, ""



def format_source_display(sources: List[Any]) -> str:
    """Форматирует список источников в HTML для отображения
    
    Args:
        sources: Список источников любого формата (словари или списки)
        
    Returns:
        HTML-строка с отформатированными источниками
    """
    if not sources or len(sources) == 0:
        return "<div>Источники не найдены</div>"
    
    html_parts = ["<div style='margin-top: 20px;'><h4>Источники:</h4><div style='padding-left: 20px;'>"]
    logger.info(f"Количество источников: {len(sources)}")
    logger.info(f"Источники: {sources}")
    
    try:
        # Обработка источников, учитывая возможные разные форматы данных
        for i, source in enumerate(sources[:5]):  # Ограничиваем количество отображаемых источников
            try:
                # Проверка типа источника и обработка соответствующим образом
                if isinstance(source, dict):
                    # Стандартный формат источника (словарь)
                    metadata = source.get('metadata', {})
                    sub_metadata = metadata.get('metadata', {}) if isinstance(metadata, dict) else {}
                    source_name = sub_metadata.get('source', 'Неизвестный источник') if isinstance(sub_metadata, dict) else 'Неизвестный источник'
                    page = sub_metadata.get('page', '') if isinstance(sub_metadata, dict) else ''
                    text = source.get('text', '')[:100] if isinstance(source.get('text', ''), str) else str(source)[:100]  # Берем первые 100 символов текста
                    
                    # Формируем отображаемое имя источника
                    display_name = f"{source_name}"
                    if page:
                        display_name += f" (стр. {page})"
                    
                    # Добавляем ссылку, если есть
                    link_start = f"<a href='{metadata['url']}' target='_blank'>" if isinstance(metadata, dict) and 'url' in metadata else ""
                    link_end = "</a>" if isinstance(metadata, dict) and 'url' in metadata else ""
                elif isinstance(source, list) and len(source) >= 2:
                    # Формат [query, answer]
                    display_name = "Результат запроса"
                    text = str(source[1])[:100]
                    link_start = ""
                    link_end = ""
                else:
                    # Неизвестный формат - пытаемся вывести хоть что-то
                    display_name = "Источник"
                    text = str(source)[:100]
                    link_start = ""
                    link_end = ""
                
                # Формируем элемент списка с текстом
                html_parts.append(f"<div><div>{link_start}{display_name}{link_end}</div><div style='color: #666; font-size: 0.9em; margin-top: 3px;'>{text}...</div></div>")
            except Exception as source_err:
                logger.error(f"Ошибка при форматировании источника {i}: {source_err}")
                html_parts.append(f"<div>Источник {i+1}: [Ошибка форматирования]</div>")
    except Exception as format_err:
        logger.error(f"Общая ошибка при форматировании источников: {format_err}")
        return "<div>Ошибка при форматировании источников</div>"
    
    html_parts.append("</div>")
    html_parts.append("</div>")
    return "".join(html_parts)

# Функции для работы с Oracle Text2SQL
oracle_tool = None
def extract_tables_from_sql(sql_query: str) -> List[str]:
    """
    Извлекает имена таблиц из SQL запроса
    
    Args:
        sql_query: SQL запрос
        
    Returns:
        Список найденных таблиц
    """
    # Простая реализация поиска таблиц после FROM и JOIN
    tables = set()
    sql_lower = sql_query.lower()
    
    # Поиск после FROM
    from_parts = sql_lower.split(' from ')
    if len(from_parts) > 1:
        for i in range(1, len(from_parts)):
            # Получаем текст после FROM
            after_from = from_parts[i].strip()
            # Останавливаемся на первом слове, разделенном пробелами или знаками пунктуации
            table_name = ''
            for char in after_from:
                if char.isalnum() or char == '_' or char == '.':
                    table_name += char
                else:
                    break
            if table_name:
                # Убираем имя схемы, если оно есть
                if '.' in table_name:
                    table_name = table_name.split('.')[-1]
                tables.add(table_name)
    
    # Поиск после JOIN
    join_keywords = [' join ', ' inner join ', ' left join ', ' right join ', ' outer join ', ' full join ']
    for keyword in join_keywords:
        join_parts = sql_lower.split(keyword)
        if len(join_parts) > 1:
            for i in range(1, len(join_parts)):
                after_join = join_parts[i].strip()
                table_name = ''
                for char in after_join:
                    if char.isalnum() or char == '_' or char == '.':
                        table_name += char
                    else:
                        break
                if table_name:
                    if '.' in table_name:
                        table_name = table_name.split('.')[-1]
                    tables.add(table_name)
    
    return list(tables)


def initialize_oracle_tool(model_type: str, model_name: str) -> str:
    """
    Инициализирует инструмент Oracle Text2SQL
    
    Args:
        model_type: Тип модели (openai или local)
        model_name: Название модели
        
    Returns:
        Статус инициализации
    """
    global oracle_tool
    
    try:
        # Создаем экземпляр инструмента
        oracle_tool = OracleText2SQL(
            model_type=model_type,
            model_name=model_name,
            temperature=0.0  # Используем низкую температуру для большей точности SQL
        )
        
        # Проверяем подключение к Oracle
        if oracle_tool.connect():
            return "Oracle Text2SQL инструмент успешно инициализирован и подключен к БД"
        else:
            return "Ошибка подключения к Oracle. Проверьте параметры подключения в .env файле."
        
    except Exception as e:
        logger.error(f"Ошибка инициализации Oracle Text2SQL: {str(e)}")
        return f"Ошибка: {str(e)}"


def get_schema_info(tables_str=None):
    """
    Получает информацию о схеме базы данных
    
    Args:
        tables_str: Список таблиц через запятую (опционально)
    
    Returns:
        Текст со схемой БД
    """
    global oracle_tool
    
    if not oracle_tool:
        return "Ошибка: Oracle Text2SQL не инициализирован. Нажмите 'Инициализировать Oracle'."
    
    try:
        # Парсим таблицы из строки, если указаны
        tables = None
        if tables_str and tables_str.strip():
            tables = [t.strip() for t in tables_str.split(',') if t.strip()]
        
        # Получаем информацию о схеме
        schema_info = oracle_tool.get_schema_info(tables=tables)
        return schema_info
        
    except Exception as e:
        logger.error(f"Ошибка при получении схемы: {str(e)}")
        return f"Ошибка: {str(e)}"


def process_text2sql_query(text_query, tables_str=None, execute=True):
    """
    Обрабатывает текстовый запрос к БД, преобразуя его в SQL и выполняя
    
    Args:
        text_query: Запрос на естественном языке
        tables_str: Список таблиц через запятую (опционально)
        execute: Выполнять ли сгенерированный SQL запрос
    
    Returns:
        Три значения: SQL запрос, результаты в виде HTML таблицы и сообщение о статусе
    """
    global oracle_tool
    
    if not oracle_tool:
        return "", "", "Ошибка: Oracle Text2SQL не инициализирован. Нажмите 'Инициализировать Oracle'."
    
    if not text_query or text_query.strip() == "":
        return "", "", "Пожалуйста, введите запрос на естественном языке."
    
    try:
        # Парсим таблицы из строки, если указаны
        tables = None
        if tables_str and tables_str.strip():
            tables = [t.strip() for t in tables_str.split(',') if t.strip()]
        
        # Обрабатываем запрос
        result = oracle_tool.process_text_query(
            text_query=text_query,
            tables=tables,
            execute=execute
        )
        
        # Форматируем результаты в виде HTML таблицы
        sql_query = result["sql"]
        results_html = format_results_as_html(result["results"]) if execute else ""
        
        # Формируем сообщение о статусе
        if result["error"]:
            status_msg = f"Ошибка: {result['error']}"
        else:
            count = len(result["results"]) if execute else 0
            status_msg = f"SQL запрос успешно сгенерирован" + (f" и выполнен. Получено {count} результатов." if execute else ".")
        
        return sql_query, results_html, status_msg
        
    except Exception as e:
        logger.error(f"Ошибка при обработке Text2SQL запроса: {str(e)}")
        return "", "", f"Ошибка: {str(e)}"


def format_results_as_html(results):
    """
    Форматирует результаты запроса в HTML таблицу
    
    Args:
        results: Результаты запроса в виде списка словарей
        
    Returns:
        HTML код таблицы с результатами
    """
    if not results or not isinstance(results, list) or not results:
        return "<div>Нет результатов</div>"
    
    # Проверяем, что первый элемент списка - словарь
    if not results or not isinstance(results[0], dict):
        return "<div>Результаты в неожиданном формате</div>"
    
    html = ["<div style='overflow-x:auto;'><table style='width:100%; border-collapse:collapse;'>"]
    
    # Заголовок таблицы
    columns = list(results[0].keys())
    html.append("<thead><tr>")
    for col in columns:
        html.append(f"<th style='border:1px solid #ddd; padding:8px; text-align:left; background-color:#f2f2f2;'>{col}</th>")
    html.append("</tr></thead>")
    
    # Данные таблицы
    html.append("<tbody>")
    for row in results:
        html.append("<tr>")
        for col in columns:
            html.append(f"<td style='border:1px solid #ddd; padding:8px;'>{str(row.get(col, ''))}</td>")
        html.append("</tr>")
    html.append("</tbody></table></div>")
    
    return "".join(html)


def clear_chat():
    """Очищает историю чата и возвращает пустые значения для всех элементов"""
    logger.info("Очистка истории чата")
    
    # Сбрасываем глобальный run_id при очистке чата
    global LAST_RUN_ID
    LAST_RUN_ID = None
    logger.info("Сброс LAST_RUN_ID при очистке чата")
    
    return [], "", ""  # Пустые значения для chat_history, sources_display, feedback_status

def submit_feedback(rating, comments, history):
    """Отправляет обратную связь о последнем ответе и отправляет её в LangSmith"""
    logger.info(f"Получен отзыв: оценка={rating}, комментарий={comments}")
    
    if not history or len(history) < 2:
        logger.warning("Не удалось сохранить отзыв: история сообщений пуста или недостаточно сообщений")
        return "Не удалось сохранить отзыв: история сообщений пуста"
    
    try:
        # Получаем последние сообщения пользователя и ассистента
        last_query = history[-2]["content"]  # Предпоследнее - сообщение пользователя
        last_response = history[-1]["content"]  # Последнее - ответ ассистента
        
        # Используем глобальную переменную для получения последнего run_id
        global LAST_RUN_ID
        run_id = LAST_RUN_ID
        
        if run_id:
            logger.info(f"Найден run_id для отправки отзыва в LangSmith: {run_id}")
        else:
            logger.warning(f"LAST_RUN_ID не установлен, отзыв будет сохранен только локально")
        
        # Сохраняем отзыв, передавая run_id для отправки в LangSmith
        success = rag_assistant.save_feedback(last_query, last_response, rating, comments, run_id)
        
        if success:
            message = f"Спасибо за вашу оценку: {rating}! Ваш отзыв поможет улучшить систему."
            if run_id:
                message += " Отзыв также отправлен в LangSmith."
            logger.info("Отзыв успешно сохранен")
            return message
        else:
            logger.error("Не удалось сохранить отзыв")
            return "Не удалось сохранить отзыв. Пожалуйста, попробуйте позже."
    
    except Exception as e:
        logger.error(f"Ошибка при сохранении отзыва: {str(e)}", exc_info=True)
        return "Произошла ошибка при сохранении отзыва. Пожалуйста, попробуйте позже."

def create_demo(rag_assistant):
    """Создает демонстрационный интерфейс Gradio"""
    # Создаем псевдоним для удобства доступа к ассистенту
    assistant = rag_assistant.assistant
    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        gr.Markdown("# 🔍 Система вопросов и ответов с гибридным поиском")        
        
        # Создаем вкладки для основного интерфейса и настроек
        with gr.Tabs() as tabs:
            with gr.TabItem("Чат") as chat_tab:
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
            # Добавляем вкладку "Настройки модели"
            
            with gr.TabItem("Настройки модели") as settings_tab:
                with gr.Group():
                    gr.Markdown("### Выбор модели")
                    with gr.Row():
                        model_type = gr.Dropdown(
                            choices=["local", "openai"],
                            value=assistant.model_type,
                            label="Тип модели",
                            info="Локальная или OpenAI"
                        )
                    
                    # Создаем обновляемый список моделей
                    local_models = assistant.AVAILABLE_MODELS[assistant.MODEL_TYPE_LOCAL]
                    openai_models = assistant.AVAILABLE_MODELS[assistant.MODEL_TYPE_OPENAI]
                    
                    # Сначала показываем модели текущего типа
                    current_models = local_models if assistant.model_type == "local" else openai_models
                    
                    model_name = gr.Dropdown(
                        choices=current_models,
                        value=assistant.model_name,
                        label="Модель",
                        info="Доступные модели выбранного типа"
                    )
                    
                    # Функция для обновления списка моделей при смене типа
                    def update_model_list(selected_type):
                        if selected_type == "local":
                            return gr.update(choices=local_models, value=local_models[0])
                        else:
                            return gr.update(choices=openai_models, value=openai_models[0])
                    
                    # Связываем изменение типа модели с обновлением списка моделей
                    model_type.change(update_model_list, inputs=[model_type], outputs=[model_name])
                    
                    gr.Markdown("### Параметры модели")
                    temperature = gr.Slider(
                        minimum=0.01,
                        maximum=1.0,
                        step=0.01,
                        value=assistant.model_params["temperature"],
                        label="Temperature",
                        info="Чем выше значение, тем более творческие ответы"
                    )
                    max_tokens = gr.Slider(
                        minimum=16,
                        maximum=4096,
                        step=16,
                        value=assistant.model_params["max_tokens"],
                        label="Max Tokens",
                        info="Максимальная длина вывода модели"
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        step=0.05,
                        value=assistant.model_params["top_p"],
                        label="Top P",
                        info="Параметр для нуклеусной выборки"
                    )
                    n_ctx = gr.Slider(
                        minimum=512,
                        maximum=4096,
                        step=512,
                        value=assistant.model_params["n_ctx"],
                        label="Контекст",
                        info="Размер контекстного окна"
                    )
                    n_threads = gr.Slider(
                        minimum=1,
                        maximum=12,
                        step=1,
                        value=assistant.model_params["n_threads"],
                        label="Потоки",
                        info="Количество потоков CPU"
                    )
                    n_gpu_layers = gr.Slider(
                        minimum=0,
                        maximum=32,
                        step=1,
                        value=assistant.model_params["n_gpu_layers"],
                        label="GPU слои",
                        info="Количество слоев для GPU"
                    )
                    verbose = gr.Checkbox(
                        value=assistant.model_params["verbose"],
                        label="Подробный режим",
                        info="Включает подробный вывод процесса генерации"
                    )
                    settings_submit_btn = gr.Button("Сохранить настройки", variant="primary")
                    settings_status = gr.Markdown()

        # Мостик для перехода между вкладками
        def change_tab_to_settings():
            return gr.update(selected="Настройки модели")
        
        # Функция для обновления настроек модели
        def update_model_settings(model_type_val, model_name_val, temp, max_tok, top_p_val, ctx, threads, gpu_layers, verb):
            try:
                # Обновляем настройки модели включая тип и название модели
                success = rag_assistant.assistant.update_model_params(
                    model_type=model_type_val,
                    model_name=model_name_val,
                    temperature=temp,
                    max_tokens=max_tok,
                    top_p=top_p_val,
                    n_ctx=ctx,
                    n_threads=threads,
                    n_gpu_layers=gpu_layers,
                    verbose=verb
                )
                
                if success:
                    return "✅ Настройки модели успешно обновлены"
                else:
                    return "❌ Не удалось обновить настройки модели"
            except Exception as e:
                return f"❌ Ошибка: {str(e)}"
        
        # Функция для изменения вкладки
        def select_tab_settings():
            return gr.update(selected="Настройки модели")
        
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
                    
        settings_submit_btn.click(
            update_model_settings,
            inputs=[model_type, model_name, temperature, max_tokens, top_p, n_ctx, n_threads, n_gpu_layers, verbose],
            outputs=[settings_status]
        )
        
        # Инициализируем Oracle Text2SQL при запуске приложения 
        # для использования в качестве инструмента агента
        
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
        
        # Инициализация RAG-ассистента
        rag_assistant = RAGAssistant()
        
        # Примечание: RAGAssistant сам создает HybridSearch внутри
        
        # Создаем экземпляр демонстрации
        demo = create_demo(rag_assistant)
        demo.launch(server_name="0.0.0.0", share=False, server_port=7862)
    
    except Exception as e:
        logger.error(f"Ошибка при запуске приложения: {str(e)}")
        print(f"Ошибка при запуске приложения: {str(e)}")
        sys.exit(1)
