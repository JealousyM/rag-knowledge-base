import sys
import logging
import json
import time
from typing import List, Dict, Any, Tuple

import gradio as gr
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Local imports
from prompts import SYSTEM_PROMPT

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
EMBEDDING_MODEL = SentenceTransformer(MODEL_PATH)

class HybridSearch:
    """Класс для выполнения гибридного поиска (векторный + ключевой)"""
    
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection_name = collection_name
        self.client = QdrantClient(host="localhost", port=6333)
        logger.info(f"Инициализирован HybridSearch с коллекцией: {collection_name}")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Выполняет гибридный поиск, комбинируя семантический и ключевой поиск
        """
        start_time = time.time()
        try:
            results = get_hybrid_search(query, k)
            logger.info(f"Найдено {len(results)} релевантных документов за {time.time() - start_time:.2f} сек")
            return results
        except Exception as e:
            logger.error(f"Ошибка при поиске: {str(e)}")
            return []

from llama_cpp import Llama

class GGUFModelAssistant:
    """Класс для работы с локальной GGUF моделью"""
    
    def __init__(self, model_path: str = "model/T-lite-it-1.0-Q4_K_M-GGUF/t-lite-it-1.0-q4_k_m.gguf"):
        self.model_path = model_path
        self.llm = None
        self._load_model()
        logger.info(f"Инициализирован ассистент с моделью: {model_path}")
    
    def _load_model(self):
        """Загружает GGUF модель"""
        try:
            # Инициализируем модель с настройками для CPU
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,  # Контекстное окно
                n_threads=6,  # Количество потоков для обработки
                n_gpu_layers=0,  # 0 для использования только CPU
                verbose=False
            )
            logger.info("Модель успешно загружена")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {str(e)}")
            raise RuntimeError(f"Не удалось загрузить модель: {str(e)}")
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Генерирует ответ от языковой модели
        """
        try:
            if not self.llm:
                logger.error("Попытка генерации ответа при незагруженной модели")
                raise RuntimeError("Модель не загружена")
            
            logger.info("Получен запрос на генерацию ответа")
            logger.debug(f"Входные сообщения: {messages}")
            
            # Форматируем сообщения в промпт
            prompt = self._format_messages(messages)
            logger.debug(f"Сформированный промпт: {prompt[:200]}...")  # Логируем начало промпта
            
            # Генерируем ответ
            logger.info("Запуск генерации ответа...")
            start_time = time.time()
            
            response = self.llm(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                echo=False,
                stop=["</s>", "<|im_end|>"]
            )
            
            # Извлекаем сгенерированный текст
            generated_text = response['choices'][0]['text'].strip()
            
            end_time = time.time()
            logger.info(f"Ответ сгенерирован за {end_time - start_time:.2f} секунд")
            logger.debug(f"Сгенерированный ответ: {generated_text[:200]}...")  # Логируем начало ответа
            
            return generated_text
            
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

class RAGAssistant:
    """
    Основной класс для системы поиска и ответов на вопросы
    Интегрирует гибридный поиск и локальную GGUF модель
    """
    
    def __init__(self):
        logger.info("Инициализация RAGAssistant...")
        try:
            self.search_engine = HybridSearch()
            logger.info("Инициализирован поисковый движок")
            
            logger.info("Загрузка GGUF модели...")
            self.assistant = GGUFModelAssistant()
            logger.info("GGUF модель успешно загружена")
            
            self.feedback_data = []  # Для хранения обратной связи
            logger.info("RAGAssistant успешно инициализирован с локальной GGUF моделью")
        except Exception as e:
            logger.error(f"Ошибка при инициализации RAGAssistant: {str(e)}", exc_info=True)
            raise
    
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
        Отвечает на вопрос с использованием RAG
        
        Args:
            query: Текущий запрос пользователя
            history: История диалога в формате [[user_msg1, assistant_msg1], [user_msg2, assistant_msg2], ...]
            
        Returns:
            Кортеж (ответ, список источников)
        """
        try:
            # Выполняем поиск по запросу
            search_results = self.search_engine.search(query)
            
            if not search_results:
                return RETRIEVAL_ERROR_MESSAGE, []
            
            # Форматируем результаты поиска
            context, context_chunks = self.format_search_results(search_results)
            
            # Создаем системное сообщение с контекстом и вопросом
            system_message = {
                "role": "system",
                "content": SYSTEM_PROMPT.format(context=context, question=query)
            }
            messages = [system_message]
            
            # Добавляем историю диалога (не больше MAX_HISTORY_LENGTH сообщений)
            for user_msg, assistant_msg in history[-MAX_HISTORY_LENGTH:]:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
            
            # Добавляем текущий запрос
            messages.append({"role": "user", "content": query})
            
            # Получаем ответ от модели
            response = self.assistant.generate_response(messages)
            
            return response, context_chunks
        
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {str(e)}", exc_info=True)
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
    
    Args:
        query: Search query string
        k: Number of results to return
        
    Returns:
        List of search results with combined scores
    """
    client = QdrantClient(host="localhost", port=6333)
    
    try:
        logger.info(f"Performing hybrid search for query: {query}")
        
        # Get query embedding with search_query prefix
        try:
            query_vector = get_embeddings_with_prefix(query, task="search_query")
            if not isinstance(query_vector, list) or len(query_vector) == 0:
                raise ValueError("Invalid query vector format")
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return []
        
        try:
            # Create a text search condition
            from qdrant_client import models
            
            # Split query into words for better matching
            query_terms = query.split()
            
            # Create a filter for text search (at least one term must match)
            text_conditions = []
            for term in query_terms:
                if len(term) > 2:  # Only include terms longer than 2 characters
                    text_conditions.append(
                        models.FieldCondition(
                            key="text",
                            match=models.MatchText(text=term)
                        )
                    )
            
            # Perform hybrid search with text conditions if any
            if text_conditions:
                search_results = client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector[0],
                    query_filter=models.Filter(
                        should=text_conditions,  # At least one term should match
                        must=[
                            models.Filter(
                                should=text_conditions,
                            )
                        ]
                    ),
                    limit=k*2,
                    with_payload=True,
                    score_threshold=0.3  # Minimum similarity score
                )
            else:
                # If no text conditions, just do vector search
                search_results = client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector[0],
                    limit=k*2,
                    with_payload=True,
                    score_threshold=0.3
                )
            
            logger.info(f"Found {len(search_results)} hybrid search results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            # Fallback to simple vector search if hybrid fails
            try:
                search_results = client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector[0],
                    limit=k*2,
                    with_payload=True
                )
                logger.info(f"Falling back to vector search, found {len(search_results)} results")
                return search_results
            except Exception as fallback_error:
                logger.error(f"Error in fallback vector search: {str(fallback_error)}")
                return []
                
    except Exception as e:
        logger.error(f"Unexpected error in hybrid search: {str(e)}")
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
