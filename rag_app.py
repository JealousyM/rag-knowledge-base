import sys
import logging
import uuid
from typing import List
import gradio as gr
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Local imports
from text_utils import format_source_display
from file_utils import clean_static_images
from memory_storage import MemoryStorage
from rag_assistant import RAGAssistant
from lang_chain_assistant import LangChainAssistant
from oracle_text2sql import OracleText2SQL
from constants import *

def update_model_choices(model_type):
    """Обновляет список доступных моделей при смене типа модели."""
    choices = LangChainAssistant.AVAILABLE_MODELS.get(model_type, [])
    # Устанавливаем значение по умолчанию на первое в списке, если список не пуст
    default_value = choices[0] if choices else None
    return gr.Dropdown.update(choices=choices, value=default_value)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Инициализация компонентов
# Загружаем модель для эмбеддингов из data_processing.py
# Так как мы уже загрузили модель в data_processing.py, мы используем ее напрямую
from data_processing import EMBEDDING_MODEL

# Инициализация хранилища памяти
memory_storage = MemoryStorage()

        
def chat_with_feedback(message, history, session_state):
    """
    Обрабатывает взаимодействие пользователя с чатом,
    возвращает ответ с информацией об источниках
    и очищает директорию static/images после ответа
    """
    import os
    import re
    
    logger.info(f"Получен запрос от пользователя: {message[:100]}...")
    logger.debug(f"Текущая история чата (до обработки): {history}")
    
    if not message or message.strip() == "":
        logger.warning("Получен пустой запрос от пользователя")
        return "", history, "Пожалуйста, введите ваш вопрос.", session_state
    
    if session_state is None:
        session_state = {}
    
    session_id = session_state.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session_state["session_id"] = session_id

    try:
        # Add the user's message to the history immediately
        history.append({"role": "user", "content": message})

        # Now, check the cache
        cached_answer, cached_sources = memory_storage.get_exact_answer(session_id, message)
        if cached_answer:
            logger.info("Найден точный ответ в кэше.")
            sources_html = format_source_display(cached_sources)
            # User message is already in history, just add assistant's response
            history.append({"role": "assistant", "content": cached_answer})
            return "", history, sources_html, session_state



        # Pass the complete, correct history to the assistant
        response, sources, run_id = rag_assistant.answer_query(message, history)
        logger.info(f"Ответ: {response}")
        logger.info(f"Источники: {sources}")
        logger.info(f"ID запроса: {run_id}")
        # Запоминаем ID запроса для отзывов
        global LAST_RUN_ID
        LAST_RUN_ID = run_id
        
        # Форматируем источники для отображения
        sources_html = format_source_display(sources)
        
        # Формируем ответ ассистента, включая обработку изображений
        assistant_response_html = ""

        # Проверяем наличие тегов с путём к base64 файлу
        b64file_pattern = r'<B64FILE>(.*?)</B64FILE>'
        b64file_matches = re.findall(b64file_pattern, response)

        # Удаляем теги из основного текста ответа
        cleaned_response = re.sub(b64file_pattern, '', response).strip()
        if cleaned_response:
            assistant_response_html += f"<div>{cleaned_response}</div>"

        if b64file_matches:
            logger.info(f"Обнаружены маркеры B64FILE: {len(b64file_matches)}")
            for file_path in b64file_matches:
                try:
                    if not os.path.exists(file_path):
                        logger.error(f"Файл не найден: {file_path}")
                        assistant_response_html += f"<div>Ошибка: Изображение не найдено ({file_path})</div>"
                        continue

                    with open(file_path, 'r') as f:
                        file_content = f.read().strip()

                    if not file_content.startswith('data:image/png;base64,'):
                        base64_data = f"data:image/png;base64,{file_content}"
                    else:
                        base64_data = file_content

                    assistant_response_html += f"<img src='{base64_data}' alt='Image' style='max-width: 100%; height: auto;'>"
                except Exception as img_err:
                    logger.error(f"Ошибка при обработке base64 файла {file_path}: {str(img_err)}")
                    assistant_response_html += f"<div>Ошибка при загрузке изображения: {file_path}</div>"

        # Append the final assistant response to the history
        history.append({"role": "assistant", "content": assistant_response_html})
        
        # Ограничиваем длину истории
        if len(history) > MAX_HISTORY_LENGTH:
            history = history[-MAX_HISTORY_LENGTH:]
            logger.debug(f"История чата обрезана до {MAX_HISTORY_LENGTH} последних сообщений")

        # Очищаем директорию с изображениями после ответа
        clean_static_images()
        logger.info(f"Директория с изображениями очищена")

        # Сохраняем диалог в базу данных
        memory_storage.add_message(session_id, message, response, sources) # Сохраняем оригинальный ответ с источниками
        logger.info(f"Диалог сохранен в базу данных")

        # Возвращаем ответ и обновляем историю
        return "", history, sources_html, session_state
    
    except Exception as e:
        error_msg = f"Ошибка при обработке запроса: {str(e)}"
        logger.error(error_msg, exc_info=True)
        history.append({"role": "assistant", "content": "Извините, произошла ошибка при обработке вашего запроса."})
        return "", history, "", session_state

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
    """Очищает историю чата и возвращает пустые значения для всех элементов, включая состояние сессии"""
    logger.info("Очистка истории чата и состояния сессии")
    
    # Сбрасываем глобальный run_id при очистке чата
    global LAST_RUN_ID
    LAST_RUN_ID = None
    logger.info("Сброс LAST_RUN_ID при очистке чата")
    
    # Возвращаем пустые значения для всех элементов, включая состояние сессии
    # chatbot, sources_display, feedback_status, session_state
    return [], "", "", {}

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


def create_demo(rag_assistant, memory_storage):
    """
    Создает демонстрационный интерфейс Gradio
    
    Args:
        rag_assistant: Экземпляр RAGAssistant
        memory_storage: Экземпляр MemoryStorage
    """
    # Загружаем текущую конфигурацию LLM при запуске
    current_config = rag_assistant.get_llm_config()
    logger.info(f"Загружена конфигурация LLM для UI: {current_config}")
    
    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        session_state = gr.State({})
        gr.Markdown("# 🔍 Система вопросов и ответов с гибридным поиском")
        
        with gr.Tab("Основной чат"):
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        [],
                        elem_id="chatbot",
                        type="messages",
                        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "images/avatar.png"))),
                        height=600
                    )
                    with gr.Row():
                        msg = gr.Textbox(show_label=False, placeholder="Введите ваш вопрос и нажмите Enter", container=False, scale=8)
                        submit_btn = gr.Button("Отправить", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("Очистить чат")
                
                with gr.Column(scale=2):
                    sources_display = gr.HTML("Источники будут отображены здесь", elem_id="sources")
                    
                    with gr.Accordion("Обратная связь", open=False):
                        feedback_rating = gr.Radio(["👍", "👎"], label="Оцените ответ")
                        feedback_comments = gr.Textbox(label="Комментарии (опционально)")
                        feedback_btn = gr.Button("Отправить отзыв")
                        feedback_status = gr.Textbox(label="Статус отзыва", interactive=False)

        with gr.Tab("Настройки модели"):
            gr.Markdown("Здесь можно настроить параметры языковой модели.")
            with gr.Row():
                model_type_dd = gr.Dropdown(
                    label="Тип модели", 
                    choices=list(LangChainAssistant.AVAILABLE_MODELS.keys()), 
                    value=current_config.get('model_type', 'openai'),
                    interactive=True
                )
                # Динамически устанавливаем список доступных моделей и выбранное значение
                available_models_for_type = LangChainAssistant.AVAILABLE_MODELS.get(current_config.get('model_type', 'openai'), [])
                model_name_dd = gr.Dropdown(
                    label="Название модели", 
                    choices=available_models_for_type, 
                    value=current_config.get('model_name', 'gpt-3.5-turbo'),
                    interactive=True
                )
            temperature_slider = gr.Slider(minimum=0.0, maximum=2.0, value=current_config.get('temperature', 0.7), step=0.1, label="Температура", interactive=True)
            top_p_slider = gr.Slider(minimum=0.0, maximum=1.0, value=current_config.get('top_p', 1.0), step=0.1, label="Top-P", interactive=True)
            max_tokens_slider = gr.Slider(minimum=1, maximum=4096, value=current_config.get('max_tokens', 1024), step=1, label="Макс. токены", interactive=True)
            model_path_tb = gr.Textbox(label="Путь к модели (для локальных моделей)", value=current_config.get('model_path', ''), interactive=True)
            save_settings_btn = gr.Button("Сохранить настройки")
            settings_status_md = gr.Markdown()

            # Динамическое обновление списка моделей
            model_type_dd.change(fn=update_model_choices, inputs=model_type_dd, outputs=model_name_dd)
            
            oracle_init_status = gr.Textbox(label="Статус инициализации Oracle", interactive=False)
            
            gr.Markdown("### Информация о схеме БД")
            with gr.Row():
                schema_tables_textbox = gr.Textbox(label="Таблицы (через запятую, опционально)")
                get_schema_btn = gr.Button("Получить схему")
            
            schema_display = gr.Textbox(label="Схема БД", lines=10, interactive=False)
            
            gr.Markdown("### Выполнение Text2SQL запроса")
            with gr.Row():
                text2sql_query_textbox = gr.Textbox(label="Запрос на естественном языке")
                text2sql_tables_textbox = gr.Textbox(label="Таблицы для запроса (опционально)")
            
            execute_sql_checkbox = gr.Checkbox(label="Выполнить SQL запрос", value=True)
            process_text2sql_btn = gr.Button("Выполнить Text2SQL")
            
            sql_display = gr.Textbox(label="Сгенерированный SQL", lines=5, interactive=False)
            results_display = gr.HTML(label="Результаты запроса")
            text2sql_status = gr.Textbox(label="Статус выполнения", interactive=False)

        # Обработчик для кнопки сохранения настроек
        save_settings_btn.click(
            fn=rag_assistant.update_llm_config,
            inputs=[model_type_dd, model_name_dd, temperature_slider, top_p_slider, max_tokens_slider, model_path_tb],
            outputs=[settings_status_md]
        )

        get_schema_btn.click(
            get_schema_info, 
            inputs=[schema_tables_textbox], 
            outputs=schema_display
        )
        
        process_text2sql_btn.click(
            process_text2sql_query, 
            inputs=[text2sql_query_textbox, text2sql_tables_textbox, execute_sql_checkbox], 
            outputs=[sql_display, results_display, text2sql_status]
        )

        # Обработчики событий
        # The user's message is the first input, and the history is the second.
        # The history is updated and returned to the chatbot component.
        submit_btn.click(chat_with_feedback, 
                         inputs=[msg, chatbot, session_state], 
                         outputs=[msg, chatbot, sources_display, session_state])
        msg.submit(chat_with_feedback, 
                   inputs=[msg, chatbot, session_state], 
                   outputs=[msg, chatbot, sources_display, session_state])
        
        # Кнопка очистки теперь также сбрасывает состояние сессии
        clear_btn.click(clear_chat, outputs=[chatbot, sources_display, feedback_status, session_state])
        
        feedback_btn.click(
            submit_feedback,
            inputs=[feedback_rating, feedback_comments, chatbot],
            outputs=feedback_status
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
        
        # Инициализация RAG-ассистента
        rag_assistant = RAGAssistant(memory_storage=memory_storage)

        # Инициализируем Oracle Text2SQL до запуска интерфейса
        logger.info("Initializing Oracle Text2SQL tool...")
        oracle_init_status = initialize_oracle_tool("openai", "gpt-4-turbo-preview")
        logger.info(f"Oracle Text2SQL status: {oracle_init_status}")
        
        # Создаем и запускаем Gradio интерфейс
        demo = create_demo(rag_assistant, memory_storage)

        # Запуск Gradio приложения
        logger.info("Запуск Gradio интерфейса...")
        demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7860, debug=True)
    
    except Exception as e:
        logger.error(f"Ошибка при запуске приложения: {str(e)}")
        print(f"Ошибка при запуске приложения: {str(e)}")
        sys.exit(1)
