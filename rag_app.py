import sys
import logging
from typing import List
import gradio as gr
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Local imports
from text_utils import format_source_display
from file_utils import clean_static_images
# Импорт класса для работы с Oracle Text2SQL
from oracle_text2sql import OracleText2SQL
from constants import *

from rag_assistant import RAGAssistant

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Инициализация компонентов
# Загружаем модель для эмбеддингов из data_processing.py
# Так как мы уже загрузили модель в data_processing.py, мы используем ее напрямую
from data_processing import EMBEDDING_MODEL
        
def chat_with_feedback(message, history):
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
        logger.info(f"Ответ: {response}")
        logger.info(f"Источники: {sources}")
        logger.info(f"ID запроса: {run_id}")
        # Запоминаем ID запроса для отзывов
        global LAST_RUN_ID
        LAST_RUN_ID = run_id
        
        # Форматируем источники для отображения
        sources_html = format_source_display(sources)
        
        # Добавляем вопрос пользователя в историю
        history.append({"role": "user", "content": message})
        
        # Проверяем наличие маркеров изображений в тексте ответа
        import re
        
        # Проверяем наличие тегов с путём к base64 файлу
        b64file_pattern = r'<B64FILE>(.*?)</B64FILE>'
        b64file_matches = re.findall(b64file_pattern, response)
        
        if b64file_matches:
            logger.info(f"Обнаружены маркеры B64FILE: {len(b64file_matches)}")
            
            # Удаляем теги из текста ответа
            cleaned_response = re.sub(b64file_pattern, '', response)
            history.append({"role": "assistant", "content": cleaned_response.strip()})
            
            # Обрабатываем каждый маркер
            for file_path in b64file_matches:
                try:
                    # Проверяем наличие файла
                    if not os.path.exists(file_path):
                        logger.error(f"Файл не найден: {file_path}")
                        # Добавляем текстовое сообщение об ошибке вместо отсутствующего изображения
                        history.append({"role": "assistant", "content": f"Ошибка: Изображение не найдено ({file_path})"})
                        continue
                    
                    # Стандартизируем путь к файлу для совместимости с разными ОС
                    normalized_path = os.path.normpath(file_path)
                    logger.info(f"Normalized path: {normalized_path}")
                    
                    # Читаем данные base64 из файла
                    with open(normalized_path, 'r') as f:
                        file_content = f.read().strip()
                    
                    logger.info(f"Успешно прочитан файл base64: {file_path} (длина: {len(file_content)} символов)")
                    
                    # Добавляем префикс, если его нет, т.к. он необходим для Gradio
                    if not file_content.startswith('data:image/png;base64,'):
                        base64_only = f"data:image/png;base64,{file_content}"
                        logger.info(f"Добавлен префикс 'data:image/png;base64,' к данным изображения")
                    else:
                        base64_only = file_content
                        logger.info(f"Данные изображения уже с префиксом data:image/png;base64,")
                    
                    logger.info(f"Подготовлены данные изображения с правильным форматом (длина: {len(base64_only)})")
                    logger.info(f"Добавляем изображение в историю чата")
                    logger.info(base64_only)
                    # Добавляем как изображение в историю чата
                    # Для Gradio Chatbot необходимо указать пустую строку вместо None для поля content
                    img = "<img src='" + base64_only + "' alt='Image'>"
                    history.append({"role": "assistant", "content": img})
                    logger.info(f"Добавлено изображение в чат (base64 данные длиной: {len(base64_only)})")
                except Exception as img_err:
                    logger.error(f"Ошибка при обработке base64 файла {file_path}: {str(img_err)}")
        else:
            logger.info(f"Добавляем ответ как текст, есть проблема с изображением")
            # Если нет маркеров, просто добавляем ответ как текст
            history.append({"role": "assistant", "content": response})
        
        # Ограничиваем длину истории
        if len(history) > MAX_HISTORY_LENGTH * 2:  # Умножаем на 2, так как каждая пара вопрос-ответ - это 2 элемента
            removed_count = len(history) - MAX_HISTORY_LENGTH * 2
            history = history[-MAX_HISTORY_LENGTH * 2:]
            logger.debug(f"История чата обрезана. Удалено {removed_count} старых сообщений")
        logger.info(f"История чата: {history}")
        logger.info(f"Источники: {sources_html}")
        # Очищаем директорию с изображениями после ответа
        clean_static_images()
        
        # Возвращаем ответ и обновляем историю
        return "", history, sources_html
    
    except Exception as e:
        error_msg = f"Ошибка при обработке запроса: {str(e)}"
        logger.error(error_msg, exc_info=True)
        history.append({"role": "assistant", "content": "Извините, произошла ошибка при обработке вашего запроса."})
        return "", history, ""

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
                            lines=1
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
        
        # Запуск Gradio интерфейса с поддержкой статических файлов
        # Используем другой порт, так как 7862 может быть занят
        demo.launch(server_port=7862, server_name="0.0.0.0", share=False, allowed_paths=["./static"])
    
    except Exception as e:
        logger.error(f"Ошибка при запуске приложения: {str(e)}")
        print(f"Ошибка при запуске приложения: {str(e)}")
        sys.exit(1)
