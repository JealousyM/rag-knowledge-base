import os

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
