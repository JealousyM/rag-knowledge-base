from __future__ import annotations

"""LangChain StructuredTool wrapper вокруг OracleText2SQL.
Позволяет агенту с function-calling автоматически генерировать SQL и выполнять запросы.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from langchain.tools import StructuredTool
from pydantic import BaseModel

# Setup logging
logger = logging.getLogger(__name__)

# Избегаем циклического импорта - oracle_tool будет передан извне
# через функцию set_oracle_tool

# Глобальная переменная для хранения ссылки на инструмент
_oracle_tool_instance = None

def set_oracle_tool(tool_instance):
    """Установить глобальный экземпляр Oracle Text2SQL инструмента"""
    global _oracle_tool_instance
    _oracle_tool_instance = tool_instance
    logger.info(f"Oracle Tool успешно установлен: {tool_instance is not None}")
    return tool_instance is not None

def get_oracle_tool():
    """Получить текущий экземпляр Oracle Text2SQL инструмента"""
    return _oracle_tool_instance

logger = logging.getLogger(__name__)

class SQLQueryInput(BaseModel):
    """Схема входных данных для oracle_text2sql tool"""

    question: str


def _run_sql_tool(question: str) -> str:
    """Генерирует SQL запрос на основе вопроса и возвращает результаты выполнения.

    Формат результата — готовый для вывода в чат Markdown-текст.
    """
    # Получаем текущий экземпляр Oracle Text2SQL инструмента
    oracle_tool_instance = get_oracle_tool()

    if oracle_tool_instance is None:
        return "Oracle Text2SQL не инициализирован."

    # Получаем текстовую схему БД (можно оптимизировать кеширование)
    # Закомментировано из-за ошибки ORA-01795: maximum number of expressions in a list is 1000
    # schema_info = oracle_tool_instance.get_schema_info()

    try:
        # Генерируем SQL напрямую, без схемы
        sql_query = oracle_tool_instance.generate_sql(question, schema_override="")
        sql_query = sql_query.strip().rstrip(";")  # убираем ; в конце
        logger.info("\u0421\u0433\u0435\u043d\u0435\u0440\u0438\u0440\u043e\u0432\u0430\u043d SQL \u0434\u043b\u044f question '%s': %s", question, sql_query)

        rows, error = oracle_tool_instance.execute_sql(sql_query)
        if error:
            return f"Ошибка при выполнении запроса: {error}"

        # Формируем JSON-строку результатов, чтобы агент мог при желании проанализировать
        rows_json = json.dumps(rows, ensure_ascii=False, indent=2)
        answer = (
            f"SQL запрос:\n```sql\n{sql_query}\n```\n\n"
            f"Результаты (JSON):\n```json\n{rows_json}\n```"
        )
        return answer
    except Exception as exc:  # noqa: BLE001
        logger.exception("oracle_text2sql tool error")
        return f"Ошибка: {exc}"


def get_oracle_sql_tool() -> StructuredTool:
    """Возвращает объект StructuredTool для регистрации в агентах."""

    return StructuredTool.from_function(
        name="oracle_text2sql",
        description=(
            "Генерирует SQL запрос к корпоративной Oracle БД по вопросу на русском/английском языке и "
            "возвращает результаты выполнения в формате JSON."
        ),
        func=_run_sql_tool,
        args_schema=SQLQueryInput,
    )
