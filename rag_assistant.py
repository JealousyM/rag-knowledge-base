import json
import time
import os
import shutil
from typing import List, Dict, Any, Optional, Tuple, TypedDict
import uuid

from lang_chain_assistant import LangChainAssistant
from hybrid_search import HybridSearch
from rag_state import RAGState
from sql_tool import set_oracle_tool, get_oracle_tool

from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

# LangChain imports
from langchain_community.llms import LlamaCpp
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import initialize_agent, create_react_agent, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages, format_to_openai_functions
from langchain.tools import tool
from langchain.tools.convert_to_openai import format_tool_to_openai_function

# LangGraph imports
from langgraph.graph import END, StateGraph

# LangSmith tracing
from langsmith import trace

from prompts import SYSTEM_PROMPT
from oracle_text2sql import OracleText2SQL
from text_utils import format_search_results, format_source_display
from constants import *


from logging import getLogger
logger = getLogger(__name__)

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
            
            # Initialize Oracle Text2SQL with the main LLM model
            logger.info("Initializing Oracle Text2SQL...")
            oracle_tool_instance = None
            
            try:
                # Create Oracle tool with our main LLM model
                oracle_tool_instance = OracleText2SQL(
                    llm=self.llm,  # Pass the existing model
                    temperature=0.0
                )
                logger.info("Oracle Text2SQL tool created, connecting to database...")
                
                # Try to connect to the database with detailed logging
                connected = oracle_tool_instance.connect()
                
                if connected:
                    logger.info("Successfully connected to Oracle database")
                    
                    # Verify we can execute a simple query
                    try:
                        test_query = "SELECT 1 FROM DUAL"
                        results, error = oracle_tool_instance.execute_sql(test_query)
                        if error:
                            raise Exception(f"Query execution failed: {error}")
                        logger.info(f"Database connection test query successful. Result: {results}")
                        
                        # Register the tool in the sql_tool module
                        set_oracle_tool(oracle_tool_instance)
                        logger.info("Oracle tool registered successfully")
                        
                    except Exception as test_error:
                        logger.error(f"Database connection test query failed: {str(test_error)}")
                        logger.warning("Oracle tool will be disabled due to connection test failure")
                        oracle_tool_instance = None
                        
                else:
                    logger.error("Failed to establish database connection. Check connection parameters.")
                    oracle_tool_instance = None
                    
            except ImportError as ie:
                logger.error(f"Failed to import Oracle Text2SQL module: {str(ie)}")
                oracle_tool_instance = None
                
            except Exception as db_err:
                logger.error(f"Error initializing Oracle Text2SQL: {str(db_err)}", exc_info=True)
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
                        
                        # Закомментировано из-за ошибки ORA-01795: maximum number of expressions in a list is 1000
                        # schema_info = oracle_instance.get_schema_info()
                        # sql_query = oracle_instance.generate_sql(question, schema_info)
                        
                        # Генерируем SQL напрямую, без схемы
                        sql_query = oracle_instance.generate_sql(question, schema_override="")
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
            
            # Store the database_tool as an instance variable for direct access
            self.database_tool = database_tool
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
                            # Инициализируем переменные для хранения контента и метаданных
                            content = None
                            source_info = {}
                            
                            # Обработка ScoredPoint объектов на основе отладочных данных
                            if hasattr(doc, 'payload') and isinstance(doc.payload, dict):
                                # Извлекаем контент из payload
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
                                
                                # Извлекаем метаданные из payload
                                if 'metadata' in doc.payload and isinstance(doc.payload['metadata'], dict):
                                    source_info = doc.payload['metadata']
                                else:
                                    # Ищем ключи метаданных непосредственно в payload
                                    metadata_keys = ['source', 'title', 'url', 'filename', 'path', 'document_id']
                                    for key in metadata_keys:
                                        if key in doc.payload:
                                            source_info[key] = doc.payload[key]
                            
                            # Обработка стандартных документов LangChain
                            elif hasattr(doc, 'page_content'):
                                content = doc.page_content
                                # Пытаемся получить метаданные
                                if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                                    source_info = doc.metadata
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
                            
                            # Проверяем, если у объекта есть поле score, добавляем его в метаданные
                            if hasattr(doc, 'score'):
                                source_info['score'] = doc.score
                                
                        except Exception as e:
                            logger.error(f"Ошибка при извлечении содержимого документа: {str(e)}")
                            content = f"[Ошибка при извлечении текста: {str(e)}]"
                            source_info = {}
                        
                        # Форматируем источник с метаданными
                        source_header = f"**Источник {i}**"
                        
                        # Добавляем информацию об источнике, если она есть
                        if source_info:
                            source_details = []
                            if 'source' in source_info:
                                source_details.append(f"Источник: {source_info['source']}")
                            if 'title' in source_info:
                                source_details.append(f"Название: {source_info['title']}")
                            if 'url' in source_info:
                                source_details.append(f"URL: {source_info['url']}")
                            if 'filename' in source_info:
                                source_details.append(f"Файл: {source_info['filename']}")
                            if 'path' in source_info:
                                source_details.append(f"Путь: {source_info['path']}")
                            if 'document_id' in source_info:
                                source_details.append(f"ID документа: {source_info['document_id']}")
                            if 'score' in source_info:
                                source_details.append(f"Релевантность: {source_info['score']:.2f}")
                            
                            # Если есть детали источника, добавляем их
                            if source_details:
                                source_header += "\n" + "\n".join(source_details)
                        
                        context_text += f"{source_header}\n{content}\n\n"
                    
                    return context_text
            except Exception as ex:
                logger.error(f"Ошибка в rag_tool: {str(ex)}", exc_info=True)
                return f"Произошла ошибка при поиске в документах: {str(ex)}"
        
        tools.append(rag_tool)
        logger.info("Инструмент rag_tool создан и добавлен")
        
        # Создаем инструмент для визуализации графа
        @tool
        def show_graph_tool(query: str = "") -> str:
            """Показывает структуру графа агентов и инструментов в системе. 
            Используй этот инструмент, когда пользователь хочет увидеть архитектуру системы или как связаны компоненты."""
            logger.info(f"Вызов инструмента show_graph_tool")
        
            # Визуализируем граф и сохраняем его как изображение для отображения в чате
            try:
                import os
                import base64
                from pathlib import Path
                from datetime import datetime
            
                # Путь для сохранения статического изображения
                static_dir = Path("./static/images")
                static_dir.mkdir(exist_ok=True, parents=True)
            
                # Создаем уникальное имя файла с тиместампом
                file_name = f"graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                image_path = static_dir / file_name
            
                # Отображаем визуализацию графа, если граф доступен
                if self.graph and hasattr(self.graph, 'get_graph'):
                    logger.info("Генерация визуализации графа и сохранение в файл")
                
                    # Сохраняем изображение графа
                    png_data = self.graph.get_graph().draw_mermaid_png()
            
                    # Сохраняем PNG в файл
                    with open(image_path, "wb") as f:
                        f.write(png_data)
                    
                    logger.info(f"Изображение графа сохранено: {image_path}")
                    
                    # Кодируем в base64 и сохраняем в отдельный файл
                    b64_data = base64.b64encode(png_data).decode('ascii')
                    base64_file_name = f"graph_b64_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    base64_file_path = static_dir / base64_file_name
                    
                    # Сохраняем base64 данные в файл
                    with open(base64_file_path, "w") as f:
                        f.write(f"data:image/png;base64,{b64_data}")
                    
                    # Преобразуем путь к файлу для корректной работы с ОС
                    standard_path = str(base64_file_path).replace('\\', '/')
                    
                    logger.info(f"Base64 данные изображения сохранены в: {base64_file_path}")
                    
                    # Добавляем специальный маркер с путем к файлу base64
                    text_description = f"""
## Структура системы RAG

<B64FILE>{standard_path}</B64FILE>

### Агенты
- **Роутер-агент** - Определяет тип запроса и выбирает специализированного агента
- **SQL-агент** - Запросы к базе данных через Oracle
- **RAG-агент** - Поиск в документах через векторное хранилище

### Инструменты
- **database_tool** - Запросы к Oracle базе данных
- **rag_tool** - Векторный поиск в документах
- **show_graph_tool** - Визуализация графа системы

### Технологии
- LangChain + LangGraph для оркестрации агентов
- Oracle Text2SQL для работы с базой данных
- Qdrant + BM25 для гибридного поиска
"""
                
                    return text_description
                else:
                    logger.warning("Граф недоступен для визуализации")
                    return "Извините, граф системы недоступен для визуализации."
            except Exception as e:
                logger.error(f"Ошибка при создании визуализации графа: {str(e)}", exc_info=True)
                return f"### Ошибка при визуализации графа\n\n```\n{str(e)}\n```\n\nПроверьте логи для получения дополнительной информации."
        
        # Store the show_graph_tool as an instance variable for direct access
        self.show_graph_tool = show_graph_tool
        tools.append(show_graph_tool)
        logger.info("Инструмент show_graph_tool создан и добавлен")
        
        return tools
    
    def _initialize_agents(self):
        """
        Инициализирует роутер и специализированные ReAct агенты с улучшенной обработкой ошибок и логированием.
        
        Создает:
        - Роутер для определения типа запроса (database/documents/direct)
        - SQL агент для запросов к базе данных (если доступен)
        - RAG агент для работы с документами
        - Общий агент для прямых ответов
        """
        logger.info("Начало инициализации агентов...")
        
        # Инициализируем все агенты в None
        self.router_chain = None
        self.sql_agent_executor = None
        self.rag_agent_executor = None
        self.general_agent_executor = None
        
        try:
            
            # 1. Инициализация роутера для маршрутизации запросов
            logger.info("Инициализация роутера запросов...")
            
            try:
                if self.supports_functions:
                    logger.info("Создание роутера с поддержкой function-calling")
                    
                    # Импортируем необходимые компоненты
                    from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
                    from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
                    from prompts import ROUTER_AGENT_PROMPT
                    
                    # Создаем промпт для роутера с валидацией
                    if not hasattr(self, 'llm') or self.llm is None:
                        raise ValueError("LLM не инициализирован для роутера")
                        
                    logger.debug("Создание шаблона промпта для роутера")
                    router_prompt = ChatPromptTemplate.from_messages([
                        ("system", ROUTER_AGENT_PROMPT),
                        ("human", "{question}"),
                        ("ai", "{agent_scratchpad}")
                    ])
                    
                    # Определяем схему функции для маршрутизации
                    logger.debug("Определение схемы функции маршрутизации")
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
                                        "description": "Объяснение выбора маршрута"
                                    }
                                },
                                "required": ["route", "reasoning"]
                            }
                        }
                    ]
                    
                    # Создаем цепочку роутера с обработкой ошибок
                    try:
                        logger.debug("Привязка функций к LLM для роутера")
                        self.llm_with_routing = self.llm.bind(functions=router_functions)
                        
                        logger.debug("Создание цепочки роутера")
                        self.router_chain = (
                            {"question": RunnablePassthrough(), "agent_scratchpad": lambda x: ""} 
                            | router_prompt 
                            | self.llm_with_routing 
                            | JsonOutputFunctionsParser()
                        )
                        
                        # Проверяем, что роутер был создан корректно
                        if not hasattr(self, 'router_chain') or self.router_chain is None:
                            raise RuntimeError("Не удалось создать цепочку роутера")
                            
                        logger.info("Роутер с function-calling успешно инициализирован")
                        
                    except Exception as chain_err:
                        logger.error(f"Ошибка при создании цепочки роутера: {str(chain_err)}", exc_info=True)
                        raise RuntimeError(f"Не удалось создать цепочку роутера: {str(chain_err)}")
                    
                else:
                    # Fallback для моделей без поддержки function-calling
                    logger.warning("Функция function-calling не поддерживается, используется упрощенный роутер")
                    from langchain.prompts import PromptTemplate
                    from langchain.chains import LLMChain
                    
                    router_template = """
                    Определи тип запроса и верни один из вариантов: database, documents или direct.
                    
                    Примеры:
                    - Запрос: "Сколько пользователей в системе?" → database
                    - Запрос: "Найди информацию о политике конфиденциальности" → documents
                    - Запрос: "Привет, как дела?" → direct
                    
                    Запрос: {question}
                    Ответ (только одно слово: database, documents или direct):
                    """
                    
                    router_prompt = PromptTemplate(
                        template=router_template,
                        input_variables=["question"]
                    )
                    
                    self.router_chain = LLMChain(
                        llm=self.llm,
                        prompt=router_prompt,
                        verbose=True
                    )
                    logger.info("Упрощенный роутер успешно инициализирован")
                    
            except Exception as router_err:
                logger.error(f"Критическая ошибка при инициализации роутера: {str(router_err)}", exc_info=True)
                # Создаем заглушку для роутера, которая всегда возвращает 'direct'
                self.router_chain = lambda x: {"route": "direct", "reasoning": "Ошибка инициализации роутера"}
                logger.warning("Роутер заменен на заглушку, все запросы будут направлены на общий агент")
                
            # 2. Инициализация SQL агента для работы с базой данных
            if self.supports_functions and self.oracle_tool:
                logger.info("Инициализация SQL ReAct агента...")
                
                try:
                    from prompts import SQL_AGENT_PROMPT
                    
                    # Проверяем наличие необходимых инструментов
                    sql_tools = [tool for tool in self.tools if hasattr(tool, 'name') and tool.name == "database_tool"]
                    if not sql_tools:
                        logger.warning("Не найдены инструменты для работы с базой данных")
                        raise ValueError("Не найдены инструменты для работы с базой данных")
                    
                    logger.debug("Создание промпта для SQL агента")
                    sql_prompt = ChatPromptTemplate.from_messages([
                        ("system", SQL_AGENT_PROMPT),
                        ("human", "{input}"),
                        ("ai", "{agent_scratchpad}")
                    ])
                    
                    # Функция для форматирования скретчпада агента
                    def _format_sql_scratchpad(steps):
                        try:
                            return format_to_openai_function_messages(steps)
                        except Exception as e:
                            logger.error(f"Ошибка при форматировании скретчпада SQL агента: {str(e)}", exc_info=True)
                            return []
                    
                    # Создаем и собираем агента с обработкой ошибок
                    try:
                        logger.debug("Создание SQL агента")
                        sql_agent = (
                            {
                                "input": lambda x: x["input"],
                                "agent_scratchpad": lambda x: _format_sql_scratchpad(x.get("intermediate_steps", [])),
                                "tools": lambda x: sql_tools
                            }
                            | sql_prompt
                            | self.llm.bind(functions=[format_tool_to_openai_function(tool) for tool in sql_tools])
                            | OpenAIFunctionsAgentOutputParser()
                        )
                        
                        # Создаем исполнитель агента с обработкой ошибок
                        logger.debug("Создание исполнителя SQL агента")
                        self.sql_agent_executor = AgentExecutor(
                            agent=sql_agent, 
                            tools=sql_tools, 
                            verbose=True,
                            handle_parsing_errors=True,
                            max_iterations=10,
                            early_stopping_method="generate"
                        )
                        
                        # Проверяем, что агент был создан корректно
                        if not hasattr(self, 'sql_agent_executor') or self.sql_agent_executor is None:
                            raise RuntimeError("Не удалось создать исполнителя SQL агента")
                            
                        logger.info("SQL ReAct агент успешно инициализирован")
                        
                    except Exception as agent_err:
                        logger.error(f"Ошибка при создании SQL агента: {str(agent_err)}", exc_info=True)
                        self.sql_agent_executor = None
                        raise RuntimeError(f"Не удалось инициализировать SQL агент: {str(agent_err)}")
                    
                except ImportError as ie:
                    logger.error(f"Не удалось импортировать необходимые модули для SQL агента: {str(ie)}", exc_info=True)
                    self.sql_agent_executor = None
                    logger.warning("SQL агент отключен из-за ошибки импорта")
                    
                except Exception as e:
                    logger.error(f"Критическая ошибка при инициализации SQL агента: {str(e)}", exc_info=True)
                    self.sql_agent_executor = None
                    logger.warning("SQL агент отключен из-за ошибки инициализации")
            else:
                reason = ""
                if not self.supports_functions:
                    reason = " (отсутствует поддержка function-calling)"
                elif not self.oracle_tool:
                    reason = " (инструмент Oracle не инициализирован)"
                logger.info(f"SQL ReAct агент не будет создан{reason}")
                self.sql_agent_executor = None
                
            # 3. Инициализация RAG агента для работы с документами
            if self.supports_functions:
                logger.info("Инициализация RAG ReAct агента...")
                
                try:
                    from prompts import GENERAL_AGENT_PROMPT
                    
                    # Проверяем наличие необходимых инструментов
                    rag_tools = [tool for tool in self.tools if hasattr(tool, 'name') and tool.name == "rag_tool"]
                    if not rag_tools:
                        logger.warning("Не найдены инструменты для работы с документами")
                        raise ValueError("Не найдены инструменты для работы с документами")
                    
                    logger.debug("Создание промпта для RAG агента")
                    rag_prompt = ChatPromptTemplate.from_messages([
                        ("system", GENERAL_AGENT_PROMPT),
                        ("human", "{input}"),
                        ("ai", "{agent_scratchpad}")
                    ])
                    
                    # Функция для форматирования скретчпада агента
                    def _format_rag_scratchpad(steps):
                        try:
                            return format_to_openai_function_messages(steps)
                        except Exception as e:
                            logger.error(f"Ошибка при форматировании скретчпада RAG агента: {str(e)}", exc_info=True)
                            return []
                    
                    # Создаем и собираем агента с обработкой ошибок
                    try:
                        logger.debug("Создание RAG агента")
                        rag_agent = (
                            {
                                "input": lambda x: x["input"],
                                "agent_scratchpad": lambda x: _format_rag_scratchpad(x.get("intermediate_steps", [])),
                                "tools": lambda x: rag_tools
                            }
                            | rag_prompt
                            | self.llm.bind(functions=[format_tool_to_openai_function(tool) for tool in rag_tools])
                            | OpenAIFunctionsAgentOutputParser()
                        )
                        
                        # Создаем исполнитель агента с обработкой ошибок
                        logger.debug("Создание исполнителя RAG агента")
                        self.rag_agent_executor = AgentExecutor(
                            agent=rag_agent, 
                            tools=rag_tools, 
                            verbose=True,
                            handle_parsing_errors=True,
                            max_iterations=10,
                            early_stopping_method="generate"
                        )
                        
                        # Проверяем, что агент был создан корректно
                        if not hasattr(self, 'rag_agent_executor') or self.rag_agent_executor is None:
                            raise RuntimeError("Не удалось создать исполнителя RAG агента")
                            
                        logger.info("RAG ReAct агент успешно инициализирован")
                        
                    except Exception as agent_err:
                        logger.error(f"Ошибка при создании RAG агента: {str(agent_err)}", exc_info=True)
                        self.rag_agent_executor = None
                        raise RuntimeError(f"Не удалось инициализировать RAG агент: {str(agent_err)}")
                    
                except ImportError as ie:
                    logger.error(f"Не удалось импортировать необходимые модули для RAG агента: {str(ie)}", exc_info=True)
                    self.rag_agent_executor = None
                    logger.warning("RAG агент отключен из-за ошибки импорта")
                    
                except Exception as e:
                    logger.error(f"Критическая ошибка при инициализации RAG агента: {str(e)}", exc_info=True)
                    self.rag_agent_executor = None
                    logger.warning("RAG агент отключен из-за ошибки инициализации")
            else:
                logger.info("RAG ReAct агент не будет создан (отсутствует поддержка function-calling)")
                self.rag_agent_executor = None
            
            # 4. Инициализация общего агента для прямых ответов
            logger.info("Инициализация общего ReAct агента...")
            
            try:
                from prompts import GENERAL_AGENT_PROMPT
                
                # Проверяем наличие необходимых инструментов
                if not hasattr(self, 'tools') or not self.tools:
                    logger.warning("Не найдены инструменты для общего агента")
                    general_tools = []
                else:
                    general_tools = self.tools
                
                logger.debug("Создание промпта для общего агента")
                general_prompt = ChatPromptTemplate.from_messages([
                    ("system", GENERAL_AGENT_PROMPT),
                    ("human", "{input}"),
                    ("ai", "{agent_scratchpad}")
                ])
                
                # Функция для форматирования скретчпада агента
                def _format_general_scratchpad(steps):
                    try:
                        return format_to_openai_function_messages(steps)
                    except Exception as e:
                        logger.error(f"Ошибка при форматировании скретчпада общего агента: {str(e)}", exc_info=True)
                        return []
                
                # Создаем и собираем агента с обработкой ошибок
                try:
                    logger.debug("Создание общего агента")
                    general_agent = (
                        {
                            "input": lambda x: x["input"],
                            "agent_scratchpad": lambda x: _format_general_scratchpad(x.get("intermediate_steps", [])),
                            "tools": lambda x: general_tools
                        }
                        | general_prompt
                        | self.llm.bind(functions=[format_tool_to_openai_function(tool) for tool in general_tools])
                        | OpenAIFunctionsAgentOutputParser()
                    )
                    
                    # Создаем исполнитель агента с обработкой ошибок
                    logger.debug("Создание исполнителя общего агента")
                    self.general_agent_executor = AgentExecutor(
                        agent=general_agent, 
                        tools=general_tools, 
                        verbose=True,
                        handle_parsing_errors=True,
                        max_iterations=10,
                        early_stopping_method="generate"
                    )
                    
                    # Проверяем, что агент был создан корректно
                    if not hasattr(self, 'general_agent_executor') or self.general_agent_executor is None:
                        raise RuntimeError("Не удалось создать исполнителя общего агента")
                        
                    logger.info("Общий ReAct агент успешно инициализирован")
                    
                except Exception as agent_err:
                    logger.error(f"Ошибка при создании общего агента: {str(agent_err)}", exc_info=True)
                    self.general_agent_executor = None
                    raise RuntimeError(f"Не удалось инициализировать общий агент: {str(agent_err)}")
                
            except ImportError as ie:
                logger.error(f"Не удалось импортировать необходимые модули для общего агента: {str(ie)}", exc_info=True)
                self.general_agent_executor = None
                logger.warning("Общий агент отключен из-за ошибки импорта")
                
            except Exception as e:
                logger.error(f"Критическая ошибка при инициализации общего агента: {str(e)}", exc_info=True)
                self.general_agent_executor = None
                logger.warning("Общий агент отключен из-за ошибки инициализации")
            else:
                # Для моделей без поддержки функций используем fallback на LangGraph
                self.general_agent_executor = None
                logger.info("Общий ReAct агент не создан (нет поддержки function-calling)")
            
            # Проверяем, что хотя бы один агент был создан успешно
            active_agents = [
                ("SQL", self.sql_agent_executor),
                ("RAG", self.rag_agent_executor),
                ("General", self.general_agent_executor)
            ]
            
            active_count = sum(1 for name, agent in active_agents if agent is not None)
            
            if active_count == 0:
                logger.error("Ни один агент не был инициализирован успешно")
                raise RuntimeError("Не удалось инициализировать ни одного агента. Проверьте логи для деталей.")
            
            # Логируем статус инициализации агентов
            logger.info("=" * 50)
            logger.info("СТАТУС ИНИЦИАЛИЗАЦИИ АГЕНТОВ:")
            for name, agent in active_agents:
                status = "АКТИВЕН" if agent is not None else "НЕ АКТИВЕН"
                logger.info(f"- {name} агент: {status}")
            logger.info("=" * 50)
            
            logger.info(f"Успешно инициализировано {active_count} из {len(active_agents)} агентов")
        
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
                try:
                    logger.info("Инициализация общего агента...")
                    
                    # Создаем промпт для общего агента
                    general_prompt = ChatPromptTemplate.from_messages([
                        ("system", GENERAL_AGENT_PROMPT),
                        ("human", "{input}"),
                        ("ai", "{agent_scratchpad}")
                    ])
                    
                    # Определяем инструменты для общего агента (все инструменты)
                    general_tools = self.tools if hasattr(self, 'tools') and self.tools else []
                    
                    # Форматируем скретчпад агента с обработкой ошибок
                    def _format_general_scratchpad(steps):
                        try:
                            return format_to_openai_function_messages(steps)
                        except Exception as format_err:
                            logger.error(f"Ошибка форматирования скретчпада: {str(format_err)}", exc_info=True)
                            return []
                    
                    # Создаем и собираем агента
                    logger.debug("Создание цепочки общего агента")
                    general_agent = (
                        {
                            "input": lambda x: x["input"],
                            "agent_scratchpad": lambda x: _format_general_scratchpad(x.get("intermediate_steps", [])),
                            "tools": lambda x: general_tools
                        }
                        | general_prompt
                        | self.llm.bind(functions=[format_tool_to_openai_function(tool) for tool in general_tools])
                        | OpenAIFunctionsAgentOutputParser()
                    )
                    
                    # Создаем исполнитель агента с обработкой ошибок
                    logger.debug("Создание исполнителя общего агента")
                    self.general_agent_executor = AgentExecutor(
                        agent=general_agent,
                        tools=general_tools,
                        verbose=True,
                        handle_parsing_errors=True,
                        max_iterations=10,
                        early_stopping_method="generate"
                    )
                    
                    logger.info("Общий агент успешно инициализирован")
                    
                except Exception as agent_err:
                    logger.error(f"Ошибка при инициализации общего агента: {str(agent_err)}", exc_info=True)
                    self.general_agent_executor = None
                    
                    # Пытаемся создать хотя бы базового агента в качестве запасного варианта
                    try:
                        logger.warning("Попытка создать базового агента в качестве запасного варианта...")
                        from langchain.agents import initialize_agent, AgentType
                        from langchain.chat_models import ChatOpenAI
                        
                        llm = ChatOpenAI(temperature=0, model_name="gpt-4") if self.llm is None else self.llm
                        self.general_agent_executor = initialize_agent(
                            [],  # Без инструментов
                            llm,
                            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                            verbose=True
                        )
                        logger.info("Базовый запасной агент успешно создан")
                    except Exception as fallback_err:
                        logger.critical(f"Не удалось создать запасного агента: {str(fallback_err)}", exc_info=True)
                        raise RuntimeError("Не удалось инициализировать ни одного агента") from agent_err
        
        except Exception as e:
            logger.critical(f"Критическая ошибка при инициализации агентов: {str(e)}", exc_info=True)
            
            # Проверяем, что хотя бы один агент был создан успешно
            active_agents = [
                ("SQL", getattr(self, 'sql_agent_executor', None)),
                ("RAG", getattr(self, 'rag_agent_executor', None)),
                ("General", getattr(self, 'general_agent_executor', None))
            ]
            
            active_count = sum(1 for name, agent in active_agents if agent is not None)
            
            if active_count == 0:
                logger.error("Ни один агент не был инициализирован успешно")
                raise RuntimeError("Не удалось инициализировать ни одного агента. Проверьте логи для деталей.") from e
            
            # Логируем статус инициализации агентов
            logger.warning("Некоторые агенты не были инициализированы, но работа продолжается...")
            logger.info("=" * 50)
            logger.info("СТАТУС ИНИЦИАЛИЗАЦИИ АГЕНТОВ:")
            for name, agent in active_agents:
                status = "АКТИВЕН" if agent is not None else "НЕ АКТИВЕН"
                logger.info(f"- {name} агент: {status}")
            logger.info("=" * 50)
        
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
                formatted_context, sources = format_search_results(search_results)
                
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
        
    def _create_rag_graph(self):
        """
        Создает и настраивает граф LangGraph для RAG процесса
        
        Returns:
            Скомпилированный граф LangGraph
        """
        logger.info("Создание графа LangGraph для RAG процесса...")
        
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
                formatted_context, sources = format_search_results(search_results)
                
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
        
        # Создаем новый граф
        from langgraph.graph import StateGraph, END
        
        # Инициализируем граф с нашим состоянием
        workflow = StateGraph(RAGState)
        
        # Добавляем узлы в граф
        workflow.add_node("retrieve", retrieve_documents)
        workflow.add_node("generate", generate_answer)
        
        # Определяем порядок выполнения
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # Создаем и компилируем граф
        try:
            graph = workflow.compile()
            logger.info("Создан и скомпилирован LangGraph для RAG")
            return graph
        except Exception as e:
            logger.error(f"Ошибка при компиляции графа: {str(e)}")
            raise
    
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
        b64file_markers = []  # Для сохранения маркеров <B64FILE>
        
        # Debug logging for agent initialization status
        logger.info("\n" + "="*50)
        logger.info("AGENT INITIALIZATION STATUS:")
        logger.info(f"Oracle tool initialized: {self.oracle_tool is not None}")
        logger.info(f"Supports functions: {self.supports_functions}")
        logger.info(f"SQL Agent Executor: {self.sql_agent_executor is not None}")
        logger.info(f"RAG Agent Executor: {self.rag_agent_executor is not None}")
        logger.info(f"General Agent Executor: {self.general_agent_executor is not None}")
        logger.info("="*50 + "\n")
        b64file_markers = []  # Для сохранения маркеров <B64FILE>
        
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
            
            # Проверяем, содержит ли запрос просьбу показать граф или визуализацию системы
            show_graph_keywords = [
                'показать граф', 'показать структуру', 'визуализировать систему', 
                'архитектура системы', 'диаграмма системы', 'как устроена система',
                'покажи граф', 'визуализация графа', 'структура агентов'
            ]
            is_graph_request = any(keyword.lower() in query.lower() for keyword in show_graph_keywords)
            
            # Если запрос явно о графе, принудительно использовать общий агент с show_graph_tool
            if is_graph_request:
                logger.info("Обнаружен запрос на визуализацию графа системы, принудительное использование show_graph_tool")
                route = "graph"  # Принудительно используем общий агент
            
            # Шаг 2: Выполнение запроса с соответствующим агентом
            if route == "database":
                logger.info(f"Маршрутизация к базе данных: {query[:100]}...")
                
                # Используем сохраненный экземпляр database_tool
                if hasattr(self, 'database_tool') and self.database_tool:
                    try:
                        logger.info("Использование database_tool для выполнения запроса к БД")
                        with trace(name="database_tool_execution") as db_run:
                            # Вызываем database_tool напрямую с вопросом пользователя
                            db_response = self.database_tool(query)
                            answer = str(db_response)
                            if db_run:
                                run_id = db_run.id
                        
                        # Если ответ содержит ошибку, логируем и пробуем fallback
                        if any(err in answer.lower() for err in ["error", "ошибка", "exception"]):
                            logger.warning(f"Database tool вернул ошибку: {answer}")
                            raise Exception(answer)
                            
                        # Добавляем префикс к ответу, чтобы было ясно, что он пришел из БД
                        answer = f"🔍 Результат запроса к базе данных:\n\n{answer}"
                        
                    except Exception as e:
                        logger.error(f"Ошибка при выполнении запроса через database_tool: {str(e)}", exc_info=True)
                        # Fallback к SQL агенту, если доступен
                        if self.sql_agent_executor:
                            try:
                                logger.info("Попытка использовать SQL агент как fallback...")
                                with trace(name="fallback_sql_agent_execution") as sql_run:
                                    response = self.sql_agent_executor.invoke({"input": query})
                                    answer = response.get("output", "")
                                    if sql_run:
                                        run_id = sql_run.id
                            except Exception as sql_err:
                                logger.error(f"Ошибка при выполнении SQL агента: {str(sql_err)}", exc_info=True)
                                answer = f"Ошибка при обработке запроса к базе данных: {str(e)}"
                                route = "direct"  # Fallback к общему агенту
                        else:
                            answer = f"Ошибка при обработке запроса к базе данных: {str(e)}"
                            route = "direct"  # Fallback к общему агенту
                else:
                    logger.warning("database_tool не найден в списке инструментов")
                    answer = "Ошибка: инструмент для работы с базой данных недоступен"
                    route = "direct"  # Fallback к общему агенту
                
                # Если после обработки маршрут изменился на direct, используем общий агент
                """ if route == "direct" and self.general_agent_executor:
                    try:
                        with trace(name="fallback_general_agent_execution") as fallback_run:
                            response = self.general_agent_executor.invoke({"input": query})
                            answer = response.get("output", answer)
                            if fallback_run:
                                run_id = fallback_run.id
                    except Exception as fallback_err:
                        logger.error(f"Ошибка при выполнении общего агента: {str(fallback_err)}", exc_info=True)
                        answer = f"Произошла ошибка при обработке вашего запроса: {str(fallback_err)}"
 """                
                # Для запросов визуализации графа мы вызываем show_graph_tool напрямую и возвращаем его результат
                # Для обычных запросов используем стандартное выполнение агента
                with trace(name="general_agent_execution") as general_run:
                    response = self.general_agent_executor.invoke({"input": query})
                    
                    # Проверяем промежуточные шаги и логи
                    original_tool_output = None
                    if 'intermediate_steps' in response:
                        logger.info("Проверка промежуточных шагов выполнения агента для маркеров B64FILE")
                        for step in response['intermediate_steps']:
                            if len(step) >= 2:
                                tool_name = getattr(step[0], 'tool', None) or getattr(step[0], 'name', 'unknown')
                                tool_output = str(step[1])
                                
                                logger.info(f"Проверка вывода инструмента {tool_name} на наличие B64FILE маркеров")
                                
                                # Если это вывод с графическим содержимым, ищем маркеры
                                import re
                                b64_matches = re.findall(r'<B64FILE>(.*?)</B64FILE>', tool_output)
                                if b64_matches:
                                    logger.info(f"Найдены B64FILE маркеры в выводе инструмента: {b64_matches}")
                                    b64file_markers.extend(b64_matches)
                                    # Сохраняем оригинальный вывод инструмента
                                    original_tool_output = tool_output
                                    
                    # Получаем ответ от агента
                    answer = response.get("output", "")
                    
                    # Если были найдены маркеры B64FILE, но их нет в ответе,
                    # добавляем их в ответ
                    if b64file_markers and not re.search(r'<B64FILE>', answer):
                        logger.info("В ответе нет B64FILE маркеров, но они были найдены в выводе инструментов")
                        
                        # Скорее всего, лучше вернуть оригинальный вывод инструмента
                        if original_tool_output:
                            logger.info("Заменяем ответ модели оригинальным выводом инструмента")
                            answer = original_tool_output
                        else:
                            # Добавляем маркеры в ответ
                            for marker in b64file_markers:
                                answer = f"<B64FILE>{marker}</B64FILE>\n\n{answer}"
                            logger.info(f"Модифицированный ответ с маркерами: {answer[:100]}...")
                    
                    if general_run:
                        run_id = general_run.id
                
                if "sources" in response:
                    sources = response["sources"]
                if general_run:
                    run_id = general_run.id
            elif route == "graph":
                logger.info(f"Маршрутизация к визуализации графа: {query[:100]}...")
                logger.info(hasattr(self, 'show_graph_tool'))
                logger.info(self.show_graph_tool)
                # Используем сохраненный экземпляр show_graph_tool
                if hasattr(self, 'show_graph_tool') and self.show_graph_tool:
                    try:
                        logger.info("Использование show_graph_tool для визуализации графа")
                        with trace(name="show_graph_tool_execution") as graph_run:
                            # Вызываем show_graph_tool напрямую с вопросом пользователя
                            graph_response = self.show_graph_tool(query)
                            answer = str(graph_response)
                            if graph_run:
                                run_id = graph_run.id
                        
                        # Если ответ содержит ошибку, логируем и пробуем fallback
                        if any(err in answer.lower() for err in ["error", "ошибка", "exception"]):
                            logger.warning(f"Show graph tool вернул ошибку: {answer}")
                            answer = f"Произошла ошибка при обработке вашего запроса: {str(err)}"
                            run_id = None
                    except Exception as e:
                        logger.error(f"Ошибка при выполнении show_graph_tool: {str(e)}", exc_info=True)
                        answer = f"Произошла ошибка при обработке вашего запроса: {str(e)}"
                        run_id = None
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
            logger.info(f"Источники для отображения: {sources_html}")
            logger.info(f"Ответ: {answer}")
            logger.info(f"Источники: {sources}")
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
