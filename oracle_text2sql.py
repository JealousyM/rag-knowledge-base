import os
import logging
import json
import oracledb
from typing import List, Dict, Any, Optional, Union, Tuple

from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langsmith import Client, trace

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Шаблон промпта для генерации SQL запросов
SQL_GENERATION_PROMPT = """
Ты - опытный SQL разработчик, специализирующийся на базах данных Oracle.
Твоя задача - преобразовать запрос на естественном языке в корректный SQL-запрос для базы данных Oracle.

### Схема базы данных:
{schema}

### Правила:
1. Используй синтаксис Oracle SQL
2. Убедись, что запрос синтаксически корректен и оптимизирован
3. Используй соответствующие JOIN-ы, когда необходимо
4. Добавь комментарии к сложным частям запроса
5. Возвращай ТОЛЬКО SQL-запрос без дополнительных пояснений или форматирования

### Запрос на естественном языке:
{query}

### SQL запрос:
"""

class OracleText2SQL:
    """
    Класс для подключения к Oracle и выполнения преобразования текста в SQL запросы
    с последующим выполнением этих запросов.
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = 1521,
        service_name: str = None,
        user: str = None,
        password: str = None,
        llm=None,
        model_type: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        llm_local_path: str = None
    ):
        """
        Инициализация подключения к Oracle и настройка LLM для генерации SQL запросов.
        
        Args:
            host: Хост Oracle сервера
            port: Порт Oracle сервера (по умолчанию 1521)
            service_name: Имя сервиса Oracle
            user: Имя пользователя
            password: Пароль
            llm: Готовый экземпляр LLM (если передан, другие параметры модели игнорируются)
            model_type: Тип модели для генерации SQL (openai или local) если llm=None
            model_name: Название модели если llm=None
            temperature: Температура для генерации (рекомендуется 0.0) если llm=None
            llm_local_path: Путь к локальной модели (если model_type='local') если llm=None
        """
        # Параметры подключения к Oracle
        self.host = host or os.environ.get("ORACLE_HOST")
        self.port = port or int(os.environ.get("ORACLE_PORT", 1521))
        self.service_name = service_name or os.environ.get("ORACLE_SERVICE_NAME")
        self.user = user or os.environ.get("ORACLE_USER")
        self.password = password or os.environ.get("ORACLE_PASSWORD")
        
        # Параметры LLM
        self.model_type = model_type
        self.model_name = model_name
        self.temperature = temperature
        
        # Используем переданную модель или инициализируем новую
        if llm is not None:
            logger.info(f"Используется переданный экземпляр LLM: {type(llm).__name__}")
            self.llm = llm
        else:
            logger.info(f"Создание нового экземпляра LLM: {model_name}")
            self.llm = self._initialize_llm(model_type, model_name, temperature, llm_local_path)
        
        # Сохраняем схему БД
        self.schema = None
        self.connection = None
        self.is_connected = False
    
    def _initialize_llm(self, model_type: str, model_name: str, temperature: float, llm_local_path: str = None):
        """Инициализирует модель для генерации SQL запросов"""
        if model_type == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY не установлен в переменных окружения")
            return ChatOpenAI(model=model_name, temperature=temperature)
        elif model_type == "local":
            if not llm_local_path:
                raise ValueError("Не указан путь к локальной модели")
            return LlamaCpp(
                model_path=llm_local_path,
                temperature=temperature,
                max_tokens=1024,
                n_ctx=4096,
                verbose=False
            )
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
    
    def connect(self):
        """Устанавливает соединение с базой данных Oracle"""
        try:
            dsn = f"{self.host}:{self.port}/{self.service_name}"
            self.connection = oracledb.connect(user=self.user, password=self.password, dsn=dsn)
            self.is_connected = True
            logger.info(f"Успешное подключение к Oracle DB: {dsn}")
            return True
        except Exception as e:
            self.connection = None
            self.is_connected = False
            logger.error(f"Ошибка подключения к Oracle: {str(e)}")
            return False
    
    def disconnect(self):
        """Закрывает соединение с базой данных Oracle"""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.is_connected = False
            logger.info("Соединение с Oracle закрыто")
    
    def get_schema_info(self, tables: Optional[List[str]] = None) -> str:
        """
        Получает информацию о схеме базы данных Oracle
        
        Args:
            tables: Список таблиц для получения схемы. Если None, получаем все таблицы.
            
        Returns:
            Строка с информацией о схеме базы данных в формате для LLM
        """
        if not self.connection:
            if not self.connect():
                return "Невозможно получить схему: нет подключения к базе данных"
        
        try:
            cursor = self.connection.cursor()
            schema_info = []
            
            # Получаем список таблиц
            if tables:
                tables_condition = "AND TABLE_NAME IN ({})".format(
                    ",".join([f"'{t.upper()}'" for t in tables])
                )
            else:
                tables_condition = ""
            
            # Запрос для получения списка таблиц
            tables_query = f"""
                SELECT TABLE_NAME 
                FROM USER_TABLES 
                WHERE 1=1 {tables_condition}
                ORDER BY TABLE_NAME
            """
            
            cursor.execute(tables_query)
            all_tables = [row[0] for row in cursor.fetchall()]
            
            # Для каждой таблицы получаем структуру
            for table_name in all_tables:
                # Получаем столбцы
                columns_query = f"""
                    SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH, NULLABLE
                    FROM USER_TAB_COLUMNS
                    WHERE TABLE_NAME = '{table_name}'
                    ORDER BY COLUMN_ID
                """
                cursor.execute(columns_query)
                columns = cursor.fetchall()
                
                # Получаем первичный ключ
                pk_query = f"""
                    SELECT cols.column_name
                    FROM user_constraints cons, user_cons_columns cols
                    WHERE cons.constraint_type = 'P'
                    AND cons.constraint_name = cols.constraint_name
                    AND cons.table_name = '{table_name}'
                """
                cursor.execute(pk_query)
                pk_columns = [row[0] for row in cursor.fetchall()]
                
                # Получаем внешние ключи
                fk_query = f"""
                    SELECT a.column_name, c_pk.table_name r_table_name, b.column_name r_column_name
                    FROM user_cons_columns a
                    JOIN user_constraints c ON a.constraint_name = c.constraint_name
                    JOIN user_constraints c_pk ON c.r_constraint_name = c_pk.constraint_name
                    JOIN user_cons_columns b ON c_pk.constraint_name = b.constraint_name
                    WHERE c.constraint_type = 'R' AND a.table_name = '{table_name}'
                """
                cursor.execute(fk_query)
                fk_relations = cursor.fetchall()
                
                # Формируем описание таблицы
                table_info = [f"Таблица {table_name}:"]
                table_info.append("Столбцы:")
                
                for col in columns:
                    col_name, data_type, length, nullable = col
                    pk_marker = " (PK)" if col_name in pk_columns else ""
                    null_marker = " NULL" if nullable == 'Y' else " NOT NULL"
                    table_info.append(f"  - {col_name}: {data_type}{pk_marker}{null_marker}")
                
                if fk_relations:
                    table_info.append("Внешние ключи:")
                    for fk in fk_relations:
                        col_name, ref_table, ref_col = fk
                        table_info.append(f"  - {col_name} -> {ref_table}({ref_col})")
                
                schema_info.append("\n".join(table_info))
            
            # Сохраняем схему для повторного использования
            self.schema = "\n\n".join(schema_info)
            
            cursor.close()
            return self.schema
        
        except Exception as e:
            logger.error(f"Ошибка при получении схемы базы данных: {str(e)}")
            return f"Ошибка при получении схемы: {str(e)}"
    
    def generate_sql(self, query: str, schema_override: Optional[str] = None) -> str:
        """
        Генерирует SQL запрос из текстового запроса
        
        Args:
            query: Запрос на естественном языке
            schema_override: Если передана строка, она будет использована вместо получения схемы из БД
            
        Returns:
            SQL запрос
        """
        # Используем переданную схему, если она предоставлена
        if schema_override is not None:
            schema_to_use = schema_override
        # Иначе пытаемся получить схему из БД, если она еще не получена
        elif self.schema is None:
            self.schema = self.get_schema_info()
            schema_to_use = self.schema
        else:
            schema_to_use = self.schema
        
        try:
            # Создаем цепочку для генерации SQL запроса
            # Используем блок with trace вместо декоратора @trace по рекомендации для LangGraph
            with trace("generate_sql_query"):
                prompt = ChatPromptTemplate.from_template(SQL_GENERATION_PROMPT)
                
                # Создаем цепочку: промпт -> LLM -> парсер
                chain = (
                    prompt 
                    | self.llm 
                    | StrOutputParser()
                )
                
                # Выполняем генерацию
                sql_query = chain.invoke({"schema": schema_to_use, "query": query})
                
                # Логируем результат
                logger.info(f"Сгенерирован SQL запрос для: '{query[:50]}...'")
                logger.debug(f"SQL запрос: {sql_query}")
                
                return sql_query
                
        except Exception as e:
            logger.error(f"Ошибка при генерации SQL запроса: {str(e)}")
            return f"Ошибка при генерации SQL: {str(e)}"
    
    def execute_sql(self, sql_query: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Выполняет SQL запрос и возвращает результаты
        
        Args:
            sql_query: SQL запрос для выполнения
            
        Returns:
            Кортеж (результаты запроса в виде списка словарей, сообщение об ошибке или None)
        """
        if not self.connection:
            if not self.connect():
                return [], "Нет подключения к базе данных"
        
        try:
            cursor = self.connection.cursor()
            
            # Устанавливаем имена столбцов в нижнем регистре
            cursor.execute(sql_query)
            
            # Получаем названия колонок
            columns = [col[0].lower() for col in cursor.description] if cursor.description else []
            
            # Формируем результат
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            cursor.close()
            return results, None
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении SQL запроса: {str(e)}")
            return [], f"Ошибка при выполнении SQL: {str(e)}"
    
    def process_text_query(self, text_query: str, tables: Optional[List[str]] = None, execute: bool = True) -> Dict:
        """
        Обрабатывает текстовый запрос: генерирует SQL запрос и опционально выполняет его
        
        Args:
            text_query: Запрос на естественном языке
            tables: Опционально список таблиц для уточнения схемы
            execute: Выполнять ли сгенерированный SQL запрос
            
        Returns:
            Словарь с результатами запроса, SQL запросом и сообщением об ошибке
        """
        # Генерируем SQL запрос
        sql_query = self.generate_sql(text_query, tables)
        
        results = []
        error_message = None
        
        # Выполняем запрос, если нужно
        if execute and not sql_query.startswith("Ошибка"):
            results, error_message = self.execute_sql(sql_query)
        
        return {
            "query": text_query,
            "sql": sql_query,
            "results": results,
            "error": error_message
        }
