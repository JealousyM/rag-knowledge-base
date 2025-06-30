import os
import sys
import json
import streamlit as st
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from data_processing import COLLECTION_NAME
from ollama_embeddings import OllamaEmbeddings
from prompts import SYSTEM_PROMPT, ERROR_NO_CONTEXT, ERROR_MODEL_UNAVAILABLE, ERROR_GENERIC

# Критическое исправление для Windows + Streamlit + PyTorch
if sys.platform == "win32":
    os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
    os.environ['TORCH_DISABLE_FILE_WATCHER'] = '1'
    os.environ['PYTHONASYNCIODEBUG'] = '0'

# Загрузка переменных окружения
load_dotenv()

# Проверка токена
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    st.error("❌ Токен Hugging Face не найден в .env файле")
    st.stop()

# Настройка пользовательского интерфейса
def setup_ui():
    st.set_page_config(layout="wide")
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📘 AI Ассистент")
    st.markdown("---")

class VectorSearch:
    def __init__(self):
        self.client = QdrantClient(
            host="localhost",
            port=6333,
            timeout=10,
            prefer_grpc=False
        )
        self.embeddings = OllamaEmbeddings(model_name="nomic-embed-text")
        self.collection_name = COLLECTION_NAME
        
        # Проверяем подключение и коллекцию при инициализации
        self._check_collection()
        
    def _check_collection(self):
        """Проверяем наличие коллекции и выводим информацию о ней"""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            print(f"Доступные коллекции: {collection_names}")
            
            if self.collection_name not in collection_names:
                print(f"Ошибка: Коллекция {self.collection_name} не найдена!")
                return
                
            # Получаем информацию о коллекции
            collection_info = self.client.get_collection(self.collection_name)
            print(f"\nИнформация о коллекции {self.collection_name}:")
            print(f"Количество точек: {collection_info.vectors_count}")
            
            # Получаем несколько случайных документов для проверки
            try:
                points = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=3,
                    with_vectors=False,
                    with_payload=True
                )
                
                print("\nПримеры документов в коллекции:")
                for i, point in enumerate(points[0], 1):
                    print(f"\nДокумент {i}:")
                    print(f"ID: {point.id}")
                    print(f"Метаданные: {point.payload}")
                    content = point.payload.get('page_content', 'Нет содержимого')
                    print(f"Содержимое: {content[:200]}..." if len(content) > 200 else f"Содержимое: {content}")
                    
            except Exception as e:
                print(f"Не удалось получить примеры документов: {str(e)}")
                
        except Exception as e:
            print(f"Ошибка при проверке коллекции: {str(e)}")

    def search(self, query, k=5):
        """Поиск релевантных документов"""
        try:
            print(f"🔍 Выполняем поиск по запросу: {query}")
            
            # Выполняем поиск с разными стратегиями
            search_queries = [
                query,  # Оригинальный запрос
                query.upper(),  # Верхний регистр (на случай аббревиатур)
                query.lower(),  # Нижний регистр
                query.replace('.', ' '),  # Без точек
                query.replace(' ', '')     # Без пробелов
            ]
            
            # Храним все найденные документы
            all_results = []
            seen_ids = set()
            
            # Выполняем поиск для каждого варианта запроса
            for search_query in search_queries:
                try:
                    # Получаем вектор запроса
                    query_vector = self.embeddings.embed_query(search_query)
                    
                    # Выполняем поиск в Qdrant с указанием имени вектора
                    results = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=("text", query_vector),  # Указываем имя вектора 'text'
                        limit=k,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    # Добавляем результаты, удаляя дубликаты
                    for result in results:
                        if result.id not in seen_ids:
                            seen_ids.add(result.id)
                            all_results.append({
                                'id': result.id,
                                'score': result.score,
                                'payload': result.payload
                            })
                except Exception as e:
                    print(f"Ошибка при поиске с запросом '{search_query}': {str(e)}")
            
            # Сортируем результаты по релевантности
            all_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Выводим результаты
            print(f"\nНайдено уникальных результатов: {len(all_results)}")
            for i, result in enumerate(all_results, 1):
                print(f"\nДокумент {i} (сходство: {result['score']:.4f}):")
                print(f"ID: {result['id']}")
                print(f"Метаданные: {result['payload']}")
                
                # Получаем текст из метаданных
                text = result['payload'].get('text', 'Нет содержимого')
                print(f"Текст: {text[:500]}...")
            
            # Возвращаем только уникальные результаты
            return [doc['payload'] for doc in all_results]
            
        except Exception as e:
            print(f"Ошибка при поиске: {str(e)}")
            return []
            st.error(error_msg)
            import traceback
            traceback.print_exc()
            return []
            return []

class OllamaAssistant:
    def __init__(self):
        import requests
        import time
        
        self.base_url = "http://localhost:11434"
        self.api_url = f"{self.base_url}/api"
        self.model_name = "mistral"  # Имя модели в Ollama
        self.headers = {'Content-Type': 'application/json'}
        self.max_retries = 3
        self.retry_delay = 5
        
        # Проверяем доступность Ollama
        for attempt in range(self.max_retries):
            try:
                # Проверяем доступность API
                response = requests.get(
                    f"{self.api_url}/tags",
                    headers=self.headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    # Проверяем, что модель доступна
                    models = response.json().get('models', [])
                    model_exists = any(m.get('name', '').startswith(self.model_name) for m in models)
                    
                    if not model_exists:
                        print(f"Модель {self.model_name} не найдена. Доступные модели: {[m.get('name') for m in models]}")
                        print(f"Попробуйте загрузить модель: ollama pull {self.model_name}")
                        raise RuntimeError(f"Модель {self.model_name} не найдена")
                        
                    print("Успешное подключение к Ollama!")
                    return  # Выходим, если подключение успешно
                    
                print(f"Ошибка API: {response.status_code} - {response.text}")
                
            except requests.exceptions.RequestException as e:
                print(f"Попытка {attempt + 1} не удалась: {str(e)}")
            
            if attempt < self.max_retries - 1:
                print(f"Повторная попытка через {self.retry_delay} секунд...")
                time.sleep(self.retry_delay)
        
        # Если дошли сюда, значит все попытки не удались
        error_msg = """
Не удалось подключиться к Ollama. Убедитесь, что:
1. Ollama запущен (проверьте в трее или запустите 'ollama serve')
2. Модель загружена (запустите 'ollama pull mistral')
3. Ожидайте несколько минут после запуска Ollama
4. Проверьте, что порт 11434 не занят другим приложением
"""
        print(error_msg)
        raise RuntimeError("Не удалось подключиться к Ollama. Проверьте консоль для подробностей.")
        
    def _create_prompt(self, user_query, context):
        """Создаем промпт для модели Mistral"""
        return SYSTEM_PROMPT.format(context=context, question=user_query)

    def generate_response(self, user_query, context):
        """Генерация ответа с использованием модели Mistral через Ollama"""
        import requests
        import json
        import time
        
        prompt = self._create_prompt(user_query, context)
        
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 512
            }
        }
        
        for attempt in range(self.max_retries):
            try:
                print(f"Отправка запроса к {self.api_url}/generate")
                print(f"Данные запроса: {json.dumps(data, ensure_ascii=False, indent=2)}")
                
                response = requests.post(
                    f"{self.api_url}/generate",
                    headers=self.headers,
                    json=data,
                    timeout=120  # Увеличиваем таймаут для больших ответов
                )
                
                print(f"Статус ответа: {response.status_code}")
                print(f"Заголовки ответа: {response.headers}")
                print(f"Тело ответа: {response.text[:500]}...")  # Выводим первые 500 символов ответа
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        response_text = result.get('response', '').strip()
                        
                        if not response_text:
                            print("Получен пустой ответ от модели")
                            return ERROR_NO_CONTEXT
                            
                        # Удаляем дублирующийся промпт из ответа, если он есть
                        if response_text.startswith(prompt):
                            response_text = response_text[len(prompt):].strip()
                            
                        print(f"Успешно сгенерирован ответ длиной {len(response_text)} символов")
                        return response_text
                        
                    except json.JSONDecodeError as e:
                        print(f"Ошибка декодирования JSON: {str(e)}")
                        print(f"Невалидный JSON: {response.text}")
                        raise RuntimeError("Ошибка при обработке ответа от сервера")
                        
                else:
                    error_msg = f"Ошибка API Ollama: {response.status_code} - {response.text}"
                    print(error_msg)  # Логируем ошибку в консоль
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    raise RuntimeError(f"Не удалось получить ответ от модели: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"Ошибка сети: {str(e)}"
                print(error_msg)  # Логируем ошибку в консоль
                if attempt == self.max_retries - 1:
                    st.error("Не удалось подключиться к серверу Ollama. Пожалуйста, проверьте, что Ollama запущен.")
                    return ERROR_MODEL_UNAVAILABLE
                time.sleep(self.retry_delay)
                
            except Exception as e:
                error_msg = f"Неожиданная ошибка: {str(e)}"
                print(error_msg)  # Логируем ошибку в консоль
                import traceback
                traceback.print_exc()  # Выводим полный стектрейс
                
                if attempt == self.max_retries - 1:
                    st.error("Произошла непредвиденная ошибка при генерации ответа.")
                    return ERROR_GENERIC
                time.sleep(self.retry_delay)
        
        return ERROR_GENERIC

def main():
    setup_ui()
    
    st.title("💬 Чат-бот с RAG")
    st.caption("Задайте вопрос, и я постараюсь ответить на основе загруженных документов")
    
    # Добавляем кнопку для проверки коллекции
    if st.sidebar.button("🔍 Проверить коллекцию"):
        with st.sidebar.expander("Информация о коллекции"):
            search_engine = VectorSearch()
            # Метод _check_collection выведет информацию в консоль
            st.info("Проверка коллекции выполнена. Смотрите вывод в консоли.")
    
    # Инициализация компонентов
    search_engine = VectorSearch()
    assistant = OllamaAssistant()

    if query := st.chat_input("Ваш вопрос:"):
        with st.chat_message("user"):
            st.write(query)

        # Поиск релевантных документов
        with st.spinner("🔍 Ищем информацию в базе знаний..."):
            try:
                context_docs = search_engine.search(query)
                
                if not context_docs:
                    st.warning("Не найдено подходящей информации в базе знаний. Попробуйте переформулировать вопрос.")
                    return
                    
                # Формируем контекст с нумерацией источников
                context_parts = []
                for i, doc in enumerate(context_docs, 1):
                    text = doc.get('text', '')
                    context_parts.append(f"[Источник {i}]\n{text}")
                
                context = "\n\n".join(context_parts)
                print(f"Сформирован контекст: {context[:500]}...")
                
            except Exception as e:
                st.error(f"Ошибка при поиске информации: {str(e)}")
                return

        # Генерация ответа
        with st.spinner("🤖 Формирую ответ..."):
            try:
                response = assistant.generate_response(query, context)
                
                # Отображаем ответ
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(response)
                
                # Отображаем контекст для отладки
                with st.expander("📚 Использованные источники информации:", expanded=True):
                    if not context_docs:
                        st.warning("Не найдено ни одного релевантного документа!")
                        # Показываем подробную информацию о поиске
                        st.error("""
                        Возможные причины:
                        1. В базе нет документов, соответствующих запросу
                        2. Порог релевантности слишком высокий
                        3. Проблемы с эмбеддингами
                        """)
                    else:
                        for i, doc in enumerate(context_docs, 1):
                            score = doc.get('score', 'N/A')
                            st.markdown(f"**Источник {i}** (релевантность: {score if isinstance(score, str) else f'{score:.4f}'})")
                            st.text(f"Метаданные: {doc.get('metadata', 'Нет данных')}")
                            st.text_area("Содержимое:", 
                                      value=doc.get('text', '')[:1000] + ("..." if len(doc.get('text', '')) > 1000 else ""),
                                      height=200,
                                      key=f"doc_{i}")
                            st.write("---")
                
            except Exception as e:
                st.error(f"Ошибка при генерации ответа: {str(e)}")
                st.error("Попробуйте переформулировать вопрос или повторить попытку позже.")

if __name__ == "__main__":
    # Дополнительные настройки для Windows
    if sys.platform == "win32":
        import multiprocessing
        multiprocessing.freeze_support()
    
    main()