from typing import List, Dict, Any, Optional, Tuple, TypedDict
import logging
logger = logging.getLogger(__name__)
import os
import time

from langsmith import trace

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