import re
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Настройка базового логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def clean_text_basic(text: str) -> str:
    """
    Базовая очистка текста для подготовки к эмбеддингам
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Удаляем HTML теги
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Заменяем все типы кавычек на стандартные
    text = re.sub(r'[«»"""]', '"', text)
    
    # Удаляем множественные пробелы и переносы строк
    text = re.sub(r'\s+', ' ', text)
    
    # Приводим к нижнему регистру
    text = text.lower().strip()
    
    # Заменяем распространенные сокращения
    replacements = {
        'н-р': 'например',
        'т.е.': 'то есть',
        'т.к.': 'так как',
        'т.д.': 'так далее',
        'т.п.': 'тому подобное',
        'и т.д.': 'и так далее',
        'и т.п.': 'и тому подобное',
        'др.': 'другие',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

def remove_stopwords(text: str, stopwords: Optional[List[str]] = None) -> str:
    """
    Удаление стоп-слов из текста
    """
    if not stopwords:
        # Базовый набор русских стоп-слов
        stopwords = ['и', 'в', 'на', 'с', 'по', 'для', 'за', 'из', 'о', 'от', 
                    'к', 'до', 'при', 'во', 'а', 'но', 'да', 'или', 'ни', 
                    'как', 'что', 'когда', 'где', 'зачем', 'почему', 'же', 
                    'очень', 'просто', 'так', 'вот', 'ведь', 'кстати']
    
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered_words)

def optimize_for_embedding(text: str) -> str:
    """
    Оптимизация текста для создания более качественных эмбеддингов
    """
    # Базовая очистка
    text = clean_text_basic(text)
    
    # Удаление стоп-слов
    text = remove_stopwords(text)
    
    # Дополнительная обработка для улучшения семантики
    # Выделение ключевых сущностей (можно расширить при необходимости)
    entities = {
        r'\b(python|javascript|java|c\+\+|c#)\b': r'\1: язык программирования',
        r'\b(linux|windows|macos)\b': r'\1: операционная система',
        r'\b(nlp|ml|ai|cv)\b': r'\1: область искусственного интеллекта'
    }
    
    for pattern, replacement in entities.items():
        text = re.sub(pattern, replacement, text)
    
    return text

def split_into_semantic_chunks(text: str, max_chunk_size: int = 400, overlap: int = 100) -> List[str]:
    """
    Разделение текста на семантические чанки с учетом структуры текста
    """
    # Сначала разбиваем по абзацам
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # Если абзац сам по себе слишком большой, разбиваем по предложениям
        if len(paragraph) > max_chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= max_chunk_size:
                    if current_chunk:
                        current_chunk += " "
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
        else:
            # Если добавление абзаца не превышает максимальный размер чанка
            if len(current_chunk) + len(paragraph) <= max_chunk_size:
                if current_chunk:
                    current_chunk += " "
                current_chunk += paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
    
    # Добавляем последний чанк
    if current_chunk:
        chunks.append(current_chunk)
    
    # Добавляем перекрытие между чанками для сохранения контекста
    if overlap > 0 and len(chunks) > 1:
        chunks_with_overlap = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            curr_chunk = chunks[i]
            
            # Добавляем последние 'overlap' символов из предыдущего чанка
            if len(prev_chunk) > overlap:
                overlap_text = prev_chunk[-overlap:]
                chunks_with_overlap.append(overlap_text + " " + curr_chunk)
            else:
                chunks_with_overlap.append(curr_chunk)
                
        return chunks_with_overlap
        
    return chunks

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