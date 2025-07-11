import os
import logging
logger = logging.getLogger(__name__)

def clean_static_images():
    """Очищает содержимое директории static/images"""
    try:
        images_dir = os.path.join('static', 'images')
        if os.path.exists(images_dir):
            # Удаляем все файлы в директории
            for filename in os.listdir(images_dir):
                file_path = os.path.join(images_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                        logger.debug(f"Удален файл: {file_path}")
                except Exception as e:
                    logger.error(f"Ошибка при удалении файла {file_path}: {e}")
            return True
        return False
    except Exception as e:
        logger.error(f"Ошибка при очистке директории static/images: {e}")
        return False