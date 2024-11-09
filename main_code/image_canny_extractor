import cv2
import numpy as np
from PIL import Image

def detect_image_with_canny(image_pil, area_threshold=5000):
    """
    Находит и вырезает изображение на скриншоте с использованием метода Canny для определения контуров.

    Parameters:
    - image_pil (PIL.Image): Входное изображение в формате PIL.
    - area_threshold (int): Минимальная площадь для контуров, чтобы они считались изображением (по умолчанию 5000).

    Returns:
    - PIL.Image или None: Вырезанное изображение, если контур найден; иначе None.
    """
    # Преобразование изображения из PIL в формат numpy
    image = np.array(image_pil)

    # Конвертация из RGB в BGR, если изображение цветное, так как OpenCV работает в BGR
    if image.shape[-1] == 3:  # Проверка, что изображение цветное
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Преобразование изображения в оттенки серого для упрощения обработки
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение размытия для уменьшения шума и улучшения качества контуров
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Применение алгоритма Canny для выделения краев
    edges = cv2.Canny(blurred, 50, 150)
    
    # Расширение краев с использованием дилатации для лучшего выявления контуров
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Поиск контуров на изображении
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Инициализация переменных для поиска контура с наибольшей площадью
    largest_contour = None
    max_area = 0
    
    # Проход по всем найденным контурам
    for contour in contours:
        area = cv2.contourArea(contour)  # Вычисление площади контура
        # Проверка, что контур больше пороговой площади и является самым большим на данный момент
        if area > area_threshold and area > max_area:
            max_area = area
            largest_contour = contour

    # Если найден подходящий контур, выполняем обрезку изображения
    if largest_contour is not None:
        # Создаем ограничивающий прямоугольник вокруг контура
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y + h, x:x + w]

        # Преобразуем обрезанное изображение в RGB и возвращаем его в формате PIL
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cropped_image_rgb).convert("RGB")
    else:
        # Возвращаем None, если изображение не найдено
        return None
