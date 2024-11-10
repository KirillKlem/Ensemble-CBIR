import cv2
import numpy as np
from PIL import Image


def detect_image_with_canny(image_pil, area_threshold=5000):
    """
    Метод для поиска и вырезки изображения на скриншоте с использованием метода Canny.

    Parameters:
    - input_image_path: путь к исходному изображению.
    - output_image_path: путь для сохранения вырезанного изображения.
    - area_threshold: минимальная площадь, чтобы контур считался изображением.
    """
    # Загрузка изображения и преобразование в оттенки серого
    image = np.array(image_pil)

    # Конвертируйте из RGB в BGR, если изображение в цвете
    if image.shape[-1] == 3:  # Если изображение цветное
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение размытия для улучшения контуров
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Применение метода Canny для выделения контуров
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Нахождение контуров
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Поиск контура с наибольшей площадью, подходящего для изображения
    largest_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold and area > max_area:
            max_area = area
            largest_contour = contour

    if largest_contour is not None:
        # Создание прямоугольной рамки вокруг найденного контура
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y + h, x:x + w]

        # Сохранение как JPEG
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cropped_image_rgb).convert("RGB")
        # Image.fromarray(cropped_image_rgb).convert("RGB").save(output_image_path, "JPEG")
        # print(f"Изображение сохранено по пути: {output_image_path}")
    else:
        return None
        # print("Изображение не найдено.")
