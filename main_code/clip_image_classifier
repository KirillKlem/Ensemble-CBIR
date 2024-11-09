import os
import numpy as np
import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from tqdm import tqdm

# Включаем прогресс-бар для pandas
tqdm.pandas()

# Функция для получения текстовых эмбеддингов классов
def get_class_embeddings(class_names):
    """
    Создает текстовые эмбеддинги для каждого класса с помощью модели CLIP.

    Args:
        class_names (list): Список названий классов (тексты).

    Returns:
        torch.Tensor: Эмбеддинги классов (на устройстве).
    """
    # Подготовка текста для CLIP и преобразование в эмбеддинги
    inputs = processor(text=class_names, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embeddings = model.get_text_features(**inputs).to(dtype=torch.float32)
    return text_embeddings


def classify_image(query_embedding, class_embeddings, class_names, top_n=3):
    """
    Классифицирует изображение, находя наиболее вероятные классы для данного эмбеддинга.

    Args:
        query_embedding (torch.Tensor): Эмбеддинг изображения-запроса.
        class_embeddings (torch.Tensor): Эмбеддинги каждого класса.
        class_names (list): Названия классов, соответствующие class_embeddings.
        top_n (int): Количество наиболее вероятных классов для возврата.

    Returns:
        tuple: Список названий top-N классов и их значения схожести.
    """
    # Нормализация эмбеддинга запроса и эмбеддингов классов для корректного расчета косинусной схожести
    query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    class_embeddings_normalized = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    # Вычисление косинусного сходства между эмбеддингом запроса и эмбеддингами классов
    similarities = class_embeddings_normalized @ query_embedding.T
    values, top_classes_idx = torch.topk(similarities, top_n)
    # Извлечение названий классов с наибольшим сходством
    top_classes = [class_names[i] for i in top_classes_idx]
    return top_classes, values


def get_similar_images(query_embedding, image_df, class_embeddings, class_names, top_n_classes=3, top_n_images=10):
    """
    Находит top-N похожих изображений для эмбеддинга запроса среди изображений наиболее вероятных классов.

    Args:
        query_embedding (torch.Tensor): Эмбеддинг изображения-запроса.
        image_df (pd.DataFrame): DataFrame с данными об изображениях (должен содержать 'file_name', 'embedding' и 'label').
        class_embeddings (torch.Tensor): Эмбеддинги каждого класса.
        class_names (list): Список названий классов.
        top_n_classes (int): Количество классов для фильтрации.
        top_n_images (int): Количество top-N похожих изображений для возврата.

    Returns:
        pd.DataFrame: DataFrame с top-N похожими изображениями (содержит 'file_name', 'similarity', 'embedding').
    """
    # Шаг 1: Получение наиболее вероятных классов для запроса
    top_classes, values = classify_image(query_embedding, class_embeddings, class_names, top_n=top_n_classes)
    print(top_classes)
    print(values)

    # Шаг 2: Фильтрация изображений DataFrame по выбранным классам
    filtered_df = image_df[image_df['label'].isin(top_classes)].copy()

    # Шаг 3: Нормализация эмбеддинга запроса и эмбеддингов изображений
    query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    filtered_df['embedding'] = filtered_df['embedding'].apply(lambda emb: emb / emb.norm(dim=-1, keepdim=True))

    # Шаг 4: Вычисление косинусного сходства между запросом и каждым изображением из отфильтрованных
    filtered_df['similarity'] = filtered_df['embedding'].apply(
        lambda emb: torch.cosine_similarity(query_embedding, emb.unsqueeze(0)).item())

    # Шаг 5: Сортировка по убыванию схожести и выбор top-N изображений
    top_similar_images = filtered_df.nlargest(top_n_images, 'similarity')[['file_name', 'similarity', 'embedding']]

    return top_similar_images


# Определение устройства (GPU или CPU) и загрузка модели CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Загрузка и подготовка классов (список названий классов из файла)
with open("class_names.txt", "r") as file:
    class_names = [line.strip() for line in file.readlines()]

# Загрузка эмбеддингов изображений из CSV и преобразование их в тензоры
embeddings = pd.read_csv('embeddings.csv')
embeddings['embedding'] = embeddings['embedding'].progress_apply(
    lambda x: torch.tensor(np.fromstring(x.strip('[]'), sep=' ')).to(device, dtype=torch.float32))
print(embeddings.head())

# Получение текстовых эмбеддингов классов с использованием функции get_class_embeddings
class_embeddings = get_class_embeddings(class_names)

# Выбор изображения для запроса и нахождение top-N похожих изображений
query_image_name = '0b2cd961d8d29af9.jpg'
query_image_embed = embeddings[embeddings['file_name'] == query_image_name]['embedding'].values[0]
print(type(query_image_embed))
top_similar = get_similar_images(query_image_embed, embeddings, class_embeddings, class_names)
print(top_similar)
