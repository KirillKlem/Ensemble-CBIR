# Импорт необходимых библиотек
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
import torch
import torchvision.transforms as transforms

# Установка библиотеки Annoy для поиска ближайших соседей
!pip install annoy

# Загружаем предобученные эмбеддинги изображений
df = pd.read_csv('/content/drive/MyDrive/embeddings_CLIP.csv')

# Базовый путь к директории с изображениями
base_path = '/content/drive/MyDrive/data/'

# Формируем полный путь к файлам, используя метку класса и имя файла
df['file_name'] = base_path + df['label'] + '/' + df['file_name']

# Импорт дополнительных библиотек для построения модели и работы с изображениями
import torch
from torchvision import models
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
import random
from sklearn.decomposition import PCA
from annoy import AnnoyIndex
from PIL import Image

# Определение класса для извлечения признаков с использованием ResNet
class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super(FeatureExtractor, self).__init__()
        # Убираем последние слои модели ResNet для получения только признаков
        self.base = nn.Sequential(*list(base_model.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Глобальный average pooling

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)  # Преобразуем к вектору

# Создание модели и загрузка весов предобученного экстрактора признаков
device = torch.device("cuda")
base_model = models.resnet50(weights="IMAGENET1K_V2")
feature_extractor = FeatureExtractor(base_model).to(device)
feature_extractor.load_state_dict(torch.load("/content/drive/MyDrive/feature_extractor.pth", map_location=device))  # Предобученный экстрактор признаков
feature_extractor.eval()  # Переключаем модель в режим оценки

from joblib import load

# Загрузка предобученной модели PCA для уменьшения размерности признаков
pca = load("/content/drive/MyDrive/pca.joblib")

# Загрузка предобученного индекса Annoy для быстрого поиска ближайших соседей
dimension = 128  # Размерность признаков после PCA
annoy_index = AnnoyIndex(dimension, 'euclidean')
annoy_index.load("/content/drive/MyDrive/annoy_index.ann")

# Функция для поиска похожих изображений
def find_similar_images_for_random_image(feature, index, image_paths, n=10):
    indices, distances = index.get_nns_by_vector(feature, n, include_distances=True)
    similar_images = [(image_paths[idx].split('/')[-1], dist) for idx, dist in zip(indices, distances)]
    return similar_images

# Функция загрузки и предобработки изображения
def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')  # Преобразуем изображение в RGB
    image = transform(image).unsqueeze(0)  # Добавляем размерность для batch
    return image

# Основная функция для создания рекомендаций
def make_rec(image_path, feature_extractor, pca, annoy_index, all_image_paths, device):
    all_recs = {}
    # Загружаем и предобрабатываем изображение
    image_tensor = load_and_preprocess_image(image_path).to(device)
    feature_extractor.eval()
    with torch.no_grad():
        feature = feature_extractor(image_tensor).cpu().numpy().flatten()  # Извлекаем признаки

    # Преобразуем признаки с использованием PCA
    feature_reduced = pca.transform([feature])[0]
    recommendations = find_similar_images_for_random_image(feature_reduced, annoy_index, all_image_paths, n=n_recommendations)

    # Формируем DataFrame с результатами рекомендаций
    image_name = image_path.split('/')[-1]
    return (pd.DataFrame({'file_name': [image_name]*len(recommendations),
                         'neighbor_file_name': [r[0] for r in recommendations],
                         'metric_learning_dist':  [r[1] for r in recommendations]}).iloc[1:], recommendations)

# Загрузка всех путей к изображениям
with open('all_image_paths.txt') as file:
    all_image_paths = [line.rstrip() for line in file]

# Функция для обновления путей
def update_paths(path_list, old_prefix, new_prefix):
    updated_list = [path.replace(old_prefix, new_prefix) for path in path_list]
    return updated_list

# Обновляем пути изображений для работы с новой директорией
all_image_paths = update_paths(all_image_paths, "dataset/", "/content/drive/MyDrive/data/")

from sklearn.model_selection import train_test_split

# Разделение данных на обучающую и валидационную выборки
train_df, valid_df = train_test_split(df, test_size=0.01, random_state=42, stratify=df['label'])
test_paths = list(valid_df['file_name'])

# Указываем количество рекомендаций
n_recommendations = 50
all_recommendations = []

# Формируем рекомендации для каждого изображения из валидационного набора
for _, row in tqdm(valid_df.iterrows()):
    image_path = row['file_name']
    recommendations_df, recommendations = make_rec(image_path, feature_extractor, pca, annoy_index, all_image_paths, device)
    all_recommendations.append(recommendations_df)

# Объединение всех рекомендаций в один DataFrame
valid_predictions_df = pd.concat(all_recommendations, ignore_index=True)

import os
import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from tqdm import tqdm
tqdm.pandas()

# Функция для получения эмбеддингов классов
def get_class_embeddings(class_names):
    inputs = processor(text=class_names, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embeddings = model.get_text_features(**inputs).to(dtype=torch.float32)
    return text_embeddings

# Функция для классификации изображения на основе сходства с классами
def classify_image(query_embedding, class_embeddings, class_names, top_n=3):
    query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    class_embeddings_normalized = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    similarities = class_embeddings_normalized @ query_embedding.T
    values, top_classes_idx = torch.topk(similarities, top_n)
    top_classes = [class_names[i] for i in top_classes_idx]
