import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
import numpy as np
import random
import streamlit as st
import pandas as pd
from annoy import AnnoyIndex
from tqdm import tqdm
import os
import io
from sklearn.decomposition import PCA
from PIL import Image
from joblib import load



from crop import detect_image_with_canny

n_recommendations = 10
with open('all_image_paths_deploy.txt', 'r') as f:
    all_image_paths = [line.rstrip() for line in f]



base_path = 'dataset'
def find_image_path(image_name):
    # Iterate over each subfolder in base_path
    for root, _, files in os.walk(base_path):
        if image_name in files:
            return os.path.join(root, image_name)
    return None  # Return None if the file is not found


class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super(FeatureExtractor, self).__init__()
        # Удаляем последние слои
        self.base = nn.Sequential(*list(base_model.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Глобальный average pooling

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)  # Преобразование в вектор

def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Добавляем размерность для batch
    return image

def find_similar_images_for_random_image(feature, index, image_paths, n=10):
    indices = index.get_nns_by_vector(feature, n)
    similar_images = [image_paths[idx] for idx in indices]
    return similar_images


def get_recs(image, feature_extractor, pca, annoy_index, all_image_paths,
                                     device):
    images = []
    # Загрузка и предобработка изображения
    image_tensor = preprocess(image).to(device)
    # Извлечение признаков для тестового изображения
    feature_extractor.eval()
    with torch.no_grad():
        feature = feature_extractor(image_tensor).cpu().numpy().flatten()  # Извлекаем признаки

    # Преобразование признака с использованием обученного PCA
    feature_reduced = pca.transform([feature])[0]

    # Поиск рекомендаций
    recommendations = find_similar_images_for_random_image(feature_reduced, annoy_index, all_image_paths, n=n_recommendations)
    for ind, rec_image_path in enumerate(recommendations):
        img = Image.open(rec_image_path).convert('RGB')
        images.append(img)

    return images


# Функция обрезки изображения (замените на фактическую реализацию)
def crop_image(image):
    cropped_image = detect_image_with_canny(image)
    return cropped_image


# Функция для преобразования изображения в байты для загрузки
def image_to_bytes(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


# Интерфейс Streamlit
st.title("Image Similarity Finder")
st.write("Загрузите изображение, и приложение найдет похожие изображения в базе данных.")

# Загрузка изображения
uploaded_files = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                                  key="123")

# Чекбокс для обрезки изображения
apply_crop = st.checkbox("Обрезать изображение перед поиском похожих")

if uploaded_files:
    for uploaded_file_path in uploaded_files:
        # Загружаем и отображаем исходное изображение
        uploaded_image = Image.open(uploaded_file_path).convert("RGB")

        # Проверяем чекбокс и обрезаем изображение, если он активен
        if apply_crop:
            processed_image = crop_image(uploaded_image)
        else:
            processed_image = uploaded_image

        # Отображаем загруженное изображение
        id = uploaded_file_path.file_id
        st.write("### Загруженное изображение")
        st.image(processed_image, caption="Исходное изображение", use_container_width=True)

        # Слайдеры для порога схожести и максимального числа похожих изображений
        # threshold = st.slider("Установите порог схожести", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        #                       key=id + "_th")
        max_images = st.slider("Максимальное количество похожих изображений для отображения", min_value=1, max_value=10,
                               value=5, step=1, key=id + "_sl")

        # Получаем похожие изображения и их оценки с помощью функции get_similar
        pca = load("pca.joblib")
        dimension = 128  # Размерность признаков после PCA
        annoy_index = AnnoyIndex(dimension, 'euclidean')
        annoy_index.load("annoy_index.ann")
        # return recommendations

        # Создание модели и загрузка весов
        device = torch.device("cpu")
        base_model = models.resnet50(weights="IMAGENET1K_V2")
        feature_extractor = FeatureExtractor(base_model).to(device)
        feature_extractor.load_state_dict(torch.load("feature_extractor.pth", map_location=device))
        feature_extractor.eval()  # Переключение в режим оценки
        similar_images_and_scores = get_recs(processed_image, feature_extractor, pca, annoy_index, all_image_paths, device)

        # Фильтруем и ограничиваем количество похожих изображений в соответствии с порогом и максимумом
        filtered_images = similar_images_and_scores[:max_images]

        # Отображаем похожие изображения, по два в строке, с оценками и кнопками загрузки
        st.write("### Похожие изображения")
        for i in range(0, len(filtered_images), 2):
            cols = st.columns(2)
            for j, similar_image in enumerate(filtered_images[i:i + 2]):
                with cols[j]:
                    st.image(similar_image, use_container_width=True)
                    # Добавляем кнопку для загрузки изображения
                    img_bytes = image_to_bytes(similar_image)
                    st.download_button(
                        label="Скачать изображение",
                        data=img_bytes,
                        file_name=f"similar_image_{i + j + 1}.jpg",
                        mime="image/jpeg",
                        key=id + f"_b{i}{j}"
                    )


