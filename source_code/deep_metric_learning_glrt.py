import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
import numpy as np
import random
import pandas as pd
import faiss
from tqdm import tqdm
import os

# Установка семян для воспроизводимости результатов
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# Определение устройства (CPU или GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# Задание параметров модели и обучения
data_dir = "dataset2/"  # Путь к папке с изображениями
batch_size = 32
n_clusters = 102  # Количество классов
n_epochs = 10  # Количество эпох для обучения
learning_rate = 1e-4
n_recommendations = 10  # Количество рекомендаций для каждого запроса

# Преобразования для изображений (предобработка для нейронной сети)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Загрузка изображений из указанного каталога с применением преобразований
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# Определение модели для извлечения признаков
class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super(FeatureExtractor, self).__init__()
        # Удаляем последние слои для использования модели как feature extractor
        self.base = nn.Sequential(*list(base_model.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Добавляем глобальный average pooling

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)  # Преобразуем в одномерный вектор

# Инициализация модели и перенос на устройство
base_model = models.resnet50(weights="IMAGENET1K_V2")
feature_extractor = FeatureExtractor(base_model).to(device)
feature_extractor.train()  # Переводим в режим обучения

# Присвоение псевдо-меток с помощью кластеризации K-Means
def adapt_with_clustering(features, n_clusters):
    """
    Присваивает псевдо-метки каждому объекту на основе кластеризации K-Means.

    Args:
        features (np.array): Массив признаков для кластеризации.
        n_clusters (int): Количество кластеров (классов).

    Returns:
        np.array: Массив псевдо-меток для каждого объекта.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pseudo_labels = kmeans.fit_predict(features)
    return pseudo_labels

# Определение функции потерь GLRTML для обучения модели
class GLRTMLLoss(nn.Module):
    def __init__(self):
        super(GLRTMLLoss, self).__init__()

    def forward(self, features, labels):
        positive_pairs, negative_pairs = self.sample_pairs(features, labels)
        pos_loss = self.compute_glrt_loss(positive_pairs)
        neg_loss = self.compute_glrt_loss(negative_pairs)
        return pos_loss - neg_loss  # Разница потерь для положительных и отрицательных пар

    def sample_pairs(self, features, labels):
        positive_pairs, negative_pairs = [], []
        label_to_indices = {}
        # Создание словаря индексов по меткам
        for idx, label in enumerate(labels):
            label_to_indices.setdefault(label.item(), []).append(idx)

        # Формирование положительных пар из изображений одного класса
        for label, indices in label_to_indices.items():
            if len(indices) < 2:
                continue
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    positive_pairs.append((features[indices[i]], features[indices[j]]))

        # Формирование отрицательных пар из разных классов (рандомно)
        all_labels = list(label_to_indices.keys())
        for _ in range(len(positive_pairs)):
            label1, label2 = random.sample(all_labels, 2)
            idx1 = random.choice(label_to_indices[label1])
            idx2 = random.choice(label_to_indices[label2])
            negative_pairs.append((features[idx1], features[idx2]))

        return positive_pairs, negative_pairs

    def compute_glrt_loss(self, pairs):
        if not pairs:
            return torch.tensor(0.0, device=device, requires_grad=True)
        x1 = torch.stack([pair[0] for pair in pairs]).to(device)
        x2 = torch.stack([pair[1] for pair in pairs]).to(device)
        distances = torch.norm(x1 - x2, p=2, dim=1)
        return distances.mean()

# Инициализация функции потерь и оптимизатора
glrtml_loss_fn = GLRTMLLoss()
optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=learning_rate)

# Функция для извлечения признаков из всех изображений
def extract_features(loader, model, device):
    model.eval()
    all_features = []
    all_image_paths = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Извлечение признаков"):
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())
            start_idx = len(all_image_paths)
            all_image_paths.extend([dataset.imgs[i][0] for i in range(start_idx, start_idx + len(images))])
    all_features = np.vstack(all_features)
    return all_features, all_image_paths

print("Извлечение признаков для всех изображений...")
all_features, all_image_paths = extract_features(dataloader, feature_extractor, device)

# Присвоение псевдо-меток путем кластеризации признаков
print("Кластеризация с использованием K-Means...")
pseudo_labels = adapt_with_clustering(all_features, n_clusters)
print("Псевдо-метки присвоены.")

# Обучение PCA для уменьшения размерности признаков
from sklearn.decomposition import PCA
from annoy import AnnoyIndex
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Уменьшение размерности признаков с помощью PCA
pca = PCA(n_components=128)
all_features_reduced = pca.fit_transform(all_features)

# Создание и построение Annoy индекса
dimension = all_features_reduced.shape[1]
annoy_index = AnnoyIndex(dimension, 'euclidean')
for i, feature in enumerate(all_features_reduced):
    annoy_index.add_item(i, feature)
annoy_index.build(10)

# Параметры для тестового изображения
image_path = "shakalcamera.jpeg"  # Путь к изображению для теста
n_recommendations = 10  # Количество рекомендаций

# Загрузка и предобработка изображения
def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

# Загрузка и извлечение признаков тестового изображения
image_tensor = load_and_preprocess_image(image_path).to(device)
feature_extractor.eval()
with torch.no_grad():
    feature = feature_extractor(image_tensor).cpu().numpy().flatten()

# Преобразование признаков тестового изображения с помощью обученного PCA
feature_reduced = pca.transform([feature])[0]

# Поиск похожих изображений с помощью Annoy
def find_similar_images_for_random_image(feature, index, image_paths, n=10):
    indices = index.get_nns_by_vector(feature, n)
    similar_images = [image_paths[idx] for idx in indices]
    return similar_images

# Получение рекомендаций для тестового изображения
recommendations = find_similar_images_for_random_image(feature_reduced, annoy_index, all_image_paths, n=n_recommendations)

# Функция для отображения исходного изображения и рекомендаций
def display_similar_images(original_image_path, recommended_image_paths):
    plt.figure(figsize=(15, 5))
    original_image = Image.open(original_image_path).convert('RGB')
    plt.subplot(1, n_recommendations + 1, 1)
    plt.imshow(original_image)
    plt.axis('off')
    plt.title("Original Image")
    for i, rec_image_path in enumerate(recommended_image_paths):
        rec_image = Image.open(rec_image_path).convert('RGB')
        plt.subplot(1, n_recommendations + 1, i + 2)
        plt.imshow(rec_image)
        plt.axis('off')
        plt.title(f"Recommendation {i + 1}")
    plt.show()

# Определение папки с тестовыми изображениями
test_folder = "laptops_t"
test_image_paths = glob.glob(os.path.join(test_folder, "*.jpeg"))

# Тестирование и отображение рекомендаций
def test_and_display_recommendations(test_image_paths, feature_extractor, pca, annoy_index, all_image_paths, device):
    for image_path in test_image_paths:
        image_tensor = load_and_preprocess_image(image_path).to(device)
        feature_extractor.eval()
        with torch.no_grad():
            feature = feature_extractor(image_tensor).cpu().numpy().flatten()
        feature_reduced = pca.transform([feature])[0]
        recommendations = find_similar_images_for_random_image(feature_reduced, annoy_index, all_image_paths, n=n_recommendations)
        display_similar_images(image_path, recommendations)

# Запуск тестирования
test_and_display_recommendations(test_image_paths, feature_extractor, pca, annoy_index, all_image_paths, device)
