# Импорт необходимых библиотек
import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.applications.convnext import preprocess_input
from tqdm import tqdm

# Путь к директории с данными
data_dir = '/content/drive/MyDrive/data'

# Инициализация списков для изображений и меток
images = []
labels = []

# Загрузка изображений и меток из директории данных
for label in tqdm(os.listdir(data_dir)):
    label_path = os.path.join(data_dir, label)
    if os.path.isdir(label_path):
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            try:
                # Открытие изображения, конвертация в RGB и изменение размера до 224x224
                image = Image.open(image_path).convert('RGB').resize((224, 224))
                images.append(np.array(image))  # Добавление изображения в список
                labels.append(label)  # Добавление метки в список
            except:
                print(f"Не удалось загрузить изображение {image_path}")

# Преобразование списков изображений и меток в массивы numpy
images = np.array(images)
labels = np.array(labels)

import pickle

# Преобразование текстовых меток в числовые
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)  # Преобразование в one-hot представление
# Сохранение кодировщика меток
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Предобработка изображений с помощью функции ConvNeXt
images = preprocess_input(images)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

# Получение меток в виде индексов для стратифицированного разбиения на обучение и валидацию
y_labels = np.argmax(y_train, axis=1)

# Разделение обучающей выборки на обучающую и валидационную с сохранением пропорций классов
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, 
    test_size=0.1,        # 10% данных для валидации
    random_state=42,      # Фиксируем случайное состояние для воспроизводимости
    stratify=y_labels     # Стратифицированное разделение для сохранения пропорций классов
)

# Загрузка предобученной модели ConvNeXt (размер XLarge) без верхних слоев
base_model = ConvNeXtXLarge(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Аугментация данных: создание генератора изображений с различными трансформациями
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Замораживание начальных слоев базовой модели для тренировки только последних 50 слоев
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Добавление собственных слоев для классификации
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)  # Полносвязный слой с регуляризацией
x = BatchNormalization()(x)  # Нормализация
x = Dropout(0.6)(x)  # Dropout для борьбы с переобучением
predictions = Dense(len(label_encoder.classes_), activation='softmax')(x)  # Выходной слой

# Создание полной модели на основе базовой модели ConvNeXt
model = Model(inputs=base_model.input, outputs=predictions)

# Компиляция модели с оптимизатором AdamW
model.compile(optimizer=AdamW(learning_rate=0.0005, weight_decay=1e-5), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Настройка коллбэков для обучения
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('/content/drive/MyDrive/best_model.keras', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',    # Метрика для отслеживания улучшений
    factor=0.5,            # Коэффициент для уменьшения learning rate
    patience=1,            # Число эпох без улучшения перед снижением learning rate
    min_lr=1e-6,           # Минимальное значение learning rate
    verbose=1
)
callbacks = [early_stop, model_checkpoint, reduce_lr]

# Обучение модели на тренировочной выборке с использованием генератора аугментированных изображений
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    steps_per_epoch=len(X_train) // 16,
    epochs=25, 
    validation_data=(X_val, y_val),
    callbacks=callbacks,
)

# Оценка модели на тестовой выборке
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Точность на тестовых данных: {test_accuracy * 100:.2f}%")

# Путь к папкам с тестовыми и тренировочными данными
test_data_dir = '/content/drive/MyDrive/dataset'
train_data_dir = '/content/drive/MyDrive/data'

# Функция для предобработки изображения для предсказания
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB').resize((224, 224))
    image_array = np.expand_dims(np.array(image), axis=0)
    return preprocess_input(image_array)

# Функция для предсказания метки изображения и получения 5 похожих изображений
def predict_and_retrieve_similar_images(image_path):
    # Предобработка изображения и предсказание
    image_array = preprocess_image(image_path)
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class_index)[0]

    # Путь к папке с изображениями предсказанной метки
    label_folder_path = os.path.join(train_data_dir, predicted_label)
    similar_images = []

    # Выбор 5 случайных изображений из папки с предсказанной меткой
    if os.path.exists(label_folder_path):
        all_images = os.listdir(label_folder_path)
        similar_images = random.sample(all_images, min(10, len(all_images)))

    # Форматирование: создание строки с именами файлов, разделенными запятыми
    formatted_images = ",".join(similar_images)
    return formatted_images

# Пример использования функции для всех изображений из тестовой папки
s = []
for image_file in os.listdir(test_data_dir):
    print(image_file)
    image_path = os.path.join(test_data_dir, image_file)
    formatted_images = predict_and_retrieve_similar_images(image_path)
    s.append(pd.DataFrame({'image': [image_file], 'recs': [formatted_images]}))

# Сохранение результатов предсказаний в CSV
pd.concat(s).to_csv('submission.csv', index=False)
