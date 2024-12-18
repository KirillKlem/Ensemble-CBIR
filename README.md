# 🎨 Проект "Поиск смысловых копий изображений" (Content-Based Image Retrieval - CBIR)

В данном проекте представлена инновационная система поиска изображений в формате **Content-Based Image Retrieval (CBIR)**, которая позволяет находить визуально или смыслово схожие изображения. 🎯 Основная цель системы — по входному изображению предоставить до **10 релевантных рекомендаций** из обучающего множества, где релевантность определяется принадлежностью к одному смысловому классу. 

> 📊 **Эффективность, точность и масштабируемость** — главные достоинства этой системы, что делает её мощным инструментом для анализа и поиска по изображениям!

---

## 🗂 Основные компоненты проекта

#### 📌 `app.py`
Основной файл, который запускает всю систему, объединяя этапы обработки изображений и поиска. **app.py** поддерживает интеграцию с веб-сервисом или интерфейс для локального использования. Здесь реализована основная логика обработки изображений с помощью предобученных моделей, а также использована стратегия ансамбля моделей, повышающая надежность и точность рекомендаций. 

#### 📌 `cnn_image_classifier_with_retrieval.py`
Скрипт, объединяющий задачи классификации и поиска смысловых копий. Использует мощные сверточные сети, такие как **ConvNeXt** и **ResNet50v2**, для высокоточной классификации и поиска схожих изображений. 📈 Благодаря ансамблю моделей система адаптируется к данным, повышая прозрачность и обоснованность выбранного метода.

#### 📌 `crop.py`
Модуль для предобработки изображений, удаляющий лишние области, чтобы сосредоточиться на главных признаках объектов. Этот этап улучшает точность и производительность системы при масштабируемой обработке данных.

#### 📌 `requirements.txt`
Файл с зависимостями, позволяющий быстро установить все необходимые библиотеки и обеспечить корректную работу системы. 

---

## 📁 Директории

#### 🔹 `helpers`
Содержит вспомогательные модули, предобученные модели и утилиты для обработки изображений и работы с эмбеддингами. Эти модули поддерживают стабильную и эффективную работу системы, а также её гибкость при интеграции новых методов.

#### 🔹 `source_code`
Включает черновые ноутбуки и код, использовавшийся на начальных этапах разработки. Здесь проводились первичная обработка данных, предобучение моделей и настройка алгоритмов CBIR. Эта папка — **источник знаний и идей** для дальнейшего развития проекта.

---

## 🔍 Задача проекта

Создание системы поиска изображений по смысловым копиям, которая возвращает до 10 релевантных изображений из обучающего набора, принадлежащих к тому же смысловому классу. Система нацелена на использование метрики **MAP@10** для оценки качества рекомендаций.

---

## ⚙️ Методология и реализация

![image](https://github.com/user-attachments/assets/e16fd03f-25fe-45ca-8517-ea7ff68e80ea)


### 📊 Использование ансамбля моделей

Для повышения точности и адаптивности системы используется ансамбль из **ConvNeXt**, **ResNet50v2**, **ViT-CLIP**, а также метод **GLRT-Based Metric Learning**. Ансамблевый подход позволяет объединять предсказания от нескольких архитектур, что увеличивает надежность поиска и адаптивность к различным данным.

### 🧩 GLRT-Based Metric Learning

**GLRT-Based Metric Learning** — метод обучения метрик, позволяющий системе лучше различать классы, минимизируя внутриклассовые различия и увеличивая межклассовые. Он использует **Generalized Likelihood Ratio Test (GLRT)** для точного поиска, что улучшает прозрачность и точность результатов. 🌟

### 📐 Предварительно рассчитанные эмбеддинги

Для повышения производительности и масштабируемости используются заранее рассчитанные эмбеддинги изображений. Это исключает необходимость пересчёта признаков на этапе поиска и позволяет системе обрабатывать большие объёмы данных с минимальными затратами ресурсов.

### 🏎️ Индексы FAISS и Annoy

Для быстрого поиска ближайших соседей в большом пространстве признаков применяются **FAISS** и **Annoy**. 🖥️ **FAISS** оптимизирован для работы на GPU, что ускоряет обработку на больших наборах данных, а **Annoy** обеспечивает высокую скорость поиска в памяти, что полезно при обработке огромных объемов данных. Комбинация FAISS и Annoy делает систему гибкой и высокоэффективной для задач, требующих масштабируемости.

---

## 🚀 Ключевые преимущества

1. **Обоснованность выбранного метода**  
   Использование проверенных моделей, таких как ConvNeXt, ResNet50v2, ViT-CLIP и GLRT-Based Metric Learning, гарантирует высокое качество и надежность. Ансамбль моделей позволяет системе адаптироваться к разным типам данных и выдавать точные рекомендации.

2. **Прозрачность решения**  
   Логическое разделение на компоненты делает проект легко воспроизводимым и доступным для понимания. Модули предобработки, аугментации и классификации обеспечивают полный контроль над качеством рекомендаций.

3. **Адаптивность и масштабируемость**  
   Благодаря FAISS, Annoy и предварительно рассчитанным эмбеддингам система готова к работе с большими наборами изображений, обрабатывая данные с высокой скоростью и гибкостью. При этом точность остаётся неизменно высокой.

4. **Релевантность поставленной задаче**  
   Весь стек технологий и архитектурные решения ориентированы на оптимизацию поиска и повышение релевантности, делая проект незаменимым для поиска по смыслу в больших изображениях.

5. **Реализация дополнительных идей**  
   Использование аугментации, замораживания слоев и снижения размерности с помощью PCA добавляет устойчивость и улучшает точность поиска. Эти техники обеспечивают высокое качество рекомендаций, особенно при работе с разнообразными данными.

---

## 📈 Заключение

Этот проект демонстрирует мощь и гибкость CBIR-системы, нацеленной на точность, масштабируемость и высокую производительность. 💡 Комбинация ConvNeXt, ViT-CLIP и GLRT-Based Metric Learning обеспечивает высочайшее качество поиска смысловых копий, делая проект эффективным инструментом для анализа и поиска по изображениям.

> 💬 **Готовы к тестированию и внедрению?** Этот проект CBIR предлагает уникальный подход, обеспечивая отличные результаты в области поиска по изображению!
