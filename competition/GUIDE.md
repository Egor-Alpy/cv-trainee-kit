# Dr. Vision — Полное учебное пособие по Computer Vision
### От пикселя до нейросети. Всё что нужно знать и понимать.

---

## Содержание

1. [Изображение как данные](#1-изображение-как-данные)
2. [Цветовые пространства](#2-цветовые-пространства)
3. [Нормализация](#3-нормализация)
4. [Аугментация](#4-аугментация)
5. [Эмбеддинги и CLIP](#5-эмбеддинги-и-clip)
6. [Кластеризация](#6-кластеризация)
7. [Нейросети и Transfer Learning](#7-нейросети-и-transfer-learning)
8. [Цикл обучения](#8-цикл-обучения)
9. [Метрики качества](#9-метрики-качества)
10. [Диагностика проблем](#10-диагностика-проблем)
11. [Чеклист соревнования](#11-чеклист-соревнования)

---

## 1. Изображение как данные

### Интуиция

Представь экран монитора вблизи. Ты видишь миллионы маленьких точек — пикселей.
Каждый пиксель светится тремя цветами: красным, зелёным, синим (RGB).
Смешивая их в разных пропорциях, получаем любой цвет.

### Что видит Python

```python
from PIL import Image
import numpy as np

img = Image.open("wheat.jpg").convert("RGB")
arr = np.array(img)

print(arr.shape)   # (480, 640, 3)
#                     H    W   C
#                     ↑    ↑   ↑
#                  высота ширина каналы (R,G,B)

print(arr.dtype)   # uint8  — числа от 0 до 255
print(arr[0, 0])   # [128, 64, 200]  — первый пиксель: R=128, G=64, B=200
```

### Как устроен массив

```
arr[y, x, c]
      ↑  ↑  ↑
      │  │  └── канал: 0=R, 1=G, 2=B
      │  └───── столбец (координата X)
      └──────── строка (координата Y, начало сверху!)
```

Важно: в NumPy ось Y идёт СВЕРХУ ВНИЗ. arr[0, 0] — это левый ВЕРХНИЙ пиксель.

### Зачем всегда .convert("RGB")

`Image.open()` не гарантирует формат:
- PNG с прозрачностью → RGBA (4 канала)
- Серое фото → L (1 канал)
- Старые форматы → P (палитра)

Нейросеть ожидает ровно 3 канала. `.convert("RGB")` приводит любой формат к нужному.

```python
img = Image.open("photo.png").convert("RGB")  # ВСЕГДА делай так
```

### PIL vs OpenCV — важное отличие

```python
import cv2
img_cv = cv2.imread("photo.jpg")   # читает как BGR (!), не RGB

# Красный и синий канал перепутаны:
# PIL:     R=0, G=1, B=2
# OpenCV:  B=0, G=1, R=2

# Конвертация:
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# Правило: всегда читай через PIL — он всегда RGB
```

---

## 2. Цветовые пространства

### RGB — что это

Три оси: красный (0–255), зелёный (0–255), синий (0–255).
Любой цвет = точка в кубе 256×256×256.

```
(255, 0,   0)   = чистый красный
(0,   255, 0)   = чистый зелёный
(0,   0,   255) = чистый синий
(255, 255, 0)   = жёлтый (R+G)
(255, 255, 255) = белый
(0,   0,   0)   = чёрный
(128, 128, 128) = серый
```

### Как определить цветное/серое изображение

Если изображение серое — R = G = B для каждого пикселя.
Мы измеряем среднее отличие между каналами:

```python
img = np.array(Image.open("photo.jpg").convert("RGB"))
r = img[:, :, 0].astype(int)
g = img[:, :, 1].astype(int)
b = img[:, :, 2].astype(int)

# Среднее попиксельное отличие между каналами
diff = (np.mean(np.abs(r - g)) +
        np.mean(np.abs(r - b)) +
        np.mean(np.abs(g - b))) / 3

is_color = diff > 8   # порог 8 — хорошо работает на практике

# Почему не просто r == g == b?
# Потому что JPEG-сжатие вносит небольшие артефакты.
# Даже серое фото после JPEG имеет R ≠ G ≠ B на ±2-3.
```

### HSV — удобен для анализа цвета

RGB неудобен: нельзя сразу сказать "насыщенный ли цвет".
HSV разделяет три независимых свойства:

```
H (Hue)        — оттенок:      0°=красный, 60°=жёлтый, 120°=зелёный, 240°=синий
S (Saturation) — насыщенность: 0=серый,    128=средне,  255=очень яркий
V (Value)      — яркость:      0=чёрный,   128=средне,  255=яркий
```

```python
img_hsv = np.array(Image.open("photo.jpg").convert("HSV"))

hue        = img_hsv[:, :, 0]   # оттенок
saturation = img_hsv[:, :, 1]   # насыщенность
value      = img_hsv[:, :, 2]   # яркость

avg_saturation = saturation.mean()

if avg_saturation < 30:    category = "бесцветный"    # почти серый
elif avg_saturation < 100: category = "средней насыщенности"
else:                      category = "насыщенный"

# Практический пример:
# Зелёные листья → высокая насыщенность (~180)
# Серые камни   → низкая насыщенность (~15)
# Жёлтые семена → средняя (~80)
```

### Grayscale — когда нужен один канал

```python
# PIL
gray = Image.open("photo.jpg").convert("L")   # L = luminance (яркость)
arr  = np.array(gray)
# arr.shape = (480, 640)   — один канал, а не три

# Формула преобразования (взвешенное среднее):
# L = 0.299*R + 0.587*G + 0.114*B
# Зелёный канал вносит больше всего в яркость — это важно!
```

---

## 3. Нормализация

### Зачем нормализовать

Нейросеть обучается через градиентный спуск — веса обновляются маленькими шагами.
Если входные числа большие (0–255), градиенты нестабильны и обучение расходится.

### Шаг 1 — делим на 255

```
uint8 [0, 255]  →  float32 [0.0, 1.0]
```

`torchvision.transforms.ToTensor()` делает это автоматически.
В albumentations это делается через `ToTensorV2()` + предварительный `Normalize`.

### Шаг 2 — стандартизация: (x - mean) / std

**mean** — центр данных. Вычитаем → данные симметричны вокруг нуля.
**std** — разброс данных. Делим → все каналы одного масштаба.

```
Пиксель: [128, 64, 200]
После /255: [0.502, 0.251, 0.784]

ImageNet mean: [0.485, 0.456, 0.406]
ImageNet std:  [0.229, 0.224, 0.225]

После нормализации:
R: (0.502 - 0.485) / 0.229 = +0.07
G: (0.251 - 0.456) / 0.224 = -0.92
B: (0.784 - 0.406) / 0.225 = +1.68

Результат: [+0.07, -0.92, +1.68]  ← числа около нуля, диапазон ~[-2, 2]
```

### Откуда числа ImageNet

Среднее и стандартное отклонение посчитаны по всему датасету ImageNet
(1.2 миллиона фотографий). Если используешь предобученную модель —
**обязательно** нормализуй теми же числами, что она видела при обучении.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2(),   # HWC numpy → CHW torch tensor
])

# Важно: albumentations принимает numpy array, не PIL Image
img    = np.array(Image.open("photo.jpg").convert("RGB"))
result = transform(image=img)["image"]
# result.shape = torch.Size([3, 224, 224])
# result.dtype = torch.float32
```

### Частая ошибка

```python
# НЕПРАВИЛЬНО: забыли нормализовать перед инференсом
img    = np.array(Image.open("photo.jpg").convert("RGB"))
tensor = torch.tensor(img).permute(2, 0, 1).float()   # [0, 255] range!
pred   = model(tensor.unsqueeze(0))   # модель видит неверные числа → мусор

# ПРАВИЛЬНО:
transform = A.Compose([A.Resize(224,224), A.Normalize(...), ToTensorV2()])
tensor = transform(image=img)["image"].unsqueeze(0)
pred   = model(tensor)
```

---

## 4. Аугментация

### Интуиция

Представь: ты учишь ребёнка распознавать кошку.
Ты показываешь ему одно фото кошки → он запомнил именно это фото.
Если показать ту же кошку чуть повёрнутой — не узнает.

Аугментация = показываем модели каждый раз немного изменённую версию фото.
Модель учится распознавать объект независимо от положения, цвета, освещения.

### Только на обучении, не на валидации!

```
Train: фото с аугментацией → разнообразие → обобщение
Val:   фото без аугментации → честная оценка качества
```

### Три уровня аугментации

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ── LIGHT: для объектов с чёткой ориентацией ─────────────
# (монеты, документы, штрих-коды)
light = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),          # отразить по горизонтали (50% шанс)
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

# ── MEDIUM: универсальный вариант ────────────────────────
# (семена, листья, общие объекты)
medium = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),           # случайный кроп — разный масштаб
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),            # семена симметричны → можно флипать
    A.RandomRotate90(p=0.5),          # повернуть на 90°/180°/270°
    A.HueSaturationValue(p=0.4),      # изменить оттенок/насыщенность
    A.RandomBrightnessContrast(p=0.4),# разное освещение
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

# ── HEAVY: для маленьких датасетов (<200 фото/класс) ─────
heavy = A.Compose([
    A.Resize(300, 300),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(                # смещение, масштаб, поворот
        scale_limit=0.2,               # ±20% размер
        rotate_limit=30,               # ±30 градусов
        p=0.5,
    ),
    A.HueSaturationValue(p=0.4),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.2),              # добавить шум (имитирует плохую камеру)
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

# ── VAL: только resize + normalize ──────────────────────
val = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])
```

### Применение

```python
# albumentations работает с numpy array (не PIL!)
img    = np.array(Image.open("wheat.jpg").convert("RGB"))   # shape: (H, W, 3)
result = medium(image=img)["image"]                          # ключ "image"!
# result: torch.Tensor, shape (3, 224, 224)
```

### Какую аугментацию выбрать для семян

| Аугментация | Подходит для семян? | Почему |
|-------------|---------------------|--------|
| HorizontalFlip | ✅ Да | Семя не имеет право/лево |
| VerticalFlip | ✅ Да | Семя симметрично |
| RandomRotate90 | ✅ Да | Ориентация не важна |
| ShiftScaleRotate | ✅ Да | Помогает при малом датасете |
| HueSaturationValue | ⚠️ Осторожно | Цвет важен для различения видов |
| GaussNoise | ✅ Да | Имитирует разные камеры |

---

## 5. Эмбеддинги и CLIP

### Что такое эмбеддинг

Эмбеддинг — это **числовой вектор** (список чисел), описывающий содержание изображения.

```
Фото пшеницы  → [0.23, -0.41, 0.87, 0.12, ..., -0.05]  (512 чисел)
Другое фото пшеницы → [0.21, -0.39, 0.85, 0.14, ..., -0.03]  (похожи!)
Фото кукурузы → [0.71,  0.82, -0.31, 0.55, ...,  0.67]  (другие числа)
```

**Ключевая идея:** похожие изображения → похожие векторы.
Можно измерить "расстояние" между двумя изображениями математически.

### Косинусное расстояние

```python
import numpy as np

# Как измерить похожесть двух эмбеддингов:
emb_a = np.array([0.23, -0.41, 0.87])
emb_b = np.array([0.21, -0.39, 0.85])
emb_c = np.array([0.71,  0.82, -0.31])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cosine_similarity(emb_a, emb_b))   # ≈ 0.99  (очень похожи)
print(cosine_similarity(emb_a, emb_c))   # ≈ 0.20  (очень разные)

# Значения от -1 до 1:
# 1.0  = одинаковые
# 0.0  = не связаны
# -1.0 = противоположные
```

### Что такое CLIP

CLIP (Contrastive Language-Image Pretraining, OpenAI, 2021) —
нейросеть, обученная на **400 миллионах** пар (изображение, текстовое описание).

Она умеет сравнивать изображение с произвольным текстом.

```
Архитектура CLIP:
┌─────────────────┐     ┌─────────────────────┐
│  Image Encoder  │     │    Text Encoder     │
│  (ViT или CNN)  │     │   (Transformer)     │
└────────┬────────┘     └──────────┬──────────┘
         │                         │
         ▼                         ▼
   вектор 512D              вектор 512D
         │                         │
         └──────── сравниваем ──────┘
                   (dot product)
```

### Zero-shot классификация — магия CLIP

Без единого обучающего примера можно классифицировать что угодно:

```python
from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image

# Загружаем модель (первый раз скачивается ~500 МБ)
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = model.to(device).eval()

# Описываем классы текстом
class_texts = [
    "a photograph of wheat grain seeds",
    "a photograph of corn maize kernels",
    "a photograph of sunflower seeds",
]

image = Image.open("unknown_seed.jpg").convert("RGB")

# Прогоняем через модель
with torch.no_grad():
    inputs = processor(
        text=class_texts,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    outputs = model(**inputs)
    probs   = outputs.logits_per_image.softmax(dim=1)[0].cpu().tolist()

# probs = [0.87, 0.08, 0.05]
# → это пшеница с вероятностью 87%
for text, prob in zip(class_texts, probs):
    print(f"{text}: {prob:.1%}")
```

### Извлечение эмбеддингов

Для кластеризации нам нужны именно числовые векторы, не вероятности:

```python
@torch.no_grad()
def get_embedding(image_path: str) -> np.ndarray:
    image  = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    # get_image_features извлекает только визуальный вектор
    features = model.get_image_features(**inputs)

    # Нормализуем до единичной длины (важно для cosine similarity!)
    features = features / features.norm(dim=-1, keepdim=True)

    return features[0].cpu().numpy()   # shape: (512,)

# Для всех изображений:
all_paths  = ["wheat/001.jpg", "wheat/002.jpg", "corn/001.jpg", ...]
embeddings = np.array([get_embedding(p) for p in all_paths])
# embeddings.shape = (N, 512)
```

### Зачем нормализовать эмбеддинги

```python
# Без нормализации — вектора разной длины
v1 = np.array([3.0, 4.0])   # длина = 5
v2 = np.array([0.6, 0.8])   # длина = 1

# Косинусное сходство одинаково (направление то же), но
# евклидово расстояние разное (||v1-v2|| = 4.0)

# После нормализации все векторы единичной длины →
# евклидово расстояние = косинусное ∝ угол между ними
v1_norm = v1 / np.linalg.norm(v1)   # [0.6, 0.8]
v2_norm = v2 / np.linalg.norm(v2)   # [0.6, 0.8]  ← одинаковые!
```

---

## 6. Кластеризация

### Интуиция

Представь 1000 шаров в пространстве. Некоторые стоят близко группами.
Кластеризация = найти эти группы автоматически, без меток.

### KMeans — разделить на K групп

**Алгоритм:**
1. Случайно поставь K центров (центроидов)
2. Каждую точку присвой ближайшему центру
3. Пересчитай центры как среднее точек своего кластера
4. Повтори шаги 2-3 до сходимости

```python
from sklearn.cluster import KMeans

# n_init=10 означает запустить 10 раз с разными начальными точками
# и взять лучший результат (избегаем плохой инициализации)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)
# labels[i] = номер кластера для i-го изображения

print(labels[:10])   # [0, 2, 0, 1, 3, 0, 1, 1, 2, 3]
```

**Слабость KMeans:**
- Предполагает сферические кластеры одного размера
- Нужно заранее знать K
- Чувствителен к выбросам

### Метод локтя — как найти K

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

inertias   = []   # сумма квадратов расстояний до центроидов
sil_scores = []   # силуэт: насколько хорошо разделены кластеры

K_range = range(2, 15)

for k in K_range:
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(embeddings, labels))

# Оптимальное K:
# 1. По методу локтя: K где инерция начинает падать медленнее
# 2. По силуэту: K где силуэт максимален (всегда используй этот)
optimal_k = list(K_range)[int(np.argmax(sil_scores))]
print(f"Оптимальное K = {optimal_k}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(list(K_range), inertias, "bo-")
ax1.set_title("Метод локтя — ищем 'колено'")
ax2.plot(list(K_range), sil_scores, "ro-")
ax2.set_title("Silhouette — ищем максимум")
plt.show()
```

### Silhouette Score — как интерпретировать

```
Silhouette Score для точки i:
    s(i) = (b(i) - a(i)) / max(a(i), b(i))

где:
    a(i) = среднее расстояние до точек СВОЕГО кластера
    b(i) = среднее расстояние до ближайшего ЧУЖОГО кластера

Значение:
    +1.0 = точка глубоко внутри своего кластера (отлично)
     0.0 = точка на границе (неоднозначно)
    -1.0 = точка ближе к чужому кластеру (плохо)

Среднее по всем точкам:
    > 0.7  = отличная кластеризация
    > 0.5  = хорошая
    > 0.25 = структура есть, но слабая
    < 0.25 = кластеров нет или перекрываются
```

### DBSCAN — находит произвольные формы

DBSCAN не нужен K. Ищет **плотные области** точек.

```
Параметры:
  eps       = радиус окрестности (насколько близко = "сосед")
  min_samples = минимум соседей чтобы стать "ядровой точкой"

Результат:
  0, 1, 2, ... = номера кластеров
  -1           = выброс (шум, не принадлежит ни одному кластеру)
```

```python
from sklearn.cluster import DBSCAN

db     = DBSCAN(eps=0.5, min_samples=3, metric="cosine")
labels = db.fit_predict(embeddings)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = sum(labels == -1)
print(f"Кластеров: {n_clusters}, выбросов: {n_noise}")

# Подбор eps: запускай с разными значениями (0.2, 0.3, 0.5, 0.7, 1.0)
# и смотри что получается
```

**Когда использовать DBSCAN:**
- Кластеры неправильной формы
- Есть явные выбросы
- Не знаешь K

### Agglomerative — иерархическая кластеризация

Начинает с каждой точки как отдельного кластера.
Шаг за шагом объединяет самые близкие пары.

```python
from sklearn.cluster import AgglomerativeClustering

# linkage = критерий близости между кластерами:
#   "ward"     = минимизирует дисперсию внутри кластеров (лучший для компактных)
#   "complete" = расстояние между самыми дальними точками (консервативный)
#   "average"  = среднее расстояние между всеми парами точек

ag     = AgglomerativeClustering(n_clusters=4, linkage="ward")
labels = ag.fit_predict(embeddings)
```

### Визуализация — t-SNE 2D

512-мерное пространство нельзя нарисовать. t-SNE сжимает его до 2D,
**сохраняя локальные расстояния**: близкие точки остаются близкими.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# perplexity ≈ "количество ожидаемых соседей"
# Правило: perplexity = min(30, N/5) где N = число точек
tsne  = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
emb2d = tsne.fit_transform(embeddings)   # (N, 512) → (N, 2)

# Красим по кластеру
plt.figure(figsize=(10, 8))
scatter = plt.scatter(emb2d[:, 0], emb2d[:, 1],
                      c=labels, cmap="tab10", s=20, alpha=0.8)
plt.colorbar(scatter, label="Кластер")
plt.title("t-SNE 2D визуализация")
plt.savefig("tsne.png", dpi=150)
plt.show()

# ВАЖНО: t-SNE медленный для больших датасетов (>5000 точек)
# Для больших данных используй UMAP
```

### Визуализация — UMAP 3D

UMAP быстрее t-SNE и лучше сохраняет глобальную структуру.

```python
import umap
import plotly.express as px

reducer = umap.UMAP(n_components=3, random_state=42,
                    n_neighbors=15, min_dist=0.1)
emb3d   = reducer.fit_transform(embeddings)   # (N, 512) → (N, 3)

fig = px.scatter_3d(
    x=emb3d[:, 0], y=emb3d[:, 1], z=emb3d[:, 2],
    color=class_names,
    hover_name=filenames,
    title="UMAP 3D",
)
fig.update_traces(marker_size=3)
fig.write_html("umap_3d.html")   # интерактивный HTML!
fig.show()
```

---

## 7. Нейросети и Transfer Learning

### Что такое нейросеть — интуиция

Нейросеть = функция `f(x) → y` где:
- `x` = входное изображение
- `y` = вероятности классов
- функция задаётся миллионами параметров (весов)
- веса подбираются автоматически в процессе обучения

### Свёрточный слой — как он работает

Маленький фильтр 3×3 скользит по изображению и вычисляет "похожесть":

```
Исходное изображение:    Фильтр (3×3):     Результат:
1 2 1 0 0                -1 -1 -1           ?  ?  ?  ?
0 2 1 0 0         *       0  0  0     →     ?  ?  ?  ?
0 1 2 1 0                 1  1  1           ?  ?  ?  ?
0 1 3 2 0
0 0 1 1 0

Вычисление одного значения результата:
Накладываем фильтр на зону 3×3 изображения и считаем сумму произведений:
(1·-1 + 2·-1 + 1·-1) + (0·0 + 2·0 + 1·0) + (0·1 + 1·1 + 2·1) = (-4) + (0) + (3) = -1
```

**Что учат фильтры:**
- Слои 1-2: края, градиенты яркости (универсально для всех фото)
- Слои 3-5: текстуры, части объектов (прожилки, зёрна)
- Слои 6+: сложные паттерны (форма семени, тип поверхности)

### EfficientNet — почему именно он

EfficientNet (Google, 2019) — семейство архитектур, найденных алгоритмом Neural Architecture Search. Оптимальное соотношение точность/скорость/размер.

```
Модели EfficientNet:
B0 → 5.3M параметров,  77% top-1 на ImageNet
B3 → 12M параметров,   81% top-1 на ImageNet  ← начинаем здесь
B4 → 19M параметров,   83% top-1 на ImageNet
B7 → 66M параметров,   84% top-1 на ImageNet
```

### Transfer Learning — почему работает

```
БЕЗ transfer learning:
  200 фото семян → 50-60% точности (плохо, нет данных)

С transfer learning:
  200 фото семян → 90-95% точности (отлично!)

Почему так?
EfficientNet уже умеет видеть:
  ├── Слои 1-3: края, цветовые переходы  ← работает для любых фото
  ├── Слои 4-6: текстуры, формы         ← работает для семян
  └── Последний слой: Linear(1536, 1000) ← ЗАМЕНЯЕМ на нашу задачу
                                           Linear(1536, N_классов)
```

```python
import timm

# timm = библиотека с сотнями предобученных моделей
model = timm.create_model(
    "efficientnet_b3",   # название модели
    pretrained=True,     # загрузить веса ImageNet (скачает ~50 МБ)
    num_classes=5,       # наше число классов (timm сам заменяет последний слой)
)

# Проверить что модель правильно создана:
dummy = torch.randn(1, 3, 224, 224)   # батч из 1 изображения
output = model(dummy)
print(output.shape)   # torch.Size([1, 5])  ← 5 вероятностей
```

### WeightedRandomSampler — борьба с дисбалансом

Если класс A — 200 фото, класс B — 20 фото:
- Модель видит A в 10 раз чаще → учится предсказывать A на всё
- Решение: давать редким классам больше веса при выборке

```python
from torch.utils.data import WeightedRandomSampler
from collections import Counter

labels_list = [0, 0, 0, 1, 1, 2, ...]   # метки всего датасета
counts  = Counter(labels_list)            # {0: 200, 1: 50, 2: 20}

# Вес каждого примера = 1 / частота его класса
weights = [1.0 / counts[l] for l in labels_list]

sampler = WeightedRandomSampler(
    weights,
    num_samples=len(weights),
    replacement=True,   # с повторением — редкие классы будут показываться чаще
)

# Используем вместо shuffle=True:
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
# Теперь shuffle нельзя ставить одновременно с sampler!
```

### Optimizer и Scheduler

**AdamW** — лучший выбор для задач CV:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,          # learning rate: шаг обновления весов
    weight_decay=1e-4 # L2-регуляризация: штраф за большие веса (против переобучения)
)
```

**CosineAnnealingLR** — плавно снижает lr:

```
Epoch 1:  lr = 3e-4  ← большой шаг, быстро двигаемся к оптимуму
Epoch 10: lr = 1.5e-4
Epoch 20: lr ≈ 0     ← маленький шаг, точно находим минимум
```

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=20   # число эпох
)
# Вызывать scheduler.step() после каждой эпохи
```

**CrossEntropyLoss** — функция потерь для классификации:

```python
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1   # вместо "100% пшеница" учим "90% пшеница, 10% другие"
                           # защита от переуверенности модели
)
```

---

## 8. Цикл обучения

### Полная схема

```python
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

best_f1 = 0.0

for epoch in range(EPOCHS):

    # ════════════════════════════════════
    # TRAIN — обновляем веса
    # ════════════════════════════════════
    model.train()   # включает Dropout, BatchNorm в режим обучения
    train_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()           # 1. Обнуляем градиенты прошлого шага
        outputs = model(images)         # 2. Forward pass: предсказание
        loss = criterion(outputs, labels)  # 3. Считаем ошибку
        loss.backward()                 # 4. Backward pass: вычисляем градиенты
        optimizer.step()                # 5. Обновляем веса

        train_loss += loss.item()
        preds       = outputs.detach().argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    train_acc  = correct / total
    train_loss = train_loss / len(train_loader)

    # ════════════════════════════════════
    # VALIDATION — только оцениваем
    # ════════════════════════════════════
    model.eval()    # выключает Dropout, BatchNorm в режим инференса
    val_preds, val_true = [], []

    with torch.no_grad():   # не вычисляем градиенты → экономим память и время
        for images, labels in val_loader:
            outputs = model(images.to(DEVICE))
            val_preds.extend(outputs.argmax(1).cpu().numpy())
            val_true.extend(labels.numpy())

    val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)

    # ════════════════════════════════════
    # Сохраняем лучший чекпоинт
    # ════════════════════════════════════
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  → Новый лучший! F1={val_f1:.4f}")

    scheduler.step()   # уменьшаем lr

    print(f"Epoch {epoch+1:02d} | "
          f"train_loss={train_loss:.3f} train_acc={train_acc:.3f} | "
          f"val_F1={val_f1:.3f}")

print(f"\nЛучший F1: {best_f1:.4f}")
```

### Загрузка модели для инференса

```python
import json

# Загружаем маппинг классов
with open("class_names.json") as f:
    idx_to_class = json.load(f)   # {"0": "wheat", "1": "corn", ...}

# Создаём модель и загружаем веса
model = timm.create_model("efficientnet_b3", pretrained=False,
                           num_classes=len(idx_to_class))
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

# Предсказание для одного изображения:
val_transform = A.Compose([A.Resize(224,224), A.Normalize(...), ToTensorV2()])
img    = np.array(Image.open("new_seed.jpg").convert("RGB"))
tensor = val_transform(image=img)["image"].unsqueeze(0)   # добавляем batch dim

with torch.no_grad():
    probs = torch.softmax(model(tensor), dim=1)[0]

top3 = probs.topk(3)
for prob, idx in zip(top3.values, top3.indices):
    print(f"{idx_to_class[str(idx.item())]}: {prob.item():.1%}")
```

---

## 9. Метрики качества

### Матрица ошибок (Confusion Matrix)

Строка = истинный класс. Столбец = предсказанный класс.
Диагональ = правильные ответы. Остальное = ошибки.

```
               wheat  corn  sunflower
wheat   →  [   45      3       2   ]   45 правильно, 5 ошибок
corn    →  [    2     48       0   ]   48 правильно, 2 ошибки
sunflower→ [    1      0      44   ]   44 правильно, 1 ошибка

Читаем строку: из всех реальных пшениц — 45 нашли правильно
Читаем столбец: из всего что назвали пшеницей — 45+2+1=48, и 45 правда пшеница
```

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.ylabel("Истинный класс")
plt.xlabel("Предсказанный класс")
plt.show()
```

### Precision, Recall, F1

```
TP = True Positive  = правильно нашли
FP = False Positive = ложная тревога (сказали "пшеница" — не пшеница)
FN = False Negative = пропустили (пшеница, но сказали что-то другое)

Precision = TP / (TP + FP)
  "Из того что назвал пшеницей — сколько реально пшеница?"
  Важна когда цена ложной тревоги высока.

Recall = TP / (TP + FN)
  "Из всей реальной пшеницы — сколько нашёл?"
  Важна когда нельзя пропускать примеры.

F1 = 2 * Precision * Recall / (Precision + Recall)
  Баланс между Precision и Recall.

F1 macro = среднее F1 по всем классам.
  Штрафует за плохую работу на ЛЮБОМ классе — правильная метрика при дисбалансе.
```

```python
from sklearn.metrics import classification_report, f1_score

print(classification_report(y_true, y_pred, target_names=classes))

# Пример вывода:
#               precision  recall  f1-score  support
# wheat             0.94    0.90      0.92       50
# corn              0.94    0.96      0.95       50
# sunflower         0.96    0.88      0.92       50
# macro avg         0.94    0.91      0.93      150   ← F1 macro = 0.93
```

### ROC-AUC

```
ROC кривая = TPR (recall) против FPR при разных порогах вероятности
AUC = площадь под кривой:
  1.0  = идеальная модель
  0.9+ = отличная
  0.8+ = хорошая
  0.5  = случайное угадывание

Строим отдельно для каждого класса (one-vs-rest):
  класс i vs все остальные
```

```python
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Бинаризуем метки (нужно для multi-class ROC)
y_bin   = label_binarize(y_true, classes=list(range(len(classes))))
# y_bin.shape = (N, num_classes)

y_probs = model_softmax_outputs   # вероятности из модели
# y_probs.shape = (N, num_classes)

plt.figure(figsize=(10, 7))
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=1, label="Случайная модель")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC-AUC кривые (one-vs-rest)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
```

---

## 10. Диагностика проблем

### Смотри на learning curves

```
learning_curves.png покажет одну из четырёх картин:

1. НОРМАЛЬНО:
   train_loss ↓   val_loss ↓   обе падают и стабилизируются
   train_acc  ↑   val_acc  ↑   → продолжай обучение

2. ПЕРЕОБУЧЕНИЕ (overfitting):
   train_loss ↓ продолжает падать
   val_loss   ↑ начинает расти
   → Решение:
      AUGMENTATION = "heavy"
      weight_decay = 1e-3 (увеличить)
      добавь больше фото

3. НЕДООБУЧЕНИЕ (underfitting):
   train_loss высокий и падает медленно
   → Решение:
      больше EPOCHS (30-50)
      LR = 1e-3 (увеличить)
      попробуй efficientnet_b4

4. СКАЧКИ loss:
   loss прыгает вверх-вниз
   → LR слишком большой
      LR = 1e-4 (уменьшить)
```

### Смотри на confusion matrix

```
Что ищем:

1. ДИАГОНАЛЬ ЯРКАЯ = хорошо, всё правильно

2. СТРОКА ТЁМНАЯ = модель плохо распознаёт этот класс
   → мало примеров? добавь фото
   → похож на другой класс? добавь разнообразие

3. ДВА КЛАССА ПУТАЮТСЯ (симметричные ячейки вне диагонали):
   wheat ↔ barley = 15 ошибок в каждую сторону
   → эти два класса визуально похожи
   → решение: больше фото именно этих двух классов
              разные условия съёмки, разные ракурсы
```

### Частые ошибки и их решения

```
❌ model.train() забыл перед train_epoch
   → Dropout отключён, BatchNorm в режиме инференса → плохое обучение

❌ model.eval() забыл перед val_epoch
   → Dropout активен на валидации → случайные результаты

❌ optimizer.zero_grad() не вызвал
   → Градиенты накапливаются → взрыв градиентов → loss = NaN

❌ with torch.no_grad() не использовал на валидации
   → Хранит все промежуточные результаты → out of memory

❌ Не нормализовал при инференсе
   → Модель видит числа 0-255, обучена на ~[-2, 2] → мусор на выходе

❌ shuffle=True одновременно с sampler
   → ValueError: sampler уже перемешивает, убери shuffle

❌ CPU вместо GPU
   → Проверь: torch.cuda.is_available()
   → images.to(DEVICE) и model.to(DEVICE) — оба должны быть на одном устройстве
```

---

## 11. Чеклист соревнования

### До начала

```
□ pip install -r requirements.txt
□ python -c "import torch; print(torch.cuda.is_available())"  → True
□ python -c "import timm, transformers, umap, plotly; print('OK')"
```

### Получил данные

```
□ Посмотрел структуру папок — уже по классам или нет?
□ Посчитал сколько фото на класс
□ Поставил PARTICIPANT_ID в обоих скриптах
□ Поставил правильный SEEDS_DIR
□ Запустил module_a.py — сразу!
□ Пока module_a работает — изучаешь данные глазами
```

### После module_a.py

```
□ Image_Library_Description.csv создан
□ Labeled_Set/ создан с фото по классам
□ Все 4 гистограммы сохранены
□ Цветность и естественность посчитаны
□ Аномалии? (один класс сильно меньше других → AUGMENTATION = "heavy")
```

### Запуск module_b.py

```
□ Проверил что output_a/ существует
□ Поставил нужную аугментацию (смотри на количество фото)
□ Запустил — пока обучается (5-15 мин) пишешь отчёт по Модулю A
```

### После module_b.py

```
□ best_model.pth создан
□ confusion_matrix.png — проверил ошибки
□ learning_curves.png — нет переобучения?
□ F1 > 0.8? → отлично, сдавай
□ F1 < 0.6? → смотри какие классы путаются, решаешь проблему
□ ROC-AUC > 0.9? → хорошо
□ Все файлы в Day1_MB_{id}/
```

### Что делать если F1 плохой

```
F1 = 0.4-0.6:
  1. Смотри confusion matrix — что с чем путается
  2. AUGMENTATION = "heavy"
  3. EPOCHS = 30
  4. Запусти ещё раз

F1 < 0.4:
  1. Проверь данные — правильно ли разложены по папкам?
  2. Мало данных (< 30 фото/класс)? → сфотографируй больше
  3. Попробуй LR = 1e-4 (меньше)
```

---

## Шпаргалка — ключевые команды

```python
# Загрузить изображение
img = np.array(Image.open("photo.jpg").convert("RGB"))

# CLIP эмбеддинг
feat = clip_model.get_image_features(**processor(images=Image.open(p), return_tensors="pt").to(device))
emb  = (feat / feat.norm()).squeeze().cpu().numpy()

# Кластеризация
labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(embeddings)
sil    = silhouette_score(embeddings, labels)

# t-SNE
emb2d = TSNE(n_components=2, random_state=42).fit_transform(embeddings)

# Создать модель
model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=N)

# Предсказание
with torch.no_grad():
    probs = torch.softmax(model(tensor.to(device)), dim=1)[0]

# Метрики
print(classification_report(y_true, y_pred, target_names=classes))
f1 = f1_score(y_true, y_pred, average="macro")
```

---

*Dr. Vision — "Понять один раз глубоко лучше, чем знать поверхностно много."*
