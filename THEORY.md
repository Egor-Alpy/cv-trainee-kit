# Компьютерное зрение — полный разбор

Всё что нужно понимать и уметь. От пикселя до готовой модели.

---

## Содержание

1. [Изображение как данные](#1-изображение-как-данные)
2. [Нормализация — зачем и как](#2-нормализация--зачем-и-как)
3. [Классика: HOG](#3-классика-hog)
4. [Свёрточная нейросеть (CNN)](#4-свёрточная-нейросеть-cnn)
5. [Transfer Learning — главный инструмент](#5-transfer-learning--главный-инструмент)
6. [Датасет и DataLoader](#6-датасет-и-dataloader)
7. [Аугментация](#7-аугментация)
8. [Дисбаланс классов](#8-дисбаланс-классов)
9. [Цикл обучения](#9-цикл-обучения)
10. [Метрики и диагностика](#10-метрики-и-диагностика)
11. [Особые случаи](#11-особые-случаи)

---

## 1. Изображение как данные

Изображение в Python — это трёхмерный NumPy-массив формы `(H, W, C)`:

```
H — высота в пикселях
W — ширина в пикселях
C — каналы: RGB → 3, серое → 1
```

Каждый пиксель — три числа от 0 до 255 (тип `uint8`):

```
Пиксель (100, 200, 50) → тёмно-зелёный
         R    G    B
```

```python
from PIL import Image
import numpy as np

img = Image.open("photo.jpg").convert("RGB")  # всегда .convert("RGB") — см. ниже
arr = np.array(img)

print(arr.shape)   # (480, 640, 3)
print(arr.dtype)   # uint8
print(arr.min(), arr.max())  # 0, 255

# Отдельный канал — срез по третьей оси
red   = arr[:, :, 0]   # shape (480, 640)
green = arr[:, :, 1]
blue  = arr[:, :, 2]

# arr[:, :, 0] читается как:
# : — все строки
# : — все столбцы
# 0 — канал 0 (Red)
```

**Зачем `.convert("RGB")`**

`Image.open()` не гарантирует формат. Файл может быть:
- `RGBA` — PNG с прозрачностью (4 канала)
- `L` — серое (1 канал)
- `P` — палитровый режим

Нейросеть ожидает ровно 3 канала. `.convert("RGB")` приводит любой формат к RGB.
Если картинка серая — канал дублируется трижды: `(128) → (128, 128, 128)`.

**PIL vs OpenCV**

```python
import cv2

img_cv = cv2.imread("photo.jpg")   # читает в BGR (!)
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# OpenCV хранит каналы в порядке Blue-Green-Red.
# Нейросеть ожидает RGB. Перепутаешь — модель видит "неправильные" цвета.
# Правило: читай через PIL, он всегда RGB.
```

---

## 2. Нормализация — зачем и как

Нейросеть обучается градиентным спуском — веса обновляются маленькими шагами.
Если входные числа большие (0–255), градиенты огромные и нестабильные → модель не сходится.

Нормализация делается в два шага:

### Шаг 1 — делим на 255

```
uint8 [0, 255] → float [0, 1]
```

`transforms.ToTensor()` делает это автоматически.

### Шаг 2 — стандартизация: (x - mean) / std

**Среднее (mean)** — центр данных. Вычитаем его → данные симметричны вокруг нуля:

```
До:    [0.3, 0.5, 0.7]  центр в 0.5
После: [-0.2, 0.0, +0.2]  центр в 0
```

**Стандартное отклонение (std)** — средний размер разброса. Как его считать:

```
Данные: [0.2, 0.4, 0.5, 0.6, 0.8]
mean = 0.5

Отклонения: [-0.3, -0.1, 0.0, +0.1, +0.3]
Квадраты:   [0.09, 0.01, 0.0, 0.01, 0.09]
Среднее:    0.04
std = √0.04 = 0.2
```

Делим на std → все каналы приводятся к одному масштабу разброса ~1.
Иначе канал с большим разбросом доминирует в градиентах, хотя это не значит что он важнее.

**Откуда числа ImageNet:**

```python
mean = [0.485, 0.456, 0.406]  # среднее R, G, B по 1.2 млн фото
std  = [0.229, 0.224, 0.225]  # стандартное отклонение
```

Посчитаны один раз по всему ImageNet. Используешь предобученную модель → подаёшь данные
в том же формате что она видела при обучении.

**Итоговая картина:**

```
Пиксель      →  / 255      →  (x - mean) / std
[128, 64, 200]  [0.50, 0.25, 0.78]  [+0.07, -0.92, +1.66]
uint8 0-255     float 0-1           float ~[-2, 2]  ← подаём в модель
```

**Код:**

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),           # PIL (H,W,C) uint8 → Tensor (C,H,W) float [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

tensor = transform(img)   # torch.Size([3, 224, 224]), диапазон ~[-2.1, 2.6]
```

---

## 3. Классика: HOG

HOG (Histogram of Oriented Gradients) — метод описания изображения через направления границ.
Используется до нейросетей или когда данных очень мало (< 100 фото).

**Алгоритм:**

```
1. Найти перепады яркости (градиенты) по горизонтали и вертикали
2. Разбить изображение на ячейки 8×8 пикселей
3. В каждой ячейке: гистограмма направлений (9 направлений 0°–180°)
4. Объединить соседние ячейки в блоки 2×2, нормализовать
5. Собрать все гистограммы → вектор ~1764 числа
```

Этот вектор — описание формы объекта. Похожие объекты → похожие векторы.

```python
from skimage.feature import hog
from skimage import color
from PIL import Image
import numpy as np

def extract_hog(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((128, 128))
    gray = color.rgb2gray(np.array(img))
    features, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True
    )
    return features, hog_image   # вектор ~1764 числа, визуализация

# Сравнить два изображения
features_a, _ = extract_hog("oak.jpg")
features_b, _ = extract_hog("maple.jpg")
distance = np.linalg.norm(features_a - features_b)
# Маленькое расстояние → похожие объекты
```

**HOG + SVM — классический пайплайн до DL:**

```python
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10)),
])
clf.fit(X_hog_train, y_train)
clf.predict(X_hog_test)
```

---

## 4. Свёрточная нейросеть (CNN)

CNN делает то же что HOG, но **автоматически учит** какие признаки важны.

### Свёртка (Conv2d)

Маленький фильтр (3×3 или 5×5) скользит по изображению и вычисляет "похожесть"
каждой зоны на этот фильтр:

```
Фильтр для вертикальных рёбер:    Фильтр для горизонтальных:
 [-1, 0, 1]                         [-1, -1, -1]
 [-1, 0, 1]                         [ 0,  0,  0]
 [-1, 0, 1]                         [ 1,  1,  1]
```

Сеть сама учит значения фильтров в процессе обучения.

### MaxPooling

Уменьшает размер карты признаков вдвое, беря максимум в каждом окне 2×2:

```
До MaxPool:         После MaxPool:
4 3 2 1             4 3
5 6 1 2     →       8 4
8 7 4 3
1 2 3 4
```

Зачем: уменьшает вычисления и делает признаки инвариантными к небольшим сдвигам.

### Архитектура CNN

```
Вход (3, 224, 224)
  ↓
Conv(3→32) + ReLU → (32, 224, 224)   — 32 карты признаков (фильтра)
  ↓
MaxPool            → (32, 112, 112)   — уменьшить вдвое
  ↓
Conv(32→64) + ReLU → (64, 112, 112)
  ↓
MaxPool            → (64, 56, 56)
  ↓
Flatten            → (204800,)        — вытянуть в вектор
  ↓
Linear(204800→256) + ReLU
  ↓
Linear(256→num_classes)               — вероятность каждого класса
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 56 * 56, 256)
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):                         # x: (batch, 3, 224, 224)
        x = self.pool(F.relu(self.conv1(x)))      # → (batch, 32, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))      # → (batch, 64, 56, 56)
        x = x.flatten(1)                          # → (batch, 204800)
        x = F.relu(self.fc1(x))
        return self.fc2(x)                        # → (batch, num_classes)
```

**Проблема:** обучить CNN с нуля хорошо — нужны сотни тысяч фото и часы на GPU.
При маленьком датасете → transfer learning.

---

## 5. Transfer Learning — главный инструмент

Берём модель, обученную на ImageNet (1.2 млн фото, 1000 классов).
Она уже умеет видеть края, текстуры, формы. Заменяем последний слой под наши классы
и дообучаем на нашем датасете.

```
Без transfer learning:  200 фото → ~50% точность
С transfer learning:    200 фото → ~90% точность
```

**Что происходит внутри:**

```
ResNet50 (обученная на ImageNet)
├── Слои 1-3: края, градиенты яркости (универсально для любых фото)
├── Слои 4-6: текстуры, части объектов (прожилки листа, шерсть, кирпич)
├── Слои 7-9: сложные паттерны (форма листа, морда животного)
└── Последний слой: Linear(2048, 1000) ← заменяем на Linear(2048, N_классов)
```

### timm — лучшая библиотека для моделей

```python
import timm

model = timm.create_model(
    "efficientnet_b3",  # название модели
    pretrained=True,    # скачать веса ImageNet
    num_classes=5,      # наше число классов — timm сам заменит последний слой
)
```

### Какую модель выбирать

| Модель | Параметры | Точность ImageNet | Когда |
|--------|-----------|-------------------|-------|
| `resnet18` | 11M | 70% | нет GPU, быстрый прототип |
| `efficientnet_b0` | 5M | 77% | нет GPU, маленький датасет |
| **`efficientnet_b3`** | **12M** | **81%** | **стартовая точка** |
| `efficientnet_b4` | 19M | 83% | есть время и данные |
| `convnext_tiny` | 28M | 82% | стабильная, хорошо обобщает |

Начинай с `efficientnet_b3`. Если есть время — пробуй `convnext_tiny`.

### Полный код создания модели

```python
import timm
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=5)
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
# CosineAnnealingLR: LR плавно снижается от 3e-4 до ~0 за 20 эпох
# Это лучше постоянного LR — модель сходится точнее в конце

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# label_smoothing: вместо "100% дуб" учим "90% дуб, 10% другие"
# Защищает от переуверенности модели, улучшает обобщение на ~0.5-1%
```

---

## 6. Датасет и DataLoader

### Структура папок

```
dataset/
├── train/
│   ├── oak/
│   │   ├── IMG_001.jpg
│   │   └── IMG_002.jpg
│   ├── maple/
│   └── birch/
└── val/
    ├── oak/
    ├── maple/
    └── birch/
```

`ImageFolder` читает эту структуру автоматически и создаёт маппинг `class_name → int`:

```python
from torchvision.datasets import ImageFolder

dataset = ImageFolder("dataset/train", transform=transform)
print(dataset.class_to_idx)  # {"birch": 0, "maple": 1, "oak": 2}
print(dataset.classes)       # ["birch", "maple", "oak"]
```

### Кастомный Dataset (если структура другая)

```python
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class MyDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[int], transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]  # albumentations API
        return img, self.labels[idx]
```

### DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,      # перемешивать при обучении (не нужен если есть sampler)
    num_workers=4,     # параллельная загрузка — не ставь > кол-ва ядер CPU
    pin_memory=True,   # ускорение передачи данных на GPU
    drop_last=True,    # отбросить последний неполный батч (только train)
)
```

### Разбивка train/val

```python
from sklearn.model_selection import train_test_split

train_paths, val_paths, train_labels, val_labels = train_test_split(
    paths, labels,
    test_size=0.2,
    stratify=labels,   # обязательно — иначе маленькие классы могут не попасть в val
    random_state=42,
)
```

---

## 7. Аугментация

Аугментация — случайные трансформации при обучении. Модель каждый раз видит
чуть разный вариант одного фото → защита от переобучения.

**Важно:** на валидации аугментация не нужна — только resize и нормализация.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# albumentations принимает np.ndarray, не PIL
# в __getitem__: img = np.array(Image.open(path).convert("RGB"))

train_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),          # случайный кроп — разный масштаб/угол
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, p=0.5),
    A.HueSaturationValue(p=0.4),     # разные цвета (сезоны, освещение)
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Применение:
img = np.array(Image.open("photo.jpg").convert("RGB"))
result = train_transform(image=img)["image"]   # ключ "image" — особенность albumentations
```

**Что использовать для разных задач:**

| Задача | Рекомендации |
|--------|-------------|
| Листья, природа | Все флипы, VerticalFlip, HueSaturation (разные сезоны) |
| Документы, текст | Только HorizontalFlip=False, небольшой поворот |
| Монеты, симметричные объекты | RandomRotate90, все флипы |
| Дефекты на производстве | Осторожно с цветом — цвет дефекта важен |

---

## 8. Дисбаланс классов

Если класс A — 200 фото, класс B — 20 фото, модель выучит предсказывать A на всё.

### WeightedRandomSampler — редкие классы выбираются чаще

```python
from torch.utils.data import WeightedRandomSampler
from collections import Counter

label_counts = Counter(labels)        # {0: 200, 1: 50, 2: 20}
weights = [1.0 / label_counts[l] for l in labels]
# Вес обратно пропорционален частоте класса → редкие классы чаще попадают в батч

sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)  # shuffle убрать!
```

### class_weight в loss — штраф за ошибки на редких классах

```python
import torch
import torch.nn as nn

counts = torch.tensor([label_counts[i] for i in range(num_classes)], dtype=torch.float)
class_weights = 1.0 / counts
class_weights = class_weights / class_weights.sum()

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

**Правило:** дисбаланс > 3:1 → WeightedRandomSampler. Не помогает → добавь class_weight в loss.

---

## 9. Цикл обучения

```python
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()   # включает dropout, batch norm в режим обучения
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # обнуляем накопленные градиенты
        outputs = model(images)        # forward pass: предсказание
        loss = criterion(outputs, labels)
        loss.backward()                # backward pass: считаем градиенты
        optimizer.step()               # обновляем веса

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def val_epoch(model, loader, criterion, device):
    model.eval()    # выключает dropout, batch norm в режим инференса
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():   # не считаем градиенты — экономим память
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


# Основной цикл
best_val_acc = 0.0
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss, val_acc     = val_epoch(model, val_loader, criterion, DEVICE)
    scheduler.step()

    print(f"Epoch {epoch+1:02d} | "
          f"train loss={train_loss:.3f} acc={train_acc:.3f} | "
          f"val loss={val_loss:.3f} acc={val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")


# Сохранить маппинг классов — нужен при инференсе
import json
idx_to_class = {str(v): k for k, v in train_dataset.class_to_idx.items()}
with open("class_names.json", "w") as f:
    json.dump(idx_to_class, f, ensure_ascii=False)
# {"0": "birch", "1": "maple", "2": "oak"}


# Загрузить модель для инференса
model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=5)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()
```

### Early stopping

```python
class EarlyStopping:
    def __init__(self, patience: int = 7):
        self.patience = patience
        self.best = None
        self.counter = 0

    def __call__(self, score: float) -> bool:
        if self.best is None or score > self.best + 0.001:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience   # True = стоп

early_stop = EarlyStopping(patience=7)
for epoch in range(100):
    ...
    if early_stop(val_f1):
        print(f"Early stopping на эпохе {epoch+1}")
        break
```

---

## 10. Метрики и диагностика

### Accuracy vs F1

**Accuracy** — доля правильных ответов. Бесполезна при дисбалансе:

```
9 фото дуба, 1 фото клёна. Модель говорит "дуб" на всё → accuracy = 90%.
Но клён она не умеет распознавать.
```

**F1 macro** — считает F1 отдельно для каждого класса, берёт среднее.
Штрафует за плохую работу на редких классах. Используй как основную метрику.

```python
from sklearn.metrics import f1_score, classification_report, confusion_matrix

preds  = model(images).argmax(1).cpu().numpy()
labels = labels.cpu().numpy()

print(f1_score(labels, preds, average="macro"))
print(classification_report(labels, preds, target_names=class_names))
# precision — из предсказанных "дуб", сколько реально дубов
# recall    — из всех реальных дубов, сколько нашла модель
# f1-score  — баланс precision и recall
# support   — сколько примеров в тесте
```

### Confusion matrix

Показывает что с чем путается:

```python
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names, yticklabels=class_names, cmap="Blues")
# Строка = истинный класс, столбец = предсказанный
# Диагональ = правильные ответы
# Вне диагонали = ошибки: какой класс с каким путается
```

### Learning curves — диагностика проблем

```
Хорошее обучение:
  train_loss ↓  val_loss ↓  — обе падают и стабилизируются

Переобучение (overfitting):
  train_loss ↓  продолжает падать
  val_loss   ↑  начинает расти
  → добавь аугментацию, увеличь weight_decay, включи Dropout

Недообучение:
  train_loss высокий и медленно падает
  → больше эпох, больший LR, более мощная модель

Скачки:
  loss прыгает вверх-вниз
  → уменьши LR
```

---

## 11. Особые случаи

### Очень мало данных (< 50 фото на класс)

```python
# 1. Заморозь все слои кроме последнего — обучай только голову
for param in model.parameters():
    param.requires_grad = False
for param in model.get_classifier().parameters():
    param.requires_grad = True

# 2. После 5 эпох — разморозь и дообучи с маленьким LR
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
```

### Похожие классы плохо различаются

Смотри confusion matrix: если два класса путаются → специфичная аугментация.
Добавь больше разнообразия именно для этих классов. Или добавь данных.

### Ансамбль — финальный буст +1–3%

Обучи 2–3 разных модели, усредни вероятности:

```python
models = [model_1, model_2, model_3]

probs_list = []
for model in models:
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(images), dim=1).cpu().numpy()
    probs_list.append(probs)

avg_probs = np.mean(probs_list, axis=0)
preds = avg_probs.argmax(axis=1)
```

### Инференс на одном фото

```python
import json
import torch
import timm
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Загрузить модель
with open("class_names.json") as f:
    idx_to_class = json.load(f)

model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=len(idx_to_class))
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Предсказание
img = np.array(Image.open("photo.jpg").convert("RGB"))
tensor = transform(image=img)["image"].unsqueeze(0)  # добавляем batch dimension

with torch.no_grad():
    probs = torch.softmax(model(tensor), dim=1)[0]

top3 = probs.topk(3)
for prob, idx in zip(top3.values, top3.indices):
    print(f"{idx_to_class[str(idx.item())]}: {prob.item():.1%}")
```

### Порядок улучшений на соревновании

```
1. baseline: efficientnet_b3, medium аугментация, 20 эпох
2. + WeightedRandomSampler (если дисбаланс > 3:1)
3. + heavy аугментация (если переобучение)
4. + efficientnet_b4 или convnext_tiny
5. + ансамбль двух моделей разных архитектур
```
