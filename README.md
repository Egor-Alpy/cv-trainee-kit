# CV Trainee Kit

Универсальный шаблон по компьютерному зрению.
Работает для любой задачи классификации изображений.

---

## Быстрый старт — что менять под свою задачу

Открой `config.py` и измени три строки:

```python
TASK_NAME = "Распознавание листьев"          # название задачи, отображается в UI
CLASSES   = ["oak", "maple", "birch"]        # твои классы
NUM_CLASSES уже считается автоматически из CLASSES
```

Больше ничего трогать не нужно.

---

## Установка

```bash
pip install torch torchvision timm albumentations opencv-python scikit-learn scikit-image matplotlib seaborn tqdm gradio
```

---

## Структура проекта

```
cv-trainee-kit/
├── config.py              ← ЕДИНСТВЕННЫЙ файл который меняешь
├── train.py               ← обучение модели
├── predict.py             ← предсказание для одного фото
├── app.py                 ← красивый UI для презентации (Gradio)
├── webcam_demo.py         ← демо с веб-камеры
├── dataset/
│   ├── organize.py        ← разбивка фото на train/val
│   └── augmentations.py   ← наборы аугментаций (light/medium/heavy)
└── tools/
    ├── generate_markers.py ← генерация ArUco-маркеров для авторазметки
    └── auto_label.py       ← авторазметка по маркерам
```

---

## Пошаговый план на соревновании

### Блок 1 — Датасет

Фотографируй объекты и раскладывай по папкам:

```
photos_raw/
├── class_1/
│   ├── IMG_001.jpg
│   └── IMG_002.jpg
├── class_2/
└── class_3/
```

Советы по съёмке:
- Однородный фон (белый лист бумаги)
- Равномерный свет, без теней
- Объект занимает 60–80% кадра
- 10–20 разных фото на класс, не один и тот же объект 10 раз

Разбить на train/val (80/20):

```bash
python dataset/organize.py photos_raw/ dataset/
```

### Блок 2 — Разметка

**Вариант А — ArUco-маркеры (автоматически):**

```bash
# Шаг 1: сгенерировать маркеры и распечатать
python tools/generate_markers.py

# Шаг 2: при съёмке класть маркер рядом с объектом

# Шаг 3: авторазметка
python tools/auto_label.py photos/ labeled/
```

**Вариант Б — вручную:** просто раскладывай фото по папкам при съёмке.

### Блок 3 — Обучение

```bash
python train.py
```

Результат: `best_model.pth`, `class_names.json`, `learning_curves.png`, `confusion_matrix.png`

Если метрика плохая:
- F1 < 0.5 → смотри `confusion_matrix.png`, что с чем путается
- Переобучение (val_loss растёт) → смени `AUGMENTATION = "heavy"` в config.py
- Мало данных → добавь фото из открытых источников в нужные классы

### Блок 4 — Демо

Запустить UI для презентации:

```bash
python app.py
# Открывается в браузере: http://localhost:7860
```

Демо с камеры:

```bash
python webcam_demo.py
# Q — выход, S — скриншот
```

Предсказание для одного фото:

```bash
python predict.py фото.jpg
```

---

## Настройка аугментации

В `config.py`:

```python
AUGMENTATION = "medium"  # light | medium | heavy
```

| Значение | Когда использовать |
|----------|-------------------|
| `light`  | Объекты с чёткой ориентацией (документы, монеты) |
| `medium` | Универсальный вариант, начинай с него |
| `heavy`  | Маленький датасет (< 200 фото на класс) |

---

## Чеклист

```
Блок 1:
  □ Сфотографировал 10–20 объектов каждого класса
  □ Разложил по папкам class_name/image.jpg
  □ Запустил dataset/organize.py → получил dataset/train и dataset/val

Блок 2:
  □ Разметил датасет (маркеры или вручную)
  □ Проверил: примерно равное количество фото в каждом классе

Блок 3:
  □ Обновил CLASSES и TASK_NAME в config.py
  □ Запустил train.py → получил best_model.pth
  □ Посмотрел confusion_matrix.png
  □ F1 macro > 0.8

Блок 4:
  □ app.py работает в браузере
  □ webcam_demo.py показывает предсказание в реальном времени
```
