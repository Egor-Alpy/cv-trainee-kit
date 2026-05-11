# ╔══════════════════════════════════════════════════════════════════╗
# ║                  ЕДИНЫЙ КОНФИГ — МЕНЯЙ ТОЛЬКО ЗДЕСЬ             ║
# ╚══════════════════════════════════════════════════════════════════╝
# Всё остальное (train.py, predict.py, app.py) трогать не нужно.

# Название задачи — отображается в UI
TASK_NAME = "Распознавание листьев"

# Список классов — в том же порядке что папки в датасете
# Пример: ["oak", "maple", "birch"] или ["cat", "dog"] или ["crack", "no_crack"]
CLASSES = ["oak", "maple", "birch", "pine", "linden"]

# Датасет (структура: class_name/image.jpg)
TRAIN_DIR = "dataset/train"
VAL_DIR   = "dataset/val"

# Модель из timm — менять не нужно, efficientnet_b3 оптимален
MODEL_NAME  = "efficientnet_b3"
MODEL_PATH  = "best_model.pth"
NAMES_PATH  = "class_names.json"

# Обучение
EPOCHS      = 20
BATCH_SIZE  = 32
LR          = 3e-4

# Аугментация: "light" | "medium" | "heavy"
# light  — только flip, для объектов с чёткой ориентацией (монеты, документы)
# medium — flip + rotate + цвет, универсальный вариант
# heavy  — всё выше + шум + сдвиги, для маленьких датасетов (<200 фото)
AUGMENTATION = "medium"
