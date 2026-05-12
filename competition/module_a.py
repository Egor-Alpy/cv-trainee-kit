"""
МОДУЛЬ A — Классификация семян: подготовка данных и анализ
===========================================================
Запуск:  python module_a.py

Что нужно на входе:
    SEEDS_DIR/ — папка с подпапками по классам семян
        wheat/
            img001.jpg
            img002.jpg
        corn/
            img001.jpg
        sunflower/
            ...

Что получишь на выходе:
    output_a/
        Image_Library_Description.csv   ← главная таблица
        Labeled_Set/                    ← по 30 фото на класс
        hist_size.png
        hist_color.png
        hist_natural.png
        sample_grid.png
    Day1_MA_{PARTICIPANT_ID}/           ← папка для сдачи (копия)
"""

# ============================================================
# CONFIG — меняй только здесь
# ============================================================
SEEDS_DIR      = "seeds"       # папка с подпапками-классами
OUTPUT_DIR     = "output_a"
PARTICIPANT_ID = "1"
MAX_PER_CLASS  = 30            # сколько брать в Labeled_Set
# ============================================================

import os, shutil, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
from transformers import CLIPProcessor, CLIPModel

# ── Папки ───────────────────────────────────────────────────
OUT        = Path(OUTPUT_DIR)
LABELED    = OUT / "Labeled_Set"
REPORT_DIR = Path(f"Day1_MA_{PARTICIPANT_ID}")
OUT.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

print(f"Устройство: {DEVICE}")
print(f"Папка семян: {SEEDS_DIR}")

# ════════════════════════════════════════════════════════════
# 1.1  ЗАГРУЗКА И ОРГАНИЗАЦИЯ ИЗОБРАЖЕНИЙ
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1.1  ЗАГРУЗКА ИЗОБРАЖЕНИЙ")
print("=" * 60)

seeds_path = Path(SEEDS_DIR)
classes    = sorted([d.name for d in seeds_path.iterdir() if d.is_dir()])
print(f"Найдено классов: {len(classes)}  →  {classes}")

records = []
for cls in classes:
    cls_dir = seeds_path / cls
    images  = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    for img_path in images:
        try:
            pil = Image.open(img_path).convert("RGB")
            w, h = pil.size
            size_b = img_path.stat().st_size
            records.append({
                "filename":   img_path.name,
                "path":       str(img_path),
                "class_name": cls,
                "width_px":   w,
                "height_px":  h,
                "size_bytes": size_b,
                "size_mb":    round(size_b / (1024 * 1024), 4),
            })
        except Exception:
            continue
    print(f"  {cls:<20} {len(images)} фото")

df = pd.DataFrame(records)
print(f"\nВсего изображений: {len(df)}")
print(df["class_name"].value_counts().to_string())

# Создаём Labeled_Set — по MAX_PER_CLASS на класс
for cls in classes:
    (LABELED / cls).mkdir(parents=True, exist_ok=True)

labeled_count = Counter()
for _, row in df.iterrows():
    cls = row["class_name"]
    if labeled_count[cls] < MAX_PER_CLASS:
        dest = LABELED / cls / row["filename"]
        try:
            shutil.copy2(row["path"], dest)
            labeled_count[cls] += 1
        except Exception:
            pass

print("\nLabeled_Set:")
for cls, cnt in labeled_count.items():
    print(f"  {cls:<20} {cnt} фото")

# ════════════════════════════════════════════════════════════
# 1.2  IMAGE_LIBRARY_DESCRIPTION
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1.2  IMAGE_LIBRARY_DESCRIPTION")
print("=" * 60)

SIZE_BINS   = [0, 1, 5, 10, float("inf")]
SIZE_LABELS = ["< 1 МБ", "1–5 МБ", "5–10 МБ", "> 10 МБ"]

df["size_category"] = pd.cut(
    df["size_mb"], bins=SIZE_BINS, labels=SIZE_LABELS, right=False
).astype(str)

# Цветность
def is_color(path: str) -> bool:
    try:
        img = np.array(Image.open(path).convert("RGB"))
        r, g, b = img[:,:,0].astype(int), img[:,:,1].astype(int), img[:,:,2].astype(int)
        diff = (np.mean(np.abs(r-g)) + np.mean(np.abs(r-b)) + np.mean(np.abs(g-b))) / 3
        return float(diff) > 8.0
    except Exception:
        return False

print("Определяем цветность...")
df["is_color"]    = [is_color(p) for p in tqdm(df["path"])]
df["color_label"] = df["is_color"].map({True: "Цветное", False: "Полутоновое"})

# Насыщенность
def saturation_cat(path: str) -> str:
    try:
        sat = np.array(Image.open(path).convert("HSV"))[:,:,1].mean()
        if sat < 30:   return "бесцветный"
        if sat < 100:  return "средней насыщенности"
        return "насыщенный"
    except Exception:
        return "бесцветный"

print("Определяем насыщенность...")
df["saturation"] = [saturation_cat(p) for p in tqdm(df["path"])]

# Сохраняем промежуточно
CSV_PATH = OUT / "Image_Library_Description.csv"
df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
print(f"CSV сохранён: {CSV_PATH}  ({len(df)} строк, {len(df.columns)} столбцов)")

# ════════════════════════════════════════════════════════════
# 1.3  АНАЛИЗ И ВИЗУАЛИЗАЦИЯ
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1.3  АНАЛИЗ")
print("=" * 60)

# ── Загружаем CLIP один раз ─────────────────────────────────
print("Загружаем CLIP...")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model     = clip_model.to(DEVICE).eval()
print(f"CLIP загружен на {DEVICE}")

NATURAL_TEXTS = [
    "a photograph of real seeds, grains, or plant parts",
    "a diagram, chart, drawing, or artificial illustration",
]

@torch.no_grad()
def naturalness(path: str):
    try:
        img    = Image.open(path).convert("RGB")
        inputs = clip_processor(
            text=NATURAL_TEXTS, images=img,
            return_tensors="pt", padding=True, truncation=True,
        ).to(DEVICE)
        probs = clip_model(**inputs).logits_per_image.softmax(dim=1)[0].cpu().tolist()
        return probs[0] > probs[1], round(probs[0], 4)
    except Exception:
        return True, 1.0

print("Определяем естественность...")
flags, probs = [], []
for p in tqdm(df["path"].tolist()):
    f, pr = naturalness(p)
    flags.append(f)
    probs.append(pr)

df["is_natural"]    = flags
df["natural_prob"]  = probs
df["natural_label"] = df["is_natural"].map({True: "Естественное", False: "Искусственное"})
df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

# ── Описательная статистика ─────────────────────────────────
print("\nСтатистика размеров файлов по классу:")
stats = df.groupby("class_name")["size_mb"].describe().round(4)
print(stats.to_string())

print("\nЦветность по классу:")
print(df.groupby(["class_name", "color_label"]).size().to_string())

print("\nЕстественность по классу:")
print(df.groupby(["class_name", "natural_label"]).size().to_string())

# ── Гистограмма 1: количество фото по классам ───────────────
fig, ax = plt.subplots(figsize=(10, 5))
counts  = df["class_name"].value_counts()
counts.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(counts)),
            edgecolor="white", width=0.7)
ax.set_title("Количество изображений по классам семян", fontsize=14)
ax.set_xlabel("Класс"); ax.set_ylabel("Количество")
ax.tick_params(axis="x", rotation=30)
ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(counts):
    ax.text(i, v + 0.3, str(v), ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(OUT / "hist_classes.png", dpi=150)
plt.savefig(REPORT_DIR / "hist_classes.png", dpi=150)
plt.show()

# ── Гистограмма 2: цветность по классам ─────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
color_data = df.groupby(["class_name", "color_label"]).size().unstack(fill_value=0)
for col in ["Цветное", "Полутоновое"]:
    if col not in color_data.columns:
        color_data[col] = 0
color_data[["Цветное", "Полутоновое"]].plot(
    kind="bar", ax=ax, color=["#42A5F5", "#90A4AE"], edgecolor="white", width=0.7
)
ax.set_title("Цветность изображений по классам", fontsize=14)
ax.set_xlabel("Класс"); ax.set_ylabel("Количество")
ax.tick_params(axis="x", rotation=30)
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "hist_color.png", dpi=150)
plt.savefig(REPORT_DIR / "hist_color.png", dpi=150)
plt.show()

# ── Гистограмма 3: естественность по классам ────────────────
fig, ax = plt.subplots(figsize=(12, 5))
nat_data = df.groupby(["class_name", "natural_label"]).size().unstack(fill_value=0)
for col in ["Естественное", "Искусственное"]:
    if col not in nat_data.columns:
        nat_data[col] = 0
nat_data[["Естественное", "Искусственное"]].plot(
    kind="bar", ax=ax, color=["#66BB6A", "#EF5350"], edgecolor="white", width=0.7
)
ax.set_title("Естественные vs Искусственные изображения по классам", fontsize=14)
ax.set_xlabel("Класс"); ax.set_ylabel("Количество")
ax.tick_params(axis="x", rotation=30)
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "hist_natural.png", dpi=150)
plt.savefig(REPORT_DIR / "hist_natural.png", dpi=150)
plt.show()

# ── Гистограмма 4: размеры файлов по классам ────────────────
fig, ax = plt.subplots(figsize=(12, 5))
size_data = df.groupby(["class_name", "size_category"]).size().unstack(fill_value=0)
for col in SIZE_LABELS:
    if col not in size_data.columns:
        size_data[col] = 0
size_data[SIZE_LABELS].plot(kind="bar", ax=ax, colormap="Set3", edgecolor="white", width=0.7)
ax.set_title("Распределение размеров файлов по классам", fontsize=14)
ax.set_xlabel("Класс"); ax.set_ylabel("Количество")
ax.tick_params(axis="x", rotation=30)
ax.legend(title="Размер"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "hist_size.png", dpi=150)
plt.savefig(REPORT_DIR / "hist_size.png", dpi=150)
plt.show()

# ── Сетка примеров (по 5 фото на класс) ─────────────────────
n_cls  = len(classes)
n_cols = 5
fig, axes = plt.subplots(n_cls, n_cols, figsize=(n_cols * 2.5, n_cls * 2.5))
if n_cls == 1:
    axes = [axes]

for row_i, cls in enumerate(classes):
    cls_rows = df[df["class_name"] == cls].head(n_cols)
    for col_i in range(n_cols):
        ax = axes[row_i][col_i]
        ax.axis("off")
        if col_i < len(cls_rows):
            try:
                img = Image.open(cls_rows.iloc[col_i]["path"]).convert("RGB")
                ax.imshow(img)
            except Exception:
                pass
        if col_i == 0:
            ax.set_ylabel(cls, fontsize=11, labelpad=4)
            ax.axis("on")
            ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("Примеры изображений по классам семян", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUT / "sample_grid.png", dpi=120, bbox_inches="tight")
plt.savefig(REPORT_DIR / "sample_grid.png", dpi=120, bbox_inches="tight")
plt.show()

# ── Финальный CSV ───────────────────────────────────────────
df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
shutil.copy2(CSV_PATH, REPORT_DIR / CSV_PATH.name)

# ════════════════════════════════════════════════════════════
# ИТОГИ
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ИТОГИ МОДУЛЯ A")
print("=" * 60)
print(f"Классов:              {len(classes)}  →  {classes}")
print(f"Всего изображений:    {len(df)}")
print(f"Цветных:              {df['is_color'].sum()} ({df['is_color'].mean():.1%})")
print(f"Естественных:         {df['is_natural'].sum()} ({df['is_natural'].mean():.1%})")
print(f"\nImage_Library_Description: {CSV_PATH}")
print(f"Labeled_Set:               {LABELED}")
print(f"Папка отчёта:              {REPORT_DIR}")
print("\nГотово. Запускай module_b.py")
