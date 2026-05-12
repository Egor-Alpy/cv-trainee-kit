"""
МОДУЛЬ B — Классификация семян: кластеризация и ML-модели
=========================================================
Запуск:  python module_b.py
Требует: output_a/ от module_a.py

Что получишь на выходе:
    output_b/
        elbow.png
        tsne_2d.png
        umap_3d.html
        clusters_description.txt
        confusion_matrix.png
        roc_auc.png
        classification_report.txt
        binary_metrics.png
        binary_roc.png
    best_model.pth              ← обученная модель EfficientNet
    class_names.json
    Day1_MB_{PARTICIPANT_ID}/   ← папка для сдачи
"""

# ============================================================
# CONFIG — меняй только здесь
# ============================================================
SEEDS_DIR      = "seeds"       # та же папка что в module_a.py
MODULE_A_OUT   = "output_a"    # выход module_a.py
OUTPUT_DIR     = "output_b"
PARTICIPANT_ID = "1"
EPOCHS         = 20
BATCH_SIZE     = 32
LR             = 3e-4
AUGMENTATION   = "medium"      # light | medium | heavy
# ============================================================

import os, json, shutil, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import timm
from transformers import CLIPProcessor, CLIPModel

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_curve, auc, roc_auc_score,
)
import umap as umap_lib

# ── Пути ────────────────────────────────────────────────────
OUT        = Path(OUTPUT_DIR)
OUT.mkdir(exist_ok=True)
REPORT_DIR = Path(f"Day1_MB_{PARTICIPANT_ID}")
REPORT_DIR.mkdir(exist_ok=True)

CSV_PATH   = Path(MODULE_A_OUT) / "Image_Library_Description.csv"
LABELED    = Path(MODULE_A_OUT) / "Labeled_Set"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

print(f"Устройство: {DEVICE}")

# ── Загружаем DataFrame ─────────────────────────────────────
df = pd.read_csv(CSV_PATH)
classes = sorted(df["class_name"].unique().tolist())
print(f"Изображений: {len(df)}, классов: {len(classes)}  →  {classes}")

# ── Загружаем CLIP ───────────────────────────────────────────
print("\nЗагружаем CLIP...")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model     = clip_model.to(DEVICE).eval()
print(f"CLIP на {DEVICE}")


_embed_error_shown = False

@torch.no_grad()
def get_embedding(path: str) -> np.ndarray:
    global _embed_error_shown
    try:
        img    = Image.open(path).convert("RGB")
        inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
        vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        feat = clip_model.visual_projection(vision_out.pooler_output)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].cpu().numpy()
    except Exception as e:
        if not _embed_error_shown:
            print(f"\n[DEBUG] Ошибка загрузки '{path}': {e}")
            _embed_error_shown = True
        return np.zeros(512, dtype=np.float32)


# ════════════════════════════════════════════════════════════
# 2.1  КЛАСТЕРИЗАЦИЯ ИЗОБРАЖЕНИЙ
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2.1  КЛАСТЕРИЗАЦИЯ ИЗОБРАЖЕНИЙ")
print("=" * 60)

print("Извлекаем CLIP-эмбеддинги...")
embeddings = np.array([get_embedding(p) for p in tqdm(df["path"].tolist())])
print(f"Матрица эмбеддингов: {embeddings.shape}")

# Диагностика
n_zeros = (np.abs(embeddings).sum(axis=1) == 0).sum()
print(f"Нулевых эмбеддингов (ошибки загрузки): {n_zeros}/{len(embeddings)}")
if n_zeros > len(embeddings) * 0.5:
    print("ВНИМАНИЕ: больше половины изображений не загрузились — проверь пути!")

# Фильтруем нулевые векторы из кластеризации
valid_mask = np.abs(embeddings).sum(axis=1) > 0
embeddings_clean = embeddings[valid_mask]
df_clean = df[valid_mask].reset_index(drop=True)
print(f"Валидных эмбеддингов для кластеризации: {len(embeddings_clean)}")
embeddings = embeddings_clean
df = df_clean

# ── Метод локтя ─────────────────────────────────────────────
K_MAX     = min(15, len(df) - 1)
K_RANGE   = range(2, K_MAX + 1)
inertias  = []
sil_list  = []

print("Метод локтя...")
for k in tqdm(K_RANGE):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lb = km.fit_predict(embeddings)
    inertias.append(km.inertia_)
    n_unique = len(set(lb))
    sil = silhouette_score(embeddings, lb) if n_unique >= 2 else 0.0
    sil_list.append(sil)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(list(K_RANGE), inertias, "bo-", lw=2)
ax1.set_xlabel("k"); ax1.set_ylabel("Инерция")
ax1.set_title("Метод локтя"); ax1.grid(True)
ax2.plot(list(K_RANGE), sil_list, "ro-", lw=2)
ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette")
ax2.set_title("Silhouette Score"); ax2.grid(True)
plt.tight_layout()
plt.savefig(OUT / "elbow.png", dpi=150)
plt.savefig(REPORT_DIR / "elbow.png", dpi=150)
plt.show()

OPTIMAL_K = list(K_RANGE)[int(np.argmax(sil_list))]
print(f"Оптимальное k = {OPTIMAL_K}")

# ── 5+ решений, 3 алгоритма ─────────────────────────────────
SOLUTIONS = []

# Алгоритм 1: KMeans
for k in sorted({max(2, OPTIMAL_K - 1), OPTIMAL_K, min(OPTIMAL_K + 1, K_MAX)}):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lb = km.fit_predict(embeddings)
    n_unique = len(set(lb))
    sil = silhouette_score(embeddings, lb) if n_unique >= 2 else 0.0
    db  = davies_bouldin_score(embeddings, lb) if n_unique >= 2 else 0.0
    SOLUTIONS.append({"name": f"KMeans k={k}", "labels": lb, "sil": sil, "db": db})
    print(f"  KMeans k={k}: Sil={sil:.3f} DB={db:.3f} → {Counter(lb)}")

# Алгоритм 2: DBSCAN
for eps in [0.3, 0.5]:
    lb = DBSCAN(eps=eps, min_samples=2, metric="cosine").fit_predict(embeddings)
    n_cls = len(set(lb)) - (1 if -1 in lb else 0)
    n_unique = len(set(lb[lb != -1])) if -1 in lb else len(set(lb))
    if n_cls > 1 and n_unique >= 2:
        sil = silhouette_score(embeddings, lb)
        SOLUTIONS.append({"name": f"DBSCAN eps={eps}", "labels": lb, "sil": sil, "db": 0.0})
        print(f"  DBSCAN eps={eps}: {n_cls} кластеров, Sil={sil:.3f}, шум={sum(lb==-1)}")
    else:
        print(f"  DBSCAN eps={eps}: пропущен (кластеров < 2)")

# Алгоритм 3: Agglomerative
for linkage, k in [("ward", OPTIMAL_K), ("complete", max(2, OPTIMAL_K - 1))]:
    lb  = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(embeddings)
    n_unique = len(set(lb))
    sil = silhouette_score(embeddings, lb) if n_unique >= 2 else 0.0
    db  = davies_bouldin_score(embeddings, lb) if n_unique >= 2 else 0.0
    SOLUTIONS.append({"name": f"Agglomerative {linkage} k={k}", "labels": lb, "sil": sil, "db": db})
    print(f"  Aggl {linkage} k={k}: Sil={sil:.3f} DB={db:.3f}")

sol_df = pd.DataFrame([{k: v for k, v in s.items() if k != "labels"} for s in SOLUTIONS])
print("\nТаблица решений:")
print(sol_df.to_string(index=False))

BEST     = SOLUTIONS[int(sol_df["sil"].idxmax())]
BEST_LBL = BEST["labels"]
df["cluster_id"] = BEST_LBL
print(f"\nЛучшее: {BEST['name']}  Sil={BEST['sil']:.4f}")

# ── t-SNE 2D ────────────────────────────────────────────────
print("t-SNE 2D...")
emb2d = TSNE(n_components=2, random_state=42,
             perplexity=min(30, len(embeddings)-1)).fit_transform(embeddings)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sc = axes[0].scatter(emb2d[:,0], emb2d[:,1], c=BEST_LBL, cmap="tab10", s=15, alpha=0.8)
plt.colorbar(sc, ax=axes[0], label="Кластер")
axes[0].set_title(f"t-SNE — кластеры ({BEST['name']})")

label_to_int = {c: i for i, c in enumerate(classes)}
true_labels  = [label_to_int[c] for c in df["class_name"]]
sc2 = axes[1].scatter(emb2d[:,0], emb2d[:,1], c=true_labels, cmap="tab20", s=15, alpha=0.8)
plt.colorbar(sc2, ax=axes[1], label="Класс")
axes[1].set_title("t-SNE — истинные классы")

plt.suptitle("t-SNE 2D визуализация", fontsize=13)
plt.tight_layout()
plt.savefig(OUT / "tsne_2d.png", dpi=150)
plt.savefig(REPORT_DIR / "tsne_2d.png", dpi=150)
plt.show()

# ── UMAP 3D ─────────────────────────────────────────────────
print("UMAP 3D...")
emb3d = umap_lib.UMAP(
    n_components=3, random_state=42,
    n_neighbors=min(15, len(embeddings)-1),
).fit_transform(embeddings)

fig3d = px.scatter_3d(
    x=emb3d[:,0], y=emb3d[:,1], z=emb3d[:,2],
    color=df["class_name"].values,
    symbol=[str(c) for c in BEST_LBL],
    hover_name=df["filename"].values,
    title="UMAP 3D — классы семян",
    width=900, height=700,
)
fig3d.update_traces(marker_size=4)
fig3d.write_html(str(OUT / "umap_3d.html"))
shutil.copy2(OUT / "umap_3d.html", REPORT_DIR / "umap_3d.html")
fig3d.show()

# ── Описание кластеров ───────────────────────────────────────
report_lines = ["=== ОПИСАНИЕ КЛАСТЕРОВ ===\n"]
for cid in sorted(set(BEST_LBL)):
    if cid == -1:
        continue
    mask  = BEST_LBL == cid
    chunk = df[mask]
    top_class = chunk["class_name"].value_counts().index[0]
    desc = (
        f"\nКластер {cid}  ({mask.sum()} изображений)\n"
        f"  Доминирующий класс: {top_class}\n"
        f"  Все классы: {chunk['class_name'].value_counts().to_dict()}\n"
        f"  Ср. размер файла: {chunk['size_mb'].mean():.3f} МБ\n"
        f"  Цветных: {chunk['is_color'].mean():.1%}\n"
        f"  Естественных: {chunk['is_natural'].mean():.1%}\n"
    )
    print(desc)
    report_lines.append(desc)

with open(OUT / "clusters_description.txt", "w", encoding="utf-8") as f:
    f.writelines(report_lines)
shutil.copy2(OUT / "clusters_description.txt", REPORT_DIR / "clusters_description.txt")

# ════════════════════════════════════════════════════════════
# 2.2  КЛАССИФИКАТОР ВИДОВ СЕМЯН (EfficientNet-B3)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2.2  ОБУЧЕНИЕ КЛАССИФИКАТОРА (EfficientNet-B3)")
print("=" * 60)

# ── Аугментации ─────────────────────────────────────────────
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

AUG_PRESETS = {
    "light": A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=MEAN, std=STD), ToTensorV2(),
    ]),
    "medium": A.Compose([
        A.Resize(256, 256), A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.HueSaturationValue(p=0.4), A.RandomBrightnessContrast(p=0.4),
        A.Normalize(mean=MEAN, std=STD), ToTensorV2(),
    ]),
    "heavy": A.Compose([
        A.Resize(300, 300), A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, p=0.5),
        A.HueSaturationValue(p=0.4), A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=MEAN, std=STD), ToTensorV2(),
    ]),
}
VAL_TF = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD), ToTensorV2(),
])


class SeedDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.tf = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]).convert("RGB"))
        return self.tf(image=img)["image"], self.labels[idx]


# ── Собираем данные из seeds/ ────────────────────────────────
all_paths, all_labels = [], []
cls_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {str(v): k for k, v in cls_to_idx.items()}
with open(OUT / "class_names.json", "w", encoding="utf-8") as f:
    json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

for cls in classes:
    cls_dir = Path(SEEDS_DIR) / cls
    for p in cls_dir.iterdir():
        if p.suffix.lower() in IMG_EXTS:
            all_paths.append(str(p))
            all_labels.append(cls_to_idx[cls])

X_tr, X_va, y_tr, y_va = train_test_split(
    all_paths, all_labels, test_size=0.2,
    random_state=42, stratify=all_labels,
)
print(f"Train: {len(X_tr)}, Val: {len(X_va)}")

counts  = Counter(y_tr)
weights = [1.0 / counts[l] for l in y_tr]
sampler = WeightedRandomSampler(weights, num_samples=len(weights))

train_loader = DataLoader(
    SeedDataset(X_tr, y_tr, AUG_PRESETS[AUGMENTATION]),
    batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=False,
)
val_loader = DataLoader(
    SeedDataset(X_va, y_va, VAL_TF),
    batch_size=64, shuffle=False, num_workers=0, pin_memory=False,
)

# ── Модель ───────────────────────────────────────────────────
model     = timm.create_model("efficientnet_b3", pretrained=True,
                               num_classes=len(classes)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

from sklearn.metrics import f1_score

history    = {"tl": [], "vl": [], "ta": [], "va": [], "f1": []}
best_f1    = 0.0
best_preds = best_true = None

for epoch in range(EPOCHS):
    # train
    model.train()
    tl, tc, tn = 0.0, 0, 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward(); optimizer.step()
        tl += loss.item(); tc += (out.detach().argmax(1) == lbls).sum().item(); tn += lbls.size(0)

    # val
    model.eval()
    vl, preds_all, true_all = 0.0, [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out = model(imgs)
            vl += criterion(out, lbls).item()
            preds_all.extend(out.argmax(1).cpu().numpy())
            true_all.extend(lbls.cpu().numpy())

    scheduler.step()
    preds_np = np.array(preds_all); true_np = np.array(true_all)
    val_acc  = (preds_np == true_np).mean()
    val_f1   = f1_score(true_np, preds_np, average="macro", zero_division=0)

    history["tl"].append(tl / len(train_loader))
    history["vl"].append(vl / len(val_loader))
    history["ta"].append(tc / tn)
    history["va"].append(val_acc)
    history["f1"].append(val_f1)

    print(f"Epoch {epoch+1:02d}  "
          f"train_loss={history['tl'][-1]:.3f} train_acc={history['ta'][-1]:.3f}  "
          f"val_loss={history['vl'][-1]:.3f} val_acc={val_acc:.3f} F1={val_f1:.3f}")

    if val_f1 > best_f1:
        best_f1    = val_f1
        best_preds = preds_np.copy()
        best_true  = true_np.copy()
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  → checkpoint (F1={val_f1:.4f})")

# Сохраняем class_names.json
idx_to_class = {str(v): k for k, v in cls_to_idx.items()}
with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

print(f"\nЛучший val F1: {best_f1:.4f}")

# ── Learning curves ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history["tl"], label="train"); ax1.plot(history["vl"], label="val")
ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True)
ax2.plot(history["ta"], label="train"); ax2.plot(history["va"], label="val")
ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.savefig(OUT / "learning_curves.png", dpi=150)
plt.savefig(REPORT_DIR / "learning_curves.png", dpi=150)
plt.show()

# ── Метрики ─────────────────────────────────────────────────
report = classification_report(best_true, best_preds,
                                target_names=classes, zero_division=0)
print("\n" + report)
with open(OUT / "classification_report.txt", "w") as f:
    f.write(report)
shutil.copy2(OUT / "classification_report.txt",
             REPORT_DIR / "classification_report.txt")

# ── Confusion matrix ─────────────────────────────────────────
cm = confusion_matrix(best_true, best_preds)
fig, ax = plt.subplots(figsize=(max(8, len(classes)), max(6, len(classes) - 1)))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=classes, yticklabels=classes)
ax.set_ylabel("Истинный класс"); ax.set_xlabel("Предсказанный класс")
ax.set_title("Матрица ошибок", fontsize=13)
plt.xticks(rotation=35, ha="right"); plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=150)
plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
plt.show()

# ── ROC-AUC ─────────────────────────────────────────────────
y_bin   = label_binarize(best_true, classes=list(range(len(classes))))
model.eval()
all_probs = []
with torch.no_grad():
    for imgs, _ in val_loader:
        out = model(imgs.to(DEVICE))
        all_probs.extend(torch.softmax(out, dim=1).cpu().numpy())
y_probs = np.array(all_probs)[:len(best_true)]

fig, ax = plt.subplots(figsize=(10, 7))
for i, cls in enumerate(classes):
    if y_bin[:, i].sum() == 0:
        continue
    fpr_, tpr_, _ = roc_curve(y_bin[:, i], y_probs[:, i])
    ax.plot(fpr_, tpr_, lw=2, label=f"{cls} (AUC={auc(fpr_,tpr_):.2f})")

ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC-AUC кривые (one-vs-rest)", fontsize=13)
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "roc_auc.png", dpi=150)
plt.savefig(REPORT_DIR / "roc_auc.png", dpi=150)
plt.show()

# ── Добавляем predicted_class в CSV ─────────────────────────
print("Добавляем предсказания в Image_Library_Description...")
all_emb_preds = []
model.eval()
with torch.no_grad():
    for imgs, _ in DataLoader(
        SeedDataset(df["path"].tolist(),
                    [0]*len(df), VAL_TF),
        batch_size=64, num_workers=0,
    ):
        out = model(imgs.to(DEVICE))
        all_emb_preds.extend(out.argmax(1).cpu().numpy())

df["predicted_class"] = [classes[p] for p in all_emb_preds]
df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
shutil.copy2(CSV_PATH, REPORT_DIR / CSV_PATH.name)

# ════════════════════════════════════════════════════════════
# 2.3  КЛАСТЕРИЗАЦИЯ ПО ВИЗУАЛЬНЫМ ГРУППАМ
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2.3  КЛАСТЕРИЗАЦИЯ ПО ВИЗУАЛЬНЫМ ГРУППАМ СЕМЯН")
print("=" * 60)

# Используем те же CLIP-эмбеддинги, но теперь с фокусом на описание кластеров
GROUP_SOLUTIONS = []

for k in range(2, min(len(classes) + 2, K_MAX + 1)):
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lb  = km.fit_predict(embeddings)
    sil = silhouette_score(embeddings, lb)
    GROUP_SOLUTIONS.append({"name": f"KMeans k={k}", "labels": lb, "sil": sil})

for linkage in ["ward", "complete", "average"]:
    k   = len(classes)
    lb  = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(embeddings)
    sil = silhouette_score(embeddings, lb)
    GROUP_SOLUTIONS.append({"name": f"Aggl-{linkage} k={k}", "labels": lb, "sil": sil})

grp_df = pd.DataFrame([{k: v for k, v in s.items() if k != "labels"}
                        for s in GROUP_SOLUTIONS])
print(grp_df.to_string(index=False))

BEST_GRP = max(GROUP_SOLUTIONS, key=lambda s: s["sil"])
print(f"\nЛучшее решение для групп: {BEST_GRP['name']} (Sil={BEST_GRP['sil']:.4f})")

grp_lines = [f"=== КЛАСТЕРИЗАЦИЯ ВИЗУАЛЬНЫХ ГРУПП СЕМЯН ===\n",
             f"Лучшее решение: {BEST_GRP['name']}\n"]
for cid in sorted(set(BEST_GRP["labels"])):
    if cid == -1: continue
    mask  = BEST_GRP["labels"] == cid
    chunk = df[mask]
    top   = chunk["class_name"].value_counts().to_dict()
    line  = f"Группа {cid}: {mask.sum()} изображений → {top}\n"
    print(line, end="")
    grp_lines.append(line)

with open(OUT / "groups_description.txt", "w", encoding="utf-8") as f:
    f.writelines(grp_lines)
shutil.copy2(OUT / "groups_description.txt", REPORT_DIR / "groups_description.txt")

# ════════════════════════════════════════════════════════════
# 2.4  БИНАРНЫЙ КЛАССИФИКАТОР
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2.4  БИНАРНЫЙ КЛАССИФИКАТОР")
print("=" * 60)

# Берём два самых многочисленных или самых путающихся класса
cls_counts  = df["class_name"].value_counts()
bin_classes = cls_counts.index[:2].tolist()
print(f"Бинарный классификатор: {bin_classes[0]} vs {bin_classes[1]}")

bin_df = df[df["class_name"].isin(bin_classes)].copy()
bin_y  = (bin_df["class_name"] == bin_classes[1]).astype(int).values
bin_X  = embeddings[bin_df.index]

N_VALUES   = [1, 3, 5, 10, 20, 50]
BIN_RESULT = {}
RNG        = np.random.default_rng(42)

# Группируем по классу
class_embs = {c: bin_X[bin_y == i] for i, c in enumerate(bin_classes)}

for N in N_VALUES:
    X_all, y_all = [], []
    for i, cls in enumerate(bin_classes):
        embs = class_embs[cls]
        if len(embs) < N:
            continue
        reps = max(5, 30 // max(1, len(embs) // N))
        for _ in range(reps):
            idx = RNG.choice(len(embs), N, replace=False)
            X_all.append(embs[idx].mean(axis=0))
            y_all.append(i)

    if len(set(y_all)) < 2 or len(y_all) < 6:
        continue

    X_a, y_a = np.array(X_all), np.array(y_all)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_a, y_a, test_size=0.25, random_state=42, stratify=y_a,
    )
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_tr, y_tr)
    preds  = clf.predict(X_te)
    probs_ = clf.predict_proba(X_te)[:, 1]
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te, preds, average="binary", zero_division=0,
    )
    try:
        auc_val = roc_auc_score(y_te, probs_)
    except Exception:
        auc_val = 0.0
    BIN_RESULT[N] = {"p": prec, "r": rec, "f1": f1, "auc": auc_val,
                     "y_te": y_te, "probs": probs_}
    print(f"  N={N:3d}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} AUC={auc_val:.3f}")

if BIN_RESULT:
    fig, ax = plt.subplots(figsize=(10, 6))
    ns = list(BIN_RESULT.keys())
    for metric, col in [("p", "#42A5F5"), ("r", "#66BB6A"),
                        ("f1", "#FFA726"), ("auc", "#EF5350")]:
        vals = [BIN_RESULT[n][metric] for n in ns]
        ax.plot(ns, vals, "o-", color=col, lw=2,
                label={"p":"Precision","r":"Recall","f1":"F1","auc":"ROC-AUC"}[metric])
    ax.set_xlabel(f"N изображений на документ")
    ax.set_ylabel("Метрика")
    ax.set_title(f"Бинарный классификатор: {bin_classes[0]} vs {bin_classes[1]}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "binary_metrics.png", dpi=150)
    plt.savefig(REPORT_DIR / "binary_metrics.png", dpi=150)
    plt.show()

    OPT_N    = max(BIN_RESULT, key=lambda n: BIN_RESULT[n]["f1"])
    best_bin = BIN_RESULT[OPT_N]
    fpr_, tpr_, _ = roc_curve(best_bin["y_te"], best_bin["probs"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_, tpr_, "b-", lw=2, label=f"AUC={auc(fpr_,tpr_):.3f}")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC-AUC бинарный (N={OPT_N})")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "binary_roc.png", dpi=150)
    plt.savefig(REPORT_DIR / "binary_roc.png", dpi=150)
    plt.show()
    print(f"\nОптимальное N={OPT_N}: F1={best_bin['f1']:.4f} AUC={best_bin['auc']:.4f}")

# ════════════════════════════════════════════════════════════
# ИТОГИ
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ИТОГИ МОДУЛЯ B")
print("=" * 60)
print(f"2.1 Кластеризация:   {BEST['name']}  Sil={BEST['sil']:.4f}")
print(f"2.2 Классификатор:   EfficientNet-B3  F1={best_f1:.4f}")
print(f"2.3 Визуал. группы: {BEST_GRP['name']}  Sil={BEST_GRP['sil']:.4f}")
if BIN_RESULT:
    print(f"2.4 Бинарный:        {bin_classes[0]} vs {bin_classes[1]}"
          f"  N={OPT_N}  F1={BIN_RESULT[OPT_N]['f1']:.4f}")
print(f"\nПапка отчёта: {REPORT_DIR}")
print("Готово!")
