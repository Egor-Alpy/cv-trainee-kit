"""
Обучение модели.

Запуск: python train.py
Настройки: config.py

Результат: best_model.pth, class_names.json, learning_curves.png, confusion_matrix.png
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from collections import Counter
from tqdm import tqdm
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from PIL import Image

from config import MODEL_NAME, MODEL_PATH, NAMES_PATH, TRAIN_DIR, VAL_DIR
from config import EPOCHS, BATCH_SIZE, LR, AUGMENTATION
from dataset.augmentations import get_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlbumentationsDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=None)
        self.alb_transform = transform

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.alb_transform:
            img = self.alb_transform(image=img)["image"]
        return img, label


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
        return self.counter >= self.patience


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.detach().argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  val  ", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            total_loss += criterion(outputs, labels).item()
            preds_all.extend(outputs.argmax(1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    preds  = np.array(preds_all)
    labels = np.array(labels_all)
    return total_loss / len(loader), (preds == labels).mean(), f1_score(labels, preds, average="macro"), preds, labels


def main():
    print(f"Устройство: {DEVICE}")

    train_ds = AlbumentationsDataset(TRAIN_DIR, transform=get_transforms("train", AUGMENTATION))
    val_ds   = AlbumentationsDataset(VAL_DIR,   transform=get_transforms("val"))
    class_names = train_ds.classes
    print(f"Классы: {class_names}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}\n")

    counts  = Counter(train_ds.targets)
    weights = [1.0 / counts[l] for l in train_ds.targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=64,         shuffle=False,   num_workers=4, pin_memory=True)

    model     = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(class_names)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    early_stop = EarlyStopping(patience=7)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_f1, best_preds, best_labels = 0.0, None, None

    for epoch in range(EPOCHS):
        print(f"Эпоха {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, val_f1, preds, labels = val_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  train loss={train_loss:.3f} acc={train_acc:.3f} | val loss={val_loss:.3f} acc={val_acc:.3f} F1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_preds, best_labels = preds, labels
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  → checkpoint (F1={val_f1:.4f})")

        if early_stop(val_f1):
            print("  Early stopping.")
            break

    idx_to_class = {str(v): k for k, v in train_ds.class_to_idx.items()}
    with open(NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

    print(f"\nЛучший val F1: {best_f1:.4f}")
    print(classification_report(best_labels, best_preds, target_names=class_names))

    # Learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in [(ax1, "loss", "Loss"), (ax2, "acc", "Accuracy")]:
        ax.plot(history[f"train_{key}"], label="train")
        ax.plot(history[f"val_{key}"],   label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150)

    # Confusion matrix
    cm = confusion_matrix(best_labels, best_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    ax.set_ylabel("Истинный класс")
    ax.set_xlabel("Предсказанный класс")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)

    print("Сохранено: best_model.pth, class_names.json, learning_curves.png, confusion_matrix.png")


if __name__ == "__main__":
    main()
