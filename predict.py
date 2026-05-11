"""
Предсказание для одного изображения.

Запуск: python predict.py путь_к_фото.jpg

Требует: best_model.pth, class_names.json (создаются после train.py)
"""
import sys
import json
import torch
import timm
import numpy as np
from PIL import Image

from config import MODEL_NAME, MODEL_PATH, NAMES_PATH
from dataset.augmentations import get_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_transforms("val")


def load_model():
    with open(NAMES_PATH, encoding="utf-8") as f:
        idx_to_class = json.load(f)
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(idx_to_class))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model, idx_to_class


def predict(image_path: str, model, idx_to_class: dict) -> tuple[str, float, list]:
    img = np.array(Image.open(image_path).convert("RGB"))
    tensor = transform(image=img)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top_k = min(3, len(idx_to_class))
    top_probs, top_idxs = probs.topk(top_k)
    top3 = [(idx_to_class[str(i.item())], p.item()) for p, i in zip(top_probs, top_idxs)]
    return top3[0][0], top3[0][1], top3


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python predict.py путь_к_фото.jpg")
        sys.exit(1)

    model, idx_to_class = load_model()
    label, conf, top3 = predict(sys.argv[1], model, idx_to_class)

    print(f"\nФайл: {sys.argv[1]}")
    print(f"Результат: {label} ({conf:.1%})\n")
    print("Топ-3:")
    for name, p in top3:
        bar = "█" * int(p * 20)
        print(f"  {name:<20} {p:.1%}  {bar}")
