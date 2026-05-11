"""
Gradio UI — демо для презентации.

Запуск: python app.py
Открывается в браузере: http://localhost:7860

Требует: best_model.pth, class_names.json (создаются после train.py)
"""
import json
import numpy as np
import torch
import timm
import gradio as gr
from PIL import Image

from config import MODEL_NAME, MODEL_PATH, NAMES_PATH, TASK_NAME
from dataset.augmentations import get_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_transforms("val")

for p in (MODEL_PATH, NAMES_PATH):
    if not __import__("pathlib").Path(p).exists():
        print(f"Файл не найден: {p}")
        print("Сначала запусти train.py — он создаст best_model.pth и class_names.json")
        exit(1)

with open(NAMES_PATH, encoding="utf-8") as f:
    idx_to_class = json.load(f)

model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(idx_to_class))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)


def classify(image: np.ndarray) -> dict:
    tensor = transform(image=image)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    return {idx_to_class[str(i)]: float(probs[i]) for i in range(len(idx_to_class))}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="numpy", label="Загрузи фото"),
    outputs=gr.Label(num_top_classes=len(idx_to_class), label="Результат"),
    title=TASK_NAME,
    description="Загрузи изображение — модель определит класс объекта.",
    theme=gr.themes.Soft(),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
