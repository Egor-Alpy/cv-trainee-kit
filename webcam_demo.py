"""
Демо с веб-камеры в реальном времени.

Запуск: python webcam_demo.py
  Q — выход
  S — сохранить скриншот

Требует: best_model.pth, class_names.json (создаются после train.py)
"""
import json
import cv2
import torch
import timm
import numpy as np
from pathlib import Path

from config import MODEL_NAME, MODEL_PATH, NAMES_PATH, TASK_NAME
from dataset.augmentations import get_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREDICT_EVERY = 8

for p in (MODEL_PATH, NAMES_PATH):
    if not Path(p).exists():
        print(f"Файл не найден: {p} — сначала запусти train.py")
        exit(1)

with open(NAMES_PATH, encoding="utf-8") as f:
    idx_to_class = json.load(f)

model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(idx_to_class))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

transform = get_transforms("val")


def predict_frame(frame_rgb: np.ndarray) -> tuple[str, float, list]:
    tensor = transform(image=frame_rgb)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top_k = min(3, len(idx_to_class))
    top_probs, top_idxs = probs.topk(top_k)
    top3 = [(idx_to_class[str(i.item())], p.item()) for p, i in zip(top_probs, top_idxs)]
    return top3[0][0], top3[0][1], top3


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Веб-камера не найдена")
    exit(1)

print(f"{TASK_NAME} — камера запущена")
print("Q — выход  |  S — скриншот")

pred, conf, top3 = "...", 0.0, []
frame_n = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_n % PREDICT_EVERY == 0:
        pred, conf, top3 = predict_frame(frame_rgb)

    frame_n += 1

    # Полупрозрачный фон
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (370, 105 + 22 * len(top3)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Цвет по уверенности
    color = (50, 220, 50) if conf >= 0.7 else (50, 165, 230) if conf >= 0.4 else (50, 50, 220)

    cv2.putText(frame, pred,        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.putText(frame, f"{conf:.0%}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    for i, (name, p) in enumerate(top3[1:], 1):
        cv2.putText(frame, f"  {i+1}. {name} {p:.0%}",
                    (10, 92 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.imshow(TASK_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        fname = f"screenshot_{frame_n}.jpg"
        cv2.imwrite(fname, frame)
        print(f"Скриншот: {fname}")

cap.release()
cv2.destroyAllWindows()
