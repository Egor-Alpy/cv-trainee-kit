"""
Генерация ArUco-маркеров для автоматической разметки датасета.

Запуск: python tools/generate_markers.py
Результат: папка markers/ с PNG-файлами — распечатай и клади рядом с объектом при съёмке.

Настройка: измени CLASSES под свои классы (должны совпадать с config.py)
"""
import cv2
import numpy as np
from pathlib import Path

from config import CLASSES

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
output_dir = Path("markers")
output_dir.mkdir(exist_ok=True)

for marker_id, class_name in enumerate(CLASSES):
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, sidePixels=400)

    canvas = np.ones((480, 460), dtype=np.uint8) * 255
    canvas[50:450, 30:430] = marker_img
    cv2.putText(canvas, f"ID={marker_id}  {class_name.upper()}",
                (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(canvas, f"Класс: {class_name}",
                (30, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)

    out_path = output_dir / f"{marker_id:02d}_{class_name}.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"  {out_path.name}")

print(f"\nГотово. Распечатай файлы из '{output_dir}/'")
print("При съёмке клади маркер нужного класса рядом с объектом.")
