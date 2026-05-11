"""
Авто-разметка фотографий по ArUco-маркерам.

Запуск: python tools/auto_label.py <папка_с_фото> <выходная_папка>

Вход:  photos/IMG_001.jpg  (объект сфотографирован рядом с маркером)
Выход: labeled/oak/IMG_001.jpg, labeled/maple/..., ...

Фото без распознанного маркера попадут в отчёт — разметь их вручную.
"""
import sys
import shutil
import cv2
from pathlib import Path

from config import CLASSES

EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

id_to_class = {i: name for i, name in enumerate(CLASSES)}
aruco_dict  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector    = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())


def detect_class(image_path: str) -> str | None:
    img = cv2.imread(image_path)
    if img is None:
        return None
    _, ids, _ = detector.detectMarkers(img)
    if ids is None:
        return None
    return id_to_class.get(int(ids[0][0]))


def run(photos_dir: str, output_dir: str) -> None:
    photos = [f for f in Path(photos_dir).rglob("*") if f.suffix in EXTENSIONS]

    if not photos:
        print(f"Фото не найдены в '{photos_dir}'")
        return

    labeled, unlabeled = 0, []

    for photo in sorted(photos):
        class_name = detect_class(str(photo))
        if class_name:
            dest = Path(output_dir) / class_name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(photo, dest / photo.name)
            print(f"  OK  {photo.name} → {class_name}/")
            labeled += 1
        else:
            unlabeled.append(photo.name)
            print(f"  --  {photo.name} → маркер не распознан")

    print(f"\nРезультат: {labeled} размечено, {len(unlabeled)} без маркера")
    if unlabeled:
        print("\nФото без маркера (разметить вручную):")
        for name in unlabeled:
            print(f"  {name}")
        print("\nПричины: маркер вне кадра / засвечен / слишком мелкий / плохой угол")


if __name__ == "__main__":
    photos_dir = sys.argv[1] if len(sys.argv) > 1 else "photos"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "labeled"
    run(photos_dir, output_dir)
