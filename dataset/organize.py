"""
Разбивка датасета на train/val.

Запуск: python dataset/organize.py <папка_с_фото> <выходная_папка>

Вход:  photos_raw/oak/IMG_001.jpg
Выход: dataset/train/oak/, dataset/val/oak/, ...

Разбивка 80/20 со стратификацией по классам.
"""
import sys
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def organize(src: str, dst: str, val_size: float = 0.2, seed: int = 42) -> None:
    src_path = Path(src)
    dst_path = Path(dst)

    classes = sorted([d.name for d in src_path.iterdir() if d.is_dir()])
    if not classes:
        print(f"Папки классов не найдены в '{src}'")
        print("Ожидаемая структура: photos_raw/class_name/image.jpg")
        sys.exit(1)

    print(f"Найдено классов: {len(classes)} → {classes}")

    for split in ("train", "val"):
        for cls in classes:
            (dst_path / split / cls).mkdir(parents=True, exist_ok=True)

    total_train, total_val = 0, 0

    for cls in classes:
        images = [f for f in (src_path / cls).iterdir() if f.suffix in EXTENSIONS]

        if len(images) < 2:
            print(f"  {cls}: слишком мало фото ({len(images)}), пропускаем")
            continue

        train_imgs, val_imgs = train_test_split(
            images, test_size=val_size, random_state=seed
        )

        for img in train_imgs:
            shutil.copy(img, dst_path / "train" / cls / img.name)
        for img in val_imgs:
            shutil.copy(img, dst_path / "val" / cls / img.name)

        print(f"  {cls}: train={len(train_imgs)}, val={len(val_imgs)}")
        total_train += len(train_imgs)
        total_val += len(val_imgs)

    print(f"\nГотово: train={total_train}, val={total_val}")
    print(f"Датасет → '{dst_path}/'")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "photos_raw"
    dst = sys.argv[2] if len(sys.argv) > 2 else "dataset"
    organize(src, dst)
