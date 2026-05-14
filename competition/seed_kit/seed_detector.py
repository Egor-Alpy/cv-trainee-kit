"""
ДЕТЕКТОР СЕМЯН — авторазметка → обучение YOLOv8 → реальное время
=================================================================
python seed_detector.py            # полный пайплайн
python seed_detector.py --label    # только авторазметка (CLIP + OpenCV)
python seed_detector.py --train    # только обучение YOLOv8n
python seed_detector.py --webcam   # только вебкамера (модель уже обучена)
"""

# ============================================================
# CONFIG — меняй только здесь
# ============================================================
UNLABELED_DIR  = "seeds_raw"                        # папка с ~200 фото без разметки
CLASSES        = ["barley", "buckwheat", "rice"]    # добавь "mixed" если есть такие фото
OUTPUT_DIR     = "output_detection"
PARTICIPANT_ID = "1"
EPOCHS         = 100
IMGSZ          = 640
BATCH          = 16
MIN_SEED_AREA  = 300    # мин. площадь контура (пикс²) чтобы считаться семенем
MAX_SEED_RATIO = 0.15   # макс. доля площади кадра для одного контура
GPU_DEVICE     = "0"    # "0" = первый GPU, "cpu" = процессор
# ============================================================

import argparse, shutil, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--label",  action="store_true", help="только авторазметка")
parser.add_argument("--train",  action="store_true", help="только обучение")
parser.add_argument("--webcam", action="store_true", help="только вебкамера")
args = parser.parse_args()

RUN_ALL    = not (args.label or args.train or args.webcam)
RUN_LABEL  = args.label  or RUN_ALL
RUN_TRAIN  = args.train  or RUN_ALL
RUN_WEBCAM = args.webcam  # вебкамера только по явному флагу

OUT       = Path(OUTPUT_DIR)
DATASET   = OUT / "yolo_dataset"
PREVIEW   = OUT / "label_preview"
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
OUT.mkdir(exist_ok=True)

COLORS_BGR = {
    "barley":    (0,   165, 255),
    "buckwheat": (255, 100, 0  ),
    "rice":      (0,   220, 80 ),
    "mixed":     (180, 0,   200),
}

# ============================================================
# АВТОРАЗМЕТКА: CLIP классификация + OpenCV детекция
# ============================================================
if RUN_LABEL:
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split
    from transformers import CLIPModel, CLIPProcessor

    print("\n" + "=" * 60)
    print("АВТОРАЗМЕТКА")
    print("=" * 60)

    CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Загружаем CLIP на {CLIP_DEVICE}...")
    clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model     = clip_model.to(CLIP_DEVICE).eval()

    CLIP_TEXTS = [f"a photo of {cls} grains on a surface" for cls in CLASSES]

    raw_dir = Path(UNLABELED_DIR)
    if not raw_dir.exists():
        print(f"Папка не найдена: {UNLABELED_DIR}")
        raise SystemExit(1)

    images = sorted([p for p in raw_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    print(f"Найдено фото: {len(images)}")

    @torch.no_grad()
    def clip_classify(path: Path) -> str:
        try:
            img    = Image.open(path).convert("RGB")
            inputs = clip_processor(
                text=CLIP_TEXTS, images=img,
                return_tensors="pt", padding=True, truncation=True,
            ).to(CLIP_DEVICE)
            probs = clip_model(**inputs).logits_per_image.softmax(dim=1)[0]
            return CLASSES[probs.argmax().item()]
        except Exception as e:
            print(f"  [CLIP ошибка] {path.name}: {e}")
            return CLASSES[0]

    print("Классифицируем фото через CLIP...")
    img_classes: dict[Path, str] = {}
    for p in tqdm(images):
        img_classes[p] = clip_classify(p)

    dist = Counter(img_classes.values())
    print("\nРаспределение по классам (CLIP):")
    for cls, cnt in dist.items():
        print(f"  {cls:<15} {cnt} фото")

    # ── OpenCV: детекция контуров ────────────────────────────
    print("\nОпределяем контуры семян через OpenCV...")

    def detect_seeds(img_bgr: np.ndarray) -> list[tuple]:
        h, w   = img_bgr.shape[:2]
        img_area = h * w
        gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur   = cv2.GaussianBlur(gray, (7, 7), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        best_boxes = []

        for inv in [True, False]:
            flag = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if inv \
                   else cv2.THRESH_BINARY + cv2.THRESH_OTSU
            _, thresh = cv2.threshold(blur, 0, 255, flag)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < MIN_SEED_AREA or area > img_area * MAX_SEED_RATIO:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                ar = bw / (bh + 1e-6)
                if ar < 0.15 or ar > 6.5:
                    continue
                boxes.append((x, y, x + bw, y + bh))

            if len(boxes) > len(best_boxes):
                best_boxes = boxes

        return best_boxes, h, w

    labeled: list[tuple] = []
    skipped = 0

    for p, cls_name in tqdm(img_classes.items()):
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            skipped += 1
            continue
        boxes, h, w = detect_seeds(img_bgr)
        if not boxes:
            skipped += 1
            print(f"  [пропуск] {p.name} — семена не найдены (настрой MIN_SEED_AREA)")
            continue
        labeled.append((p, cls_name, boxes, h, w))

    print(f"\nФото с метками: {len(labeled)}, пропущено: {skipped}")

    if not labeled:
        print("Ни одно фото не размечено. Уменьши MIN_SEED_AREA и попробуй снова.")
        raise SystemExit(1)

    # ── Train / Val split ────────────────────────────────────
    cls_to_id = {c: i for i, c in enumerate(CLASSES)}
    stratify  = [x[1] for x in labeled]

    try:
        train_idx, val_idx = train_test_split(
            range(len(labeled)), test_size=0.2,
            random_state=42, stratify=stratify,
        )
    except ValueError:
        print("  [предупреждение] стратификация невозможна — слишком мало примеров одного класса, делим без стратификации")
        train_idx, val_idx = train_test_split(
            range(len(labeled)), test_size=0.2, random_state=42,
        )

    for split in ["train", "val"]:
        (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(exist_ok=True)

    def save_split(indices, split):
        for i in indices:
            p, cls_name, boxes, h, w = labeled[i]
            shutil.copy2(p, DATASET / "images" / split / p.name)
            cls_id = cls_to_id.get(cls_name, 0)
            lines  = []
            for (x1, y1, x2, y2) in boxes:
                xc = (x1 + x2) / 2 / w
                yc = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            (DATASET / "labels" / split / (p.stem + ".txt")).write_text("\n".join(lines))

    save_split(train_idx, "train")
    save_split(val_idx,   "val")

    # data.yaml
    yaml_path = DATASET / "data.yaml"
    names_yaml = "\n".join(f"  - {c}" for c in CLASSES)
    yaml_path.write_text(
        f"path: {DATASET.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names:\n{names_yaml}\n",
        encoding="utf-8",
    )

    # Превью разметки (первые 8 фото)
    for p, cls_name, boxes, h, w in labeled[:8]:
        img_bgr = cv2.imread(str(p))
        color   = COLORS_BGR.get(cls_name, (200, 200, 200))
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_bgr, f"{cls_name}  ({len(boxes)} seeds)",
                    (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imwrite(str(PREVIEW / p.name), img_bgr)

    print(f"\nДатасет:  {DATASET}")
    print(f"Превью:   {PREVIEW}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    print("Открой превью и проверь — правильно ли CLIP определил классы!")


# ============================================================
# ОБУЧЕНИЕ YOLOv8n
# ============================================================
if RUN_TRAIN:
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ YOLOv8n")
    print("=" * 60)

    yaml_path = DATASET / "data.yaml"
    if not yaml_path.exists():
        print(f"data.yaml не найден: {yaml_path}")
        print("Запусти сначала: python seed_detector.py --label")
        raise SystemExit(1)

    model = YOLO("yolov8n.pt")
    model.train(
        data     = str(yaml_path),
        epochs   = EPOCHS,
        imgsz    = IMGSZ,
        batch    = BATCH,
        device   = GPU_DEVICE,
        patience = 20,
        save     = True,
        project  = OUTPUT_DIR,
        name     = "train",
        exist_ok = True,
    )

    best_pt  = Path(OUTPUT_DIR) / "train" / "weights" / "best.pt"
    final_pt = OUT / "best_model.pt"
    if not best_pt.exists():
        print(f"best.pt не найден в {best_pt} — обучение не сохранило модель")
        raise SystemExit(1)
    shutil.copy2(best_pt, final_pt)
    print(f"\nМодель: {final_pt}")

    print("\nОценка на валидации...")
    val_model = YOLO(str(final_pt))
    metrics   = val_model.val(data=str(yaml_path), device=GPU_DEVICE, verbose=False)
    print(f"  mAP50:    {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")
    for i, cls in enumerate(CLASSES):
        if i < len(metrics.box.p):
            print(f"  {cls:<15} P={metrics.box.p[i]:.3f}  R={metrics.box.r[i]:.3f}")

    print(f"\nConfusion matrix и графики: {OUTPUT_DIR}/train/")


# ============================================================
# ВЕБКАМЕРА — реальное время с подсчётом
# ============================================================
if RUN_WEBCAM:
    import cv2
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("ВЕБКАМЕРА — реальное время")
    print("=" * 60)
    print("Q — выход  |  S — скриншот")

    model_pt = OUT / "best_model.pt"
    if not model_pt.exists():
        print(f"Модель не найдена: {model_pt}")
        print("Запусти сначала: python seed_detector.py --train")
        raise SystemExit(1)

    model = YOLO(str(model_pt))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Вебкамера не найдена (VideoCapture(0))")
        raise SystemExit(1)

    frame_n = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.45, verbose=False, device=GPU_DEVICE)

        counts: Counter = Counter()
        for box in results[0].boxes:
            cls_id   = int(box.cls[0])
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else "?"
            conf     = float(box.conf[0])
            counts[cls_name] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLORS_BGR.get(cls_name, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Панель счётчика
        panel_h = 40 + (len(CLASSES) + 1) * 34
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (240, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "Seed Counter", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        y_off = 58
        for cls_name in CLASSES:
            cnt   = counts.get(cls_name, 0)
            color = COLORS_BGR.get(cls_name, (200, 200, 200))
            cv2.putText(frame, f"{cls_name}: {cnt}", (8, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            y_off += 34

        total = sum(counts.values())
        cv2.putText(frame, f"TOTAL: {total}", (8, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Seed Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"screenshot_{frame_n:04d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"Скриншот: {fname}")

        frame_n += 1

    cap.release()
    cv2.destroyAllWindows()
