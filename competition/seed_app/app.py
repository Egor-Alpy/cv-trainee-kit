"""
Seed Detector — веб-приложение
python app.py
"""

MODEL_PATH   = "../output_detection/best_model.pt"
CLASSES      = ["barley", "buckwheat", "rice"]
CONF_DEFAULT = 0.45

import cv2, time, threading
import numpy as np
import gradio as gr
from datetime import datetime
from collections import Counter

COLORS_BGR = {
    "barley":    (0,   165, 255),
    "buckwheat": (0,    80, 255),
    "rice":      (34,  197,  94),
}

_predict_lock = threading.Lock()
_last_ms = [0.0]

try:
    import torch
    _DEVICE_STR = "CUDA" if torch.cuda.is_available() else "CPU"
except Exception:
    _DEVICE_STR = "CPU"

try:
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    MODEL_LOADED = True
    print("Модель загружена")
except Exception as e:
    print(f"Модель не загружена: {e}")
    model = None
    MODEL_LOADED = False

# ── Детекция ─────────────────────────────────────────────────
def run_detection(img_rgb: np.ndarray, conf: float, lock_timeout: float = 15):
    if model is None or img_rgb is None:
        return img_rgb, Counter()
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb] * 3, axis=2)
    elif img_rgb.shape[2] == 4:
        img_rgb = img_rgb[:, :, :3]
    # lock_timeout=0 для стрима: если модель занята — пропускаем кадр
    if not _predict_lock.acquire(timeout=lock_timeout):
        return img_rgb, Counter()
    try:
        t0 = time.perf_counter()
        results = model.predict(img_rgb, conf=conf, verbose=False)
        _last_ms[0] = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as exc:
        print(f"predict error: {exc}")
        return img_rgb, Counter()
    finally:
        _predict_lock.release()
    out = img_rgb.copy()
    counts = Counter()
    for box in results[0].boxes:
        cls_id   = int(box.cls[0])
        cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else "?"
        conf_val = float(box.conf[0])
        counts[cls_name] += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = COLORS_BGR.get(cls_name, (128, 128, 128))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        lbl = f"{cls_name} {conf_val:.0%}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, lbl, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return out, counts

def stats_html(counts: Counter, conf: float = CONF_DEFAULT) -> str:
    total  = sum(counts.values())
    ts     = datetime.now().strftime("%H:%M:%S")
    ms_str = f"{_last_ms[0]:.0f} ms" if _last_ms[0] else "—"
    rows   = ""
    for cls in CLASSES:
        cnt = counts.get(cls, 0)
        pct = round(cnt / total * 100) if total else 0
        rows += f"""
        <div class="s-row">
          <span class="s-dot" style="background:var(--{cls})"></span>
          <div class="s-body">
            <div class="s-name">{cls} <span class="s-pct">{pct}%</span></div>
            <div class="s-bar"><i style="width:{pct}%;background:var(--{cls})"></i></div>
          </div>
          <span class="s-cnt">{cnt}</span>
        </div>"""
    return f"""
    <div class="s-panel">
      <div class="s-head">
        <span class="s-lbl">Результат</span>
        <span class="s-meta">{ts}</span>
      </div>
      <div class="s-big-wrap">
        <span class="s-big">{total if total else "—"}</span>
        <span class="s-sub">&nbsp;семян</span>
      </div>
      <div class="s-breakdown">{rows}</div>
    </div>
    <div class="s-panel" style="margin-top:10px">
      <div class="s-head">
        <span class="s-lbl">Параметры</span>
        <span class="s-meta">{_DEVICE_STR}</span>
      </div>
      <div class="s-inf-list">
        <div class="s-inf-row"><span>inference</span><span class="s-val">{ms_str}</span></div>
        <div class="s-inf-row"><span>conf</span><span class="s-val">{conf:.2f}</span></div>
        <div class="s-inf-row"><span>классов</span><span class="s-val">{len(CLASSES)}</span></div>
      </div>
    </div>"""

# ── Обработчики ──────────────────────────────────────────────
def detect_stream(img_np, conf):
    """Вызывается каждые 0.5с из стрима — не ждёт если модель занята."""
    if img_np is None:
        return None, stats_html(Counter(), conf)
    out, counts = run_detection(img_np, conf, lock_timeout=0)
    return out, stats_html(counts, conf)

def detect_upload(pil_img, conf):
    """Вызывается по кнопке — ждёт завершения inference."""
    if pil_img is None:
        return None, stats_html(Counter(), conf)
    out, counts = run_detection(np.array(pil_img.convert("RGB")), conf, lock_timeout=15)
    return out, stats_html(counts, conf)

# ── CSS ──────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

:root {
  --bg:        #F7F5F0;
  --surface:   #FFFFFF;
  --ink:       #0A0A0A;
  --ink2:      #3A3A38;
  --muted:     #8A8983;
  --line:      #E6E2D8;
  --line2:     #EFEBE0;
  --accent:    #B4541A;
  --barley:    #F97316;
  --buckwheat: #3B82F6;
  --rice:      #22C55E;
}

footer { display: none !important; }

.gradio-container {
  background: var(--bg) !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  color: var(--ink);
  max-width: 1280px !important;
  margin: 0 auto !important;
  padding: 24px 32px 64px !important;
}

/* labels */
.gradio-container label {
  color: var(--ink2) !important;
  font-weight: 500 !important;
  font-size: 12px !important;
}

/* tabs */
[role="tablist"] {
  border-bottom: 1px solid var(--line) !important;
  margin-bottom: 20px !important;
  background: transparent !important;
}
[role="tab"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  color: var(--muted) !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  font-size: 13.5px !important;
  font-weight: 500 !important;
  padding: 11px 18px !important;
  cursor: pointer !important;
  transition: color .15s, border-color .15s !important;
}
[role="tab"]:hover { color: var(--ink2) !important; }
[role="tab"][aria-selected="true"] {
  color: var(--ink) !important;
  font-weight: 600 !important;
  border-bottom-color: var(--ink) !important;
}

/* all non-tab buttons */
.gradio-container button:not([role="tab"]) {
  background: var(--surface) !important;
  color: var(--ink) !important;
  border: 1px solid var(--line) !important;
  border-radius: 8px !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 9px 18px !important;
  cursor: pointer !important;
  transition: opacity .15s !important;
  line-height: 1.4 !important;
}
.gradio-container button:not([role="tab"]):hover { opacity: .75 !important; }

/* primary button */
.btn-p button, button.btn-p {
  background: var(--ink) !important;
  color: #FFFFFF !important;
  border-color: var(--ink) !important;
}

/* slider */
.gradio-container input[type="range"] { accent-color: var(--ink); }

/* image border */
.gradio-container .image-container {
  border-color: var(--line) !important;
  border-radius: 10px !important;
}

/* mono */
.mono { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }

/* ── Topbar ── */
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding-bottom: 20px; border-bottom: 1px solid var(--line); margin-bottom: 22px;
}
.brand { display: flex; align-items: center; gap: 12px; }
.mark  { width: 34px; height: 34px; border-radius: 8px; background: var(--ink); position: relative; flex-shrink: 0; }
.mark-b { position: absolute; width: 8px; height: 12px; border-radius: 50%; background: var(--barley); top: 6px; left: 5px; transform: rotate(-25deg); }
.mark-r { position: absolute; width: 7px; height: 10px; border-radius: 50%; background: var(--rice); top: 14px; left: 20px; transform: rotate(20deg); }
.brand-name { font-size: 15px; font-weight: 600; color: var(--ink); letter-spacing: -.01em; }
.brand-sub  { font-size: 11px; color: var(--muted); margin-top: 2px; }
.pill { display: inline-flex; align-items: center; gap: 7px; padding: 5px 12px; border: 1px solid var(--line); border-radius: 999px; font-size: 12px; color: var(--ink2); background: var(--surface); }
.dot     { width: 7px; height: 7px; border-radius: 50%; background: var(--rice); box-shadow: 0 0 0 3px rgba(34,197,94,.18); }
.dot.red { background: #EF4444; box-shadow: 0 0 0 3px rgba(239,68,68,.18); }

/* ── Title ── */
.t-row  { display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; margin-bottom: 22px; }
.t-head { margin: 0; font-size: 34px; font-weight: 600; letter-spacing: -.025em; line-height: 1.1; color: var(--ink); }
.t-head em { font-style: normal; color: var(--accent); }
.t-desc { max-width: 380px; font-size: 13.5px; color: var(--ink2); line-height: 1.55; }

/* ── Warn bar ── */
.warn { background: #FEF9C3; border: 1px solid #FDE047; border-radius: 8px; padding: 10px 16px; color: #854D0E; font-size: 13px; margin-bottom: 18px; }
.warn code { color: var(--accent); font-family: 'JetBrains Mono', monospace; font-size: 12px; background: rgba(180,84,26,.1); padding: 1px 6px; border-radius: 3px; }

/* ── Panel header ── */
.p-head { display: flex; align-items: center; justify-content: space-between; padding: 10px 16px; background: var(--surface); border: 1px solid var(--line); border-bottom: 1px solid var(--line2); border-radius: 10px 10px 0 0; }
.p-lbl  { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); }
.p-meta { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--muted); }

/* ── Stats sidebar ── */
.s-panel { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
.s-head  { display: flex; justify-content: space-between; padding: 10px 16px; border-bottom: 1px solid var(--line2); }
.s-lbl   { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); }
.s-meta  { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--muted); }
.s-big-wrap { padding: 16px 18px 8px; display: flex; align-items: baseline; }
.s-big  { font-size: 44px; font-weight: 500; letter-spacing: -.03em; line-height: 1; color: var(--ink); font-family: 'JetBrains Mono', monospace; }
.s-sub  { font-size: 12px; color: var(--muted); }
.s-breakdown { padding: 2px 18px 16px; }
.s-row  { display: grid; grid-template-columns: 12px 1fr auto; gap: 10px; align-items: center; padding: 8px 0; border-top: 1px solid var(--line2); }
.s-dot  { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.s-body { min-width: 0; }
.s-name { font-size: 12.5px; color: var(--ink2); font-weight: 500; margin-bottom: 4px; }
.s-pct  { font-size: 11px; color: var(--muted); margin-left: 3px; }
.s-bar  { height: 3px; background: var(--line2); border-radius: 2px; overflow: hidden; }
.s-bar i { display: block; height: 100%; border-radius: 2px; transition: width .35s; }
.s-cnt  { font-size: 13px; font-weight: 600; color: var(--ink); font-family: 'JetBrains Mono', monospace; min-width: 22px; text-align: right; }
.s-inf-list { padding: 2px 18px 12px; }
.s-inf-row  { display: flex; justify-content: space-between; padding: 7px 0; border-top: 1px solid var(--line2); font-size: 12.5px; color: var(--ink2); }
.s-val      { font-weight: 600; color: var(--ink); font-family: 'JetBrains Mono', monospace; }
"""

# ── UI ───────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="Seed Detector") as demo:

    model_pill = (
        '<span class="pill"><span class="dot"></span>модель загружена <span class="mono">YOLOv8n</span></span>'
        if MODEL_LOADED else
        '<span class="pill"><span class="dot red"></span><span class="mono">модель не найдена</span></span>'
    )
    gr.HTML(f"""
    <div class="topbar">
      <div class="brand">
        <div class="mark"><div class="mark-b"></div><div class="mark-r"></div></div>
        <div>
          <div class="brand-name">Seed Detector</div>
          <div class="brand-sub">Computer Vision · соревнование 2026</div>
        </div>
      </div>
      <div style="display:flex;gap:8px;align-items:center">
        {model_pill}
        <span class="pill mono">v1.0</span>
      </div>
    </div>""")

    if not MODEL_LOADED:
        gr.HTML("""<div class="warn">
          Модель не найдена. Запусти <code>python seed_detector.py</code>, затем перезапусти приложение.
        </div>""")

    gr.HTML("""
    <div class="t-row">
      <h2 class="t-head">Распознавание<br>и подсчёт <em>зёрен.</em></h2>
      <div class="t-desc">
        Наведи камеру на разложенные семена или загрузи фото.
        Модель находит ячмень, гречку и рис и считает по классам.
      </div>
    </div>""")

    with gr.Tabs():

        # ── Камера ───────────────────────────────────────────
        with gr.Tab("Камера"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    gr.HTML('<div class="p-head"><span class="p-lbl">Реальное время</span><span class="p-meta">device 0</span></div>')
                    cam_in  = gr.Image(sources=["webcam"], type="numpy",
                                       streaming=True,
                                       label="", show_label=False, height=400)
                    cam_out = gr.Image(type="numpy", label="", show_label=False, height=400)
                    with gr.Row():
                        cam_conf = gr.Slider(0.1, 0.9, value=CONF_DEFAULT, step=0.05,
                                             label="Порог уверенности", scale=1)
                with gr.Column(scale=1, min_width=280):
                    cam_stats = gr.HTML(stats_html(Counter()))
            cam_in.stream(detect_stream, [cam_in, cam_conf], [cam_out, cam_stats],
                          stream_every=0.5, concurrency_limit=1)

        # ── Загрузить фото ───────────────────────────────────
        with gr.Tab("Загрузить фото"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    gr.HTML('<div class="p-head"><span class="p-lbl">Фото</span><span class="p-meta">drag &amp; drop или выбор файла</span></div>')
                    up_in   = gr.Image(sources=["upload"], type="pil",
                                       label="", show_label=False, height=400)
                    up_out  = gr.Image(type="numpy", label="", show_label=False, height=400)
                    with gr.Row():
                        up_conf = gr.Slider(0.1, 0.9, value=CONF_DEFAULT, step=0.05,
                                            label="Порог уверенности", scale=1)
                        up_btn  = gr.Button("Определить",
                                            elem_classes=["btn-p"], scale=0)
                with gr.Column(scale=1, min_width=280):
                    up_stats = gr.HTML(stats_html(Counter()))
            up_btn.click(detect_upload, [up_in, up_conf], [up_out, up_stats])

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
