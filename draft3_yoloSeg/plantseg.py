"""Stage 1 - background-invariant plant segmentation.

Finds the plant in a photo taken on ANY background and isolates it. Nothing in
here scores disease; that is stage 3 and does not exist yet. `severity` is
deliberately absent from the JSON so the website cannot accidentally show one.

Library:
    from plantseg import segment, render_panel
    res = segment("photo.jpg")          # dict, JSON-serialisable except 'mask'
    panel = render_panel(res)           # BGR image: original | outline | isolated

CLI:
    python plantseg.py photo.jpg                 -> photo_panel.jpg + _outline/_isolated/_mask
    python plantseg.py folder/*.jpg --out out/
    python plantseg.py photo.jpg --json          -> JSON only, no files written
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time

import cv2
import numpy as np

# Ultralytics will pip-install missing extras mid-request if you let it. On a
# server that means a surprise network call and a surprise dependency change.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "model", "plantseg_v2.pt")
MODEL_VERSION = "seg_v2"

# Tuned on the 31 held-out plants; see HANDOFF.md for what each one catches.
MIN_AREA_FRAC = 0.02      # smaller than this and it is probably not the subject
MAX_AREA_FRAC = 0.80      # larger than this and it has almost certainly grabbed background
MIN_CONF = 0.50
FRAME_MARGIN = 2          # px band counted as "touching the edge"

_model = None
_weights_sha = None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(weights=WEIGHTS, device="cpu"):
    """Loaded once and cached - ~1.3 s to load, then ~0.1-0.2 s/image on CPU."""
    global _model, _weights_sha
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(weights)
        _model.to(device)
        _weights_sha = _sha256(weights)
    return _model


def read_image(src):
    """Path, bytes or numpy array -> BGR uint8. HEIC works (phones shoot it)."""
    if isinstance(src, np.ndarray):
        return src
    if isinstance(src, (bytes, bytearray)):
        data = bytes(src)
    else:
        with open(src, "rb") as fh:
            data = fh.read()
    if not data:
        raise ValueError("empty image")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:                                     # HEIC / HEIF fallback
        try:
            from PIL import Image
            import pillow_heif
            pillow_heif.register_heif_opener()
            pil = Image.open(io.BytesIO(data)).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError("could not decode image (not a supported image format)")
    return img


def segment(src, conf=0.25, imgsz=640, device="cpu"):
    """Segment the plant. Returns a dict; res['mask'] is a uint8 0/255 array."""
    t0 = time.time()
    img = read_image(src)
    h, w = img.shape[:2]
    model = load_model(device=device)

    r = model.predict(img, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]

    mask = np.zeros((h, w), np.uint8)
    scores = []
    if r.masks is not None and len(r.masks.data):
        scores = [float(x) for x in r.boxes.conf.cpu().numpy()]
        for mk in r.masks.data.cpu().numpy():
            mk = cv2.resize(mk, (w, h), interpolation=cv2.INTER_LINEAR)
            mask |= (mk > 0.5).astype(np.uint8) * 255

    # Connected components tell us whether we found one plant or confetti.
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    areas = sorted(stats[1:, cv2.CC_STAT_AREA].tolist(), reverse=True) if n_lbl > 1 else []
    total = float(sum(areas))
    area_frac = total / (w * h)
    largest_frac = (areas[0] / total) if total else 0.0
    confidence = max(scores) if scores else 0.0

    touches = False
    if total:
        m = mask > 0
        b = FRAME_MARGIN
        touches = bool(m[:b].any() or m[-b:].any() or m[:, :b].any() or m[:, -b:].any())

    warnings = []
    if not total:
        warnings.append("no_plant_found")
    else:
        if area_frac < MIN_AREA_FRAC:
            warnings.append("plant_too_small")
        if area_frac > MAX_AREA_FRAC:
            warnings.append("mask_covers_most_of_frame")
        if confidence < MIN_CONF:
            warnings.append("low_confidence")
        if largest_frac < 0.85 and len(areas) > 1:
            warnings.append("fragmented_mask")
        if touches:
            warnings.append("plant_touches_frame_edge")

    return {
        "ok": True,
        "plant_found": bool(total),
        "confidence": round(confidence, 4),
        "plant_area_fraction": round(area_frac, 4),
        "largest_component_fraction": round(largest_frac, 4),
        "n_components": max(0, n_lbl - 1),
        "touches_frame_edge": touches,
        "usable": bool(total) and not warnings,
        "warnings": warnings,
        "severity": None,          # stage 3 not built - never fill this in downstream
        "image": {"width": w, "height": h},
        "model": {"name": "plantseg", "version": MODEL_VERSION,
                  "weights_sha256": _weights_sha, "device": device},
        "timing_ms": int((time.time() - t0) * 1000),
        "mask": mask,
        "_img": img,
    }


# ---------------------------------------------------------------- renders

def _thick(img):
    return max(2, int(round(min(img.shape[:2]) / 250)))


def render_outline(img, mask, color=(0, 255, 120), dim=0.45):
    """Original photo, everything outside the plant dimmed, boundary traced."""
    out = img.copy()
    m = mask > 0
    out[~m] = (out[~m] * dim).astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, _thick(img))
    return out


def render_isolated(img, mask):
    """BGRA cutout - only plant pixels survive, the rest is transparent."""
    out = np.dstack([img, (mask > 0).astype(np.uint8) * 255])
    out[mask == 0] = 0
    return out


def _checkerboard(h, w, size=None, light=(238, 238, 238), dark=(212, 212, 212)):
    size = size or max(8, min(h, w) // 40)
    board = np.zeros((h, w, 3), np.uint8)
    board[:] = light
    yy, xx = np.mgrid[0:h, 0:w]
    board[((yy // size + xx // size) % 2) == 1] = dark
    return board


def _flatten(bgra):
    """RGBA cutout over a checkerboard, so 'transparent' is visible in a JPEG."""
    a = bgra[:, :, 3:4].astype(np.float32) / 255.0
    return (bgra[:, :, :3] * a + _checkerboard(*bgra.shape[:2]) * (1 - a)).astype(np.uint8)


def render_panel(res, width=1500, title=None):
    """The three-panel figure: input photo | outline found | plant isolated.

    Panel 3 is exactly what stage 2/3 will consume - nothing else reaches them.
    """
    img, mask = res["_img"], res["mask"]
    panels = [img, render_outline(img, mask), _flatten(render_isolated(img, mask))]

    pw = width // 3
    ph = int(round(pw * img.shape[0] / img.shape[1]))
    panels = [cv2.resize(p, (pw, ph), interpolation=cv2.INTER_AREA) for p in panels]

    bar = 46
    top = 40 if title else 0
    canvas = np.full((top + ph + bar, pw * 3, 3), 24, np.uint8)
    labels = ["1. Input photo",
              "2. Plant found (outline)",
              "3. Isolated plant -> stage 2"]
    for k, (p, lab) in enumerate(zip(panels, labels)):
        canvas[top:top + ph, k * pw:(k + 1) * pw] = p
        cv2.putText(canvas, lab, (k * pw + 12, top + ph + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
    for k in (1, 2):
        canvas[top:top + ph, k * pw - 1:k * pw + 1] = 24

    if title:
        sub = "%s   conf %.2f   plant %.1f%% of frame%s" % (
            title, res["confidence"], 100 * res["plant_area_fraction"],
            ("   [" + ", ".join(res["warnings"]) + "]") if res["warnings"] else "")
        cv2.putText(canvas, sub, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (120, 230, 255) if res["warnings"] else (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def to_json(res, renders=False, render_width=900):
    """Strip the numpy fields; optionally embed the renders as base64 PNG data URIs."""
    out = {k: v for k, v in res.items() if k not in ("mask", "_img")}
    if renders:
        img, mask = res["_img"], res["mask"]
        rw = min(render_width, img.shape[1])
        rh = int(round(img.shape[0] * rw / img.shape[1]))

        def small(a):
            return cv2.resize(a, (rw, rh), interpolation=cv2.INTER_AREA)

        def enc(a, fmt=".jpg"):
            # JPEG for the photo-like renders (5-10x smaller over the wire);
            # PNG only where we need the alpha channel or crisp binary edges.
            p = [cv2.IMWRITE_JPEG_QUALITY, 88] if fmt == ".jpg" else []
            mime = "jpeg" if fmt == ".jpg" else "png"
            return "data:image/%s;base64," % mime + base64.b64encode(
                cv2.imencode(fmt, a, p)[1].tobytes()).decode()

        out["renders"] = {
            "outline": enc(small(render_outline(img, mask))),
            "isolated": enc(small(render_isolated(img, mask)), ".png"),
            "mask": enc(small(mask), ".png"),
            "panel": enc(render_panel(res, width=rw * 2)),
        }
    return out


# ---------------------------------------------------------------- CLI

def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Segment the plant out of a photo.")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default=".", help="directory for the written images")
    ap.add_argument("--json", action="store_true", help="print JSON only, write nothing")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for path in args.images:
        res = segment(path, conf=args.conf, device=args.device)
        stem = os.path.splitext(os.path.basename(path))[0]
        if args.json:
            print(json.dumps(to_json(res), indent=2))
            continue
        cv2.imwrite(os.path.join(args.out, stem + "_panel.jpg"),
                    render_panel(res, title=stem), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(args.out, stem + "_outline.jpg"),
                    render_outline(res["_img"], res["mask"]), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(args.out, stem + "_isolated.png"),
                    render_isolated(res["_img"], res["mask"]))
        cv2.imwrite(os.path.join(args.out, stem + "_mask.png"), res["mask"])
        print("%-34s conf %.2f  area %.3f  %4d ms  %s"
              % (stem, res["confidence"], res["plant_area_fraction"], res["timing_ms"],
                 ",".join(res["warnings"]) or "ok"))


if __name__ == "__main__":
    _cli()
