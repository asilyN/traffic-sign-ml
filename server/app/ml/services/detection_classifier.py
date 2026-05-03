#!/usr/bin/env python3
"""
Traffic Sign Detection + Classification Pipeline
=================================================
Integrates YOLOv8 (detection) → TrafficSignCNN (classification).

Pipeline
--------
  1. Load a full image (PIL or numpy)
  2. YOLO detects bounding boxes of traffic signs
  3. Each crop is resized to 48×48 and normalised (ImageNet priors)
  4. TrafficSignCNN classifies each crop
  5. Structured JSON is returned

Quick start
-----------
  from traffic_sign_pipeline import load_models, detect_and_classify
  from PIL import Image

  models = load_models()                         # load once
  image  = Image.open("road_scene.jpg")
  result = detect_and_classify(image, **models)  # reuse every call
  import json
  print(json.dumps(result, indent=2))

CLI usage
---------
  python traffic_sign_pipeline.py --image road_scene.jpg
  python traffic_sign_pipeline.py --image road_scene.jpg --conf 0.35 --device cpu
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Normalisation constants (MUST match training in train_simple_classifier.py)
# ---------------------------------------------------------------------------

NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CNN_INPUT_SIZE = 48          # px — fixed by training

# ---------------------------------------------------------------------------
# Default paths (relative to this file; override via load_models())
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent

DEFAULT_PT_PATH     = _HERE / "models" / "simple_classifier.pt"
DEFAULT_YOLO_WEIGHTS = "yolov8n.pt"   # downloaded automatically on first run


# ---------------------------------------------------------------------------
# CNN model definition
# (Must exactly mirror train_simple_classifier.py so state_dict loads cleanly)
# ---------------------------------------------------------------------------

class _ResBlock4(nn.Module):
    """Residual conv block with 1×1 projection shortcut (block 4)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.pool     = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x) + self.shortcut(x))


class TrafficSignCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout1: float = 0.4,
        dropout2: float = 0.2,
        spatial_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),   # ← this was missing
            nn.MaxPool2d(2, 2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),   # ← this was missing
            nn.MaxPool2d(2, 2),
        )

        self.block4 = _ResBlock4()
        self.spatial_drop = nn.Dropout2d(p=spatial_dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout1),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.spatial_drop(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)    
    
    def __init__(
        self,
        num_classes: int,
        dropout1: float = 0.4,
        dropout2: float = 0.2,
        spatial_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # ← extra layer
            nn.BatchNorm2d(64),                            # ← extra layer
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # ← extra layer
            nn.BatchNorm2d(128),                            # ← extra layer
            nn.MaxPool2d(2, 2),
        )
        self.block4      = _ResBlock4()
        self.spatial_drop = nn.Dropout2d(p=spatial_dropout)
        self.classifier  = nn.Sequential(
            nn.Dropout(p=dropout1),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.spatial_drop(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)   

    def __init__(
        self,
        num_classes: int,
        dropout1: float = 0.4,
        dropout2: float = 0.2,
        spatial_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block4      = _ResBlock4()
        self.spatial_drop = nn.Dropout2d(p=spatial_dropout)
        self.classifier  = nn.Sequential(
            nn.Dropout(p=dropout1),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.spatial_drop(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _pil_to_rgb(img: Any) -> Image.Image:
    """Ensure input is a PIL RGB image (accepts numpy too)."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8) if img.dtype != np.uint8 else img)
    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected PIL Image or numpy array, got {type(img)}")
    return img.convert("RGB")


def _crop_and_preprocess(
    image_rgb: Image.Image,
    x1: int, y1: int, x2: int, y2: int,
    size: int = CNN_INPUT_SIZE,
) -> torch.Tensor:
    """
    Crop a bounding-box region from the full image, resize to `size`×`size`,
    apply ImageNet normalisation, and return a (1, 3, size, size) tensor.

    Normalisation: (pixel/255 − mean) / std  — identical to training.
    """
    W, H = image_rgb.size

    # Safe clip so we never go out of bounds
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(x1 + 1, min(x2, W))
    y2 = max(y1 + 1, min(y2, H))

    crop = image_rgb.crop((x1, y1, x2, y2))
    crop = crop.resize((size, size), Image.BILINEAR)

    arr = np.asarray(crop, dtype=np.float32) / 255.0   # [0, 1]
    arr = (arr - NORM_MEAN) / (NORM_STD + 1e-7)        # normalise
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # HWC → CHW
    return tensor.unsqueeze(0)                          # (1, C, H, W)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(
    pt_path: str | Path = DEFAULT_PT_PATH,
    yolo_weights: str = DEFAULT_YOLO_WEIGHTS,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Load the YOLO detector and the CNN classifier once.

    Returns a dict that is unpacked as **kwargs into detect_and_classify().

    Parameters
    ----------
    pt_path      : Path to `simple_classifier.pt` produced by training script.
    yolo_weights : YOLOv8 weight name/path. "yolov8n.pt" is auto-downloaded.
    device       : "cuda", "cpu", or None (auto-select).

    Returns
    -------
    {
        "yolo"         : ultralytics.YOLO instance,
        "cnn"          : TrafficSignCNN (eval mode, on device),
        "idx_to_label" : {int → original class_id},
        "label_meta"   : {class_id → {"class_name", "category"}},
        "image_size"   : int (48),
        "device"       : torch.device,
    }
    """
    # ── device ────────────────────────────────────────────────────────────────
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(device)
    print(f"[load_models] device={_device}")

    # ── YOLO ──────────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Run:  pip install ultralytics"
        ) from exc

    print(f"[load_models] Loading YOLO weights: {yolo_weights}")
    yolo = YOLO(yolo_weights)

    # ── CNN ───────────────────────────────────────────────────────────────────
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(
            f"CNN checkpoint not found: {pt_path}\n"
            "Train first with train_simple_classifier.py, or pass the correct path."
        )

    print(f"[load_models] Loading CNN from: {pt_path}")
    ckpt = torch.load(pt_path, map_location="cpu")

    num_classes  = ckpt["num_classes"]
    image_size   = ckpt.get("image_size", CNN_INPUT_SIZE)
    idx_to_label = {int(k): int(v) for k, v in ckpt["idx_to_label"].items()}

    cnn = TrafficSignCNN(
        num_classes=num_classes,
        dropout1=ckpt.get("dropout1", 0.4),
        dropout2=ckpt.get("dropout2", 0.2),
        spatial_dropout=ckpt.get("spatial_dropout", 0.1),
    )
    cnn.load_state_dict(ckpt["state_dict"])
    cnn.eval()
    cnn.to(_device)

    # ── label metadata ────────────────────────────────────────────────────────
    # Attempt to load from labels.json next to the checkpoint; fall back to empty.
    label_meta: dict[int, dict] = {}
    labels_json = pt_path.parent.parent / "labels.json"
    if not labels_json.exists():
        labels_json = pt_path.parent / "labels.json"
    if labels_json.exists():
        import json as _json
        raw = _json.loads(labels_json.read_text(encoding="utf-8"))
        for row in raw.get("classes", []):
            cid = int(row["class_id"])
            label_meta[cid] = {
                "class_name": str(row.get("class_name", f"class_{cid}")),
                "category":   str(row.get("category", "unknown")),
            }
    else:
        warnings.warn(
            "labels.json not found — class_name and category will be generic.",
            stacklevel=2,
        )

    print(
        f"[load_models] CNN ready — {num_classes} classes, "
        f"input {image_size}×{image_size}"
    )
    return {
        "yolo":          yolo,
        "cnn":           cnn,
        "idx_to_label":  idx_to_label,
        "label_meta":    label_meta,
        "image_size":    image_size,
        "device":        _device,
    }


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def detect_and_classify(
    image: Any,
    *,
    yolo: Any,
    cnn: TrafficSignCNN,
    idx_to_label: dict[int, int],
    label_meta: dict[int, dict],
    image_size: int = CNN_INPUT_SIZE,
    device: torch.device | str = "cpu",
    conf_threshold: float = 0.25,
    yolo_classes: list[int] | None = None,
) -> dict[str, list[dict]]:
    """
    Run the full detection → classification pipeline on one image.

    Parameters
    ----------
    image           : PIL Image or H×W×3 numpy uint8 array.
    yolo            : Loaded YOLO model (from load_models).
    cnn             : Loaded TrafficSignCNN (from load_models).
    idx_to_label    : CNN output index → original dataset class ID.
    label_meta      : class ID → {"class_name", "category"}.
    image_size      : CNN input resolution (default 48).
    device          : torch.device used during CNN inference.
    conf_threshold  : Minimum YOLO detection confidence (default 0.25).
    yolo_classes    : YOLO class IDs to keep (None = keep all detections).

    Returns
    -------
    {
      "detections": [
        {
          "bbox":                      [x1, y1, x2, y2],
          "detection_confidence":      float,   # YOLO score
          "predicted_class":           int,     # dataset class ID
          "class_name":                str,
          "category":                  str,
          "classification_confidence": float,   # CNN softmax prob
        },
        ...
      ]
    }
    """
    if isinstance(device, str):
        device = torch.device(device)

    # ── 1. Ensure PIL RGB ─────────────────────────────────────────────────────
    image_pil = _pil_to_rgb(image)
    W, H = image_pil.size

    # ── 2. YOLO detection ─────────────────────────────────────────────────────
    # Pass PIL directly; ultralytics accepts PIL, numpy, and file paths.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yolo_results = yolo(
            image_pil,
            conf=conf_threshold,
            classes=yolo_classes,   # None → all classes
            verbose=False,
        )

    # yolo_results is a list (one element per image)
    result0 = yolo_results[0]

    # Boxes tensor: (N, 6) — [x1, y1, x2, y2, conf, cls]
    if result0.boxes is None or len(result0.boxes) == 0:
        return {"detections": []}

    boxes_xyxy = result0.boxes.xyxy.cpu().numpy()   # (N, 4)
    confs      = result0.boxes.conf.cpu().numpy()   # (N,)

    detections: list[dict] = []

    cnn.eval()
    with torch.no_grad():
        for (x1, y1, x2, y2), det_conf in zip(boxes_xyxy, confs):
            # ── 3. Crop ───────────────────────────────────────────────────────
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

            # ── 4. Preprocess (MUST match training) ───────────────────────────
            tensor = _crop_and_preprocess(
                image_pil, ix1, iy1, ix2, iy2, size=image_size
            ).to(device)

            # ── 5. CNN classification ─────────────────────────────────────────
            logits = cnn(tensor)                        # (1, num_classes)
            probs  = torch.softmax(logits, dim=1)       # (1, num_classes)
            pred_idx   = int(probs.argmax(dim=1).item())
            cls_conf   = float(probs[0, pred_idx].item())

            # ── 6. Map back to original dataset class ID ──────────────────────
            pred_class = idx_to_label.get(pred_idx, pred_idx)
            meta       = label_meta.get(
                pred_class,
                {"class_name": f"class_{pred_class}", "category": "unknown"},
            )

            detections.append({
                "bbox":                      [ix1, iy1, ix2, iy2],
                "detection_confidence":      round(float(det_conf), 4),
                "predicted_class":           pred_class,
                "class_name":               meta["class_name"],
                "category":                 meta["category"],
                "classification_confidence": round(cls_conf, 4),
            })

    return {"detections": detections}


# ---------------------------------------------------------------------------
# Optional: annotated image helper
# ---------------------------------------------------------------------------

def draw_detections(
    image: Any,
    result: dict[str, list[dict]],
    font_scale: float = 0.5,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes + labels on a copy of the image.

    Returns an H×W×3 uint8 numpy array suitable for cv2.imshow / PIL.

    Requires opencv-python:  pip install opencv-python-headless
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("pip install opencv-python-headless") from exc

    img_np = np.asarray(_pil_to_rgb(image), dtype=np.uint8).copy()
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    COLOR = (0, 200, 80)   # green

    for det in result["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        label = (
            f"{det['class_name']} "
            f"det={det['detection_confidence']:.2f} "
            f"cls={det['classification_confidence']:.2f}"
        )
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), COLOR, thickness)
        cv2.putText(
            img_bgr, label,
            (x1, max(y1 - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, COLOR, thickness,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Traffic Sign Detection + Classification Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", required=True, help="Path to input image.")
    p.add_argument("--pt-path", default=str(DEFAULT_PT_PATH),
                   help="Path to simple_classifier.pt checkpoint.")
    p.add_argument("--yolo-weights", default=DEFAULT_YOLO_WEIGHTS,
                   help="YOLOv8 weight file or name (auto-downloaded).")
    p.add_argument("--conf", type=float, default=0.25,
                   help="YOLO detection confidence threshold.")
    p.add_argument("--device", default=None,
                   help="'cuda' or 'cpu'. Auto-selected if omitted.")
    p.add_argument("--save-annotated", default=None,
                   help="Optional path to save annotated image (requires opencv).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path)
    except (UnidentifiedImageError, OSError) as exc:
        raise SystemExit(f"Cannot open image: {exc}") from exc

    # ── load models once ──────────────────────────────────────────────────────
    models = load_models(
        pt_path=args.pt_path,
        yolo_weights=args.yolo_weights,
        device=args.device,
    )

    # ── run pipeline ──────────────────────────────────────────────────────────
    result = detect_and_classify(
        image,
        conf_threshold=args.conf,
        **models,
    )

    print(json.dumps(result, indent=2))
    print(f"\nTotal detections: {len(result['detections'])}")

    # ── optional annotated image ──────────────────────────────────────────────
    if args.save_annotated:
        from PIL import Image as _Image
        annotated = draw_detections(image, result)
        _Image.fromarray(annotated).save(args.save_annotated)
        print(f"Annotated image saved → {args.save_annotated}")


if __name__ == "__main__":
    main()