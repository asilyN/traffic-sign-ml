#!/usr/bin/env python3
"""
Run inference with the CNN traffic-sign classifier using PyTorch.

Input:
- image path

Output:
- frontend-ready JSON with class_name/category/confidence
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".webp"}


class SimpleConvNet(nn.Module):
    """Lightweight CNN for 32x32 grayscale traffic sign images."""

    def __init__(self, num_classes: int = 48):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict traffic sign from an image using CNN.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Dataset root directory. If omitted, common dataset locations are auto-detected.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/cnn_classifier.joblib"),
        help="Model path relative to root.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help=(
            "Path to image (absolute or relative to root). "
            "If omitted, auto-picks one sample from splits/val.txt, then splits/test.txt."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', or 'auto'.",
    )
    return parser.parse_args()


def load_image_feature(image_path: Path, size: int) -> np.ndarray:
    try:
        with Image.open(image_path) as img:
            arr = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise RuntimeError(f"Unreadable image: {image_path}") from exc
    return arr.reshape(1, 1, size, size) / 255.0


def compute_confidence(logits: np.ndarray, pred_idx: int, indices: list[int]) -> float:
    """Convert logits to confidence using softmax."""
    logits = logits - np.max(logits)
    exp_vals = np.exp(logits)
    probs = exp_vals / np.sum(exp_vals)
    
    if 0 <= pred_idx < len(probs):
        return float(probs[pred_idx])
    return float(np.max(probs))


def pick_image_from_split(root: Path, split_file: Path) -> Path | None:
    if not split_file.exists():
        return None
    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(" ", 1)
        rel = parts[0]
        img_path = (root / rel).resolve()
        if img_path.exists() and img_path.is_file() and img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return img_path
    return None


def pick_default_image(root: Path) -> Path:
    splits_dir = root / "splits"
    for name in ("val_clean.txt", "val_augmented.txt", "val.txt", "test.txt"):
        pick = pick_image_from_split(root, splits_dir / name)
        if pick is not None:
            return pick
    raise FileNotFoundError(
        "No --image provided and no usable sample found in splits/ "
        "(tried val_clean.txt, val_augmented.txt, val.txt, test.txt)."
    )


def resolve_dataset_root(root_arg: Path | None) -> Path:
    if root_arg is not None:
        root = root_arg.resolve()
        if (root / "Train").exists():
            return root
        raise FileNotFoundError(
            f"Train directory not found under provided --root: {root / 'Train'}"
        )

    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(".").resolve(),
        script_dir.parent / "dataset",
        script_dir.parent,
        Path(".").resolve() / "server" / "app" / "ml" / "dataset",
    ]
    for candidate in candidates:
        if (candidate / "Train").exists():
            return candidate.resolve()

    pretty_candidates = "\n".join(f"- {c.resolve()}" for c in candidates)
    raise FileNotFoundError(
        "Could not auto-detect dataset root (missing 'Train' directory).\n"
        "Searched paths:\n"
        f"{pretty_candidates}\n"
        "Pass --root explicitly, e.g. --root server/app/ml/dataset"
    )


def resolve_model_path(root: Path, model_path_arg: Path) -> Path:
    if model_path_arg.is_absolute():
        if model_path_arg.exists():
            return model_path_arg
        raise FileNotFoundError(f"Model not found: {model_path_arg}")

    candidates = [
        (root / model_path_arg).resolve(),
        (root / "models" / "cnn_classifier.joblib").resolve(),
        (Path(".").resolve() / model_path_arg).resolve(),
        (Path(".").resolve() / "server" / "app" / "ml" / "dataset" / "models" / "cnn_classifier.joblib").resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        "Model not found. Searched:\n"
        f"{searched}\n"
        "Run train-cnn-classifier.py or train-cnn-classifier-original.py first."
    )


def _relative_image_candidates(root: Path, image_arg: Path) -> list[Path]:
    rel = Path(*image_arg.parts)
    seen: set[Path] = set()
    out: list[Path] = []

    def add(p: Path) -> None:
        p = p.resolve()
        if p not in seen:
            seen.add(p)
            out.append(p)

    add(Path(".").resolve() / rel)
    add(root / rel)
    for i, ancestor in enumerate(root.parents):
        if i > 12:
            break
        add(ancestor / rel)
    return out


def resolve_image_path(root: Path, image_arg: Path) -> Path:
    if image_arg.is_absolute():
        return image_arg.resolve()

    candidates = _relative_image_candidates(root, image_arg)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def format_image_search_paths(root: Path, image_arg: Path) -> str:
    return "\n".join(f"- {p}" for p in _relative_image_candidates(root, image_arg))


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    root = resolve_dataset_root(args.root)
    model_path = resolve_model_path(root, args.model_path)

    if args.image is None:
        image_path = pick_default_image(root).resolve()
    else:
        image_path = resolve_image_path(root, args.image)
    if not image_path.exists():
        hint = ""
        if args.image is not None and not args.image.is_absolute():
            hint = (
                "\n\nSearched relative path in:\n"
                f"{format_image_search_paths(root, args.image)}\n"
                "Use an absolute path, or run from the project directory, or place the file under the dataset root."
            )
        raise FileNotFoundError(f"Image not found: {image_path}{hint}")

    device = get_device(args.device)

    bundle = joblib.load(model_path)
    image_size = int(bundle.get("image_size", 32))
    num_classes = int(bundle.get("num_classes", 48))
    label_metadata = bundle.get("label_metadata", {})
    labels_present = bundle.get("labels_present", list(range(1, 49)))

    model = SimpleConvNet(num_classes=num_classes).to(device)
    model.load_state_dict(bundle["model_state"])
    model.eval()

    x = torch.from_numpy(load_image_feature(image_path, image_size)).to(device)
    
    with torch.no_grad():
        logits = model(x).cpu().numpy()[0]
    
    pred_idx = int(np.argmax(logits))
    pred_class = pred_idx + 1  # Convert index (0-47) to class ID (1-48)
    confidence = compute_confidence(logits, pred_idx, list(range(num_classes)))

    meta = label_metadata.get(
        pred_class,
        {"class_id": pred_class, "class_name": f"class_{pred_class}", "category": "unknown"},
    )
    result = {
        "image_path": image_path.as_posix(),
        "class_id": pred_class,
        "class_name": str(meta.get("class_name", f"class_{pred_class}")),
        "category": str(meta.get("category", "unknown")),
        "confidence": round(confidence, 4),
        "status": "success",
    }

    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
