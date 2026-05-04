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
from typing import cast

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".webp"}


class TunableConvNet(nn.Module):
    """CNN with configurable width (must match train-cnn-classifier-tune)."""

    def __init__(
        self,
        num_classes: int,
        base_channels: int,
        dropout: float,
        fc_hidden: int,
    ):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        flat = c4 * 8 * 8
        self.classifier = nn.Sequential(
            nn.Linear(flat, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class SimpleConvNet(nn.Module):
    """Lightweight CNN for traffic sign images."""

    def __init__(self, num_classes: int = 48):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 -> 32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
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
        default=Path("models/cnn_classifier.h5"),
        help="Model path relative to root (HDF5 format).",
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
            arr = np.asarray(img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise RuntimeError(f"Unreadable image: {image_path}") from exc
    
    # Normalize using ImageNet mean/std per channel
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = arr / 255.0
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1).reshape(1, 3, size, size)


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
        (root / "models" / "cnn_classifier.h5").resolve(),
        (Path(".").resolve() / model_path_arg).resolve(),
        (Path(".").resolve() / "server" / "app" / "ml" / "dataset" / "models" / "cnn_classifier.h5").resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        "Model not found. Searched:\n"
        f"{searched}\n"
        "Run train-cnn-classifier.py first."
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


def load_labels_from_json(root: Path) -> dict[int, dict[str, str]]:
    """Load labels from labels.json file at server/app/ml/labels.json (parent of dataset)."""
    # labels.json is at root.parent/labels.json (one level up from dataset root)
    labels_file = root.parent / "labels.json"
    
    if labels_file.exists():
        try:
            payload = json.loads(labels_file.read_text(encoding="utf-8"))
            metadata: dict[int, dict[str, str]] = {}
            for row in payload.get("classes", []):
                class_id = int(row["class_id"])
                metadata[class_id] = {
                    "class_id": str(class_id),
                    "class_name": str(row.get("class_name", f"class_{class_id}")),
                    "category": str(row.get("category", "unknown")),
                    "instruction": str(row.get("instruction", "")),
                }
            return metadata
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    
    return {}


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

    # Load model from HDF5 file
    with h5py.File(model_path, "r") as f:  # type: ignore
        # Load metadata from attributes
        image_size = int(f.attrs.get("image_size", 32))  # type: ignore
        num_classes = int(f.attrs.get("num_classes", 48))  # type: ignore
        model_class_raw = f.attrs.get("model_class", "SimpleConvNet")
        if isinstance(model_class_raw, bytes):
            model_class = model_class_raw.decode("utf-8")
        else:
            model_class = str(model_class_raw)
        tunable_arch: dict[str, object] | None = None
        arch_raw = f.attrs.get("tunable_arch")
        if arch_raw is not None:
            if isinstance(arch_raw, bytes):
                arch_raw = arch_raw.decode("utf-8")
            tunable_arch = json.loads(str(arch_raw))

        # Load labels present
        labels_present_array: np.ndarray = f["labels_present"][()]  # type: ignore
        labels_present = labels_present_array.tolist()

        # Load model state dictionary
        model_state: dict[str, torch.Tensor] = {}
        model_state_group = f["model_state"]  # type: ignore
        for key in model_state_group.keys():  # type: ignore
            model_state[key] = torch.from_numpy(np.array(model_state_group[key]))  # type: ignore

    if model_class == "TunableConvNet" and tunable_arch is not None:
        model = TunableConvNet(
            num_classes=num_classes,
            base_channels=int(tunable_arch["base_channels"]),
            dropout=float(tunable_arch["dropout"]),
            fc_hidden=int(tunable_arch["fc_hidden"]),
        ).to(device)
    else:
        model = SimpleConvNet(num_classes=num_classes).to(device)
    model.load_state_dict(model_state)
    label_metadata = load_labels_from_json(root)
    if not label_metadata:
        # Fallback: try to load from HDF5
        try:
            with h5py.File(model_path, "r") as f:  # type: ignore
                label_meta_bytes: bytes = f["label_metadata"][()]  # type: ignore
                label_meta_str = label_meta_bytes.decode("utf-8")
                raw_meta = json.loads(label_meta_str)
            label_metadata = {}
            for k, v in raw_meta.items():
                try:
                    ik = int(k)
                except (TypeError, ValueError):
                    continue
                if isinstance(v, dict):
                    label_metadata[ik] = {
                        "class_id": str(v.get("class_id", ik)),
                        "class_name": str(v.get("class_name", f"class_{ik}")),
                        "category": str(v.get("category", "unknown")),
                        "instruction": str(v.get("instruction", "")),
                    }
        except Exception:
            label_metadata = {}

    model.eval()

    x = torch.from_numpy(load_image_feature(image_path, image_size)).to(device)
    
    with torch.no_grad():
        logits = model(x).cpu().numpy()[0]
    
    pred_idx = int(np.argmax(logits))
    pred_class = pred_idx + 1  # Convert index (0-47) to class ID (1-48)
    confidence = compute_confidence(logits, pred_idx, list(range(num_classes)))

    meta = label_metadata.get(
        pred_class,
        {
            "class_id": str(pred_class),
            "class_name": f"class_{pred_class}",
            "category": "unknown",
            "instruction": "",
        },
    )
    instruction_text = str(meta.get("instruction", ""))
    result = {
        "image_path": image_path.as_posix(),
        "class_id": pred_class,
        "class_name": str(meta.get("class_name", f"class_{pred_class}")),
        "category": str(meta.get("category", "unknown")),
        "confidence": round(confidence, 4),
        "instruction": instruction_text,
        "status": "success",
    }

    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
