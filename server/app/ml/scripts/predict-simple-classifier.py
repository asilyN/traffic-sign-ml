#!/usr/bin/env python3
"""
Run inference with the simple traffic-sign classifier.

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
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict traffic sign from an image.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Dataset root directory. If omitted, common dataset locations are auto-detected.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/simple_classifier.joblib"),
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
    return parser.parse_args()


def load_image_feature(image_path: Path, size: int) -> np.ndarray:
    try:
        with Image.open(image_path) as img:
            arr = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise RuntimeError(f"Unreadable image: {image_path}") from exc
    return arr.reshape(1, -1) / 255.0


def _softmax_1d(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    denom = np.sum(exp_vals)
    if not np.isfinite(denom) or denom <= 0:
        return np.zeros_like(logits, dtype=np.float64)
    return exp_vals / denom


def compute_confidence(pipeline: object, x: np.ndarray, pred_class: int) -> float:
    classes = getattr(pipeline, "classes_", None)
    class_to_idx = {int(c): i for i, c in enumerate(classes)} if classes is not None else {}

    # First choice: use predict_proba when it is numerically valid.
    if hasattr(pipeline, "predict_proba"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                probs = np.asarray(pipeline.predict_proba(x)[0], dtype=np.float64)
            if (
                probs.ndim == 1
                and probs.size > 0
                and np.all(np.isfinite(probs))
                and float(np.sum(probs)) > 0.0
            ):
                probs = probs / float(np.sum(probs))
                idx = class_to_idx.get(pred_class)
                if idx is not None and 0 <= idx < probs.size:
                    return float(probs[idx])
                return float(np.max(probs))
        except Exception:
            pass

    # Fallback for unstable probability outputs (seen with some SGDClassifier samples).
    if hasattr(pipeline, "decision_function"):
        try:
            scores = np.asarray(pipeline.decision_function(x), dtype=np.float64)
            if scores.ndim == 2:
                logits = scores[0]
            else:
                logits = scores

            if logits.ndim == 1 and logits.size > 0 and np.all(np.isfinite(logits)):
                probs = _softmax_1d(logits)
                idx = class_to_idx.get(pred_class)
                if idx is not None and 0 <= idx < probs.size:
                    return float(probs[idx])
                return float(np.max(probs))
        except Exception:
            pass

    return 0.0


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
    val_pick = pick_image_from_split(root, splits_dir / "val.txt")
    if val_pick is not None:
        return val_pick
    test_pick = pick_image_from_split(root, splits_dir / "test.txt")
    if test_pick is not None:
        return test_pick
    raise FileNotFoundError(
        "No --image provided and no usable sample found in splits/val.txt or splits/test.txt."
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
        (root / "models" / "simple_classifier.joblib").resolve(),
        (Path(".").resolve() / model_path_arg).resolve(),
        (Path(".").resolve() / "server" / "app" / "ml" / "dataset" / "models" / "simple_classifier.joblib").resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        "Model not found. Searched:\n"
        f"{searched}\n"
        "Run server/app/ml/scripts/train-simple-classifier.py first."
    )


def resolve_image_path(root: Path, image_arg: Path) -> Path:
    if image_arg.is_absolute():
        return image_arg.resolve()

    candidates = [
        (Path(".").resolve() / image_arg).resolve(),
        (root / image_arg).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Return the root-relative candidate so the error path remains predictable.
    return candidates[1]


def main() -> None:
    args = parse_args()
    root = resolve_dataset_root(args.root)
    model_path = resolve_model_path(root, args.model_path)

    if args.image is None:
        image_path = pick_default_image(root).resolve()
    else:
        image_path = resolve_image_path(root, args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    image_size = int(bundle.get("image_size", 32))
    label_metadata = bundle.get("label_metadata", {})

    x = load_image_feature(image_path, image_size)
    pred_class = int(pipeline.predict(x)[0])
    confidence = compute_confidence(pipeline, x, pred_class)

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
