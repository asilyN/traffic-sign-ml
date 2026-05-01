#!/usr/bin/env python3
"""
Simple baseline traffic-sign classifier training.

Uses (defaults):
- splits/train.txt
- splits/val.txt

Outputs:
- reports/simple_train_metrics.txt
- models/simple_classifier.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple traffic sign classifier.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Dataset root directory. If omitted, common dataset locations are auto-detected.",
    )
    parser.add_argument("--size", type=int, default=32, help="Square resize for images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--train-split",
        type=str,
        default="train.txt",
        help="Train split filename inside splits/ directory.",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="val.txt",
        help="Validation split filename inside splits/ directory.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("labels.json"),
        help="Path to labels metadata JSON relative to root.",
    )
    return parser.parse_args()


def read_split(split_path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        rel, label = line.rsplit(" ", 1)
        rows.append((rel, int(label)))
    return rows


def load_features(root: Path, rows: list[tuple[str, int]], size: int) -> tuple[np.ndarray, np.ndarray]:
    feats: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0
    for rel, label in rows:
        img_path = root / rel
        try:
            with Image.open(img_path) as img:
                arr = np.asarray(
                    img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32
                )
        except (UnidentifiedImageError, OSError, ValueError):
            skipped += 1
            continue
        feats.append(arr.reshape(-1) / 255.0)
        labels.append(label)

    if not feats:
        raise RuntimeError("No readable images found in split.")
    if skipped:
        print(f"Skipped unreadable images: {skipped}")
    return np.stack(feats), np.asarray(labels, dtype=np.int32)


def load_label_metadata(root: Path, labels_path: Path) -> dict[int, dict[str, object]]:
    labels_file = resolve_labels_path(root, labels_path)
    metadata: dict[int, dict[str, object]] = {}

    payload = json.loads(labels_file.read_text(encoding="utf-8"))
    for row in payload.get("classes", []):
        class_id = int(row["class_id"])
        metadata[class_id] = {
            "class_id": class_id,
            "class_name": str(row.get("class_name", f"class_{class_id}")),
            "category": str(row.get("category", "unknown")),
        }
    return metadata


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


def resolve_split_path(splits_dir: Path, split_name: str, fallback_name: str) -> Path:
    primary = splits_dir / split_name
    if primary.exists():
        return primary

    fallback = splits_dir / fallback_name
    if fallback.exists():
        print(f"Using fallback split: {fallback.name} (requested: {split_name})")
        return fallback

    raise FileNotFoundError(
        f"Split file not found: {primary}\n"
        f"Fallback also missing: {fallback}\n"
        "Generate splits first with clean-splits.py or pass --train-split/--val-split explicitly."
    )


def resolve_labels_path(root: Path, labels_path: Path) -> Path:
    # If user passes an absolute path, trust it directly.
    if labels_path.is_absolute():
        if labels_path.exists():
            return labels_path
        raise FileNotFoundError(f"Label metadata file not found: {labels_path}")

    candidates = [
        root / labels_path,
        root.parent / labels_path,
        root.parent / "ml" / labels_path,
        Path(".").resolve() / labels_path,
        Path(".").resolve() / "server" / "app" / "ml" / labels_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(f"- {p.resolve()}" for p in candidates)
    raise FileNotFoundError(
        "Label metadata file not found.\n"
        f"Requested path: {labels_path}\n"
        "Searched:\n"
        f"{searched}\n"
        "Pass --labels-path explicitly if your labels file is elsewhere."
    )


def main() -> None:
    args = parse_args()
    root = resolve_dataset_root(args.root)
    splits = root / "splits"
    reports = root / "reports"
    models = root / "models"
    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    train_split_path = resolve_split_path(splits, args.train_split, "train_clean.txt")
    val_split_path = resolve_split_path(splits, args.val_split, "val_clean.txt")
    train_rows = read_split(train_split_path)
    val_rows = read_split(val_split_path)

    print("Loading train features...")
    x_train, y_train = load_features(root, train_rows, args.size)
    print("Loading val features...")
    x_val, y_val = load_features(root, val_rows, args.size)
    label_metadata = load_label_metadata(root, args.labels_path)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-4,
                    max_iter=30,
                    tol=1e-3,
                    random_state=args.seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training baseline model...")
    clf.fit(x_train, y_train)

    train_pred = clf.predict(x_train)
    val_pred = clf.predict(x_val)
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    model_path = models / "simple_classifier.joblib"
    joblib.dump(
        {
            "pipeline": clf,
            "image_size": args.size,
            "labels_present": sorted(set(y_train.tolist())),
            "label_metadata": {
                class_id: label_metadata.get(
                    class_id,
                    {
                        "class_id": class_id,
                        "class_name": f"class_{class_id}",
                        "category": "unknown",
                    },
                )
                for class_id in sorted(set(y_train.tolist()))
            },
        },
        model_path,
    )

    report_text = classification_report(y_val, val_pred, digits=4, zero_division=0)
    metrics_path = reports / "simple_train_metrics.txt"
    metrics_path.write_text(
        "\n".join(
            [
                "Simple Classification Training Metrics",
                "=" * 80,
                f"Train samples: {len(y_train)}",
                f"Val samples: {len(y_val)}",
                f"Image size: {args.size}x{args.size}",
                f"Train accuracy: {train_acc:.4f}",
                f"Val accuracy: {val_acc:.4f}",
                f"Labels metadata path: {(root / args.labels_path).resolve()}",
                "",
                "Validation Classification Report",
                "-" * 80,
                report_text,
                "",
                f"Saved model: {model_path.relative_to(root).as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics:  {metrics_path}")


if __name__ == "__main__":
    main()
