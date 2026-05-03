#!/usr/bin/env python3
"""
Generate train/val splits from the Train_Augmented_Balanced dataset.

Outputs:
- splits/train_augmented.txt
- splits/val_augmented.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate splits from augmented balanced dataset.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Dataset root directory. If omitted, common dataset locations are auto-detected.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="Train_Augmented_Balanced",
        help="Name of the augmented dataset directory.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (0.0-1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-train",
        type=str,
        default="train_augmented.txt",
        help="Output train split filename.",
    )
    parser.add_argument(
        "--output-val",
        type=str,
        default="val_augmented.txt",
        help="Output validation split filename.",
    )
    return parser.parse_args()


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


def collect_images(dataset_path: Path) -> list[tuple[str, int]]:
    """Collect all image paths and class labels from dataset directory."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    rows: list[tuple[str, int]] = []
    
    # Iterate through class directories (1-48)
    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        try:
            class_id = int(class_dir.name)
        except ValueError:
            print(f"Skipping non-numeric directory: {class_dir.name}")
            continue
        
        # Find all image files in the class directory
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".webp"}
        for img_file in sorted(class_dir.iterdir()):
            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                # Relative path from dataset root
                rel_path = f"{dataset_path.name}/{class_id}/{img_file.name}"
                rows.append((rel_path, class_id))
    
    if not rows:
        raise RuntimeError(f"No images found in {dataset_path}")
    
    return rows


def split_data(rows: list[tuple[str, int]], train_ratio: float, seed: int) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Split data into train and validation sets."""
    rng = np.random.RandomState(seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    
    split_idx = int(len(rows) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_rows = [rows[i] for i in train_indices]
    val_rows = [rows[i] for i in val_indices]
    
    return train_rows, val_rows


def write_split(filepath: Path, rows: list[tuple[str, int]]) -> None:
    """Write split file in format: 'path/to/image.png class_id'"""
    with filepath.open("w", encoding="utf-8") as f:
        for rel_path, class_id in rows:
            f.write(f"{rel_path} {class_id}\n")


def main() -> None:
    args = parse_args()
    root = resolve_dataset_root(args.root)
    dataset_path = root / args.dataset_dir
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Collecting images from {dataset_path}...")
    rows = collect_images(dataset_path)
    print(f"Found {len(rows)} images across all classes")
    
    print(f"Splitting data (train: {args.train_ratio*100:.0f}%, val: {(1-args.train_ratio)*100:.0f}%)...")
    train_rows, val_rows = split_data(rows, args.train_ratio, args.seed)
    
    train_path = splits_dir / args.output_train
    val_path = splits_dir / args.output_val
    
    write_split(train_path, train_rows)
    write_split(val_path, val_rows)
    
    print(f"Train split: {len(train_rows)} samples → {train_path.relative_to(root).as_posix()}")
    print(f"Val split:   {len(val_rows)} samples → {val_path.relative_to(root).as_posix()}")
    print("Done!")


if __name__ == "__main__":
    main()
