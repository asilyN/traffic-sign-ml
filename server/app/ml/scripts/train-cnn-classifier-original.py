#!/usr/bin/env python3
"""
Train the CNN on the original Train/ dataset (paths under splits/train_clean.txt).

This is a thin wrapper around train-cnn-classifier.py with defaults that point at
non-augmented images and write a separate artifact so augmented training can coexist.

Defaults:
- splits/train_clean.txt, splits/val_clean.txt (relative paths like Train/.../image.png)
- models/cnn_classifier_original.joblib
- reports/cnn_train_metrics_original.txt

Augmented training (Train_Augmented_Balanced): python train-cnn-classifier.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_trainer_module():
    path = Path(__file__).resolve().parent / "train-cnn-classifier.py"
    spec = importlib.util.spec_from_file_location("_train_cnn_classifier_main", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    base_argv = [
        Path(__file__).name,
        "--train-split",
        "train_clean.txt",
        "--val-split",
        "val_clean.txt",
        "--strict-splits",
        "--model-out",
        "cnn_classifier_original.joblib",
        "--metrics-out",
        "cnn_train_metrics_original.txt",
    ]
    sys.argv = base_argv + sys.argv[1:]
    mod = _load_trainer_module()
    mod.main()


if __name__ == "__main__":
    main()
