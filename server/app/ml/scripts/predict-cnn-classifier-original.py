#!/usr/bin/env python3
"""
Run inference with the CNN trained on original Train/ data (cnn_classifier_original.joblib).

Thin wrapper around predict-cnn-classifier.py with the matching default model path.

Train: python train-cnn-classifier-original.py
Predict: python predict-cnn-classifier-original.py [--image PATH] [--pretty]
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_predict_module():
    path = Path(__file__).resolve().parent / "predict-cnn-classifier.py"
    spec = importlib.util.spec_from_file_location("_predict_cnn_classifier_main", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load prediction module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    base_argv = [
        Path(__file__).name,
        "--model-path",
        "models/cnn_classifier_original.joblib",
    ]
    sys.argv = base_argv + sys.argv[1:]
    mod = _load_predict_module()
    mod.main()


if __name__ == "__main__":
    main()
