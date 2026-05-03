from __future__ import annotations

import warnings
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError


class PredictionError(Exception):
    """Raised when prediction cannot be completed."""


def _softmax_1d(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    denom = np.sum(exp_vals)
    if not np.isfinite(denom) or denom <= 0:
        return np.zeros_like(logits, dtype=np.float64)
    return exp_vals / denom


class SimpleConvNet(nn.Module):
    """Lightweight CNN for 32x32 grayscale traffic sign images."""

    def __init__(self, num_classes: int = 48):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
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


def _compute_confidence(pipeline: object, x: np.ndarray, pred_class: int) -> float:
    classes = getattr(pipeline, "classes_", None)
    class_to_idx = {int(c): i for i, c in enumerate(classes)} if classes is not None else {}

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

    if hasattr(pipeline, "decision_function"):
        try:
            scores = np.asarray(pipeline.decision_function(x), dtype=np.float64)
            logits = scores[0] if scores.ndim == 2 else scores
            if logits.ndim == 1 and logits.size > 0 and np.all(np.isfinite(logits)):
                probs = _softmax_1d(logits)
                idx = class_to_idx.get(pred_class)
                if idx is not None and 0 <= idx < probs.size:
                    return float(probs[idx])
                return float(np.max(probs))
        except Exception:
            pass

    return 0.0


class PredictorService:
    def __init__(self, model_path: Path | None = None) -> None:
        server_root = Path(__file__).resolve().parents[2]  # .../server
        # Original Train/ CNN only (train-cnn-classifier-original.py); SGD fallback if missing
        cnn_path = server_root / "app" / "ml" / "dataset" / "models" / "cnn_classifier_original.joblib"
        sgd_path = server_root / "app" / "ml" / "dataset" / "models" / "sgd_classifier.joblib"
        
        if model_path:
            self.model_path = model_path
        elif cnn_path.exists():
            self.model_path = cnn_path
        else:
            self.model_path = sgd_path
        
        self._bundle = None
        self._model_type = None  # "cnn" or "sgd"
        self._pipeline = None  # For SGD
        self._model = None  # For CNN
        self._image_size = 32
        self._label_metadata: dict[int, dict[str, object]] = {}
        self._device = torch.device("cpu")

    def reload_model(self, model_path: Path | None = None) -> None:
        """Drop cached weights and optionally switch the on-disk model path."""
        if model_path is not None:
            self.model_path = Path(model_path).resolve()
        self._bundle = None
        self._model_type = None
        self._pipeline = None
        self._model = None

    def is_ready(self) -> tuple[bool, str]:
        try:
            self._ensure_loaded()
            return True, "ok"
        except Exception as exc:
            return False, str(exc)

    def _ensure_loaded(self) -> None:
        if self._bundle is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self._bundle = joblib.load(self.model_path)
        self._image_size = int(self._bundle.get("image_size", 32))
        self._label_metadata = self._bundle.get("label_metadata", {})
        
        # Detect model type
        if "model_state" in self._bundle:
            # CNN model
            self._model_type = "cnn"
            num_classes = int(self._bundle.get("num_classes", 48))
            self._model = SimpleConvNet(num_classes=num_classes).to(self._device)
            self._model.load_state_dict(self._bundle["model_state"])
            self._model.eval()
        else:
            # SGD/Scikit-learn model
            self._model_type = "sgd"
            self._pipeline = self._bundle["pipeline"]

    def _load_image_feature(self, image_bytes: bytes) -> np.ndarray:
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                arr = np.asarray(
                    img.convert("L").resize((self._image_size, self._image_size), Image.BILINEAR),
                    dtype=np.float32,
                )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise PredictionError("Unreadable image file.") from exc
        
        if self._model_type == "cnn":
            # Return as (1, 1, H, W) for CNN
            return arr.reshape(1, 1, self._image_size, self._image_size) / 255.0
        else:
            # Return as (1, H*W) for SGD
            return arr.reshape(1, -1) / 255.0

    def predict_from_bytes(self, image_bytes: bytes) -> dict[str, object]:
        if not image_bytes:
            raise PredictionError("Empty file content.")

        self._ensure_loaded()
        x = self._load_image_feature(image_bytes)

        if self._model_type == "cnn":
            # CNN prediction
            x_tensor = torch.from_numpy(x).to(self._device)
            with torch.no_grad():
                logits = self._model(x_tensor).cpu().numpy()[0]
            
            pred_idx = int(np.argmax(logits))
            pred_class = pred_idx + 1  # Convert from 0-47 to 1-48
            
            # Compute confidence from logits
            logits = logits - np.max(logits)
            exp_vals = np.exp(logits)
            probs = exp_vals / np.sum(exp_vals)
            confidence = float(probs[pred_idx])
        else:
            # SGD prediction
            pred_class = int(self._pipeline.predict(x)[0])
            confidence = _compute_confidence(self._pipeline, x, pred_class)

        meta = self._label_metadata.get(
            pred_class,
            {"class_id": pred_class, "class_name": f"class_{pred_class}", "category": "unknown"},
        )

        return {
            "prediction": str(meta.get("class_name", f"class_{pred_class}")),
            "confidence": round(float(confidence), 4),
            "label_index": pred_class,
            "category": str(meta.get("category", "unknown")),
        }