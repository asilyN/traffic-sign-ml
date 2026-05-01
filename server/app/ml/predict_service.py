from __future__ import annotations

import warnings
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
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
        self.model_path = model_path or (server_root / "app" / "ml" / "models" / "simple_classifier.joblib")
        self._bundle = None
        self._pipeline = None
        self._image_size = 32
        self._label_metadata: dict[int, dict[str, object]] = {}

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
        self._pipeline = self._bundle["pipeline"]
        self._image_size = int(self._bundle.get("image_size", 32))
        self._label_metadata = self._bundle.get("label_metadata", {})

    def _load_image_feature(self, image_bytes: bytes) -> np.ndarray:
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                arr = np.asarray(
                    img.convert("L").resize((self._image_size, self._image_size), Image.BILINEAR),
                    dtype=np.float32,
                )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise PredictionError("Unreadable image file.") from exc
        return arr.reshape(1, -1) / 255.0

    def predict_from_bytes(self, image_bytes: bytes) -> dict[str, object]:
        if not image_bytes:
            raise PredictionError("Empty file content.")

        self._ensure_loaded()
        x = self._load_image_feature(image_bytes)

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
        }