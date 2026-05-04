from __future__ import annotations

import json
import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    import tensorflow as tf
except ImportError as e:
    raise ImportError("tensorflow is required: pip install tensorflow") from e


class PredictionError(Exception):
    """Raised when prediction cannot be completed."""


class PredictorService:
    def __init__(self, model_path: Path | None = None) -> None:
        server_root     = Path(__file__).resolve().parents[2]   # .../server
        models_dir      = server_root / "app" / "ml" / "dataset" / "models"
        self.model_path = model_path or (models_dir / "simple_classifier.h5")
        self.meta_path  = models_dir / "simple_classifier_meta.json"

        self._model      = None
        self._meta       = None
        self._image_size = 48        # overridden by meta.json
        self._norm_mean  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._norm_std   = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # idx (0-based model output index) → original class_id (int)
        self._idx_to_class_id: dict[int, int] = {}

        # original class_id (int) → {class_name, category, ...}
        self._label_metadata: dict[int, dict] = {}

    
    
    # ------------------------------------------------------------------
    def is_ready(self) -> tuple[bool, str]:
        try:
            self._ensure_loaded()
            return True, "ok"
        except Exception as exc:
            return False, str(exc)

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load with custom_objects so the label-smoothing loss deserialises cleanly
        self._model = tf.keras.models.load_model(
            self.model_path, compile=False
        )

        if not self.meta_path.exists():
            raise PredictionError(
                "Meta JSON is required for inference. "
                f"Missing: {self.meta_path}"
            )

        self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))

        # ── image size & normalisation (must match training exactly) ──────────
        self._image_size = int(self._meta.get("image_size", 48))
        self._norm_mean  = np.array(
            self._meta.get("norm_mean", [0.485, 0.456, 0.406]),
            dtype=np.float32,
        )
        self._norm_std = np.array(
            self._meta.get("norm_std", [0.229, 0.224, 0.225]),
            dtype=np.float32,
        )

        # ── idx_to_label: str(model_output_idx) → original_class_id (int) ────
        # Stored in meta.json as {"0": 1, "1": 3, ...} (string keys, int values)
        self._idx_to_class_id = {
            int(k): int(v)
            for k, v in self._meta.get("idx_to_label", {}).items()
        }

        # ── label_metadata: str(original_class_id) → {class_name, category} ─
        # Stored as {"1": {"class_name": "Stop", ...}, "3": {...}, ...}
        self._label_metadata = {
            int(k): v
            for k, v in self._meta.get("label_metadata", {}).items()
        }

    # ------------------------------------------------------------------
    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        """
        Mirrors build_arrays() / load_image() in train2.py exactly:
          1. Convert to RGB
          2. Resize to image_size × image_size with BILINEAR interpolation
          3. float32, divide by 255
          4. Subtract norm_mean, divide by (norm_std + 1e-7)
          5. Add batch dim → (1, size, size, 3)
        """
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                img = img.convert("RGB").resize(
                    (self._image_size, self._image_size), Image.BILINEAR
                )
                arr = np.asarray(img, dtype=np.float32) / 255.0
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise PredictionError("Unreadable image file.") from exc

        arr = (arr - self._norm_mean) / (self._norm_std + 1e-7)
        return np.expand_dims(arr, axis=0)   # (1, H, W, 3)

    # ------------------------------------------------------------------
    def predict_from_bytes(self, image_bytes: bytes) -> dict[str, object]:
        if not image_bytes:
            raise PredictionError("Empty file content.")

        self._ensure_loaded()
        x = self._preprocess(image_bytes)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Model outputs raw logits — apply softmax to get probabilities
            logits = self._model.predict(x, verbose=0)[0]   # (num_classes,)

        probs      = tf.nn.softmax(logits).numpy()
        pred_idx   = int(np.argmax(probs))         # 0-based model output index
        confidence = float(probs[pred_idx])

        # Step 1: model output index → original class_id used in the dataset
        class_id = self._idx_to_class_id.get(pred_idx, pred_idx + 1)

        # Step 2: original class_id → human-readable metadata
        meta = self._label_metadata.get(
            class_id,
            {"class_name": f"class_{class_id}", "category": "unknown"},
        )

        return {
            "prediction":  str(meta.get("class_name", f"class_{class_id}")),
            "confidence":  round(confidence, 4),
            "label_index": class_id,   # original 1-based dataset class id
            "category":    str(meta.get("category", "unknown")),
        }