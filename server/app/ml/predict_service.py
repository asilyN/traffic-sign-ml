from __future__ import annotations

import json
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
    """Lightweight CNN for 32x32 RGB traffic sign images."""

    def __init__(self, num_classes: int = 48):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
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


class PredictorService:
    def __init__(self, model_path: Path | None = None) -> None:
        server_root = Path(__file__).resolve().parents[2]  # .../server
        # Use only cnn_classifier.h5 model
        h5_path = server_root / "app" / "ml" / "dataset" / "models" / "cnn_classifier.h5"
        
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = h5_path
        
        self._bundle = None
        self._model = None  # For CNN
        self._image_size = 32
        self._label_metadata: dict[int, dict[str, object]] = {}
        self._device = torch.device("cpu")

    def reload_model(self, model_path: Path | None = None) -> None:
        """Drop cached weights and optionally switch the on-disk model path."""
        if model_path is not None:
            self.model_path = Path(model_path).resolve()
        self._bundle = None
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
        
        # Load from HDF5 format (cnn_classifier.h5)
        self._load_from_hdf5()

    def _load_from_hdf5(self) -> None:
        """Load model from HDF5 format (cnn_classifier.h5)."""
        import h5py
        
        with h5py.File(self.model_path, "r") as f:  # type: ignore
            # Load metadata from attributes
            self._image_size = int(f.attrs.get("image_size", 32))  # type: ignore
            num_classes = int(f.attrs.get("num_classes", 48))  # type: ignore
            
            # Load labels present
            labels_present_array: np.ndarray = f["labels_present"][()]  # type: ignore
            labels_present = labels_present_array.tolist()
            
            # Load model state dictionary
            model_state: dict[str, torch.Tensor] = {}
            model_state_group = f["model_state"]  # type: ignore
            for key in model_state_group.keys():  # type: ignore
                model_state[key] = torch.from_numpy(np.array(model_state_group[key]))  # type: ignore
            
            # Load label metadata from HDF5
            try:
                label_meta_bytes: bytes = f["label_metadata"][()]  # type: ignore
                label_meta_str = label_meta_bytes.decode("utf-8")
                self._label_metadata = json.loads(label_meta_str)
            except Exception:
                self._label_metadata = {}
        
        # Try to load labels from labels.json (preferred, more up-to-date)
        try:
            server_root = Path(__file__).resolve().parents[2]  # .../server
            labels_json_path = server_root / "app" / "ml" / "labels.json"
            if labels_json_path.exists():
                with open(labels_json_path, "r", encoding="utf-8") as f:
                    labels_data = json.load(f)
                    # Convert from classes array to dict keyed by class_id
                    for row in labels_data.get("classes", []):
                        class_id = int(row["class_id"])
                        self._label_metadata[class_id] = {
                            "class_id": str(class_id),
                            "class_name": str(row.get("class_name", f"class_{class_id}")),
                            "category": str(row.get("category", "unknown")),
                        }
        except Exception:
            pass
        
        # Create and load model
        self._model_type = "cnn"
        self._model = SimpleConvNet(num_classes=num_classes).to(self._device)
        self._model.load_state_dict(model_state)
        self._model.eval()
        self._bundle = {"model_state": model_state}  # Store for reference

    def _load_image_feature(self, image_bytes: bytes) -> np.ndarray:
        """RGB preprocessing for CNN."""
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                arr = np.asarray(
                    img.convert("RGB").resize((self._image_size, self._image_size), Image.Resampling.LANCZOS),
                    dtype=np.float32,
                )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise PredictionError("Unreadable image file.") from exc
        
        # Normalize using ImageNet mean/std per channel
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = arr / 255.0
        arr = (arr - mean) / std
        # Return as (1, 3, H, W) for CNN
        return arr.transpose(2, 0, 1).reshape(1, 3, self._image_size, self._image_size)

    def predict_from_bytes(self, image_bytes: bytes) -> dict[str, object]:
        if not image_bytes:
            raise PredictionError("Empty file content.")

        self._ensure_loaded()
        x = self._load_image_feature(image_bytes)

        # CNN prediction
        assert self._model is not None, "Model not loaded"
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