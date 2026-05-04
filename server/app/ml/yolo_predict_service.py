from __future__ import annotations

import importlib.util
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


class YoloPredictorService:
    def __init__(self, model_path: Path | None = None) -> None:
        server_root = Path(__file__).resolve().parents[2]
        self.model_path = model_path or (server_root / "app" / "ml" / "dataset" / "models" / "simple_classifier.h5")
        self._models: dict[str, Any] | None = None

        # Resolve the detection_classifier module path once at init
        self._dc_path = Path(__file__).resolve().parent / "services" / "detection_classifier.py"
        if not self._dc_path.exists():
            raise FileNotFoundError(f"detection_classifier.py not found at: {self._dc_path}")

    def _import_detection_classifier(self):
        """Import detection_classifier.py by file path using importlib."""
        if "detection_classifier" in sys.modules:
            return sys.modules["detection_classifier"]

        spec = importlib.util.spec_from_file_location("detection_classifier", self._dc_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["detection_classifier"] = module
        spec.loader.exec_module(module)
        return module

    def _ensure_loaded(self) -> None:
        if self._models is not None:
            return
        dc = self._import_detection_classifier()
        self._models = dc.load_models(pt_path=self.model_path)

    def predict_from_bytes(self, image_bytes: bytes, conf_threshold: float = 0.25) -> dict:
        self._ensure_loaded()
        dc = self._import_detection_classifier()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return dc.detect_and_classify(image, conf_threshold=conf_threshold, **self._models)