from pathlib import Path

from flask import jsonify, request

from app.api import api_bp
from app.ml.predict_service import PredictionError, PredictorService

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".webp"}

_predictor = None
_yolo_predictor = None


def get_predictor():
    """Lazy so Flask can bind PORT before TensorFlow loads."""
    global _predictor
    if _predictor is None:
        from app.ml.predict_service import PredictorService

        _predictor = PredictorService()
    return _predictor


def get_yolo_predictor():
    global _yolo_predictor
    if _yolo_predictor is None:
        from app.ml.yolo_predict_service import YoloPredictorService

        _yolo_predictor = YoloPredictorService()
    return _yolo_predictor


@api_bp.route("/v1/detect", methods=["POST"])
def detect():
    """Direct classification endpoint (no YOLO8 required)."""
    if "file" not in request.files:
        return jsonify({"error": 'Missing file. Use multipart/form-data key "file".'}), 400

    file = request.files["file"]
    if file.filename is None or file.filename.strip() == "":
        return jsonify({"error": "No file selected."}), 400

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        return jsonify({"error": f"Unsupported file type. Allowed types: {allowed}"}), 400

    if file.mimetype and not file.mimetype.startswith("image/"):
        return jsonify({"error": f"Unsupported content type: {file.mimetype}"}), 400

    try:
        image_bytes = file.read()
        # Use direct CNN classification instead of YOLO8
        prediction = get_predictor().predict_from_bytes(image_bytes)
        
        # Convert to detection format (single detection with full image as bbox)
        detection = {
            "bbox": [0, 0, 1, 1],  # Normalized coordinates for full image
            "class_name": prediction["prediction"],
            "category": prediction["category"],
            "detection_confidence": prediction["confidence"],  # Use CNN confidence
            "classification_confidence": prediction["confidence"],
            "other_predictions": prediction.get("other_predictions", []),
        }
        
        return jsonify({"detections": [detection]}), 200
    except PredictionError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        import traceback

        print("=" * 60)
        print("DETECT ERROR:", exc)
        traceback.print_exc()
        print("=" * 60)
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "Backend is running"})


@api_bp.route("/v1/ready", methods=["GET"])
def ready():
    """Model readiness endpoint."""
    is_ready, reason = get_predictor().is_ready()
    if is_ready:
        return jsonify({"status": "ok", "model_ready": True}), 200
    return jsonify({"status": "error", "model_ready": False, "message": reason}), 503


@api_bp.route("/v1/predict", methods=["POST"])
def predict():
    """Prediction endpoint for traffic sign recognition."""
    if "file" not in request.files:
        return jsonify({"error": 'Missing file. Use multipart/form-data key "file".'}), 400

    file = request.files["file"]
    if file.filename is None or file.filename.strip() == "":
        return jsonify({"error": "No file selected."}), 400

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        return jsonify({"error": f"Unsupported file type. Allowed types: {allowed}"}), 400

    if file.mimetype and not file.mimetype.startswith("image/"):
        return jsonify({"error": f"Unsupported content type: {file.mimetype}"}), 400

    try:
        image_bytes = file.read()
        result = get_predictor().predict_from_bytes(image_bytes)
        return jsonify(result), 200
    except PredictionError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        return jsonify({"error": "Internal prediction error."}), 500
