from pathlib import Path

from flask import jsonify, request

from app.api import api_bp
from app.ml.predict_service import PredictionError, PredictorService

predictor = PredictorService()
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".webp"}


@api_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "Backend is running"})


@api_bp.route("/v1/ready", methods=["GET"])
def ready():
    """Model readiness endpoint."""
    is_ready, reason = predictor.is_ready()
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

    # Browsers usually send image/* for uploaded images; reject obvious mismatches early.
    if file.mimetype and not file.mimetype.startswith("image/"):
        return jsonify({"error": f"Unsupported content type: {file.mimetype}"}), 400

    try:
        image_bytes = file.read()
        result = predictor.predict_from_bytes(image_bytes)
        return jsonify(result), 200
    except PredictionError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        return jsonify({"error": "Internal prediction error."}), 500

