import os

from flask import Flask, jsonify
from flask_cors import CORS


def _parse_origins() -> list[str]:
    raw = os.getenv("FRONTEND_ORIGINS", "")
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    if origins:
        return origins
    # Safe local defaults for common frontend dev servers.
    return ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"]


def create_app():
    app = Flask(__name__)
    max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "4"))
    app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024

    CORS(
        app,
        resources={r"/api/*": {"origins": _parse_origins()}},
    )

    @app.errorhandler(413)
    def payload_too_large(_):
        return jsonify({"error": f"File too large. Max allowed size is {max_upload_mb}MB."}), 413
    
    # Register blueprints
    from app.api import api_bp
    app.register_blueprint(api_bp)
    
    return app
