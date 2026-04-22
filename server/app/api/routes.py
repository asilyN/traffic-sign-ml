from flask import jsonify, request
from app.api import api_bp

@api_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Backend is running'})

@api_bp.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint for traffic sign recognition"""
    # TODO: Implement model prediction logic
    return jsonify({'error': 'Not implemented yet'}), 501
