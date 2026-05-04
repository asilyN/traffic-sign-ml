"""Lightweight prediction errors — safe to import at Flask startup."""


class PredictionError(Exception):
    """Raised when prediction cannot be completed."""
