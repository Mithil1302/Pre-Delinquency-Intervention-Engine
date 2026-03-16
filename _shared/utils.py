"""
utils.py — Shared HTTP response helpers for all Azure Function endpoints.
"""

import json
import azure.functions as func

CORS_HEADERS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-API-Key",
}


def handle_options() -> func.HttpResponse:
    """Respond to CORS preflight OPTIONS requests."""
    return func.HttpResponse(status_code=200, headers=CORS_HEADERS)


def json_response(data, status: int = 200) -> func.HttpResponse:
    """Return a JSON HttpResponse with CORS headers."""
    return func.HttpResponse(
        body=json.dumps(data, default=str),
        status_code=status,
        mimetype="application/json",
        headers=CORS_HEADERS,
    )


def error_response(message: str, status: int = 500) -> func.HttpResponse:
    return json_response({"error": message}, status)
