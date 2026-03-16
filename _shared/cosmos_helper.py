"""
cosmos_helper.py — Singleton CosmosDB client for Azure Functions.

Uses connection_verify=False because the local emulator uses a self-signed cert.
Never create a new CosmosClient per request — this module-level singleton is
reused across all invocations of the same worker process.
"""

import os
from azure.cosmos import CosmosClient

_client = None
_db = None

ENDPOINT = os.environ.get("COSMOS_DB_ENDPOINT", "https://localhost:8081")
KEY      = os.environ.get(
    "COSMOS_DB_KEY",
    "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw=="
)
DB_NAME  = "delinquency_prevention"


def get_db():
    """Return a singleton database client."""
    global _client, _db
    if _db is None:
        _client = CosmosClient(ENDPOINT, credential=KEY, connection_verify=False)
        _db = _client.get_database_client(DB_NAME)
    return _db


def get_container(name: str):
    """Return a container client by name."""
    return get_db().get_container_client(name)
