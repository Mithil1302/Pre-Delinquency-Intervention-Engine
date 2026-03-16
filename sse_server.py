"""
sse_server.py — Real-time Server-Sent Events bridge for the RiskGuard Dashboard.

Monitors the Cosmos DB risk_scores container every 3 seconds.
When it detects that any customer's scored_at timestamp has changed it
broadcasts a 'scores_updated' event to every connected browser client.
The frontend React Query cache is then invalidated and all charts/tables
re-render automatically — no page refresh required.

Start this server (after `conda activate base` or your venv):

    uvicorn sse_server:app --host 0.0.0.0 --port 7072

Keep the Azure Functions host running on port 7071 at the same time.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ── Resolve project root so _shared helpers are importable ───────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from _shared.cosmos_helper import get_container  # noqa: E402

log = logging.getLogger("sse_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="RiskGuard SSE Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    # Allow the Vite dev server and any common local ports
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

# ── Global state ─────────────────────────────────────────────────────────────
# One asyncio.Queue per connected client; items are already-serialised SSE lines.
_clients: list[asyncio.Queue] = []

# Last known snapshot so newly connecting clients get an immediate update.
_snapshot: dict = {}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sse_line(event: dict) -> str:
    """Format a Python dict as a valid SSE 'data:' line."""
    return f"data: {json.dumps(event, default=str)}\n\n"


def _broadcast(event: dict) -> None:
    """Push an already-built event dict to every connected client's queue."""
    line = _sse_line(event)
    dead: list[asyncio.Queue] = []
    for q in _clients:
        try:
            q.put_nowait(line)
        except asyncio.QueueFull:
            # Slow client — mark for removal to avoid memory leak
            dead.append(q)
    for q in dead:
        try:
            _clients.remove(q)
        except ValueError:
            pass


# ── Background tasks ─────────────────────────────────────────────────────────

async def _monitor_cosmos() -> None:
    """
    Poll the risk_scores container every 3 seconds.
    Only fetches the four fields needed for change detection — avoids pulling
    full documents (keeps the query fast even at 1,000 records).
    When any customer's scored_at has advanced we broadcast 'scores_updated'.
    """
    global _snapshot
    log.info("Cosmos monitor started — polling every 3 s")

    while True:
        try:
            container = get_container("risk_scores")

            items = list(container.query_items(
                query="""
                    SELECT
                        c.customerId,
                        c.risk_tier,
                        c.scored_at,
                        c.pd_pit
                    FROM c
                """,
                enable_cross_partition_query=True,
            ))

            if items:
                latest_scored_at = max(
                    (i.get("scored_at") or "") for i in items
                )
                total      = len(items)
                high_count = sum(1 for i in items if i.get("risk_tier") == "HIGH")

                prev_scored_at = _snapshot.get("latest_scored_at", "")

                if latest_scored_at != prev_scored_at:
                    # Diff: which customers have a new scored_at vs last snapshot
                    prev_map: dict[str, str] = _snapshot.get("scored_at_map", {})
                    changed = [
                        i["customerId"]
                        for i in items
                        if (i.get("scored_at") or "") != prev_map.get(i.get("customerId", ""), "")
                    ]

                    scored_at_map = {
                        i.get("customerId", ""): (i.get("scored_at") or "")
                        for i in items
                    }

                    _snapshot = {
                        "latest_scored_at": latest_scored_at,
                        "total":            total,
                        "high_count":       high_count,
                        "scored_at_map":    scored_at_map,
                    }

                    event = {
                        "type":              "scores_updated",
                        "latest_scored_at":  latest_scored_at,
                        "total":             total,
                        "high_count":        high_count,
                        # Cap at 20 IDs so the SSE payload stays small
                        "changed_customers": changed[:20],
                        "ts":                _now_iso(),
                    }
                    _broadcast(event)
                    log.info(
                        "scores_updated  changed=%d  total=%d  high=%d  clients=%d",
                        len(changed), total, high_count, len(_clients),
                    )

        except Exception as exc:
            log.warning("Cosmos poll error: %s", exc)

        await asyncio.sleep(3)


async def _heartbeat() -> None:
    """
    Send a heartbeat comment every 15 seconds.
    This prevents proxies and browsers from timing out idle SSE connections.
    """
    while True:
        await asyncio.sleep(15)
        _broadcast({"type": "heartbeat", "ts": _now_iso()})


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(_monitor_cosmos())
    asyncio.create_task(_heartbeat())
    log.info("SSE server ready — listening for Cosmos changes")


# ── Per-client event generator ───────────────────────────────────────────────

async def _event_generator(queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    """
    Yields SSE lines for one client until it disconnects.
    Sends an immediate 'connected' event so the browser knows the stream is live,
    then forwards everything from the shared queue.
    """
    _clients.append(queue)
    log.info("Client connected  total=%d", len(_clients))

    try:
        # 1. Immediate 'connected' acknowledgement
        yield _sse_line({"type": "connected", "ts": _now_iso()})

        # 2. Send current snapshot so a fresh page load shows up-to-date status
        if _snapshot:
            yield _sse_line({
                "type":             "snapshot",
                "latest_scored_at": _snapshot.get("latest_scored_at", ""),
                "total":            _snapshot.get("total", 0),
                "high_count":       _snapshot.get("high_count", 0),
                "ts":               _now_iso(),
            })

        # 3. Stream live events
        while True:
            try:
                # wait_for with a 30 s timeout so we can send keep-alive comments
                # even when no real events have occurred
                line = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield line
            except asyncio.TimeoutError:
                # An SSE comment (': ...\n\n') is never shown to .onmessage
                # but it resets the browser's connection-timeout timer.
                yield ": keep-alive\n\n"

    except asyncio.CancelledError:
        pass
    finally:
        try:
            _clients.remove(queue)
        except ValueError:
            pass
        log.info("Client disconnected  total=%d", len(_clients))


# ── HTTP endpoints ────────────────────────────────────────────────────────────

@app.get("/events")
async def events(request: Request) -> StreamingResponse:
    """
    SSE endpoint.  Browser connects once; events are pushed as they occur.
    EventSource auto-reconnects if the connection drops.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    return StreamingResponse(
        _event_generator(queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control":       "no-cache, no-store",
            "X-Accel-Buffering":   "no",    # disable nginx/proxy buffering
            "Connection":          "keep-alive",
            "Transfer-Encoding":   "chunked",
        },
    )


@app.get("/health")
async def health() -> dict:
    """Quick health check — useful for the TopBar status probe."""
    return {
        "status":            "ok",
        "connected_clients": len(_clients),
        "latest_scored_at":  _snapshot.get("latest_scored_at", ""),
        "total_customers":   _snapshot.get("total", 0),
        "high_risk":         _snapshot.get("high_count", 0),
        "ts":                _now_iso(),
    }
