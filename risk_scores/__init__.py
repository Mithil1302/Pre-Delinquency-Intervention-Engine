"""
GET /api/risk-scores

Returns all risk score documents with optional filtering, sorting, and pagination.

Query parameters:
  tier        HIGH | MEDIUM | LOW
  stage       Stage 1 | Stage 2 | Stage 3
  pd_min      float  (filter on pd_pit)
  pd_max      float
  score_min   int    (filter on credit_score)
  score_max   int
  sort_by     pd_pit | ecl_12m | credit_score | lsi   (default: pd_pit)
  sort_dir    asc | desc                               (default: desc)
  limit       int 1-1000                               (default: 1000)
  offset      int                                      (default: 0)
"""

import logging
import sys
from pathlib import Path

import azure.functions as func

# Resolve shared helpers relative to this file
_root_dir = Path(__file__).resolve().parent.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

from _shared.cosmos_helper import get_container  # noqa: E402
from _shared.utils import handle_options, json_response, error_response  # noqa: E402

log = logging.getLogger(__name__)

VALID_SORT = {"pd_pit", "ecl_12m", "credit_score", "lsi", "customerId"}


def main(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return handle_options()

    try:
        params = req.params

        # ── Parse filter params ──────────────────────────────────────────────
        tier      = params.get("tier", "").upper()
        stage     = params.get("stage", "")
        pd_min    = _float(params.get("pd_min"), 0.0)
        pd_max    = _float(params.get("pd_max"), 1.0)
        score_min = _int(params.get("score_min"), 300)
        score_max = _int(params.get("score_max"), 900)
        sort_by   = params.get("sort_by", "pd_pit")
        sort_dir  = params.get("sort_dir", "desc").upper()
        limit     = min(_int(params.get("limit"), 1000), 1000)
        offset    = _int(params.get("offset"), 0)

        # ── Sort: accept both shorthand ("sort=pd_desc") and separate params ──
        sort_shorthand = params.get("sort", "")
        SHORTHAND = {
            "id_asc":      ("customerId",   "ASC"),
            "id_desc":     ("customerId",   "DESC"),
            "pd_desc":     ("pd_pit",      "DESC"),
            "pd_asc":      ("pd_pit",      "ASC"),
            "credit_asc":  ("credit_score","ASC"),
            "credit_desc": ("credit_score","DESC"),
            "ecl_desc":    ("ecl_12m",     "DESC"),
            "ecl_asc":     ("ecl_12m",     "ASC"),
            "lsi_desc":    ("lsi",         "DESC"),
            "lsi_asc":     ("lsi",         "ASC"),
        }
        if sort_shorthand in SHORTHAND:
            sort_by, sort_dir = SHORTHAND[sort_shorthand]
        # Guard against invalid sort fields
        if sort_by not in VALID_SORT:
            sort_by = "pd_pit"
        if sort_dir not in ("ASC", "DESC"):
            sort_dir = "DESC"

        conditions = [
            "c.pd_pit  >= @pd_min",
            "c.pd_pit  <= @pd_max",
            "c.credit_score >= @score_min",
            "c.credit_score <= @score_max",
        ]
        parameters = [
            {"name": "@pd_min",    "value": pd_min},
            {"name": "@pd_max",    "value": pd_max},
            {"name": "@score_min", "value": score_min},
            {"name": "@score_max", "value": score_max},
        ]

        if tier in ("HIGH", "MEDIUM", "LOW"):
            conditions.append("c.risk_tier = @tier")
            parameters.append({"name": "@tier", "value": tier})

        if stage in ("Stage 1", "Stage 2", "Stage 3"):
            conditions.append("c.ifrs9_stage = @stage")
            parameters.append({"name": "@stage", "value": stage})

        # ── Fetch all matching items, then sort + paginate in Python ─────────
        # Cosmos cross-partition ORDER BY requires a composite index.
        # Python-side sort is safe and fast for ≤ 1 000 documents.
        where  = " AND ".join(conditions)
        data_q = f"SELECT * FROM c WHERE {where}"
        items = list(get_container("risk_scores").query_items(
            query=data_q,
            parameters=parameters,
            enable_cross_partition_query=True,
        ))

        # Sort — handle both string and numeric keys
        reverse = sort_dir == "DESC"
        if sort_by == "customerId":
            items.sort(key=lambda x: str(x.get("customerId") or ""), reverse=reverse)
        else:
            items.sort(key=lambda x: x.get(sort_by, 0) or 0, reverse=reverse)

        # Count before slicing (total = number filtered, not just current page)
        total = len(items)
        items = items[offset : offset + limit]

        return json_response({"data": items, "total": total, "limit": limit, "offset": offset})

    except Exception as exc:
        log.exception("risk_scores error")
        return error_response(str(exc))


def _float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default
