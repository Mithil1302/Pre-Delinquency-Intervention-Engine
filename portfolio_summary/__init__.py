"""
GET /api/portfolio/summary?scenario=current

Aggregate summary for the Dashboard KPI strip, donut charts, IFRS 9 ladder,
and scenario bar chart.  Aggregation is done Python-side after fetching a
lightweight projection so we avoid Cosmos emulator AVG/SUM GROUP BY limits.

Scenario stress multipliers are computed using compute_macro_index from risk_engine.
"""

import logging
import sys
from collections import defaultdict
from pathlib import Path

import azure.functions as func

# Resolve shared helpers relative to this file
_root_dir = Path(__file__).resolve().parent.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

from _shared.cosmos_helper import get_container          # noqa: E402
from _shared.engine_loader import compute_macro_index    # noqa: E402
from _shared.utils import handle_options, json_response, error_response  # noqa: E402

log = logging.getLogger(__name__)

VALID_SCENARIOS = {"current", "mild_stress", "severe_stress"}
PD_BUCKETS = [
    ("0.10-0.15", 0.10, 0.15),
    ("0.15-0.20", 0.15, 0.20),
    ("0.20-0.25", 0.20, 0.25),
    ("0.25-0.30", 0.25, 0.30),
    ("0.30-0.35", 0.30, 0.35),
    ("0.35+",     0.35, 1.00),
]


def main(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return handle_options()

    try:
        scenario = req.params.get("scenario", "current")
        if scenario not in VALID_SCENARIOS:
            scenario = "current"

        container = get_container("risk_scores")

        # ── Single lightweight projection query ───────────────────────────────
        # Fetch only the fields we need — avoids GROUP BY aggregate limitations
        # on Cosmos emulator while still being fast for ~1 000 documents.
        items = list(container.query_items(
            query="""
                SELECT
                  c.customerId, c.risk_tier, c.ifrs9_stage,
                  c.pd_pit, c.pd_ttc, c.credit_score,
                  c.lsi, c.ecl_12m, c.ecl_lifetime, c.scored_at
                FROM c
            """,
            enable_cross_partition_query=True,
        ))

        if not items:
            return json_response({
                "portfolio": {"total_customers": 0, "total_ecl_12m": 0,
                              "total_ecl_lifetime": 0, "avg_pd_pit": 0,
                              "avg_pd_ttc": 0, "avg_credit_score": 0,
                              "avg_lsi": 0, "last_scored": ""},
                "tiers": {}, "ifrs9": {}, "scenarios": {}, "pd_histogram": [],
            })

        # ── Python-side aggregation ──────────────────────────────────────────
        tier_groups: dict[str, list] = defaultdict(list)
        stage_groups: dict[str, list] = defaultdict(list)
        last_scored = ""

        for item in items:
            tier  = item.get("risk_tier",  "LOW")
            stage = item.get("ifrs9_stage", "Stage 1")
            tier_groups[tier].append(item)
            stage_groups[stage].append(item)
            scored_at = item.get("scored_at", "")
            if scored_at > last_scored:
                last_scored = scored_at

        total_customers    = len(items)
        total_ecl_12m      = sum(i.get("ecl_12m",      0.0) for i in items)
        total_ecl_lifetime = sum(i.get("ecl_lifetime",  0.0) for i in items)
        avg_pd_pit    = _mean(items, "pd_pit")
        avg_pd_ttc    = _mean(items, "pd_ttc")
        avg_cs        = _mean(items, "credit_score")
        avg_lsi       = _mean(items, "lsi")

        # Tier breakdown
        tiers = {}
        for tier, group in tier_groups.items():
            tiers[tier] = {
                "count":              len(group),
                "total_ecl_12m":      round(sum(d.get("ecl_12m",      0) for d in group), 2),
                "total_ecl_lifetime": round(sum(d.get("ecl_lifetime",  0) for d in group), 2),
                "avg_pd":             round(_mean(group, "pd_pit"), 4),
            }

        # IFRS 9 stage breakdown
        stages = {}
        for stage, group in stage_groups.items():
            pct = round(len(group) / total_customers * 100, 1)
            stages[stage] = {
                "count":     len(group),
                "total_ecl": round(sum(d.get("ecl_12m", 0) for d in group), 2),
                "pct":       pct,
            }

        # ── Scenario ECL estimates ────────────────────────────────────────────
        base_macro    = compute_macro_index("current")
        scenarios_out = {}
        for sc in VALID_SCENARIOS:
            sc_macro = compute_macro_index(sc)
            factor   = 1.0 + (sc_macro - base_macro) * 2.5
            sc_ecl   = round(total_ecl_12m      * factor, 2)
            sc_ecl_lt = round(total_ecl_lifetime * factor, 2)
            sc_pd    = round(avg_pd_pit * factor, 4)
            delta    = round((factor - 1.0) * 100, 1) if sc != "current" else None
            scenarios_out[sc] = {
                "total_ecl_12m":      sc_ecl,
                "total_ecl_lifetime": sc_ecl_lt,
                "avg_pd":             sc_pd,
                "delta_pct":          delta,
            }

        # ── PD histogram (Python-side bucketing) ─────────────────────────────
        pd_histogram = []
        for label, lo, hi in PD_BUCKETS:
            count = sum(1 for i in items if lo <= i.get("pd_pit", 0) < hi)
            pd_histogram.append({"bucket": label, "count": count})

        return json_response({
            "portfolio": {
                "total_customers":    total_customers,
                "total_ecl_12m":      round(total_ecl_12m, 2),
                "total_ecl_lifetime": round(total_ecl_lifetime, 2),
                "avg_pd_pit":         round(avg_pd_pit, 4),
                "avg_pd_ttc":         round(avg_pd_ttc, 4),
                "avg_credit_score":   round(avg_cs),
                "avg_lsi":            round(avg_lsi, 4),
                "last_scored":        last_scored,
            },
            "tiers":     tiers,
            "ifrs9":     stages,
            "scenarios": scenarios_out,
            "pd_histogram": pd_histogram,
        })

    except Exception as exc:
        log.exception("portfolio_summary error")
        return error_response(str(exc))


def _mean(items: list, key: str) -> float:
    """Safe mean of a numeric field across a list of dicts."""
    vals = [float(i.get(key, 0.0)) for i in items if i.get(key) is not None]
    return sum(vals) / len(vals) if vals else 0.0
