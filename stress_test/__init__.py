"""
GET /api/risk-scores/{customerId}/stress

Re-runs layers 3–7 of the risk engine for all 3 macro scenarios without
re-running the LSTM (uses stored pd_pit + lsi from Cosmos).

Returns side-by-side scenario comparison with deltas vs the current scenario.
"""

import logging
import math
import sys
from pathlib import Path

import azure.functions as func

# Resolve shared helpers relative to this file
_root_dir = Path(__file__).resolve().parent.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

# engine_loader is loaded at module level by the shared package
from _shared.engine_loader import (                     # noqa: E402
    compute_macro_index,
    compute_macro_adj_pd,
    compute_survival_pd,
    compute_credit_score,
    log_odds,
)
from _shared.cosmos_helper import get_container         # noqa: E402
from _shared.utils import handle_options, json_response, error_response  # noqa: E402

log = logging.getLogger(__name__)

SCENARIOS = ["current", "mild_stress", "severe_stress"]
TTC_ALPHA = 0.70


def _compute_ttc_pd(pd_pit: float, pd_hist: float = 0.08) -> float:
    """Blend PIT with long-run historical average (Layer 4)."""
    return float(max(0.0, min(1.0, TTC_ALPHA * pd_pit + (1.0 - TTC_ALPHA) * pd_hist)))


def main(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return handle_options()

    customer_id = req.route_params.get("customerId", "")
    if not customer_id:
        return error_response("customerId is required", 400)

    try:
        # Read stored risk score (pd_pit, lsi, lgd, ead already computed)
        items = list(get_container("risk_scores").query_items(
            query="SELECT TOP 1 * FROM c WHERE c.customerId = @cid ORDER BY c.latest_date DESC",
            parameters=[{"name": "@cid", "value": customer_id}],
            partition_key=customer_id,
        ))
        if not items:
            return error_response(f"Customer not found: {customer_id}", 404)

        doc     = items[0]
        pd_pit  = float(doc.get("pd_pit",       0.20))
        lsi     = float(doc.get("lsi",           0.40))
        lgd     = float(doc.get("lgd",           0.45))
        ead     = float(doc.get("ead",       10000.0))
        pd_hist = float(doc.get("pd_historical", 0.08))
        lo_pit  = log_odds(pd_pit)

        results = {}
        for scenario in SCENARIOS:
            macro_idx = compute_macro_index(scenario)
            pd_adj    = compute_macro_adj_pd(lo_pit, lsi, macro_idx)
            pd_ttc    = _compute_ttc_pd(pd_adj, pd_hist)
            h, pd_3m, pd_12m  = compute_survival_pd(pd_adj, 12)
            _, _,     pd_60m  = compute_survival_pd(pd_adj, 60)
            ecl_12m   = round(pd_12m * lgd * ead, 2)
            ecl_life  = round(pd_60m * lgd * ead, 2)
            credit_sc = compute_credit_score(pd_ttc)

            results[scenario] = {
                "macro_index":   round(macro_idx, 4),
                "pd_macro_adj":  round(pd_adj, 4),
                "pd_ttc":        round(pd_ttc, 4),
                "pd_3m":         round(pd_3m, 4),
                "pd_12m":        round(pd_12m, 4),
                "ecl_12m":       ecl_12m,
                "ecl_lifetime":  ecl_life,
                "credit_score":  credit_sc,
                "delta_ecl_pct": None,
            }

        # Compute deltas relative to "current" baseline
        base_ecl = results["current"]["ecl_12m"]
        for sc in ("mild_stress", "severe_stress"):
            sc_ecl = results[sc]["ecl_12m"]
            results[sc]["delta_ecl_pct"] = round(
                (sc_ecl - base_ecl) / (base_ecl + 1e-6) * 100, 1
            )

        return json_response({"customerId": customer_id, **results})

    except Exception as exc:
        log.exception("stress_test error for %s", customer_id)
        return error_response(str(exc))
