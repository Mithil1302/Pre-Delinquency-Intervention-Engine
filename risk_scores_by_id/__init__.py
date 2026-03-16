"""
GET /api/risk-scores/{customerId}

Returns a single customer's full risk score document + a server-computed
survival curve (hazard model over months [1,3,6,12,24,36,48,60]).

Note: Cosmos docs use id = "score_{customerId}_{latest_date}".
Since the date is unknown at request time we query by partition key (fast)
rather than doing a blind point-read.
"""

import logging
import sys
from pathlib import Path

import azure.functions as func

_api_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_api_dir))

from _shared.cosmos_helper import get_container          # noqa: E402
from _shared.utils import handle_options, json_response, error_response  # noqa: E402

log = logging.getLogger(__name__)

SURVIVAL_MONTHS = [1, 3, 6, 12, 24, 36, 48, 60]


def main(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return handle_options()

    customer_id = req.route_params.get("customerId", "")
    if not customer_id:
        return error_response("customerId is required", 400)

    try:
        container = get_container("risk_scores")

        # Query within the partition — fast even without document id
        items = list(container.query_items(
            query="SELECT TOP 1 * FROM c WHERE c.customerId = @cid ORDER BY c.latest_date DESC",
            parameters=[{"name": "@cid", "value": customer_id}],
            partition_key=customer_id,
        ))

        if not items:
            return error_response(f"Customer not found: {customer_id}", 404)

        doc = items[0]

        # Compute survival curve server-side from stored hazard rate
        h = float(doc.get("hazard_rate_monthly", 0.0))
        doc["survival_curve"] = [
            {"month": m, "survival_prob": round((1.0 - h) ** m, 4)}
            for m in SURVIVAL_MONTHS
        ]

        return json_response(doc)

    except Exception as exc:
        log.exception("risk_scores_by_id error for %s", customer_id)
        return error_response(str(exc))
