"""
GET /api/customers?ids=CUST_000001,CUST_000002,...

Returns demographic + account data for up to 50 customers from the
`customers` Cosmos container.  Used by the Watch List drawer panel.

Customer documents use camelCase field names:
  monthlyIncome, currentBalance, creditLimit, creditUtilizationPct, accountType, ...
"""

import logging
import sys
from pathlib import Path

import azure.functions as func

# Resolve shared helpers relative to this file
_root_dir = Path(__file__).resolve().parent.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

from _shared.cosmos_helper import get_container          # noqa: E402
from _shared.utils import handle_options, json_response, error_response  # noqa: E402

log = logging.getLogger(__name__)


def main(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return handle_options()

    try:
        raw_ids = req.params.get("ids", "").strip()
        if not raw_ids:
            return error_response("ids query parameter is required (comma-separated)", 400)

        ids = [i.strip() for i in raw_ids.split(",") if i.strip()][:50]
        if not ids:
            return json_response({"customers": []})

        container = get_container("customers")

        results = list(container.query_items(
            query="SELECT * FROM c WHERE ARRAY_CONTAINS(@ids, c.customerId)",
            parameters=[{"name": "@ids", "value": ids}],
            enable_cross_partition_query=True,
        ))

        # Normalise output field names for the frontend
        customers = [_normalise(doc) for doc in results]

        return json_response(customers)  # return array directly — frontend uses .map()

    except Exception as exc:
        log.exception("customers error")
        return error_response(str(exc))


def _normalise(doc: dict) -> dict:
    """Return a clean subset of customer fields."""
    return {
        "customerId":          doc.get("customerId", ""),
        "customer_name":       doc.get("name", ""),
        "age":                 doc.get("age", 0),
        "monthly_income":      doc.get("monthlyIncome", 0),
        "employment_status":   doc.get("employmentStatus", ""),
        "city":                doc.get("city", ""),
        "state":               doc.get("state", ""),
        "customer_tier":       doc.get("customerTier", ""),
        "risk_tolerance":      doc.get("riskProfile", ""),
        "account_type":        doc.get("accountType", ""),
        "current_balance":     doc.get("currentBalance", 0),
        "credit_limit":        doc.get("creditLimit", 0),
        "credit_utilization":  doc.get("creditUtilizationPct", 0),
        "debt_to_income":      doc.get("debtToIncomeRatio", 0),
        "nsf_count_12m":       doc.get("nsfCount12m", 0),
        "became_delinquent":   doc.get("becameDelinquent", False),
    }
