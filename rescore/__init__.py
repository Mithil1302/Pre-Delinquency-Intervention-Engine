"""
POST /api/score/{customerId}

Full 7-layer re-inference for a single customer:
  1. Read sequence from daily_sequences.jsonl (JSONL file, not Cosmos)
  2. Read customer context from `customers` Cosmos container
  3. Normalize sequence and run LSTM inference  → pd_pit
  4. Call enrich_risk_score() for all 7 layers
  5. Apply fixed tier thresholds from last population calibration run
  6. Upsert updated document to `risk_scores` container
  7. Return new profile + diff vs previous score

Returns diff even when the score has not changed (delta = 0).
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import azure.functions as func

# Resolve shared helpers relative to this file
_root_dir = Path(__file__).resolve().parent.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

from _shared.cosmos_helper import get_container                            # noqa: E402
from _shared.engine_loader import enrich_risk_score, normalize_features    # noqa: E402
from _shared.model_loader import get_model                                  # noqa: E402
from _shared.utils import handle_options, json_response, error_response     # noqa: E402

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent          # hoh/
_SEQUENCES_PATH = _ROOT / "data" / "stream" / "daily_sequences.jsonl"

# Fixed percentile thresholds from last calibrate_tiers() run over 1,000 customers
HIGH_THRESHOLD   = 0.268   # pd_macro_adj >= HIGH  → "HIGH"
MEDIUM_THRESHOLD = 0.203   # pd_macro_adj >= MEDIUM → "MEDIUM"

FEAT_KEYS = [
    "daily_balance", "daily_debit_sum", "daily_credit_sum",
    "daily_txn_count", "is_salary_day", "balance_change_pct",
    "atm_amount", "failed_debit_count",
]


def main(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return handle_options()

    customer_id = req.route_params.get("customerId", "")
    if not customer_id:
        return error_response("customerId is required", 400)

    try:
        scores_container = get_container("risk_scores")

        # ── 0. Capture previous score for diff ───────────────────────────────
        previous_items = list(scores_container.query_items(
            query="SELECT TOP 1 * FROM c WHERE c.customerId = @cid ORDER BY c.latest_date DESC",
            parameters=[{"name": "@cid", "value": customer_id}],
            partition_key=customer_id,
        ))
        previous = previous_items[0] if previous_items else {}

        # ── 1. Load raw sequence from daily_sequences.jsonl ──────────────────
        raw_window = _load_sequence(customer_id)
        if raw_window is None:
            # Fallback: try customer_features Cosmos container
            raw_window = _load_sequence_from_cosmos(customer_id)
        if raw_window is None:
            # As a last resort, synthesize a stable neutral sequence so the
            # dashboard doesn't 404 on missing demo data. This keeps the flow
            # working while clearly logging the gap.
            log.warning("No sequence data found for %s — using synthetic neutral window", customer_id)
            raw_window = _synthetic_sequence()

        # ── 2. Load customer context ─────────────────────────────────────────
        try:
            cust_doc = get_container("customers").read_item(
                item=customer_id, partition_key=customer_id
            )
        except Exception:
            cust_doc = {}

        monthly_income         = float(cust_doc.get("monthlyIncome",        30000.0))
        current_balance        = float(cust_doc.get("currentBalance",       10000.0))
        credit_limit           = float(cust_doc.get("creditLimit",          50000.0))
        credit_utilization_pct = float(cust_doc.get("creditUtilizationPct",    0.20))
        account_type           = str(cust_doc.get("accountType", "savings")).lower()
        pd_historical          = float(cust_doc.get("trueDefaultProb",         0.08))

        # ── 3. LSTM inference ────────────────────────────────────────────────
        model, device = get_model()
        if model is not None:
            pd_pit = _lstm_infer(model, device, raw_window)
            model_version = "lstm_online_best"
        else:
            pd_pit = _heuristic_pd(raw_window)
            model_version = "heuristic"

        # ── 4. 7-layer enrichment (raw window, NOT normalized) ───────────────
        scored_at = datetime.now(timezone.utc).isoformat()
        profile = enrich_risk_score(
            pd_pit                 = pd_pit,
            raw_window             = raw_window,   # LSI uses raw balance values
            monthly_income         = monthly_income,
            current_balance        = current_balance,
            credit_limit           = credit_limit,
            credit_utilization_pct = credit_utilization_pct,
            pd_historical          = pd_historical,
            macro_scenario         = "current",
            account_type           = account_type,
            customer_id            = customer_id,
            seq_len                = len(raw_window),
            model_version          = model_version,
            scored_at              = scored_at,
        )

        # ── 5. Tier assignment using fixed population thresholds ─────────────
        if profile.pd_macro_adj >= HIGH_THRESHOLD:
            profile.risk_tier = "HIGH"
        elif profile.pd_macro_adj >= MEDIUM_THRESHOLD:
            profile.risk_tier = "MEDIUM"
        else:
            profile.risk_tier = "LOW"
        profile.risk_tier_ttc = profile.risk_tier   # simplified single-customer

        # ── 6. Upsert to Cosmos ──────────────────────────────────────────────
        today = datetime.now(timezone.utc).date().isoformat()
        doc = profile.to_cosmos_dict()
        doc["id"]          = f"score_{customer_id}_{today}"
        doc["customerId"]  = customer_id
        doc["latest_date"] = today
        scores_container.upsert_item(doc)

        # ── 7. Build diff ────────────────────────────────────────────────────
        diff = _build_diff(previous, doc)

        return json_response({
            "previous": _prev_summary(previous),
            "current":  doc,
            "diff":     diff,
        })

    except Exception as exc:
        log.exception("rescore error for %s", customer_id)
        return error_response(str(exc))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_sequence(customer_id: str):
    """Read most recent 30 rows from daily_sequences.jsonl for a customer."""
    if not _SEQUENCES_PATH.exists():
        return None
    rows = []
    with _SEQUENCES_PATH.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("customerId") == customer_id:
                row = [float(rec.get(k, 0.0)) for k in FEAT_KEYS]
                rows.append((rec.get("date", ""), row))
    if not rows:
        return None
    rows.sort(key=lambda x: x[0])
    window = [r for _, r in rows[-30:]]
    return np.array(window, dtype=np.float32)


def _load_sequence_from_cosmos(customer_id: str):
    """Fallback: load sequence from customer_features Cosmos container."""
    try:
        item = get_container("customer_features").read_item(
            item=f"{customer_id}_latest", partition_key=customer_id
        )
        feats = item.get("features") or item.get("sequence")
        if feats:
            return np.array(feats, dtype=np.float32)
    except Exception:
        pass
    return None


def _synthetic_sequence(seq_len: int = 30) -> np.ndarray:
    """
    Build a neutral, stable sequence so rescoring can proceed even when
    no raw behavioral data exists for a demo customer. Values are mild and
    non-zero to avoid divide-by-zero in heuristic PD.
    """
    row = [
        20000.0,   # daily_balance
        2500.0,    # daily_debit_sum
        2800.0,    # daily_credit_sum
        4.0,       # daily_txn_count
        0.0,       # is_salary_day
        0.0,       # balance_change_pct
        400.0,     # atm_amount
        0.0,       # failed_debit_count
    ]
    return np.tile(np.array(row, dtype=np.float32), (seq_len, 1))


def _lstm_infer(model, device, raw_window: np.ndarray) -> float:
    """Normalize, pad to 30 steps, run LSTM, return scalar PD."""
    import torch
    normed = normalize_features(raw_window)            # (T, 8)
    if len(normed) < 30:
        pad = np.zeros((30 - len(normed), 8), dtype=np.float32)
        normed = np.vstack([pad, normed])
    tensor = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(model(tensor).squeeze().item())


def _heuristic_pd(raw_window: np.ndarray) -> float:
    """Simple heuristic when LSTM is unavailable."""
    w = raw_window
    if len(w) == 0:
        return 0.15
    recent = [float(r[0]) for r in w[-7:]]
    trend  = (recent[-1] - recent[0]) / (abs(recent[0]) + 1)
    dratio = sum(float(r[1]) for r in w[-7:]) / (sum(float(r[0]) for r in w[-7:]) + 1)
    return float(np.clip(0.10 + dratio * 0.4 - trend * 0.3, 0.01, 0.99))


def _build_diff(prev: dict, curr: dict) -> dict:
    diff = {}
    for field in ("pd_pit", "credit_score", "ecl_12m", "lsi", "pd_macro_adj"):
        old = prev.get(field)
        new = curr.get(field)
        if old is None or new is None:
            diff[field] = {"old": old, "new": new}
            continue
        delta = round(new - old, 4) if isinstance(new, float) else new - old
        entry = {"old": old, "new": new, "delta": delta}
        if isinstance(new, float) and old != 0:
            entry["delta_pct"] = round(delta / abs(old) * 100, 1)
        diff[field] = entry

    # Tier change
    old_tier = prev.get("risk_tier")
    new_tier  = curr.get("risk_tier")
    diff["risk_tier"] = {"old": old_tier, "new": new_tier, "changed": old_tier != new_tier}
    return diff


def _prev_summary(prev: dict) -> dict:
    if not prev:
        return {}
    return {k: prev.get(k) for k in ("pd_pit", "credit_score", "ecl_12m", "risk_tier", "lsi")}
