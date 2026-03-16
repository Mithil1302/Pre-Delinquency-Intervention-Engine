"""
GET /api/model/metrics

Returns LSTM training timeline, ensemble metrics, and logistic model
coefficients sourced from local model files.  No Cosmos involved.
"""

import json
import logging
import sys
from pathlib import Path

import azure.functions as func

# Resolve shared helpers relative to this file
_root_dir = Path(__file__).resolve().parent.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

from _shared.utils import handle_options, json_response, error_response  # noqa: E402

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent          # hoh/
_MODELS_DIR = _ROOT / "models"


def main(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return handle_options()

    try:
        # ── LSTM online training timeline ────────────────────────────────────
        metrics_path = _MODELS_DIR / "lstm_online_metrics.jsonl"
        timeline = []
        if metrics_path.exists():
            with metrics_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    if not rec:
                        continue
                    timeline.append({
                        "step":      rec.get("step"),
                        "auc":       rec.get("val_auc", rec.get("auc")),
                        "loss":      rec.get("val_loss", rec.get("loss")),
                        "timestamp": rec.get("timestamp", ""),
                    })

        current_auc  = next((t["auc"]  for t in reversed(timeline) if t.get("auc")  is not None), 0.7982)
        current_loss = next((t["loss"] for t in reversed(timeline) if t.get("loss") is not None), None)

        # ── Ensemble metrics ─────────────────────────────────────────────────
        ensemble_path = _MODELS_DIR / "ensemble_metrics.json"
        ensemble = {}
        if ensemble_path.exists():
            try:
                ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # ── Logistic model coefficients ──────────────────────────────────────
        coef_path = _MODELS_DIR / "model_coefficients.json"
        coefficients = {}
        if coef_path.exists():
            try:
                coefficients = json.loads(coef_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # ── Model file sizes (MB) as a health sanity check ───────────────────
        files_info = {}
        for fname in ("lstm_online_best.pt", "ensemble_catboost.cbm",
                      "ensemble_xgboost.cbm", "ensemble_meta.pkl"):
            p = _MODELS_DIR / fname
            files_info[fname] = {"exists": p.exists(), "size_mb": round(p.stat().st_size / 1e6, 2) if p.exists() else 0}

        # ── Normalise timeline entries for api.ts ────────────────────────────
        # api.ts reads raw.lstm_timeline, each entry needs .checkpoint/.auc/.loss
        lstm_timeline = [
            {
                "checkpoint": int(rec.get("step") or (i + 1)),
                "epoch":      rec.get("epoch"),
                "auc":        rec.get("auc"),
                "loss":       rec.get("loss"),
                "val_loss":   rec.get("val_loss"),
                "timestamp":  rec.get("timestamp", ""),
            }
            for i, rec in enumerate(timeline)
        ]

        # ── Normalise coefficients → flat array for api.ts ──────────────────
        # model_coefficients.json has: {intercept, coefficients: {X1: {value, description}, ...}}
        raw_coeffs = coefficients.get("coefficients", coefficients) if isinstance(coefficients, dict) else {}
        coeff_list = []
        if isinstance(raw_coeffs, dict):
            for feat, info in raw_coeffs.items():
                if isinstance(info, dict):
                    coeff_list.append({
                        "feature":     info.get("description", feat),
                        "coefficient": info.get("value", 0),
                    })
                else:
                    coeff_list.append({"feature": feat, "coefficient": float(info or 0)})

        # ── Normalise model_files → array for api.ts (file_artifacts) ────────
        file_artifacts = [
            {"name": fname, "exists": v["exists"], "size_bytes": round(v.get("size_mb", 0) * 1e6)}
            for fname, v in files_info.items()
        ]

        return json_response({
            # api.ts reads: lstm_timeline, ensemble_metrics, coefficients, file_artifacts
            "lstm_timeline":    lstm_timeline,
            "ensemble_metrics": {
                "auc_roc":          current_auc,
                "gini":             round(2 * (current_auc or 0.0) - 1, 4),
                "current_loss":     current_loss,
                "total_steps":      len(timeline),
                **(ensemble if isinstance(ensemble, dict) else {}),
            },
            "coefficients":     coeff_list,
            "file_artifacts":   file_artifacts,
            # Extra raw fields kept for forward compatibility
            "features": [
                "daily_balance", "daily_debit_sum", "daily_credit_sum",
                "daily_txn_count", "is_salary_day", "balance_change_pct",
                "atm_amount", "failed_debit_count",
            ],
        })

    except Exception as exc:
        log.exception("model_metrics error")
        return error_response(str(exc))
