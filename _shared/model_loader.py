"""
model_loader.py — Singleton OnlineBiLSTM loader for Azure Functions.

The model is loaded once at module-import time (cold start), then reused
across all invocations of the same worker process.  Never call torch.load
per request — it is expensive and not thread-safe without care.
"""

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger(__name__)

_ROOT       = Path(__file__).resolve().parent.parent   # hoh/
_MODEL_PATH = _ROOT / "models" / "lstm_online_best.pt"
_SCRIPTS    = _ROOT / "scripts"

_model = None
_device = None


def _load_model_class():
    """Load OnlineBiLSTM from scripts/train_lstm_online.py via importlib."""
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    train_path = _SCRIPTS / "train_lstm_online.py"
    spec = importlib.util.spec_from_file_location("train_lstm_online", train_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["train_lstm_online"] = m
    spec.loader.exec_module(m)
    return m.OnlineBiLSTM


def get_model() -> Tuple[Optional[object], Optional[object]]:
    """
    Return (model, device).  Returns (None, None) when the checkpoint is
    absent or fails to load — callers fall back to a heuristic PD scorer.
    """
    global _model, _device
    if _model is not None:
        return _model, _device

    if not _MODEL_PATH.exists():
        log.warning("LSTM checkpoint not found at %s — using heuristic scorer", _MODEL_PATH)
        return None, None

    try:
        import torch
        _device = torch.device("cpu")

        OnlineBiLSTM = _load_model_class()
        model = OnlineBiLSTM()

        # weights_only=False: checkpoint was saved with custom class objects
        ckpt = torch.load(str(_MODEL_PATH), map_location=_device, weights_only=False)

        if isinstance(ckpt, dict):
            # Priority of checkpoint keys (update_cosmos.py uses "model_state")
            for key in ("model_state", "state_dict", "model_state_dict"):
                if key in ckpt:
                    model.load_state_dict(ckpt[key])
                    break
            else:
                # If none of the known keys match, try loading ckpt directly
                model.load_state_dict(ckpt)
        else:
            model.load_state_dict(ckpt)

        model.eval()
        _model = model
        log.info("OnlineBiLSTM loaded from %s", _MODEL_PATH)
        return _model, _device

    except Exception as exc:
        log.warning("Could not load LSTM model (%s) — using heuristic scorer", exc)
        return None, None
