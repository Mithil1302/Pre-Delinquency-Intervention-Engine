"""
engine_loader.py — Load risk_engine.py via importlib (bypasses src/shared/__init__.py
which requires the optional `opencensus` package).

Also injects scripts/ into sys.path so stream_processor can be imported normally.

All exported names are module-level — loaded once at cold start.
"""

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent   # project root (hoh/)

# ── 1. Load risk_engine ──────────────────────────────────────────────────────

_engine_path = _ROOT / "src" / "shared" / "risk_engine.py"

def _load_risk_engine():
    spec = importlib.util.spec_from_file_location("hoh_risk_engine", _engine_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["hoh_risk_engine"] = m   # required so @dataclass pickling works
    spec.loader.exec_module(m)
    return m

_engine = _load_risk_engine()

# Public symbols from risk_engine
enrich_risk_score    = _engine.enrich_risk_score
calibrate_tiers      = _engine.calibrate_tiers
assign_tiers         = _engine.assign_tiers
compute_macro_index  = _engine.compute_macro_index
compute_macro_adj_pd = _engine.compute_macro_adj_pd
compute_survival_pd  = _engine.compute_survival_pd
compute_credit_score = _engine.compute_credit_score
log_odds             = _engine.log_odds
MACRO_SCENARIOS      = _engine.MACRO_SCENARIOS


# ── 2. Load stream_processor.normalize_features ──────────────────────────────

_scripts_dir = _ROOT / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

try:
    from stream_processor import normalize_features  # type: ignore
except ImportError:
    import numpy as np

    def normalize_features(arr):   # type: ignore[misc]
        """Inline fallback if stream_processor is unavailable."""
        feat = arr.copy().astype("float32")
        for i in (0, 1, 2, 6):
            feat[:, i] = np.log1p(np.maximum(0.0, feat[:, i]))
        feat[:, 3] = np.sqrt(np.maximum(0.0, feat[:, 3]))
        feat[:, 5] = np.clip(feat[:, 5], -2.0, 2.0)
        return feat
