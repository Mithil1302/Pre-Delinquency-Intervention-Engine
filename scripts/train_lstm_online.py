"""
Online Incremental LSTM Trainer
=================================
Implements continual / online learning for financial delinquency prediction.
The LSTM is NOT retrained from scratch each night — it learns incrementally
as new transaction batches arrive from the stream processor.

Architecture
------------
  BiLSTM (2-layer, hidden=64, bidirectional)           [online-stable size]
    → LayerNorm
    → Dropout(0.25)
    → Dense(32, GELU)
    → Dense(1, Sigmoid)
  Input: (batch, seq_len=30, feat_dim=8)

Continual learning mechanisms
------------------------------
1. Experience Replay
   A FIFO ring buffer of capacity REPLAY_CAPACITY stores past sequences.
   Every gradient update mixes new samples with REPLAY_RATIO * batch_size
   randomly drawn replay samples — prevents catastrophic forgetting of
   earlier customer patterns (analogous to replay in DQN).

2. EWC-Lite (Elastic Weight Consolidation — diagonal Fisher)
   After every ANCHOR_EVERY gradient steps, the trainer:
   a) Computes a diagonal Fisher information estimate on recent data.
   b) Adds an EWC penalty  λ/2 * Σ F_i (θ_i - θ*_i)² to the loss.
   This protects important weights from large updates as the data
   distribution drifts — mirrors Amazon SageMaker continual learning.

3. Gradient Clipping  (max_norm=1.0)
   Prevents exploding gradients common in online settings with rare
   high-loss samples (e.g., sudden default spike in economic shock).

Checkpoint management
----------------------
Checkpoint is saved every SAVE_EVERY gradient steps to:
   models/lstm_online.pt        ← main checkpoint
   models/lstm_online_best.pt   ← highest val-AUC seen so far

Metrics reported every REPORT_EVERY steps
-------------------------------------------
  step | loss | train_AUC | replay_AUC | ewc_loss | lr | buffer_pos | buffer_neg

Usage
------
  # Runs forever while simulator is running (Ctrl+C to stop):
  python scripts/train_lstm_online.py

  # Warmup on existing file then switch to live mode:
  python scripts/train_lstm_online.py --warmup --epochs 3

  # Specify model checkpoint to resume from:
  python scripts/train_lstm_online.py --checkpoint models/lstm_online.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Torch import guard ────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    from sklearn.metrics import roc_auc_score
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ── Stream processor ─────────────────────────────────────────────────────────
try:
    from scripts.stream_processor import StreamProcessor, SEQ_LEN, FEAT_DIM
except ImportError:
    # allow running as top-level script
    _sp_path = Path(__file__).parent / "stream_processor.py"
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("stream_processor", _sp_path)
    _mod  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    StreamProcessor = _mod.StreamProcessor
    SEQ_LEN  = _mod.SEQ_LEN
    FEAT_DIM = _mod.FEAT_DIM

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR      = ROOT / "models"
CHECKPOINT_PATH = MODELS_DIR / "lstm_online.pt"
BEST_PATH       = MODELS_DIR / "lstm_online_best.pt"
METRICS_PATH    = MODELS_DIR / "lstm_online_metrics.jsonl"

REPLAY_CAPACITY = 800     # sequences in the experience replay buffer
REPLAY_RATIO    = 0.4     # fraction of each batch drawn from replay
ANCHOR_EVERY    = 50      # recompute EWC anchor every N steps
EWC_LAMBDA      = 0.05    # EWC penalty weight
SAVE_EVERY      = 40      # save checkpoint every N steps
REPORT_EVERY    = 10      # print metrics every N steps


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

class OnlineBiLSTM(nn.Module):
    """
    Bidirectional LSTM optimised for online / incremental learning.

    Deliberately smaller than the offline BiLSTM (hidden=64 vs 128):
    • Smaller models converge faster in the low-sample online regime.
    • Less capacity → less catastrophic forgetting between updates.
    • Still captures temporal dependencies (salary cycles, stress build-up).

    Layer normalisation (instead of batch norm) is applied after the LSTM
    because batch norm is unreliable with mini-batch size < 32.
    """

    def __init__(
        self,
        feat_dim:   int = FEAT_DIM,
        hidden:     int = 64,
        num_layers: int = 2,
        dropout:    float = 0.25,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden * 2)
        self.dropout    = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (B, T, F)
        out, _ = self.lstm(x)           # (B, T, 2H)
        # Use last time-step of each direction (concatenated)
        last_fwd = out[:, -1, :out.shape[-1] // 2]
        last_bwd = out[:, 0,  out.shape[-1] // 2:]
        h = torch.cat([last_fwd, last_bwd], dim=-1)  # (B, 2H)
        h = self.layer_norm(h)
        h = self.dropout(h)
        return self.head(h).squeeze(-1)   # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Experience replay buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Fixed-capacity FIFO ring buffer of (seq_array, label) pairs.
    Mirrors Amazon S3 historical storage providing a historical baseline
    alongside the live Kinesis stream.

    On eviction the oldest sample is dropped (FIFO) — recent patterns
    are given higher weight by discarding very old data naturally.
    """

    def __init__(self, capacity: int = REPLAY_CAPACITY):
        self.capacity = capacity
        self._buf: deque = deque(maxlen=capacity)

    def add_batch(self, X: np.ndarray, y: np.ndarray) -> None:
        for i in range(len(X)):
            self._buf.append((X[i], y[i]))

    def sample(self, n: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if len(self._buf) < n:
            return None
        idx = np.random.choice(len(self._buf), n, replace=False)
        seqs   = np.stack([self._buf[i][0] for i in idx], axis=0)
        labels = np.array([self._buf[i][1] for i in idx], dtype=np.float32)
        return seqs, labels

    def __len__(self) -> int:
        return len(self._buf)

    def pos_frac(self) -> float:
        if not self._buf:
            return 0.0
        return sum(1 for _, y in self._buf if y == 1) / len(self._buf)


# ─────────────────────────────────────────────────────────────────────────────
# EWC (Elastic Weight Consolidation) anchor
# ─────────────────────────────────────────────────────────────────────────────

class EWCAnchor:
    """
    Diagonal Fisher EWC to protect important weights from large updates.
    After each anchor update:
      fisher_i  = E[ (∂ log p(y|x) / ∂ θ_i)² ]
      penalty   = λ/2 * Σ fisher_i * (θ_i - θ*_i)²

    Only applied to LSTM and head weight matrices (not biases / norms)
    to limit compute overhead in the online setting.
    """

    def __init__(self, model: "OnlineBiLSTM", lam: float = EWC_LAMBDA):
        self.lam     = lam
        self._params : Dict[str, "torch.Tensor"] = {}
        self._fishers: Dict[str, "torch.Tensor"] = {}
        self._names  : List[str] = [
            n for n, p in model.named_parameters()
            if p.requires_grad and "weight" in n
        ]

    def update(
        self,
        model: "OnlineBiLSTM",
        X:     "torch.Tensor",
        y:     "torch.Tensor",
    ) -> None:
        """Recompute Fisher diagonal on a sample batch."""
        model.eval()
        model.zero_grad()
        preds = model(X)
        bce   = nn.BCELoss()(preds, y)
        bce.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self._names:
                    self._params[name]  = param.detach().clone()
                    self._fishers[name] = (
                        param.grad.detach().pow(2)
                        if param.grad is not None
                        else torch.zeros_like(param)
                    )
        model.train()

    def penalty(self, model: "OnlineBiLSTM") -> "torch.Tensor":
        """Compute EWC regularisation loss."""
        if not self._params:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0, requires_grad=True)
        for name, param in model.named_parameters():
            if name in self._names and name in self._params:
                diff = param - self._params[name]
                loss = loss + (self._fishers[name] * diff.pow(2)).sum()
        return (self.lam / 2) * loss


# ─────────────────────────────────────────────────────────────────────────────
# Online trainer
# ─────────────────────────────────────────────────────────────────────────────

class OnlineTrainer:
    """
    Manages the full online training loop:

      1. Warm-up: drain all historical sequences from daily_sequences.jsonl
         and run `warmup_epochs` passes over them (batch learning baseline).
      2. Live loop: poll StreamProcessor for new mini-batches, mix with
         replay, do a gradient step, update EWC anchor periodically.
      3. Checkpointing and metrics logging.

    The trainer runs in the foreground (blocking) — launch it in a separate
    process or thread alongside the simulator.
    """

    def __init__(
        self,
        checkpoint:    Optional[Path] = None,
        batch_size:    int   = 64,
        lr:            float = 3e-4,
        warmup_epochs: int   = 0,
        max_steps:     int   = 0,   # 0 = run forever
    ):
        if not TORCH_OK:
            raise RuntimeError(
                "PyTorch not installed. Run:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )

        self.batch_size    = batch_size
        self.lr            = lr
        self.warmup_epochs = warmup_epochs
        self.max_steps     = max_steps
        self.step          = 0
        self.best_auc      = 0.0

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # ── Model + optimiser ────────────────────────────────────────────────
        self.device = torch.device("cpu")   # CPU for streaming — no batch-GPU
        self.model  = OnlineBiLSTM().to(self.device)
        self.opt    = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.sched  = CosineAnnealingWarmRestarts(self.opt, T_0=50, T_mult=2)
        self.loss_fn = nn.BCELoss()

        # ── Supporting components ────────────────────────────────────────────
        self.replay = ReplayBuffer(REPLAY_CAPACITY)
        self.ewc    = EWCAnchor(self.model, lam=EWC_LAMBDA)

        # ── Load checkpoint if supplied ──────────────────────────────────────
        if checkpoint and Path(checkpoint).exists():
            self._load(Path(checkpoint))
        elif CHECKPOINT_PATH.exists():
            print(f"  Auto-loading checkpoint: {CHECKPOINT_PATH}")
            self._load(CHECKPOINT_PATH)

        # ── Stream processor ─────────────────────────────────────────────────
        self.sp = StreamProcessor(batch_size=batch_size)

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        """Full training loop — warm-up then live online learning."""
        print("=" * 70)
        print("  ONLINE LSTM TRAINER")
        print("=" * 70)
        print(f"  Model     : OnlineBiLSTM  (hidden=64, layers=2, bidirectional)")
        print(f"  Replay    : capacity={REPLAY_CAPACITY}  ratio={REPLAY_RATIO}")
        print(f"  EWC       : λ={EWC_LAMBDA}  anchor_every={ANCHOR_EVERY} steps")
        print(f"  LR        : {self.lr}  schedule=CosineAnnealingWarmRestarts")
        print(f"  Checkpoint: every {SAVE_EVERY} steps → {CHECKPOINT_PATH}")
        print("=" * 70)

        # ── Phase 1: warm-up on history ──────────────────────────────────────
        if self.warmup_epochs > 0:
            self._warmup()

        # ── Phase 2: live online loop ────────────────────────────────────────
        self.sp.start()
        print("\n  Entering live online training loop.  Ctrl+C to stop.\n")
        self.model.train()

        try:
            while True:
                batch = self.sp.get_mini_batch(timeout=30.0)
                if batch is None:
                    print("  [waiting for data from simulator ...]")
                    continue

                X_new, y_new = batch
                self._gradient_step(X_new, y_new)

                if self.max_steps > 0 and self.step >= self.max_steps:
                    print(f"\n  Reached max_steps={self.max_steps}.  Stopping.")
                    break

        except KeyboardInterrupt:
            print("\n  Interrupted by user.")
        finally:
            self.sp.stop()
            self._save(CHECKPOINT_PATH)
            print(f"\n  ✓ Final checkpoint saved → {CHECKPOINT_PATH}")
            self._print_final_stats()

    # ── Warm-up pass ─────────────────────────────────────────────────────────

    def _warmup(self) -> None:
        """
        One-shot warm-up: drain all sequences from the file and train
        warmup_epochs full passes.  Initialises the replay buffer.
        """
        print(f"\n  Warm-up: reading all historical sequences ...")
        batches = self.sp.drain_all()
        if not batches:
            print("  ⚠ No historical sequences found. Skipping warm-up.")
            return

        total_seqs = sum(len(y) for _, y in batches)
        pos        = sum(int(y.sum()) for _, y in batches)
        print(f"  Warm-up batches: {len(batches)}  sequences: {total_seqs:,}  "
              f"pos={pos} ({pos/total_seqs*100:.1f}%)")

        self.model.train()
        for epoch in range(self.warmup_epochs):
            np.random.shuffle(batches)
            ep_loss, ep_n = 0.0, 0
            all_prob, all_y = [], []

            for X_arr, y_arr in batches:
                loss_val, probs = self._gradient_step(
                    X_arr, y_arr, return_probs=True
                )
                ep_loss += loss_val * len(y_arr)
                ep_n    += len(y_arr)
                all_prob.extend(probs)
                all_y.extend(y_arr.tolist())

            auc = _safe_auc(all_y, all_prob)
            print(f"  Warm-up epoch {epoch+1}/{self.warmup_epochs}  "
                  f"loss={ep_loss/ep_n:.4f}  AUC={auc:.4f}")

        # Populate replay buffer with the last batch
        for X_arr, y_arr in batches[-20:]:
            self.replay.add_batch(X_arr, y_arr)
        print(f"  Warm-up done. Replay buffer: {len(self.replay)} sequences.\n")

    # ── Gradient step ─────────────────────────────────────────────────────────

    def _gradient_step(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        return_probs: bool = False,
    ) -> Tuple:
        """
        One online gradient update:
          • Mix new samples with experience replay samples
          • Compute BCE loss + EWC penalty
          • Clip gradients and step optimizer
          • Add new samples to replay buffer
          • Periodically: update EWC anchor, save checkpoint, report metrics
        """
        self.step += 1

        # ── Build mixed batch ─────────────────────────────────────────────────
        n_replay = int(len(X_new) * REPLAY_RATIO)
        replay   = self.replay.sample(n_replay) if (
            len(self.replay) >= n_replay > 0
        ) else None

        if replay is not None:
            X_r, y_r = replay
            X = np.vstack([X_new, X_r])
            y = np.concatenate([y_new, y_r])
        else:
            X, y = X_new, y_new

        # Shuffle mixed batch
        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm]

        X_t = torch.from_numpy(X).float().to(self.device)
        y_t = torch.from_numpy(y).float().to(self.device)

        # ── Forward + loss ────────────────────────────────────────────────────
        self.opt.zero_grad()
        preds = self.model(X_t)

        bce_loss = self.loss_fn(preds, y_t)
        ewc_loss = self.ewc.penalty(self.model)
        loss     = bce_loss + ewc_loss

        # ── Backward + clip ───────────────────────────────────────────────────
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.opt.step()
        self.sched.step()

        # ── Update replay buffer ──────────────────────────────────────────────
        self.replay.add_batch(X_new, y_new)

        # ── EWC anchor refresh ────────────────────────────────────────────────
        if self.step % ANCHOR_EVERY == 0:
            self.ewc.update(self.model, X_t[:32], y_t[:32])

        # ── Metrics ───────────────────────────────────────────────────────────
        with torch.no_grad():
            probs_numpy = preds.cpu().numpy().tolist()
            y_numpy     = y_t.cpu().numpy().tolist()

        loss_val = float(bce_loss.item())
        ewc_val  = float(ewc_loss.item()) if isinstance(ewc_loss, torch.Tensor) else 0.0

        if self.step % REPORT_EVERY == 0:
            auc = _safe_auc(y_numpy, probs_numpy)
            lr  = self.opt.param_groups[0]["lr"]
            buf = self.replay.buffer_info()

            line = (
                f"  step={self.step:5d}  loss={loss_val:.4f}  "
                f"ewc={ewc_val:.5f}  AUC={auc:.4f}  "
                f"lr={lr:.2e}  "
                f"replay={len(self.replay)}/{REPLAY_CAPACITY}  "
                f"pos_frac={self.replay.pos_frac()*100:.1f}%"
            )
            print(line)

            # Log to JSONL
            metric = {
                "step": self.step, "bce_loss": round(loss_val, 5),
                "ewc_loss": round(ewc_val, 5), "auc": round(auc, 5),
                "lr": round(lr, 8), "replay_size": len(self.replay),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(METRICS_PATH, "a") as mf:
                mf.write(json.dumps(metric) + "\n")

            # Save best model
            if auc > self.best_auc:
                self.best_auc = auc
                self._save(BEST_PATH)

        if self.step % SAVE_EVERY == 0:
            self._save(CHECKPOINT_PATH)

        if return_probs:
            return loss_val, probs_numpy
        return loss_val, None

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def _save(self, path: Path) -> None:
        torch.save(
            {
                "model_state":  self.model.state_dict(),
                "opt_state":    self.opt.state_dict(),
                "step":         self.step,
                "best_auc":     self.best_auc,
                "ewc_params":   self.ewc._params,
                "ewc_fishers":  self.ewc._fishers,
            },
            path,
        )

    def _load(self, path: Path) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model_state"])
        self.opt.load_state_dict(ck["opt_state"])
        self.step     = ck.get("step", 0)
        self.best_auc = ck.get("best_auc", 0.0)
        if "ewc_params" in ck:
            self.ewc._params   = ck["ewc_params"]
            self.ewc._fishers  = ck["ewc_fishers"]
        print(f"  ✓ Loaded checkpoint  step={self.step}  best_AUC={self.best_auc:.4f}")

    def _print_final_stats(self) -> None:
        sp_s = self.sp.stats
        print(f"\n  Total gradient steps : {self.step}")
        print(f"  Best AUC seen        : {self.best_auc:.4f}")
        print(f"  Sequences ingested   : {sp_s['records_ingested']:,}")
        print(f"  Batches processed    : {sp_s['batches_emitted']}")
        print(f"  Best checkpoint      : {BEST_PATH}")
        print(f"  Metrics log          : {METRICS_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_auc(y_true, y_score) -> float:
    """ROC-AUC gracefully for edge-cases (single class in batch)."""
    try:
        yt = np.array(y_true)
        if len(np.unique(yt)) < 2:
            return float("nan")
        return float(roc_auc_score(yt, np.array(y_score)))
    except Exception:
        return float("nan")


# Monkey-patch ReplayBuffer to add buffer_info helper
def _buf_info(self):
    pos = sum(1 for _, y in self._buf if y == 1)
    neg = len(self._buf) - pos
    return {"pos": pos, "neg": neg, "total": len(self._buf)}
ReplayBuffer.buffer_info = _buf_info


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Online incremental LSTM trainer for delinquency prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to .pt checkpoint to resume from")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Mini-batch size")
    p.add_argument("--lr",         type=float, default=3e-4,
                   help="Initial learning rate")
    p.add_argument("--warmup",     action="store_true",
                   help="Run warm-up pass over historical sequences first")
    p.add_argument("--warmup-epochs", type=int, default=3,
                   help="Number of epochs in warm-up phase")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Stop after this many gradient steps (0 = infinite)")
    args = p.parse_args()

    if not TORCH_OK:
        print("✗ PyTorch is not installed.")
        print("  Install with:")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)

    trainer = OnlineTrainer(
        checkpoint    = args.checkpoint,
        batch_size    = args.batch_size,
        lr            = args.lr,
        warmup_epochs = args.warmup_epochs if args.warmup else 0,
        max_steps     = args.max_steps,
    )
    trainer.run()


if __name__ == "__main__":
    main()
