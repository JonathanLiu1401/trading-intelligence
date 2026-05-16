"""
ArticleNet — Deep PyTorch GPU model for financial article scoring.

Architecture (multi-task encoder + 3 heads):
  TF-IDF + extras → Linear 512 → LayerNorm → GELU → Dropout(0.2)
                  → Linear 512 → LayerNorm → GELU → Dropout(0.2)
                  → Linear 256 → LayerNorm → GELU → Dropout(0.15)
                  → Linear 128 → LayerNorm → GELU → Dropout(0.1)
                  → relevance_head  (sigmoid * 10) — 0..10
                  → urgency_head    (sigmoid)      — probability of urgent
                  → uncertainty_head(sigmoid)      — predicted error magnitude

Training: multi-task — MSE on relevance + 0.5 * BCE on urgency + 0.2 * BCE on
uncertainty (target = detached |rel_pred - rel_target| / 10).

Uncertainty is exposed two ways:
  - The dedicated uncertainty head (used by recursive_labeler for active learning)
  - MC-Dropout ensemble spread (used by inference.py for grey-zone routing)

Checkpoints:
  - data/ml/model_gpu.pt        — last-trained state, fast reload
  - <USB>/ml_checkpoints/       — versioned snapshots, last 10 kept
  - <USB>/ml_checkpoints/best_model.pt — lowest validation loss across all runs
"""
import logging
import os
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Route progress prints to the project logger so daemon.log shows training
# progress without needing to scrape stdout. Falls back to root if the project
# logger isn't installed in some scripts that import ml/model directly.
try:
    from core.logger import get_logger  # type: ignore
    _log = get_logger("ml.model")
except Exception:
    _log = logging.getLogger("ml.model")

MODEL_DIR = Path(os.environ.get("DIGITAL_INTERN_ML_DIR",
                                Path(__file__).resolve().parent.parent / "data" / "ml"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "model_gpu.pt"

# Versioned checkpoints on USB drive, mirroring article_store.USB_PATH layout.
USB_PATH = Path(os.environ.get("DIGITAL_INTERN_USB", "/media/zeph/projects/digital-intern/db"))
USB_CHECKPOINT_DIR = USB_PATH / "ml_checkpoints"
BEST_MODEL_PATH = USB_CHECKPOINT_DIR / "best_model.pt"
MAX_CHECKPOINTS = 10

INPUT_DIM    = 15_000
HIDDEN_DIM   = 512
DROPOUT      = 0.2          # used for MC-Dropout passes at inference
MC_PASSES    = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _usb_available() -> bool:
    """USB checkpoint dir is usable when its parent (mount point) exists."""
    try:
        if USB_CHECKPOINT_DIR.parent.exists():
            USB_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            return True
    except Exception:
        pass
    return False


class ArticleNetModule(nn.Module):
    """Deep transformer-style encoder for financial article scoring."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        head_in = hidden_dim // 4
        # Relevance scaled to 0..10 so existing thresholds (>=8 = urgent etc.)
        # in inference.py / scorer_worker keep working unchanged.
        self.relevance_head        = nn.Linear(head_in, 1)
        self.urgency_head          = nn.Linear(head_in, 1)
        self.uncertainty_head      = nn.Linear(head_in, 1)
        # time_sensitivity: 1.0 = highly time-sensitive (earnings, breaking price moves),
        # 0.0 = timeless (macro thesis, secular trends). Consumed by briefing ranking
        # to apply intelligent recency decay per-article rather than blanket decay.
        self.time_sensitivity_head = nn.Linear(head_in, 1)

    def forward(self, x):
        h = self.encoder(x)
        relevance        = torch.sigmoid(self.relevance_head(h)) * 10.0
        urgency          = torch.sigmoid(self.urgency_head(h))           # 0..1 prob
        uncertainty      = torch.sigmoid(self.uncertainty_head(h))       # 0..1
        time_sensitivity = torch.sigmoid(self.time_sensitivity_head(h))  # 0..1
        return relevance, urgency, uncertainty, time_sensitivity


class ArticleNet:
    """PyTorch GPU model wrapping ArticleNetModule with multi-task heads
    and MC-Dropout uncertainty for the inference path."""

    def __init__(self):
        self.device = DEVICE
        if self.device.type == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(self.device)
            except Exception:
                gpu_name = "CUDA"
            print(f"[ML] Using device: cuda ({gpu_name})")
        else:
            print("[ML] Using device: cpu (no CUDA available)")
        self.net = ArticleNetModule().to(self.device)
        self._fitted = False
        self._input_dim = INPUT_DIM
        self._best_val_loss = float("inf")
        self._load()

    # ── persistence ─────────────────────────────────────────────────────────
    def _load(self):
        """Prefer best_model.pt on USB, then local best_model.pt, then last checkpoint."""
        local_best = MODEL_DIR / "best_model.pt"
        for path in (BEST_MODEL_PATH, local_best, CHECKPOINT_PATH):
            if not path.exists():
                continue
            try:
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                state = ckpt.get("state_dict", ckpt)
                ckpt_dim = ckpt.get("input_dim", INPUT_DIM)
                self._input_dim = ckpt_dim
                if self._input_dim != INPUT_DIM:
                    self.net = ArticleNetModule(input_dim=self._input_dim).to(self.device)
                # New 3-head module is incompatible with old 2-head checkpoints.
                # strict=False lets us partially load the encoder and start fresh
                # heads when the architectures don't match.
                missing, unexpected = self.net.load_state_dict(state, strict=False)
                if unexpected or any("head" in k for k in missing):
                    print(f"[ml:model] Checkpoint {path.name} has incompatible heads — "
                          f"loaded encoder only, heads reinitialized")
                    self._fitted = False
                else:
                    self._fitted = True
                ckpt_best = ckpt.get("best_val_loss", float("inf"))
                # A "best_val_loss" recorded against a tiny/broken feature space
                # (vocab=3 → 18-dim input) is incomparable to the live feature
                # space; clear it so refits aren't locked out of the best slot.
                if ckpt_dim < 100:
                    print(f"[ml:model] Ignoring legacy best_val_loss={ckpt_best:.4f} "
                          f"from {ckpt_dim}-dim checkpoint — resetting to inf")
                    self._best_val_loss = float("inf")
                else:
                    self._best_val_loss = ckpt_best
                print(f"[ml:model] Loaded model from {path} "
                      f"(device={self.device}, input_dim={self._input_dim}, "
                      f"best_val_loss={self._best_val_loss:.4f})")
                return
            except Exception as e:
                print(f"[ml:model] Load error from {path}: {e}")
        print("[ml:model] No usable checkpoint — starting fresh")

    def save(self):
        """Save the current state to the local checkpoint (fast reload)."""
        torch.save({
            "state_dict": self.net.state_dict(),
            "input_dim": self._input_dim,
            "best_val_loss": self._best_val_loss,
        }, CHECKPOINT_PATH)

    def _save_best_local(self, val_loss: float, metrics: dict) -> bool:
        """Save weights to a local best_model.pt when val_loss improves.

        Independent of USB availability so that "best" tracking works on hosts
        where DIGITAL_INTERN_USB isn't mounted. Returns True if this run was a
        new best."""
        local_best = MODEL_DIR / "best_model.pt"
        improved = val_loss < self._best_val_loss
        if improved:
            self._best_val_loss = val_loss
            try:
                torch.save({
                    "state_dict": self.net.state_dict(),
                    "input_dim": self._input_dim,
                    "best_val_loss": self._best_val_loss,
                    "metrics": metrics,
                    "saved_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                }, local_best)
                print(f"[ml:model] New local best: val_loss={val_loss:.4f} → {local_best}")
            except Exception as e:
                print(f"[ml:model] Local best save failed: {e}")
        return improved

    def _save_versioned(self, val_loss: float, metrics: dict, is_new_best: bool) -> None:
        """Save a timestamped snapshot to USB; rotate to last MAX_CHECKPOINTS.

        ``is_new_best`` is decided by the caller (``fit``) and mirrors what
        ``_save_best_local`` recorded. Recomputing the comparison here against
        ``self._best_val_loss`` would always be False because ``_save_best_local``
        already mutated it — that's how the USB ``best_model.pt`` promotion
        path used to silently no-op.
        """
        if not _usb_available():
            return
        try:
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            target = USB_CHECKPOINT_DIR / f"checkpoint_{ts}.pt"
            torch.save({
                "state_dict": self.net.state_dict(),
                "input_dim": self._input_dim,
                "val_loss": val_loss,
                "best_val_loss": self._best_val_loss,
                "metrics": metrics,
                "saved_at": ts,
            }, target)
            # Rotate — keep the newest MAX_CHECKPOINTS files.
            snaps = sorted(USB_CHECKPOINT_DIR.glob("checkpoint_*.pt"))
            for stale in snaps[:-MAX_CHECKPOINTS]:
                try:
                    stale.unlink()
                except Exception:
                    pass
            # Promote to best_model.pt on improvement.
            if is_new_best:
                torch.save({
                    "state_dict": self.net.state_dict(),
                    "input_dim": self._input_dim,
                    "best_val_loss": self._best_val_loss,
                    "metrics": metrics,
                    "saved_at": ts,
                }, BEST_MODEL_PATH)
                print(f"[ml:model] New best model: val_loss={val_loss:.4f} → {BEST_MODEL_PATH}")
        except Exception as e:
            print(f"[ml:model] Versioned save failed: {e}")

    # ── training ────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y_rel: np.ndarray, y_urg: np.ndarray,
            y_time: np.ndarray | None = None,
            epochs: int = 100, batch_size: int = 256, lr: float = 1e-3,
            verbose: bool = True, warm: bool | None = None,
            label_weight_exponent: float | None = None,
            early_stop_patience: int = 6) -> dict:
        """Multi-task GPU training. Returns metrics dict.

        ``y_rel`` is a 0..10 relevance score; ``y_urg`` is a 0..10 urgency
        score that gets binarized to ``>= 8.0`` for the BCE head.

        ``early_stop_patience`` is the number of consecutive validation
        checks without improvement after which training halts (only active
        when a held-out val set exists, i.e. n >= 100). Best-epoch weights
        are restored regardless, so this only trims wasted, overfitting
        epochs — it never changes which checkpoint is saved.
        """
        t0 = time.time()
        n, dim = X.shape

        if dim != self._input_dim:
            self._input_dim = dim
            self.net = ArticleNetModule(input_dim=dim).to(self.device)
            self._fitted = False
            # Old best_val_loss was measured in a different feature space (e.g.
            # tiny 18-dim TF-IDF). Keeping it here would block local/best-model
            # promotion forever once the embedder refits to a real vocab.
            self._best_val_loss = float("inf")

        # Warm-start LR: an already-fitted model that's near a minimum gets
        # blown off the basin if we restart Adam at LR=1e-3 every cycle. Drop
        # to a fine-tune LR for subsequent fits so cross-run val_loss actually
        # trends down instead of oscillating. Caller can override with warm=False.
        if warm is None:
            warm = self._fitted
        if warm:
            lr = min(lr, 1e-4)

        X_t      = torch.as_tensor(X, dtype=torch.float32)
        rel_t    = torch.as_tensor(y_rel, dtype=torch.float32)
        urg_bin  = torch.as_tensor((y_urg >= 8.0).astype(np.float32), dtype=torch.float32)

        # Per-sample loss weighting by label magnitude. Convex on (y_rel/10):
        # exp=2 → score 10 weighted 25x stronger than score 2 (pre-norm). We
        # normalize to mean=1 so loss scale (and effective LR) stay constant.
        # exp=None or <=0 → uniform weights (back-compat with callers that
        # don't pass the exponent).
        if label_weight_exponent is not None and label_weight_exponent > 0:
            w_np = np.power(np.clip(y_rel, 0.0, 10.0) / 10.0,
                            float(label_weight_exponent)).astype(np.float32)
            # Floor to avoid zero-weight rows (score=0 kw bootstrap samples
            # would otherwise contribute nothing to gradients).
            w_np = np.clip(w_np, 1e-3, None)
            w_mean = float(w_np.mean()) if w_np.size else 1.0
            if w_mean > 0:
                w_np = w_np / w_mean
        else:
            w_np = np.ones_like(y_rel, dtype=np.float32)
        weight_t = torch.as_tensor(w_np, dtype=torch.float32)
        # If caller didn't supply time_sensitivity labels, fall back to a neutral
        # 0.5 target with a near-zero loss weight (so we don't pull predictions
        # toward 0.5 when labels are absent — see ``time_loss_weight`` below).
        has_time = y_time is not None
        if has_time:
            time_t = torch.as_tensor(np.clip(y_time, 0.0, 1.0).astype(np.float32),
                                     dtype=torch.float32)
        else:
            time_t = torch.full((n,), 0.5, dtype=torch.float32)
        time_loss_weight = 0.3 if has_time else 0.0

        # Hold out 10% for validation when we have enough samples to make it
        # meaningful (>= 100). Otherwise train on everything.
        # Use a deterministic permutation so val_loss is comparable across
        # successive fits on the same dataset (random splits made the metric
        # noisy enough to mask real progress).
        if n >= 100:
            gen = torch.Generator().manual_seed(0xA17C1E)
            perm = torch.randperm(n, generator=gen)
            n_val = max(10, n // 10)
            val_idx = perm[:n_val]
            tr_idx  = perm[n_val:]
            X_tr, X_val = X_t[tr_idx], X_t[val_idx]
            rel_tr, rel_val = rel_t[tr_idx], rel_t[val_idx]
            urg_tr, urg_val = urg_bin[tr_idx], urg_bin[val_idx]
            time_tr, time_val = time_t[tr_idx], time_t[val_idx]
            w_tr = weight_t[tr_idx]
        else:
            X_tr, X_val = X_t, None
            rel_tr, rel_val = rel_t, None
            urg_tr, urg_val = urg_bin, None
            time_tr, time_val = time_t, None
            w_tr = weight_t

        ds = torch.utils.data.TensorDataset(X_tr, rel_tr, urg_tr, time_tr, w_tr)
        pin = (self.device.type == "cuda")
        # Singleton final batch breaks BatchNorm; LayerNorm handles N=1 fine,
        # but keep the guard to mirror the prior behavior.
        n_tr = X_tr.shape[0]
        drop_last = (n_tr > batch_size) and (n_tr % batch_size == 1)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            pin_memory=pin, num_workers=0, drop_last=drop_last,
        )

        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

        self.net.train()
        final_loss = float("nan")
        # Track best in-run state by validation loss (or train loss if no val).
        # We restore this before save() so a noisy late epoch can't overwrite a
        # good earlier checkpoint — this is what stops cross-run val_loss from
        # trending downward.
        best_in_run = float("inf")
        best_state = None
        # Early-stopping bookkeeping. A val check must beat the running best by
        # at least ``min_delta`` to count as progress, so val-loss jitter near
        # a plateau doesn't keep training alive indefinitely.
        no_improve_checks = 0
        stopped_early = False
        epochs_run = 0
        es_min_delta = 1e-4
        def _eval_val_loss() -> float:
            if X_val is None:
                return float("inf")
            self.net.eval()
            with torch.no_grad():
                xb = X_val.to(self.device)
                rb = rel_val.to(self.device)
                ub = urg_val.to(self.device)
                rel_p, urg_p, _, _ = self.net(xb)
                v_rel = F.mse_loss(rel_p.squeeze(-1), rb).item()
                v_urg = F.binary_cross_entropy(
                    urg_p.squeeze(-1).clamp(1e-6, 1 - 1e-6), ub
                ).item()
            self.net.train()
            return v_rel + 0.5 * v_urg

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for xb, rb, ub, tb, wb in loader:
                xb = xb.to(self.device, non_blocking=pin)
                rb = rb.to(self.device, non_blocking=pin)
                ub = ub.to(self.device, non_blocking=pin)
                tb = tb.to(self.device, non_blocking=pin)
                wb = wb.to(self.device, non_blocking=pin)
                opt.zero_grad(set_to_none=True)
                rel, urg, unc, tsens = self.net(xb)
                rel = rel.squeeze(-1)
                urg = urg.squeeze(-1)
                unc = unc.squeeze(-1)
                tsens = tsens.squeeze(-1)

                # Sample-weighted MSE/BCE: high-score articles dominate the
                # gradient, mirroring the "200% trade trains harder than 150%"
                # analogy. wb is pre-normalized to mean≈1 dataset-wide.
                rel_loss = (F.mse_loss(rel, rb, reduction='none') * wb).mean()
                # clamp keeps urg strictly within (eps, 1-eps) for BCE numerical safety
                urg_loss = (F.binary_cross_entropy(
                    urg.clamp(1e-6, 1 - 1e-6), ub, reduction='none'
                ) * wb).mean()
                # Train uncertainty to predict |error|/10 — aleatoric proxy.
                with torch.no_grad():
                    unc_target = (rel.detach() - rb).abs().clamp(0, 10) / 10.0
                unc_loss = F.binary_cross_entropy(unc.clamp(1e-6, 1 - 1e-6), unc_target)
                # time_sensitivity: BCE against soft 0..1 targets (works for both
                # binary heuristic labels and probabilistic ones).
                time_loss = F.binary_cross_entropy(
                    tsens.clamp(1e-6, 1 - 1e-6), tb.clamp(0.0, 1.0)
                )

                loss = rel_loss + 0.5 * urg_loss + 0.2 * unc_loss + time_loss_weight * time_loss
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1
            sched.step()
            final_loss = epoch_loss / max(n_batches, 1)
            epochs_run = epoch + 1

            # Best-epoch tracking — cheap val pass every few epochs.
            check_every = max(1, epochs // 20)
            if X_val is not None and ((epoch + 1) % check_every == 0 or epoch == epochs - 1):
                vl = _eval_val_loss()
                if vl < best_in_run - es_min_delta:
                    best_in_run = vl
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                    no_improve_checks = 0
                else:
                    # Still keep the strictly-best state even on a sub-min_delta
                    # nudge, but count this check as "no real progress".
                    if vl < best_in_run:
                        best_in_run = vl
                        best_state = {k: v.detach().clone()
                                      for k, v in self.net.state_dict().items()}
                    no_improve_checks += 1
                    if (early_stop_patience > 0
                            and no_improve_checks >= early_stop_patience
                            and epoch < epochs - 1):
                        stopped_early = True
            elif X_val is None and final_loss < best_in_run:
                best_in_run = final_loss
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}

            if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1):
                gpu_mem = ""
                if self.device.type == "cuda":
                    mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
                    peak = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                    gpu_mem = f" gpu_mem={mb:.0f}MB peak={peak:.0f}MB"
                # Compute current val_loss if we have a held-out set; reporting
                # train+val side-by-side makes overfitting visible in the log.
                vl_part = ""
                if X_val is not None:
                    vl_now = _eval_val_loss()
                    vl_part = f" val={vl_now:.4f}"
                msg = (f"[ml:model] epoch {epoch+1:>3}/{epochs} "
                       f"loss={final_loss:.4f}{vl_part} "
                       f"lr={sched.get_last_lr()[0]:.2e}{gpu_mem}")
                print(msg)
                # Mirror to the project logger so daemon.log captures progress.
                try:
                    _log.info(msg)
                except Exception:
                    pass

            if stopped_early:
                es_msg = (f"[ml:model] early stop at epoch {epoch+1}/{epochs} "
                          f"(no val improvement for {no_improve_checks} checks, "
                          f"best_val={best_in_run:.4f})")
                if verbose:
                    print(es_msg)
                try:
                    _log.info(es_msg)
                except Exception:
                    pass
                break

        # Restore best-epoch weights so we save the best, not the last.
        # Without this, late-epoch noise can overwrite a strictly-better earlier
        # state — the main reason cross-run val_loss appeared flat.
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        self._fitted = True

        # Validation pass (on restored best weights)
        val_loss = final_loss
        if X_val is not None:
            with torch.no_grad():
                xb = X_val.to(self.device)
                rb = rel_val.to(self.device)
                ub = urg_val.to(self.device)
                rel, urg, _, _ = self.net(xb)
                rel = rel.squeeze(-1)
                urg = urg.squeeze(-1)
                v_rel = F.mse_loss(rel, rb).item()
                v_urg = F.binary_cross_entropy(urg.clamp(1e-6, 1 - 1e-6), ub).item()
                val_loss = v_rel + 0.5 * v_urg

        elapsed = time.time() - t0
        metrics = {
            "final_loss": round(final_loss, 4),
            "val_loss":   round(val_loss, 4),
            "best_in_run": round(best_in_run, 4) if best_in_run != float("inf") else None,
            "epochs": epochs,
            "epochs_run": epochs_run,
            "stopped_early": stopped_early,
            "samples": n,
            "elapsed_s": round(elapsed, 1),
            "device": str(self.device),
            "warm": warm,
            "lr": lr,
        }
        improved_best = self._save_best_local(val_loss, metrics)
        metrics["new_best"] = improved_best
        self.save()
        self._save_versioned(val_loss, metrics, is_new_best=improved_best)
        print(f"[ml:model] Trained {epochs_run}/{epochs} epochs on {n} samples "
              f"in {elapsed:.1f}s (device={self.device}, val_loss={val_loss:.4f}"
              f"{', early-stopped' if stopped_early else ''})")
        return metrics

    # ── inference ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """MC-Dropout inference for backward compat with inference.py.
        Returns (rel_mean, rel_std, urg_mean, urg_std, time_sensitivity_mean),
        each shape (N,). Urgency mean/std are scaled to 0..10 so existing
        thresholds keep working. time_sensitivity stays in 0..1.
        """
        n = X.shape[0]
        if not self._fitted or X.shape[1] != self._input_dim:
            return (np.zeros(n), np.full(n, 99.0),
                    np.zeros(n), np.full(n, 99.0),
                    np.full(n, 0.5))

        # MC-Dropout: keep dropout active, LayerNorm has no train/eval split.
        self.net.eval()
        for m in self.net.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        rels, urgs, tsens_runs = [], [], []
        for _ in range(MC_PASSES):
            rel, urg, _, tsens = self.net(x)
            rels.append(rel.squeeze(-1).cpu().numpy())
            urgs.append(urg.squeeze(-1).cpu().numpy())
            tsens_runs.append(tsens.squeeze(-1).cpu().numpy())
        rel_arr = np.stack(rels, axis=0)         # (P, N)
        urg_arr = np.stack(urgs, axis=0) * 10.0  # scale prob → 0..10
        tsens_arr = np.stack(tsens_runs, axis=0) # (P, N), in 0..1

        rel_mean = np.clip(rel_arr.mean(axis=0), 0, 10)
        urg_mean = np.clip(urg_arr.mean(axis=0), 0, 10)
        rel_std  = rel_arr.std(axis=0)
        urg_std  = urg_arr.std(axis=0)
        tsens_mean = np.clip(tsens_arr.mean(axis=0), 0.0, 1.0)

        self.net.eval()
        return rel_mean, rel_std, urg_mean, urg_std, tsens_mean

    @torch.no_grad()
    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single deterministic pass — returns (relevance, urgency_prob, uncertainty).
        Used by recursive_labeler to find articles the model is unsure about."""
        n = X.shape[0]
        if not self._fitted or X.shape[1] != self._input_dim:
            return np.zeros(n), np.zeros(n), np.ones(n)

        self.net.eval()
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        rel, urg, unc, _tsens = self.net(x)
        return (
            rel.squeeze(-1).cpu().numpy(),
            urg.squeeze(-1).cpu().numpy(),
            unc.squeeze(-1).cpu().numpy(),
        )

    @property
    def fitted(self) -> bool:
        return self._fitted


_net: ArticleNet | None = None


def get_model() -> ArticleNet:
    global _net
    if _net is None:
        _net = ArticleNet()
    return _net
