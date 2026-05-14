"""
ArticleNet — PyTorch GPU model for financial article scoring.

Architecture:
  TF-IDF (15000) → Linear 512 → BN → ReLU → Dropout
                 → Linear 256 → BN → ReLU → Dropout
                 → Linear 128 → BN → ReLU → Dropout
                 → Linear 2   (relevance, urgency)

Uncertainty via Monte Carlo Dropout: 10 stochastic forward passes at inference,
return mean and std across passes. Higher std → route to LLM.

Trains on CUDA when available, falls back to CPU. Checkpoint: data/ml/model_gpu.pt.
"""
import os
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DIR = Path(os.environ.get("DIGITAL_INTERN_ML_DIR",
                                Path(__file__).resolve().parent.parent / "data" / "ml"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "model_gpu.pt"

INPUT_DIM    = 15_000
HIDDEN       = (512, 256, 128)
OUTPUT_DIM   = 2          # relevance, urgency
DROPOUT      = 0.3
MC_PASSES    = 10         # Monte Carlo Dropout passes at inference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ArticleNetModule(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden=HIDDEN,
                 output_dim: int = OUTPUT_DIM, dropout: float = DROPOUT):
        super().__init__()
        h1, h2, h3 = hidden
        self.fc1 = nn.Linear(input_dim, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        self.head = nn.Linear(h3, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        return self.head(x)


class ArticleNet:
    """PyTorch GPU model wrapping ArticleNetModule with MC-Dropout uncertainty."""

    def __init__(self):
        self.device = DEVICE
        self.net = ArticleNetModule().to(self.device)
        self._fitted = False
        self._input_dim = INPUT_DIM
        self._load()

    # ── persistence ─────────────────────────────────────────────────────────
    def _load(self):
        if CHECKPOINT_PATH.exists():
            try:
                ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device,
                                  weights_only=False)
                state = ckpt.get("state_dict", ckpt)
                self._input_dim = ckpt.get("input_dim", INPUT_DIM)
                if self._input_dim != INPUT_DIM:
                    # Rebuild module if checkpoint used a different input dim
                    self.net = ArticleNetModule(input_dim=self._input_dim).to(self.device)
                self.net.load_state_dict(state)
                self._fitted = True
                print(f"[ml:model] Loaded GPU model from {CHECKPOINT_PATH} "
                      f"(device={self.device}, input_dim={self._input_dim})")
            except Exception as e:
                print(f"[ml:model] Load error: {e} — starting fresh")
                self._fitted = False

    def save(self):
        torch.save({
            "state_dict": self.net.state_dict(),
            "input_dim": self._input_dim,
        }, CHECKPOINT_PATH)

    # ── training ────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y_rel: np.ndarray, y_urg: np.ndarray,
            epochs: int = 50, batch_size: int = 256, lr: float = 1e-3,
            verbose: bool = True) -> dict:
        """
        Train the GPU model. Returns metrics dict with final_loss, epochs, samples.
        Keeps the same (X, y_rel, y_urg) interface as the previous sklearn model.
        """
        t0 = time.time()
        n, dim = X.shape

        # Adapt the network if dim doesn't match what we have
        if dim != self._input_dim:
            self._input_dim = dim
            self.net = ArticleNetModule(input_dim=dim).to(self.device)
            self._fitted = False

        X_t   = torch.as_tensor(X, dtype=torch.float32)
        y_t   = torch.stack([
            torch.as_tensor(y_rel, dtype=torch.float32),
            torch.as_tensor(y_urg, dtype=torch.float32),
        ], dim=1)  # (N, 2)

        ds = torch.utils.data.TensorDataset(X_t, y_t)
        # pin_memory only helps for CUDA; keep CPU path simple
        pin = (self.device.type == "cuda")
        # BatchNorm1d errors out on a batch of size 1 in train mode. Drop the
        # final partial batch whenever it would be a singleton (n % bs == 1),
        # provided we have at least one full batch to train on.
        drop_last = (n > batch_size) and (n % batch_size == 1)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            pin_memory=pin, num_workers=0, drop_last=drop_last,
        )

        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        loss_fn = nn.SmoothL1Loss()

        self.net.train()
        final_loss = float("nan")
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=pin)
                yb = yb.to(self.device, non_blocking=pin)
                opt.zero_grad(set_to_none=True)
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1
            sched.step()
            final_loss = epoch_loss / max(n_batches, 1)

            if verbose and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1):
                gpu_mem = ""
                if self.device.type == "cuda":
                    mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
                    peak = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                    gpu_mem = f" gpu_mem={mb:.0f}MB peak={peak:.0f}MB"
                print(f"[ml:model] epoch {epoch+1:>3}/{epochs} "
                      f"loss={final_loss:.4f} lr={sched.get_last_lr()[0]:.2e}{gpu_mem}")

        self.net.eval()
        self._fitted = True
        self.save()
        elapsed = time.time() - t0
        print(f"[ml:model] Trained {epochs} epochs on {n} samples in {elapsed:.1f}s "
              f"(device={self.device})")
        return {
            "final_loss": round(final_loss, 4),
            "epochs": epochs,
            "samples": n,
            "elapsed_s": round(elapsed, 1),
            "device": str(self.device),
        }

    # ── inference ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout inference: MC_PASSES stochastic forward passes.
        Returns (rel_mean, rel_std, urg_mean, urg_std), each shape (N,).
        """
        n = X.shape[0]
        if not self._fitted or X.shape[1] != self._input_dim:
            return (np.zeros(n), np.full(n, 99.0),
                    np.zeros(n), np.full(n, 99.0))

        # Keep dropout active for MC sampling; BatchNorm stays in eval mode.
        self.net.eval()
        for m in self.net.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        # Single-sample inputs break BatchNorm in train mode; we forced BN to eval
        # above so this is safe.
        preds = []
        for _ in range(MC_PASSES):
            preds.append(self.net(x).cpu().numpy())
        stacked = np.stack(preds, axis=0)  # (P, N, 2)

        rel = stacked[:, :, 0]
        urg = stacked[:, :, 1]
        rel_mean = np.clip(rel.mean(axis=0), 0, 10)
        urg_mean = np.clip(urg.mean(axis=0), 0, 10)
        rel_std  = rel.std(axis=0)
        urg_std  = urg.std(axis=0)

        self.net.eval()  # restore fully eval after MC
        return rel_mean, rel_std, urg_mean, urg_std

    @property
    def fitted(self) -> bool:
        return self._fitted


_net: ArticleNet | None = None


def get_model() -> ArticleNet:
    global _net
    if _net is None:
        _net = ArticleNet()
    return _net
