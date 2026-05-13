"""
ArticleNet — lightweight MLP for financial article scoring.

Two output heads:
  - relevance_score  (0–10): how relevant is this to the portfolio/sector
  - urgency_score    (0–10): how urgently should this trigger an alert

Architecture: TF-IDF features → BatchNorm → [FC→ReLU→Dropout] × 3 → dual heads
Phase 1: TF-IDF input (fast, no GPU needed, ~0.1ms/article)
Phase 2: swap embedder for sentence-transformers (plug-in, same MLP head)
"""
import torch
import torch.nn as nn


class ArticleNet(nn.Module):
    def __init__(self, input_dim: int = 15_000, hidden: list[int] = None, dropout: float = 0.3):
        super().__init__()
        if hidden is None:
            hidden = [512, 256, 128]

        layers = []
        prev = input_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h

        self.trunk = nn.Sequential(*layers)
        self.relevance_head = nn.Sequential(nn.Linear(prev, 1), nn.Sigmoid())
        self.urgency_head   = nn.Sequential(nn.Linear(prev, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(x)
        # Scale sigmoid [0,1] → [0,10]
        relevance = self.relevance_head(feat).squeeze(-1) * 10.0
        urgency   = self.urgency_head(feat).squeeze(-1) * 10.0
        return relevance, urgency

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_passes: int = 15
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Monte Carlo Dropout — run n_passes with dropout active.

        Returns (rel_mean, rel_std, urg_mean, urg_std) all shape (batch,).
        High std → uncertain → send to LLM.
        """
        self.train()  # enable dropout
        rels, urgs = [], []
        with torch.no_grad():
            for _ in range(n_passes):
                r, u = self.forward(x)
                rels.append(r)
                urgs.append(u)
        self.eval()
        rels = torch.stack(rels)
        urgs = torch.stack(urgs)
        return rels.mean(0), rels.std(0), urgs.mean(0), urgs.std(0)
