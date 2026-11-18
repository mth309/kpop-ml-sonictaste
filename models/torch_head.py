"""Custom PyTorch LayerAttentionMLP taste predictor (Phase 4)."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import get_device


class TrackEmbeddingDataset(Dataset):
    """PyTorch Dataset wrapping (N, 13, 768) MERT layer embeddings and (N,) ratings."""

    def __init__(self, embeddings: np.ndarray, ratings: np.ndarray) -> None:
        self.x = torch.as_tensor(embeddings, dtype=torch.float32)
        self.y = torch.as_tensor(ratings, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class _BaseLayerAttention(nn.Module):
    """Shared softmax layer-attention pooling and feature normalization for 13-layer MERT heads."""

    def __init__(self, num_layers: int = 13, embed_dim: int = 768) -> None:
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.register_buffer("feat_mean", torch.zeros(embed_dim))
        self.register_buffer("feat_std", torch.ones(embed_dim))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std.clamp(min=1e-6))

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def _pool_layers(self, x_layers: torch.Tensor, normalize_mode: str = "zscore") -> torch.Tensor:
        alpha = self.layer_weights().view(1, -1, 1)
        pooled = (x_layers * alpha).sum(dim=1)
        if normalize_mode == "l2":
            return F.normalize(pooled - self.feat_mean, p=2, dim=-1)
        return (pooled - self.feat_mean) / self.feat_std


class LayerAttentionMLP(_BaseLayerAttention):
    """Learns softmax attention weights across MERT's 13 layers + a regularized MLP head."""

    def __init__(
        self,
        num_layers: int = 13,
        embed_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.35,
    ) -> None:
        super().__init__(num_layers=num_layers, embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_layers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self._pool_layers(x_layers, normalize_mode="zscore")
        out = self.head(normed)
        return out[:, 0], out[:, 1]


class LayerAttentionLinear(_BaseLayerAttention):
    """Learns softmax attention across MERT's 13 layers + dual linear heads (score + hit logit)."""

    def __init__(
        self,
        num_layers: int = 13,
        embed_dim: int = 768,
        dropout: float = 0.25,
        normalize_mode: str = "zscore",
    ) -> None:
        super().__init__(num_layers=num_layers, embed_dim=embed_dim)
        self.normalize_mode = normalize_mode
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_dim, 2)

    def forward(self, x_layers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self._pool_layers(x_layers, normalize_mode=self.normalize_mode)
        out = self.linear(self.dropout(normed))
        return out[:, 0], out[:, 1]


class TemporalAttentionLinear(nn.Module):
    """Learns softmax attention across 5-second time windows + dual linear heads."""

    def __init__(self, embed_dim: int = 768, dropout: float = 0.25) -> None:
        super().__init__()
        self.attn_scorer = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_dim, 2)

    def forward(self, x_windows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_windows: (batch, 6, 768) -> L2-normalize each 5s window
        w_norm = F.normalize(x_windows, p=2, dim=-1)
        scores = self.attn_scorer(self.dropout(w_norm)).squeeze(-1)  # (batch, 6)
        beta = torch.softmax(scores, dim=1).unsqueeze(-1)            # (batch, 6, 1)
        pooled = F.normalize((w_norm * beta).sum(dim=1), p=2, dim=-1)
        out = self.linear(self.dropout(pooled))
        return out[:, 0], out[:, 1]


class TorchTasteModel:
    """Scikit-learn-style fit/predict wrapper around LayerAttentionMLP or LayerAttentionLinear."""

    def __init__(
        self,
        arch: str = "linear",
        hidden_dim: int = 128,
        dropout: Optional[float] = None,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        epochs: Optional[int] = None,
        batch_size: int = 64,
        normalize_mode: str = "l2",
        seed: int = 42,
        device: Optional[torch.device] = None,
    ) -> None:
        self.arch = arch
        self.hidden_dim = hidden_dim
        self.dropout = dropout if dropout is not None else (0.25 if arch in ("linear", "temporal") else 0.35)
        self.lr = lr if lr is not None else (1e-2 if arch in ("linear", "temporal") else 1e-3)
        self.weight_decay = weight_decay if weight_decay is not None else (0.01 if arch in ("linear", "temporal") else 2e-2)
        self.epochs = epochs if epochs is not None else (50 if arch in ("linear", "temporal") else 35)
        self.batch_size = batch_size
        self.normalize_mode = normalize_mode
        self.seed = seed
        self.device = device or get_device()
        self.num_layers: Optional[int] = None
        self.embed_dim: Optional[int] = None
        self.model: Optional[nn.Module] = None

    def fit(self, embeddings: np.ndarray, ratings: np.ndarray) -> "TorchTasteModel":
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        num_layers, embed_dim = embeddings.shape[1], embeddings.shape[2]
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        if self.arch == "temporal":
            model: nn.Module = TemporalAttentionLinear(
                embed_dim=embed_dim,
                dropout=self.dropout,
            ).to(self.device)
        elif self.arch == "linear":
            model = LayerAttentionLinear(
                num_layers=num_layers,
                embed_dim=embed_dim,
                dropout=self.dropout,
                normalize_mode=self.normalize_mode,
            ).to(self.device)
        else:
            model = LayerAttentionMLP(
                num_layers=num_layers,
                embed_dim=embed_dim,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
            ).to(self.device)

        # Compute per-feature mean/std across training tracks (averaged over layers)
        if hasattr(model, "set_normalization"):
            x_tensor = torch.as_tensor(embeddings, dtype=torch.float32)
            flat_mean = x_tensor.mean(dim=(0, 1)).to(self.device)
            flat_std = x_tensor.mean(dim=1).std(dim=0).to(self.device)
            model.set_normalization(flat_mean, flat_std)

        dataset = TrackEmbeddingDataset(embeddings, ratings)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        # Dual-task objective: SmoothL1 for star rating + BCEWithLogits for Hit (>= 4.0*)
        reg_criterion = nn.SmoothL1Loss(beta=0.5)
        hit_criterion = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                y_hit = (yb >= 4.0).float()
                optimizer.zero_grad(set_to_none=True)
                score_pred, hit_logit = model(xb)
                loss = reg_criterion(score_pred, yb) + 0.5 * hit_criterion(hit_logit, y_hit)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        self.model = model
        return self

    def predict_with_hit_prob(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (predicted_ratings, hit_probabilities) for >=4.0* tracks."""
        if self.model is None:
            raise RuntimeError("TorchTasteModel must be fit() before calling predict()")
        self.model.eval()
        x = torch.as_tensor(embeddings, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            score_pred, hit_logit = self.model(x)
            scores = np.clip(score_pred.cpu().numpy(), 1.0, 5.0).astype(np.float32)
            probs = torch.sigmoid(hit_logit).cpu().numpy().astype(np.float32)
        return scores, probs

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        scores, _ = self.predict_with_hit_prob(embeddings)
        return scores

    def get_layer_weights(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TorchTasteModel must be fit() first")
        with torch.no_grad():
            return self.model.layer_weights().cpu().numpy()

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("No trained model to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": {
                    "arch": self.arch,
                    "hidden_dim": self.hidden_dim,
                    "dropout": self.dropout,
                    "normalize_mode": self.normalize_mode,
                    "num_layers": self.num_layers or 13,
                    "embed_dim": self.embed_dim or 768,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: Optional[torch.device] = None) -> "TorchTasteModel":
        target_device = device or get_device()
        ckpt = torch.load(path, map_location=target_device, weights_only=True)
        cfg: Dict = ckpt["config"]
        arch = cfg.get("arch", "mlp")
        norm_mode = cfg.get("normalize_mode", "zscore")
        num_layers = int(cfg.get("num_layers", 13))
        embed_dim = int(cfg.get("embed_dim", 768))
        wrapper = cls(
            arch=arch,
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
            normalize_mode=norm_mode,
            device=target_device,
        )
        wrapper.num_layers = num_layers
        wrapper.embed_dim = embed_dim
        if arch == "temporal":
            model: nn.Module = TemporalAttentionLinear(
                embed_dim=embed_dim,
                dropout=cfg["dropout"],
            ).to(target_device)
        elif arch == "linear":
            model = LayerAttentionLinear(
                num_layers=num_layers,
                embed_dim=embed_dim,
                dropout=cfg["dropout"],
                normalize_mode=norm_mode,
            ).to(target_device)
        else:
            model = LayerAttentionMLP(
                num_layers=num_layers,
                embed_dim=embed_dim,
                hidden_dim=cfg["hidden_dim"],
                dropout=cfg["dropout"],
            ).to(target_device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        wrapper.model = model
        return wrapper
