"""Group-Structured Neural Network for DL realized-volatility forecasting.

Features are split into economic groups (Core / Momentum / Macro / Spillover).
Each group is encoded by a small MLP from temporal summary statistics
(configurable via ``summary_type``), and the four group representations are
**concatenated** into a single vector passed through a final MLP head.

Group importance for economic interpretation is derived from the L2 norm of
each group's trained representation (``predict(return_gate=True)`` /
``get_gate_weights()``).  Since the reps are trained through the main loss,
their norms reflect how strongly each group activated the network — unlike a
separate gate linear layer whose weights would receive zero gradients.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...dl.dataset import SequenceDataset
from ...eval.metrics import qlike as qlike_fn
from .group_utils import build_feature_groups


PRED_FLOOR = 1e-8
SUPPORTED_ES_METRICS = ("mse", "qlike")

# Temporal summary options
# "mean_last"     : [mean, last]                          n_stats=2  (original)
# "mean_std_last" : [mean, std, last]                     n_stats=3  (volatility clustering)
# "har"           : [monthly_mean, weekly_mean, daily]    n_stats=3  (HAR-style multi-scale)
# "har_std"       : [monthly_mean, weekly_mean, daily, std] n_stats=4 (HAR + clustering)
SUMMARY_TYPES = ("mean_last", "mean_std_last", "har", "har_std")
SUMMARY_N_STATS = {"mean_last": 2, "mean_std_last": 3, "har": 3, "har_std": 4}


class QLikeLoss(nn.Module):
    """QLIKE loss for training.

    QLIKE(y, ŷ) = mean(y/ŷ - log(y/ŷ) - 1)

    Directly optimises the evaluation metric used by all models in this
    project (ML Optuna objective, DL early-stop metric).  Gradient:
        d/dŷ = (ŷ - y) / ŷ²
    which is well-behaved as long as ŷ > 0 (enforced by ``floor``).

    Parameters
    ----------
    floor : float
        Minimum predicted value; prevents log(0) / division by zero.
    """

    def __init__(self, floor: float = PRED_FLOOR):
        super().__init__()
        self.floor = floor

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(min=self.floor)
        ratio = target / pred
        return (ratio - torch.log(ratio) - 1.0).mean()


def _make_encoder(
    in_dim: int,
    hidden: int,
    dropout: float,
    use_layer_norm: bool,
) -> nn.Sequential:
    """Two-layer MLP encoder with optional LayerNorm (pre-activation)."""
    layers: list = [nn.Linear(in_dim, hidden)]
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden))
    layers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)]
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _make_head(
    in_dim: int,
    hidden: int,
    dropout: float,
    use_layer_norm: bool,
) -> nn.Sequential:
    """Final prediction head with optional LayerNorm."""
    layers: list = [nn.Linear(in_dim, hidden)]
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden))
    layers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)]
    return nn.Sequential(*layers)


class GroupNNNet(nn.Module):
    """Group-structured MLP with optional LayerNorm and cross-group attention.

    Input  : (B, L, F)
    Output : (B,) or ((B,), (B, n_groups))

    Options
    -------
    use_layer_norm : bool
        Insert LayerNorm after each Linear layer in group encoders and head.
    use_cross_attn : bool
        After individual group encoding, apply a single multi-head self-attention
        layer across the n_groups representations (with residual connection).
        Allows groups to interact before the final head.
    n_heads : int
        Number of attention heads (only used when use_cross_attn=True).
        Must divide group_hidden evenly.
    """

    def __init__(
        self,
        feature_groups: Dict[str, List[int]],
        group_hidden: int = 64,
        final_hidden: int = 64,
        dropout: float = 0.2,
        summary_type: str = "mean_std_last",
        use_layer_norm: bool = False,
        use_cross_attn: bool = False,
        n_heads: int = 4,
    ):
        if summary_type not in SUMMARY_TYPES:
            raise ValueError(f"summary_type must be one of {SUMMARY_TYPES}, got {summary_type!r}")
        if use_cross_attn and group_hidden % n_heads != 0:
            raise ValueError(
                f"group_hidden ({group_hidden}) must be divisible by n_heads ({n_heads})"
            )
        super().__init__()
        self.feature_groups = {k: list(v) for k, v in feature_groups.items()}
        self.group_names = list(self.feature_groups.keys())
        self.summary_type = summary_type
        self.use_layer_norm = use_layer_norm
        self.use_cross_attn = use_cross_attn
        n_groups = len(self.group_names)

        # ── Per-group encoders ───────────────────────────────────────────────
        n_stats = SUMMARY_N_STATS[summary_type]
        self.group_encoders = nn.ModuleDict()
        for name, indices in self.feature_groups.items():
            in_dim = len(indices) * n_stats
            self.group_encoders[name] = _make_encoder(
                in_dim, group_hidden, dropout, use_layer_norm
            )

        # ── Cross-group attention (optional) ────────────────────────────────
        # Self-attention over (B, n_groups, group_hidden) with residual.
        # Pre-norm applied when use_layer_norm=True.
        if use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=group_hidden, num_heads=n_heads,
                dropout=dropout, batch_first=True,
            )
            if use_layer_norm:
                self.cross_attn_ln = nn.LayerNorm(group_hidden)

        # ── Prediction head ──────────────────────────────────────────────────
        # Input dim = group_hidden * n_groups  (e.g. 64 * 4 = 256)
        self.head = _make_head(
            group_hidden * n_groups, final_hidden, dropout, use_layer_norm
        )

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        reps = []
        L = x.shape[1]
        for name in self.group_names:
            idx = self.feature_groups[name]
            group_x = x[:, :, idx]                              # (B, L, G_g)
            group_summary = torch.cat(
                self._summarise(group_x, L), dim=1
            )                                                    # (B, n_stats*G_g)
            rep = self.group_encoders[name](group_summary)      # (B, group_hidden)
            reps.append(rep)

        # ── Cross-group interaction (optional) ───────────────────────────────
        if self.use_cross_attn:
            stacked = torch.stack(reps, dim=1)                  # (B, n_groups, H)
            attn_in = self.cross_attn_ln(stacked) if self.use_layer_norm else stacked
            attended, _ = self.cross_attn(attn_in, attn_in, attn_in)
            stacked = stacked + attended                         # residual
            reps = [stacked[:, i, :] for i in range(len(self.group_names))]

        # ── Prediction head ──────────────────────────────────────────────────
        z = torch.cat(reps, dim=1)                              # (B, group_hidden*n_groups)
        out = self.head(z).squeeze(-1)                          # (B,)

        if return_gate:
            # Group importance: normalised L2 norm of each trained rep.
            norms = torch.stack(
                [rep.norm(dim=1) for rep in reps], dim=1       # (B, n_groups)
            )
            importance = norms / norms.sum(dim=1, keepdim=True) # (B, n_groups)
            return out, importance
        return out

    def _summarise(self, g: torch.Tensor, L: int) -> list:
        """Return temporal summary statistics for one group tensor (B, L, G_g)."""
        st = self.summary_type
        if st == "mean_last":
            return [g.mean(dim=1), g[:, -1, :]]
        elif st == "mean_std_last":
            return [g.mean(dim=1), g.std(dim=1, unbiased=False), g[:, -1, :]]
        elif st == "har":
            # HAR-style: monthly → weekly → daily (slow to fast)
            m = g[:, -min(22, L):, :].mean(dim=1)
            w = g[:, -min(5,  L):, :].mean(dim=1)
            d = g[:, -1, :]
            return [m, w, d]
        else:  # "har_std"
            m = g[:, -min(22, L):, :].mean(dim=1)
            w = g[:, -min(5,  L):, :].mean(dim=1)
            d = g[:, -1, :]
            s = g.std(dim=1, unbiased=False)
            return [m, w, d, s]


class GroupNNModel:
    """Train/predict wrapper compatible with the existing DL model interface."""

    name = "GroupNN"

    def __init__(
        self,
        feature_cols: List[str],
        L: int,
        group_hidden: int = 64,
        final_hidden: int = 64,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        max_epochs: int = 100,
        early_stop_patience: int = 10,
        early_stop_min_delta: float = 1e-5,
        early_stop_metric: str = "qlike",
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        lr_min: float = 1e-6,
        grad_clip: float = 1.0,
        seed: int = 42,
        use_qlike_loss: bool = False,
        summary_type: str = "mean_std_last",
        use_layer_norm: bool = False,
        use_cross_attn: bool = False,
        n_heads: int = 4,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        if early_stop_metric not in SUPPORTED_ES_METRICS:
            raise ValueError(
                f"early_stop_metric must be one of {SUPPORTED_ES_METRICS}, got {early_stop_metric!r}"
            )
        if summary_type not in SUMMARY_TYPES:
            raise ValueError(
                f"summary_type must be one of {SUMMARY_TYPES}, got {summary_type!r}"
            )

        self.feature_cols = list(feature_cols)
        self.feature_groups = build_feature_groups(self.feature_cols)
        self.L = L
        self.group_hidden = group_hidden
        self.final_hidden = final_hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.early_stop_metric = early_stop_metric
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.grad_clip = grad_clip
        self.seed = seed
        self.use_qlike_loss = use_qlike_loss
        self.summary_type = summary_type
        self.use_layer_norm = use_layer_norm
        self.use_cross_attn = use_cross_attn
        self.n_heads = n_heads
        self.verbose = verbose

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.net_: Optional[GroupNNNet] = None
        self.best_val_loss_: Optional[float] = None
        self.best_val_mse_: Optional[float] = None
        self.best_val_qlike_: Optional[float] = None
        self.best_epoch_: int = 0
        self.epochs_used_: int = 0
        self.train_loss_history_: List[float] = []
        self.valid_loss_history_: List[float] = []
        self.valid_mse_history_: List[float] = []
        self.valid_qlike_history_: List[float] = []
        self.lr_history_: List[float] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
    ) -> "GroupNNModel":
        self._set_seed()

        n_features = X_train.shape[2]
        if n_features != len(self.feature_cols):
            raise ValueError(
                f"X_train n_features={n_features} != len(feature_cols)={len(self.feature_cols)}"
            )

        self.net_ = GroupNNNet(
            feature_groups=self.feature_groups,
            group_hidden=self.group_hidden,
            final_hidden=self.final_hidden,
            dropout=self.dropout,
            summary_type=self.summary_type,
            use_layer_norm=self.use_layer_norm,
            use_cross_attn=self.use_cross_attn,
            n_heads=self.n_heads,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.lr_min,
        )
        loss_fn = QLikeLoss() if self.use_qlike_loss else nn.MSELoss()

        train_loader = DataLoader(
            SequenceDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        valid_loader = None
        if X_valid is not None and y_valid is not None and len(X_valid) > 0:
            valid_loader = DataLoader(
                SequenceDataset(X_valid, y_valid),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )

        best_val_metric = float("inf")
        best_val_mse = float("inf")
        best_val_qlike = float("inf")
        best_epoch = 0
        best_state: Optional[dict] = None
        patience_counter = 0
        self.train_loss_history_ = []
        self.valid_loss_history_ = []
        self.valid_mse_history_ = []
        self.valid_qlike_history_ = []
        self.lr_history_ = []

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_one_epoch(train_loader, optimizer, loss_fn)

            if valid_loader is not None:
                val_mse, val_qlike = self._eval_valid(valid_loader, loss_fn)
            else:
                val_mse = train_loss
                val_qlike = train_loss

            target = val_qlike if self.early_stop_metric == "qlike" else val_mse

            lr_now = optimizer.param_groups[0]["lr"]
            self.train_loss_history_.append(train_loss)
            self.valid_loss_history_.append(target)
            self.valid_mse_history_.append(val_mse)
            self.valid_qlike_history_.append(val_qlike)
            self.lr_history_.append(lr_now)
            scheduler.step(target)

            if target < best_val_metric - self.early_stop_min_delta:
                best_val_metric = target
                best_val_mse = val_mse
                best_val_qlike = val_qlike
                best_epoch = epoch
                best_state = copy.deepcopy(self.net_.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if self.verbose:
                loss_label = "train_qlike" if self.use_qlike_loss else "train_mse"
                print(
                    f"  epoch {epoch:3d} | {loss_label} {train_loss:.5f} | "
                    f"val_mse {val_mse:.5f} | val_qlike {val_qlike:.5f} | "
                    f"lr {lr_now:.1e} | patience {patience_counter}/{self.early_stop_patience}"
                )

            if patience_counter >= self.early_stop_patience:
                break

        self.epochs_used_ = epoch
        self.best_val_loss_ = best_val_metric
        self.best_val_mse_ = best_val_mse
        self.best_val_qlike_ = best_val_qlike
        self.best_epoch_ = best_epoch
        if best_state is not None:
            self.net_.load_state_dict(best_state)

        return self

    def history_df(self) -> "pd.DataFrame":
        import pandas as pd

        n = len(self.train_loss_history_)
        return pd.DataFrame({
            "epoch": list(range(1, n + 1)),
            "train_loss": self.train_loss_history_,
            "valid_loss": self.valid_loss_history_,
            "valid_mse": self.valid_mse_history_,
            "valid_qlike": self.valid_qlike_history_,
            "lr": self.lr_history_,
        })

    def save_checkpoint(self, path, extra: Optional[dict] = None) -> None:
        if self.net_ is None:
            raise RuntimeError("call .fit() first")
        ckpt = {
            "state_dict": self.net_.state_dict(),
            "best_val_loss": float(self.best_val_loss_) if self.best_val_loss_ is not None else None,
            "best_val_mse": float(self.best_val_mse_) if self.best_val_mse_ is not None else None,
            "best_val_qlike": float(self.best_val_qlike_) if self.best_val_qlike_ is not None else None,
            "best_epoch": int(self.best_epoch_),
            "epochs_used": int(self.epochs_used_),
            "feature_cols": list(self.feature_cols),
            "feature_groups": self.feature_groups,
            "L": int(self.L),
            "hp": {
                "group_hidden": self.group_hidden,
                "final_hidden": self.final_hidden,
                "dropout": self.dropout,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "early_stop_patience": self.early_stop_patience,
                "early_stop_min_delta": self.early_stop_min_delta,
                "early_stop_metric": self.early_stop_metric,
                "lr_patience": self.lr_patience,
                "lr_factor": self.lr_factor,
                "lr_min": self.lr_min,
                "grad_clip": self.grad_clip,
                "seed": self.seed,
                "use_qlike_loss": self.use_qlike_loss,
                "summary_type": self.summary_type,
                "use_layer_norm": self.use_layer_norm,
                "use_cross_attn": self.use_cross_attn,
                "n_heads": self.n_heads,
            },
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    @classmethod
    def from_checkpoint(cls, path, device: Optional[str] = None) -> "GroupNNModel":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=device, weights_only=False)
        hp = ckpt["hp"]
        # Backward-compat: older checkpoints lack these keys → default False/4
        hp.setdefault("use_layer_norm", False)
        hp.setdefault("use_cross_attn", False)
        hp.setdefault("n_heads", 4)
        inst = cls(
            feature_cols=ckpt["feature_cols"],
            L=ckpt["L"],
            device=device,
            **hp,
        )
        inst.net_ = GroupNNNet(
            feature_groups=inst.feature_groups,
            group_hidden=inst.group_hidden,
            final_hidden=inst.final_hidden,
            dropout=inst.dropout,
            summary_type=inst.summary_type,
            use_layer_norm=inst.use_layer_norm,
            use_cross_attn=inst.use_cross_attn,
            n_heads=inst.n_heads,
        ).to(inst.device)
        inst.net_.load_state_dict(ckpt["state_dict"])
        inst.net_.eval()
        inst.best_val_loss_ = ckpt.get("best_val_loss")
        inst.best_val_mse_ = ckpt.get("best_val_mse")
        inst.best_val_qlike_ = ckpt.get("best_val_qlike")
        inst.best_epoch_ = ckpt.get("best_epoch", 0)
        inst.epochs_used_ = ckpt.get("epochs_used", 0)
        return inst

    def predict(self, X_test: np.ndarray, return_gate: bool = False):
        if self.net_ is None:
            raise RuntimeError("call .fit() first")
        self.net_.eval()
        ds = SequenceDataset(X_test, np.zeros(len(X_test), dtype=np.float32))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        preds = []
        gates = []
        with torch.no_grad():
            for x, _ in loader:
                if return_gate:
                    out, gate = self.net_(x.to(self.device), return_gate=True)
                    gates.append(gate.cpu().numpy())
                else:
                    out = self.net_(x.to(self.device))
                preds.append(out.cpu().numpy())
        y_pred = np.concatenate(preds).astype(np.float32)
        y_pred = np.clip(y_pred, PRED_FLOOR, None)
        if return_gate:
            gate_weights = np.concatenate(gates).astype(np.float32)
            return y_pred, gate_weights
        return y_pred

    def get_gate_weights(self, X: np.ndarray) -> "pd.DataFrame":
        import pandas as pd

        _pred, gate_weights = self.predict(X, return_gate=True)
        return pd.DataFrame(gate_weights, columns=list(self.feature_groups.keys()))

    def _set_seed(self) -> None:
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)

    def _train_one_epoch(self, loader, optimizer, loss_fn) -> float:
        self.net_.train()
        total_loss, total_n = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out = self.net_(x)
            loss = loss_fn(out, y)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), self.grad_clip)
            optimizer.step()
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_n += bs
        return total_loss / max(total_n, 1)

    def _eval_valid(self, loader, loss_fn) -> tuple:
        self.net_.eval()
        total_loss, total_n = 0.0, 0
        y_pred_all = []
        y_true_all = []
        with torch.no_grad():
            for x, y in loader:
                x_dev, y_dev = x.to(self.device), y.to(self.device)
                out = self.net_(x_dev)
                loss = loss_fn(out, y_dev)
                bs = y_dev.size(0)
                total_loss += loss.item() * bs
                total_n += bs
                y_pred_all.append(out.cpu().numpy())
                y_true_all.append(y_dev.cpu().numpy())
        mse_val = total_loss / max(total_n, 1)
        y_pred = np.clip(np.concatenate(y_pred_all), PRED_FLOOR, None)
        y_true = np.concatenate(y_true_all)
        qlike_val = qlike_fn(y_true, y_pred)
        return mse_val, qlike_val
