"""Group-Structured Neural Network for DL realized-volatility forecasting.

This is the first stable GroupNN baseline: features are split into economic
groups, each group is encoded by a small MLP after temporal mean pooling, and
the concatenated group representations are passed to a final prediction head.
Gating can be layered on after this baseline is validated.
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


class GroupNNNet(nn.Module):
    """Group-structured MLP baseline.

    Input  : (B, L, F)
    Output : (B,)
    """

    def __init__(
        self,
        feature_groups: Dict[str, List[int]],
        group_hidden: int = 32,
        final_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_groups = {k: list(v) for k, v in feature_groups.items()}
        self.group_names = list(self.feature_groups.keys())
        self.group_encoders = nn.ModuleDict()

        for name, indices in self.feature_groups.items():
            in_dim = len(indices)
            self.group_encoders[name] = nn.Sequential(
                nn.Linear(in_dim, group_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(group_hidden, group_hidden),
                nn.ReLU(),
            )

        self.head = nn.Sequential(
            nn.Linear(group_hidden * len(self.group_names), final_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reps = []
        for name in self.group_names:
            idx = self.feature_groups[name]
            group_x = x[:, :, idx].mean(dim=1)
            reps.append(self.group_encoders[name](group_x))
        z = torch.cat(reps, dim=1)
        return self.head(z).squeeze(-1)


class GroupNNModel:
    """Train/predict wrapper compatible with the existing DL model interface."""

    name = "GroupNN"

    def __init__(
        self,
        feature_cols: List[str],
        L: int,
        group_hidden: int = 32,
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
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        if early_stop_metric not in SUPPORTED_ES_METRICS:
            raise ValueError(
                f"early_stop_metric must be one of {SUPPORTED_ES_METRICS}, got {early_stop_metric!r}"
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
        loss_fn = nn.MSELoss()

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
                print(
                    f"  epoch {epoch:3d} | train {train_loss:.5f} | "
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
        ).to(inst.device)
        inst.net_.load_state_dict(ckpt["state_dict"])
        inst.net_.eval()
        inst.best_val_loss_ = ckpt.get("best_val_loss")
        inst.best_val_mse_ = ckpt.get("best_val_mse")
        inst.best_val_qlike_ = ckpt.get("best_val_qlike")
        inst.best_epoch_ = ckpt.get("best_epoch", 0)
        inst.epochs_used_ = ckpt.get("epochs_used", 0)
        return inst

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.net_ is None:
            raise RuntimeError("call .fit() first")
        self.net_.eval()
        ds = SequenceDataset(X_test, np.zeros(len(X_test), dtype=np.float32))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        preds = []
        with torch.no_grad():
            for x, _ in loader:
                out = self.net_(x.to(self.device)).cpu().numpy()
                preds.append(out)
        y_pred = np.concatenate(preds).astype(np.float32)
        return np.clip(y_pred, PRED_FLOOR, None)

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
