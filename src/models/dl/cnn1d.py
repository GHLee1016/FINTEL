"""1D-CNN 모델 — PyTorch 기반 WeightNorm(Conv1d) + last timestep + 학습 루프 wrapper.

모델 구조: WeightNorm(`nn.Conv1d`) 여러 layer + ReLU + Dropout + last timestep + Linear(1).
fit / predict / save_checkpoint / from_checkpoint 인터페이스 제공.

설계 포인트:
- Input shape: (B, L, F)
- forward 첫 줄에서 `x.transpose(1, 2)` → (B, F, L) (Conv1d format)
- 마지막 timestep `out[:, :, -1]`을 head로 사용 (LSTM/TCN과 동일).
  - RV는 lag-1 의존성이 압도적 → AvgPool(전체 평균) 대신 last timestep이 적합.
- Normalization: 각 Conv1d에 `nn.utils.weight_norm` 적용 (TCN과 동일 패턴).
  - 시계열 batch 통계 불안정성 회피 (BatchNorm 비추천 도메인).
- L별 receptive field 부족은 layer 수/kernel로 조정 (hp). 현재 RF = 1 + 2·num_layers (k=3 기준).

학습 설정 (plan에 따라 고정):
- AdamW (lr=1e-3, weight_decay=1e-5)
- ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6)
- MSE loss
- Early stopping (patience=10, early_stop_metric='qlike')
- Gradient clipping (max_norm=1.0)
- Batch size 64, max_epochs 100, seed=42

Usage:
    from src.models.dl import CNN1DModel
    model = CNN1DModel(
        feature_cols=feature_cols, L=22,
        hidden_channels=64, num_layers=3, kernel_size=3, dropout=0.2,
    )
    model.fit(X_train, y_train, X_valid, y_valid)
    y_pred = model.predict(X_test)
"""

from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...dl.dataset import SequenceDataset
from ...eval.metrics import qlike as qlike_fn


PRED_FLOOR = 1e-8
SUPPORTED_ES_METRICS = ("mse", "qlike")


class CNN1DNet(nn.Module):
    """1D-CNN: stacked WeightNorm(Conv1d) → last timestep → Linear.

    Input  : (B, L, F) FloatTensor
    Output : (B,)      FloatTensor

    각 Conv 블록: WeightNorm(Conv1d(kernel=K, padding=K//2)) → ReLU → Dropout
    padding='same' 효과로 시퀀스 길이 L 유지 → L에 무관하게 작동.
    마지막 timestep `out[:, :, -1]`을 head로 사용 (LSTM/TCN과 동일).
      - RV는 lag-1 의존성이 압도적이라 AvgPool(전체 평균)보다 last가 적합.
      - AvgPool은 최근 정보를 22~252일 평균에 묻어버려 부적합 (실증적 확인됨).

    Normalization: 각 Conv1d에 `nn.utils.weight_norm` 적용 (TCN과 동일 패턴).
      - 시계열의 batch 통계 불안정성 회피 (BatchNorm 비추천 도메인).
      - 작은 dataset (위기 cell L=252는 76 sample)에서 안정 학습.
    """

    def __init__(
        self,
        n_features: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd for 'same' padding, got {kernel_size}")
        layers: List[nn.Module] = []
        in_c = n_features
        for _ in range(num_layers):
            layers.append(
                nn.utils.weight_norm(
                    nn.Conv1d(in_c, hidden_channels, kernel_size=kernel_size,
                              padding=kernel_size // 2)
                )
            )
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_c = hidden_channels
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) → Conv1d format (B, F, L)
        x = x.transpose(1, 2).contiguous()
        out = self.body(x)                       # (B, hidden, L)
        out = out[:, :, -1]                      # (B, hidden) — last timestep
        return self.head(out).squeeze(-1)        # (B,)


class CNN1DModel:
    """1D-CNN 학습/예측 wrapper.

    fit(X_train, y_train, X_valid, y_valid) / predict(X_test) /
    history_df() / save_checkpoint() / from_checkpoint() 인터페이스.
    """

    name = "1DCNN"

    def __init__(
        self,
        feature_cols: List[str],
        L: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
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
        self.L = L
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
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

        # fit 후 채워짐
        self.net_: Optional[CNN1DNet] = None
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

    # ----------------------------------------------------------------- fit
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
    ) -> "CNN1DModel":
        self._set_seed()

        n_features = X_train.shape[2]
        if n_features != len(self.feature_cols):
            raise ValueError(
                f"X_train n_features={n_features} != len(feature_cols)={len(self.feature_cols)}"
            )

        self.net_ = CNN1DNet(
            n_features=n_features,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.lr_factor,
            patience=self.lr_patience, min_lr=self.lr_min,
        )
        loss_fn = nn.MSELoss()

        train_loader = DataLoader(
            SequenceDataset(X_train, y_train),
            batch_size=self.batch_size, shuffle=False, num_workers=0,
        )
        valid_loader = None
        if X_valid is not None and y_valid is not None and len(X_valid) > 0:
            valid_loader = DataLoader(
                SequenceDataset(X_valid, y_valid),
                batch_size=self.batch_size, shuffle=False, num_workers=0,
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

    # ----------------------------------------------------------------- predict
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.net_ is None:
            raise RuntimeError("call .fit() first")
        self.net_.eval()
        ds = SequenceDataset(X_test, np.zeros(len(X_test), dtype=np.float32))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        preds = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                out = self.net_(x).cpu().numpy()
                preds.append(out)
        y_pred = np.concatenate(preds).astype(np.float32)
        return np.clip(y_pred, PRED_FLOOR, None)

    # ----------------------------------------------------------------- helpers
    def _set_seed(self):
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

    def _eval_loss(self, loader, loss_fn) -> float:
        self.net_.eval()
        total_loss, total_n = 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.net_(x)
                loss = loss_fn(out, y)
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

    # ----------------------------------------------------------------- history
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

    # ----------------------------------------------------------------- checkpoint
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
            "L": int(self.L),
            "hp": {
                "hidden_channels": self.hidden_channels,
                "num_layers": self.num_layers,
                "kernel_size": self.kernel_size,
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
    def from_checkpoint(cls, path, device: Optional[str] = None) -> "CNN1DModel":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=device, weights_only=False)
        hp = ckpt["hp"]
        inst = cls(feature_cols=ckpt["feature_cols"], L=ckpt["L"], device=device, **hp)
        inst.net_ = CNN1DNet(
            n_features=len(ckpt["feature_cols"]),
            hidden_channels=inst.hidden_channels,
            num_layers=inst.num_layers,
            kernel_size=inst.kernel_size,
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
