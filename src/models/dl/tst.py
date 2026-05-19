"""TST 모델 — Time Series Transformer (Encoder-only).

모델 구조: Linear(F→d_model) + Sinusoidal PE + TransformerEncoder × N + last timestep + Linear(1).
fit / predict / save_checkpoint / from_checkpoint 인터페이스 제공.

설계 포인트:
- Input shape: (B, L, F) — 다른 DL 모델과 동일
- **Encoder-only**: regression task (sequence-to-one)라 decoder 불필요.
  multi-step forecasting이 아니라 1-step ahead RV 예측이므로 encoder만으로 충분.
- **Sinusoidal positional encoding**: parameter-free, L=22/60/252 모두 generalize.
  Learnable PE는 작은 dataset (위기 cell ~76개)에서 학습 어려움.
- **Pre-norm** (norm_first=True): 작은 dataset에서 학습 안정성. residual이 LN 거치지 않고
  직접 흐름. 시계열 transformer 통념 (Wang et al. 2019, "On Layer Normalization in
  the Transformer Architecture").
- **마지막 timestep head**: LSTM/TCN/CNN1D(수정 후)와 일관. RV lag-1 신호 직접 활용.
  AvgPool은 RV의 lag-1 신호를 L 전체 평균에 묻어버려 부적합.
- **Causal mask 없음**: input window 자체가 sliding window로 미래 정보 차단된 상태.
  window 내부는 양방향 attention OK.

학습 설정 (LSTM/CNN1D/TCN과 동일):
- AdamW (lr=1e-3, weight_decay=1e-5)
- ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6)
- MSE loss (early stop은 QLIKE 기준)
- Early stopping (patience=10, early_stop_metric='qlike')
- Gradient clipping (max_norm=1.0)
- Batch size 64, max_epochs 100, seed=42

Usage:
    from src.models.dl import TSTModel
    model = TSTModel(
        feature_cols=feature_cols, L=22,
        d_model=64, nhead=4, num_layers=2,
        dim_feedforward=128, dropout=0.2,
    )
    model.fit(X_train, y_train, X_valid, y_valid)
    y_pred = model.predict(X_test)
"""

from __future__ import annotations

import copy
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...dl.dataset import SequenceDataset
from ...eval.metrics import qlike as qlike_fn


PRED_FLOOR = 1e-8
SUPPORTED_ES_METRICS = ("mse", "qlike")


class _SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    `register_buffer`로 저장 → 학습 안 됨 + state_dict에는 자동 포함.
    """

    def __init__(self, d_model: int, max_len: int = 300):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, max_len, d_model)로 unsqueeze → batch_first 호환
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model) → x + pe[:, :L, :]
        return x + self.pe[:, : x.size(1), :]


class TSTNet(nn.Module):
    """Time Series Transformer (encoder-only).

    Input  : (B, L, F) FloatTensor
    Output : (B,)      FloatTensor

    Pipeline:
      Linear(F→d_model)
        → + SinusoidalPE → Dropout
        → TransformerEncoderLayer × num_layers (pre-norm, batch_first)
        → final LayerNorm
        → out[:, -1, :]  (마지막 timestep)
        → Linear(d_model, 1)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        activation: str = "gelu",
        norm_first: bool = True,
        max_len: int = 300,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model({d_model}) must be divisible by nhead({nhead})"
            )

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = _SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # pre-norm 사용 시 final LayerNorm 필요 (PyTorch 내장에는 없음)
        self.final_norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        x = self.input_proj(x)               # (B, L, d_model)
        x = self.pos_encoding(x)             # + sinusoidal PE
        x = self.embed_dropout(x)            # dropout 0.2
        out = self.encoder(x)                # (B, L, d_model)
        out = self.final_norm(out)
        last = out[:, -1, :]                 # (B, d_model) — 마지막 timestep
        return self.head(last).squeeze(-1)   # (B,)


class TSTModel:
    """TST 학습/예측 wrapper.

    fit(X_train, y_train, X_valid, y_valid) / predict(X_test) /
    history_df() / save_checkpoint() / from_checkpoint() 인터페이스.
    """

    name = "TST"

    def __init__(
        self,
        feature_cols: List[str],
        L: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        activation: str = "gelu",
        norm_first: bool = True,
        max_len: int = 300,
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
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        self.max_len = max_len
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
        self.net_: Optional[TSTNet] = None
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
    ) -> "TSTModel":
        self._set_seed()

        n_features = X_train.shape[2]
        if n_features != len(self.feature_cols):
            raise ValueError(
                f"X_train n_features={n_features} != len(feature_cols)={len(self.feature_cols)}"
            )
        if X_train.shape[1] > self.max_len:
            raise ValueError(
                f"X_train L={X_train.shape[1]} > max_len={self.max_len} "
                f"(SinusoidalPE 미준비). max_len을 늘리세요."
            )

        self.net_ = TSTNet(
            n_features=n_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            norm_first=self.norm_first,
            max_len=self.max_len,
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
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout,
                "activation": self.activation,
                "norm_first": self.norm_first,
                "max_len": self.max_len,
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
    def from_checkpoint(cls, path, device: Optional[str] = None) -> "TSTModel":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=device, weights_only=False)
        hp = ckpt["hp"]
        inst = cls(feature_cols=ckpt["feature_cols"], L=ckpt["L"], device=device, **hp)
        inst.net_ = TSTNet(
            n_features=len(ckpt["feature_cols"]),
            d_model=inst.d_model,
            nhead=inst.nhead,
            num_layers=inst.num_layers,
            dim_feedforward=inst.dim_feedforward,
            dropout=inst.dropout,
            activation=inst.activation,
            norm_first=inst.norm_first,
            max_len=inst.max_len,
        ).to(inst.device)
        inst.net_.load_state_dict(ckpt["state_dict"])
        inst.net_.eval()
        inst.best_val_loss_ = ckpt.get("best_val_loss")
        inst.best_val_mse_ = ckpt.get("best_val_mse")
        inst.best_val_qlike_ = ckpt.get("best_val_qlike")
        inst.best_epoch_ = ckpt.get("best_epoch", 0)
        inst.epochs_used_ = ckpt.get("epochs_used", 0)
        return inst
