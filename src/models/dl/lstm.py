"""LSTM 모델 — PyTorch 기반 LSTM + 학습 루프 wrapper.

ML 모델(src/models/ml.py)과 호환되는 인터페이스 (fit/predict).
다만 입력이 시퀀스 (N, L, F) 라 numpy 직접 받음 (DataFrame 아님).

학습 설정 (plan에 따라 고정):
- AdamW (lr=1e-3, weight_decay=1e-5)
- ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6)
- MSE loss
- Early stopping (patience=10, min_delta=1e-5)
- Gradient clipping (max_norm=1.0)
- Batch size 64, max_epochs 100
- Seed 42 (deterministic)

Usage:
    from src.models.dl import LSTMModel

    model = LSTMModel(
        feature_cols=feature_cols,
        L=22,
        hidden_size=64, num_layers=1, dropout=0.2,
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


PRED_FLOOR = 1e-8     # QLIKE 폭발 방지 — ML과 동일
SUPPORTED_ES_METRICS = ("mse", "qlike")


class LSTMNet(nn.Module):
    """단순 LSTM → 마지막 timestep hidden → ReLU → Linear(1).

    Input  : (B, L, F) FloatTensor
    Output : (B,)      FloatTensor
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.act = nn.ReLU()
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                    # (B, L, H)
        last = out[:, -1, :]                     # (B, H) — 마지막 timestep
        return self.head(self.act(last)).squeeze(-1)   # (B,)


class LSTMModel:
    """LSTM 학습/예측 wrapper.

    ML _BaseMLModel과 비슷한 인터페이스이지만 입력이 numpy ndarray (이미 scaling됨).

    Attributes (fit 후 채워짐)
    --------------------------
    net_ : LSTMNet              학습된 nn.Module
    best_val_loss_ : float      early stop 시점의 best metric 값 (early_stop_metric=mse/qlike)
    best_val_mse_ : float       best epoch의 valid MSE
    best_val_qlike_ : float     best epoch의 valid QLIKE
    best_epoch_ : int           best metric을 기록한 epoch 번호 (1-based)
    epochs_used_ : int          early stop이 발동된 epoch 번호 (또는 max_epochs)
    train_loss_history_ : list  epoch별 train MSE
    valid_loss_history_ : list  epoch별 early_stop_metric 값 (mse 또는 qlike)
    valid_mse_history_ : list   epoch별 valid MSE (항상 기록)
    valid_qlike_history_ : list epoch별 valid QLIKE (항상 기록)
    lr_history_ : list          epoch별 optimizer lr

    Methods
    -------
    history_df() → pd.DataFrame
        epoch별 (epoch, train_loss, valid_mse, valid_qlike, lr) 로그.
    """

    name = "LSTM"

    def __init__(
        self,
        feature_cols: List[str],
        L: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        max_epochs: int = 100,
        early_stop_patience: int = 10,
        early_stop_min_delta: float = 1e-5,
        early_stop_metric: str = "qlike",   # 'mse' or 'qlike' (default qlike → ML 정합)
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
        self.hidden_size = hidden_size
        self.num_layers = num_layers
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
        self.net_: Optional[LSTMNet] = None
        self.best_val_loss_: Optional[float] = None       # early_stop_metric 의 best 값
        self.best_val_mse_: Optional[float] = None        # best epoch의 valid MSE
        self.best_val_qlike_: Optional[float] = None      # best epoch의 valid QLIKE
        self.best_epoch_: int = 0
        self.epochs_used_: int = 0
        self.train_loss_history_: List[float] = []
        self.valid_loss_history_: List[float] = []        # early_stop_metric 의 epoch별 값
        self.valid_mse_history_: List[float] = []
        self.valid_qlike_history_: List[float] = []
        self.lr_history_: List[float] = []

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
    ) -> "LSTMModel":
        """학습 (early stopping 위해 X_valid/y_valid 권장).

        Parameters
        ----------
        X_train : (N_train, L, F) numpy float32  — 이미 scaling된 시퀀스
        y_train : (N_train,) numpy float32       — 타깃 (이미 t+1 RV)
        X_valid, y_valid : 검증용 (None이면 train loss로 early stop)
        """
        self._set_seed()

        # 1. Net + Optimizer + Scheduler + Loss
        n_features = X_train.shape[2]
        if n_features != len(self.feature_cols):
            raise ValueError(
                f"X_train n_features={n_features} != len(feature_cols)={len(self.feature_cols)}"
            )

        self.net_ = LSTMNet(
            n_features=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
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

        # 2. DataLoader (shuffle=False 시계열 순서 보존)
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

        # 3. Early stopping state
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

        # 4. Training loop
        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_one_epoch(train_loader, optimizer, loss_fn)

            if valid_loader is not None:
                val_mse, val_qlike = self._eval_valid(valid_loader, loss_fn)
            else:
                val_mse = train_loss
                val_qlike = train_loss   # qlike 측정 불가 → train loss로 대체

            # early_stop_metric에 따라 target 선택
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
        self.best_val_loss_ = best_val_metric    # early_stop_metric의 best
        self.best_val_mse_ = best_val_mse
        self.best_val_qlike_ = best_val_qlike
        self.best_epoch_ = best_epoch
        if best_state is not None:
            self.net_.load_state_dict(best_state)

        return self

    # --------------------------------------------------------------- history
    def history_df(self) -> "pd.DataFrame":
        """epoch별 (epoch, train_loss, valid_mse, valid_qlike, lr) 로그를 DataFrame으로.

        외부에서 한 DataFrame으로 합쳐 csv 저장하기 편함.
        valid_loss = early_stop_metric (mse 또는 qlike) — best 선택에 쓰인 값.
        valid_mse / valid_qlike — 항상 별도 기록.
        """
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

    # --------------------------------------------------------------- checkpoint
    def save_checkpoint(self, path, extra: Optional[dict] = None) -> None:
        """self-contained checkpoint 저장.

        state_dict + 학습 결과(best_val_loss, best_epoch 등) + 재현용 HP/feature_cols.
        `extra` dict로 임의 메타데이터 추가 가능 (regime, country, tier, L 등).

        나중에 `LSTMModel.from_checkpoint(path)`로 학습 환경 없이도 복원.
        """
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
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
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
    def from_checkpoint(cls, path, device: Optional[str] = None) -> "LSTMModel":
        """저장된 checkpoint에서 LSTMModel 인스턴스 복원.

        fit 없이 바로 predict 가능. 학습 메타데이터(best_val_loss 등)도 복원.
        """
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
        inst.net_ = LSTMNet(
            n_features=len(ckpt["feature_cols"]),
            hidden_size=inst.hidden_size,
            num_layers=inst.num_layers,
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

    # --------------------------------------------------------------- predict
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """배치 단위로 추론. (N_test,) float32 numpy 반환 (1e-8 floor 적용)."""
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
        """valid set에서 MSE + QLIKE 둘 다 측정 (forward 1회).

        QLIKE은 numpy 기반 (src.eval.metrics.qlike) — y_pred는 PRED_FLOOR로 clip.
        """
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
