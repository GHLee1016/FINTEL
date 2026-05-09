"""
dl_trainer.py
-------------
DL 모델 학습 / 예측 / 튜닝 / 프로토콜 실행.

주요 함수
---------
qlike_loss       : QLIKE 손실 함수
train_dl_model   : EarlyStopping 포함 학습
predict_dl       : 예측 → numpy
tune_dl_model    : Optuna 튜닝
run_dl_static    : static 프로토콜
run_dl_expanding : expanding 프로토콜 (refit_every=5)
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from torch.utils.data import DataLoader

from src.models.dl import make_dl_model, DL_SEQ_LEN
from src.preprocess import CustomScaler
from src.preprocess.sequence import build_loaders

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _scale_frames(
    train_df: pd.DataFrame,
    other_dfs: list[pd.DataFrame],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[pd.DataFrame], CustomScaler]:
    """
    ML과 동일한 CustomScaler를 train_df에 fit하고 나머지 DataFrame에는 transform만 적용.

    Returns
    -------
    scaled_train, scaled_others, scaler
    """
    scaler = CustomScaler()

    scaled_train = train_df.copy()
    scaled_train.loc[:, feature_cols] = scaler.fit_transform(
        train_df[feature_cols].values,
        feature_cols,
    )

    scaled_others: list[pd.DataFrame] = []
    for df in other_dfs:
        scaled_df = df.copy()
        scaled_df.loc[:, feature_cols] = scaler.transform(
            df[feature_cols].values,
            feature_cols,
        )
        scaled_others.append(scaled_df)

    return scaled_train, scaled_others, scaler


# ──────────────────────────────────────────────
# 1. QLIKE Loss
# ──────────────────────────────────────────────
def qlike_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    eps   : float = 1e-8,
) -> torch.Tensor:
    """
    QLIKE = mean( log(ŷ) + y / ŷ )

    softplus 출력으로 ŷ > 0 보장되나 eps로 추가 안전장치.
    """
    y_pred = y_pred.clamp(min=eps)
    return (torch.log(y_pred) + y_true / y_pred).mean()


# ──────────────────────────────────────────────
# 2. 학습 루프
# ──────────────────────────────────────────────
def train_dl_model(
    model       : nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    max_epochs  : int   = 100,
    patience    : int   = 10,
    lr          : float = 1e-3,
    device      : torch.device = DEVICE,
) -> tuple[nn.Module, float]:
    """
    QLIKE loss 학습 + valid QLIKE 기준 EarlyStopping.

    Returns
    -------
    best_model     : valid QLIKE 최소 시점 모델 (deepcopy)
    best_val_qlike : 최소 valid QLIKE 값
    """
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=max(2, patience // 2),
    )

    best_val   = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for _ in range(max_epochs):
        # ── Train ──
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = qlike_loss(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ── Validation ──
        val_q = _eval_qlike(model, valid_loader, device)
        scheduler.step(val_q)

        if val_q < best_val:
            best_val   = val_q
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val


def _eval_qlike(
    model : nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            loss   = qlike_loss(model(X_b), y_b)
            total += loss.item() * len(y_b)
            n     += len(y_b)
    return total / n if n > 0 else float('inf')


# ──────────────────────────────────────────────
# 3. 예측
# ──────────────────────────────────────────────
def predict_dl(
    model : nn.Module,
    loader: DataLoader,
    device: torch.device = DEVICE,
) -> np.ndarray:
    """DataLoader 전체 예측 → numpy 1-D array."""
    model.eval()
    preds = []
    with torch.no_grad():
        for X_b, _ in loader:
            preds.append(model(X_b.to(device)).cpu().numpy())
    return np.concatenate(preds)


# ──────────────────────────────────────────────
# 4. Optuna 탐색 공간
# ──────────────────────────────────────────────
def _search_space(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """모델별 하이퍼파라미터 탐색 공간."""
    # 공통
    params: dict[str, Any] = {
        'lr'        : trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'dropout'   : trial.suggest_float('dropout', 0.0, 0.3),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
    }

    if model_name == '1DCNN':
        params['num_filters'] = trial.suggest_categorical('num_filters', [32, 64, 128])
        params['kernel_size'] = trial.suggest_categorical('kernel_size', [3, 5, 7])

    elif model_name == 'TCN':
        params['num_layers'] = trial.suggest_int('num_layers', 2, 4)
        params['kernel_size'] = trial.suggest_categorical('kernel_size', [2, 3, 5])

    return params


# ──────────────────────────────────────────────
# 5. Optuna 튜닝
# ──────────────────────────────────────────────
def tune_dl_model(
    model_name  : str,
    train_df    : pd.DataFrame,
    valid_df    : pd.DataFrame,
    feature_cols: list[str],
    n_trials    : int   = 20,
    max_epochs  : int   = 100,
    patience    : int   = 10,
    batch_size  : int   = 64,
    seed        : int   = 42,
    device      : torch.device = DEVICE,
) -> tuple[dict[str, Any], float]:
    """
    Optuna TPE로 DL 하이퍼파라미터 튜닝.

    Returns
    -------
    best_params    : 최적 파라미터 dict  (lr 포함)
    best_val_qlike : 최소 valid QLIKE
    """
    seq_len   = DL_SEQ_LEN[model_name]
    input_dim = len(feature_cols)

    train_scaled, [valid_scaled], _ = _scale_frames(
        train_df,
        [valid_df],
        feature_cols,
    )

    # 튜닝 시 test_loader는 불필요 → valid_df 재사용
    train_loader, valid_loader, _, _ = build_loaders(
        train_scaled, valid_scaled, valid_scaled,
        feature_cols, seq_len, batch_size,
    )

    def objective(trial: optuna.Trial) -> float:
        params = _search_space(trial, model_name)
        lr     = params.pop('lr')

        torch.manual_seed(seed)
        model = make_dl_model(model_name, input_dim, **params).to(device)
        _, val_q = train_dl_model(
            model, train_loader, valid_loader,
            max_epochs=max_epochs, patience=patience,
            lr=lr, device=device,
        )
        return val_q

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params.copy(), study.best_value


# ──────────────────────────────────────────────
# 6. Static 프로토콜
# ──────────────────────────────────────────────
def run_dl_static(
    model_name  : str,
    combined_df : pd.DataFrame,
    test_df     : pd.DataFrame,
    feature_cols: list[str],
    best_params : dict[str, Any],
    max_epochs  : int   = 100,
    patience    : int   = 10,
    batch_size  : int   = 64,
    seed        : int   = 42,
    device      : torch.device = DEVICE,
) -> tuple[pd.Series, nn.Module]:

    seq_len   = DL_SEQ_LEN[model_name]
    input_dim = len(feature_cols)

    split     = int(len(combined_df) * 0.8)
    inner_tr  = combined_df.iloc[:split]
    inner_va  = combined_df.iloc[split:]

    inner_tr_scaled, [inner_va_scaled, test_scaled], _ = _scale_frames(
        inner_tr,
        [inner_va, test_df],
        feature_cols,
    )

    train_loader, valid_loader, test_loader, test_idx = build_loaders(
        inner_tr_scaled, inner_va_scaled, test_scaled,
        feature_cols, seq_len, batch_size,
    )

    lr       = best_params.get('lr', 1e-3)
    model_hp = {k: v for k, v in best_params.items() if k != 'lr'}

    torch.manual_seed(seed)
    model = make_dl_model(model_name, input_dim, **model_hp).to(device)
    model, _ = train_dl_model(
        model, train_loader, valid_loader,
        max_epochs=max_epochs, patience=patience,
        lr=lr, device=device,
    )

    preds  = predict_dl(model, test_loader, device)

    # ✅ 수정: test_df 전체 인덱스에 맞게 앞을 NaN으로 채움
    y_pred_full = pd.Series(np.nan, index=test_df.index, name=model_name)
    y_pred_full.loc[test_idx] = preds
    
    return y_pred_full, model



# ──────────────────────────────────────────────
# 7. Expanding 프로토콜
# ──────────────────────────────────────────────
def run_dl_expanding(
    model_name  : str,
    combined_df : pd.DataFrame,   # 초기 history = train + valid
    test_df     : pd.DataFrame,
    feature_cols: list[str],
    best_params : dict[str, Any],
    refit_every : int   = 20,
    max_epochs  : int   = 10,     # expanding은 빠르게
    patience    : int   = 3,
    batch_size  : int   = 64,
    warm_start  : bool  = True,
    seed        : int   = 42,
    device      : torch.device = DEVICE,
) -> pd.Series:
    """
    Expanding window 예측.

    - 초기 history = combined (train+valid)
    - test를 1행씩 추가하며 refit_every마다 재학습
    - 재학습 시 history 내부 8:2 분할로 EarlyStopping
    - warm_start=True면 이전 가중치에서 이어 학습하여 속도를 줄임

    Returns
    -------
    y_pred : pd.Series (index = test_df.index)
    """
    seq_len   = DL_SEQ_LEN[model_name]
    input_dim = len(feature_cols)
    lr        = best_params.get('lr', 1e-3)
    model_hp  = {k: v for k, v in best_params.items() if k != 'lr'}

    history = combined_df.copy()
    model   : nn.Module | None = None
    scaler  : CustomScaler | None = None
    preds   : list[float] = []

    for i, (date, row) in enumerate(test_df.iterrows()):

        # ── refit 조건 ──
        if i % refit_every == 0:
            split    = int(len(history) * 0.8)
            inner_tr = history.iloc[:split]
            inner_va = history.iloc[split:]

            # history가 너무 짧으면 skip
            min_len = seq_len + 1
            if len(inner_tr) < min_len or len(inner_va) < min_len:
                preds.append(float('nan'))
                history = pd.concat([history, test_df.iloc[[i]]])
                continue

            inner_tr_scaled, [inner_va_scaled], scaler = _scale_frames(
                inner_tr,
                [inner_va],
                feature_cols,
            )

            tr_loader, va_loader, _, _ = build_loaders(
                inner_tr_scaled, inner_va_scaled, inner_va_scaled,   # test 자리 dummy
                feature_cols, seq_len, batch_size,
            )

            if model is None or not warm_start:
                torch.manual_seed(seed + i)
                model = make_dl_model(model_name, input_dim, **model_hp).to(device)
            model, _ = train_dl_model(
                model, tr_loader, va_loader,
                max_epochs=max_epochs, patience=patience,
                lr=lr, device=device,
            )

        # ── 단일 시점 예측 ──
        if model is None or len(history) < seq_len:
            preds.append(float('nan'))
        else:
            assert scaler is not None
            history_tail = history[feature_cols].iloc[-seq_len:]
            history_tail_scaled = scaler.transform(
                history_tail.values,
                feature_cols,
            )
            feat_window = (
                history_tail_scaled.astype(np.float32)
            )                                                   # (seq_len, F)
            X_t = torch.tensor(feat_window).unsqueeze(0).to(device)  # (1, seq_len, F)
            model.eval()
            with torch.no_grad():
                preds.append(model(X_t).item())

        # history에 현재 test 행 추가
        history = pd.concat([history, test_df.iloc[[i]]])

    return pd.Series(preds, index=test_df.index, name=model_name)
