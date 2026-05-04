"""
평가 지표 — 모든 모델·구간·phase에 동일하게 적용.

지표 4종:
- RMSE     : sqrt(mean((y_true - y_pred)^2))
- MAE      : mean(|y_true - y_pred|)
- QLIKE    : mean(ratio - log(ratio) - 1),  ratio = y_true / y_pred
             (y_true, y_pred > 0 가정. 변동성 예측 표준 손실.)
- RMSE_CV  : RMSE / mean(y_true)
             (국가 간 RV 스케일 차이 정규화 — coefficient of variation.)

QLIKE 정의:
    ratio = y_true / y_pred
    QLIKE = mean( ratio - log(ratio) - 1 )

이 식은 ratio = 1에서 0이고, 양쪽으로 비대칭 양수 손실. 분산 단위 양수 입력 가정.
y_true, y_pred는 1e-12 이하로 clip하여 log(0) 폭발 방지.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.Series]


def _align(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """양쪽을 numpy로 변환하고 NaN 행을 동시에 제거."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    yt, yp = _align(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    yt, yp = _align(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def qlike(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-12) -> float:
    """
    QLIKE 변동성 예측 손실. y_true·y_pred 모두 분산 단위(양수)로 가정.
    eps로 0 근방 보호.
    """
    yt, yp = _align(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    yt = np.clip(yt, eps, None)
    yp = np.clip(yp, eps, None)
    ratio = yt / yp
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def rmse_cv(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """RMSE / mean(y_true) — coefficient of variation."""
    yt, yp = _align(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    mean_yt = np.mean(yt)
    if mean_yt == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)) / mean_yt)


def evaluate(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """4지표를 한 번에 dict로 반환. 컬럼 순서: RMSE / MAE / QLIKE / RMSE_CV."""
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "QLIKE": qlike(y_true, y_pred),
        "RMSE_CV": rmse_cv(y_true, y_pred),
    }
