"""
평가 프로토콜 — 모든 모델에 동일하게 적용 (금융 + ML).

두 가지 모드:
- run_static    : 1회 fit (train) → 전체 test 예측
- run_expanding : walk-forward — train으로 시작해 매 step train에 한 행씩 추가하며 refit·예측

모델 인터페이스 가정: model.fit(train_df) → model,  model.predict(test_df) → pd.Series.
모든 .predict 출력은 test_df의 인덱스와 동일한 순서·길이를 가져야 함.

ML 모델 호출 예 (feature_cols와 best_params를 model_kwargs로 전달):
    run_expanding(
        LightGBMModel, train, test, refit_every=5,
        feature_cols=feats, **best_params,   # best_params는 Optuna로 미리 튜닝
    )

기본 refit_every:
- 금융모형 (HAR/GARCH): 1 권장 (fit 비용 작음)
- ML 모델: 5 권장 (cost 절감, 성능 손실 미미)
- DL 모델: 20+ 권장 (fit 비용 큼)
"""

from __future__ import annotations

from typing import Type, Union

import pandas as pd


ModelLike = Union[type, object]  # 클래스 또는 인스턴스 모두 허용


def _instantiate(model: ModelLike):
    """클래스가 들어오면 새 인스턴스 생성, 인스턴스면 그대로."""
    if isinstance(model, type):
        return model()
    return model


# ---------------------------------------------------------------------------
# Static (cold)
# ---------------------------------------------------------------------------

def run_static(model: ModelLike, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    """
    Train 전체에 1회 fit → test 전체에 대해 예측.

    Returns
    -------
    pd.Series  — index가 test_df.index와 동일한 예측값.
    """
    m = _instantiate(model)
    m.fit(train_df)
    preds = m.predict(test_df)
    if not isinstance(preds, pd.Series):
        preds = pd.Series(preds, index=test_df.index)
    preds = preds.reindex(test_df.index)
    return preds


# ---------------------------------------------------------------------------
# Expanding window (walk-forward)
# ---------------------------------------------------------------------------

def run_expanding(
    model_cls: Type,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    refit_every: int = 1,
    **model_kwargs,
) -> pd.Series:
    """
    Walk-forward: 각 test step에서 (지금까지 본 데이터) 로 refit 후 1-step 예측.

    Parameters
    ----------
    model_cls : 모델 클래스 (인스턴스 아님). 매 refit마다 새 인스턴스 생성.
    train_df, test_df : 시계열 분할.
    refit_every : N step마다 refit. 기본 1 (매 step). 비용 절감 시 N>1로.
    **model_kwargs : model_cls(**kwargs)에 전달.

    Returns
    -------
    pd.Series  — index가 test_df.index와 동일한 예측값.
    """
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")

    history = train_df.copy()
    preds = pd.Series(index=test_df.index, dtype=float, name="pred")

    fitted = None  # 직전에 fit된 모델
    steps_since_fit = refit_every  # 첫 step에서 무조건 fit하도록

    for i, (ts, row) in enumerate(test_df.iterrows()):
        if steps_since_fit >= refit_every:
            fitted = model_cls(**model_kwargs).fit(history)
            steps_since_fit = 0

        # 1-step 예측: test_df의 ts 행 하나를 모델에 넣어 예측
        single = test_df.loc[[ts]]
        y_hat = fitted.predict(single)
        preds.iloc[i] = float(y_hat.iloc[0])

        # 이번 관측치를 history에 추가 (다음 refit용)
        history = pd.concat([history, single])
        steps_since_fit += 1

    return preds
