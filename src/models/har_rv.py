"""
HAR-RV wrapper — Heterogeneous Autoregressive Realized Volatility.

회귀식:
    RV_t = β₀ + β₁ · RV_d + β₂ · RV_w + β₃ · RV_m + ε_t

설명변수 (모두 CSV에 사전 계산되어 있음):
- RV_d : 직전 1일 RV
- RV_w : 직전 5일 RV 평균 (주간)
- RV_m : 직전 22일 RV 평균 (월간)

추정: statsmodels OLS, HC1 robust standard errors.
예측: y_hat = β₀ + β₁·RV_d + β₂·RV_w + β₃·RV_m, QLIKE 보호를 위해 1e-8로 clip.

단위: RV_target과 동일 스케일로 입력·출력 (별도 변환 없음).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


HAR_FEATURES = ["RV_d", "RV_w", "RV_m"]
TARGET = "RV_target"
PRED_FLOOR = 1e-8  # QLIKE 계산 시 0/음수 방지


class HARRVModel:
    """공통 인터페이스: .fit(train_df) → self,  .predict(test_df) → pd.Series."""

    def __init__(self, cov_type: str = "HC1"):
        self.cov_type = cov_type
        self.result_ = None  # statsmodels OLSResults

    # ---------------------------------------------------------------- fit
    def fit(self, train_df: pd.DataFrame) -> "HARRVModel":
        missing = set(HAR_FEATURES + [TARGET]) - set(train_df.columns)
        if missing:
            raise ValueError(f"train_df missing columns: {missing}")

        df = train_df[HAR_FEATURES + [TARGET]].dropna()
        y = df[TARGET].astype(float)
        X = sm.add_constant(df[HAR_FEATURES].astype(float))
        self.result_ = sm.OLS(y, X).fit(cov_type=self.cov_type)
        return self

    # ------------------------------------------------------------ predict
    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        if self.result_ is None:
            raise RuntimeError("call fit() before predict()")

        missing = set(HAR_FEATURES) - set(test_df.columns)
        if missing:
            raise ValueError(f"test_df missing columns: {missing}")

        X_test = sm.add_constant(test_df[HAR_FEATURES].astype(float), has_constant="add")
        # 컬럼 순서를 train과 동일하게 보장
        X_test = X_test[self.result_.model.exog_names]
        preds = self.result_.predict(X_test)
        preds = preds.clip(lower=PRED_FLOOR)
        preds.name = "har_rv_pred"
        return preds

    # ------------------------------------------------------------- params
    @property
    def params_(self) -> Optional[pd.Series]:
        if self.result_ is None:
            return None
        return self.result_.params
