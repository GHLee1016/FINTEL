"""
ML 모델 wrapper — Ridge / ElasticNet / Huber / LightGBM / XGBoost.

공통 인터페이스 (HARRVModel/GARCHModel과 동일):
- model.fit(train_df) → self
- model.predict(test_df) → pd.Series  (인덱스는 test_df.index)

선형 모델 (Ridge, ElasticNet, Huber):
- 내부에 CustomScaler를 보유. fit 시 scaler를 train으로 fit, predict 시 transform.
- scaling 정책은 src/preprocess/scaler.py의 SCALING_CONFIG (4 그룹).

트리 모델 (LightGBM, XGBoost):
- raw feature 입력 (트리는 monotonic 변환에 불변이라 scaling 효과 없음).

타깃: RV_target (h=1, 다음 날 RV 예측). train_df의 RV_target을 회귀 타깃으로 직접 사용.
※ Multi-horizon (h=5, h=22)은 현재 단계에서 미지원.

각 모델은 인스턴스 생성 시 feature_cols 리스트를 받음:
    RidgeModel(feature_cols=feats, alpha=0.5)

하이퍼파라미터 탐색 공간은 src/eval/tuning.py의 _suggest_params 참조.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None  # type: ignore

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None  # type: ignore

from ..preprocess.scaler import CustomScaler

TARGET_COLUMN = "RV_target"
PRED_FLOOR = 1e-8

RANDOM_STATE = 42

LINEAR_MODEL_NAMES = ("Ridge", "ElasticNet", "Huber")
TREE_MODEL_NAMES = ("LightGBM", "XGBoost")
ALL_MODEL_NAMES = LINEAR_MODEL_NAMES + TREE_MODEL_NAMES


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class _BaseMLModel:
    """공통 동작: fit/predict 인터페이스 + 선형 모델용 scaler 보유."""

    name: str = "Base"
    use_scaler: bool = False

    def __init__(self, feature_cols: List[str], **hyperparams):
        if not feature_cols:
            raise ValueError("feature_cols가 비어 있음.")
        if TARGET_COLUMN in feature_cols:
            raise ValueError(f"feature_cols에 타깃 {TARGET_COLUMN!r} 포함 — 제외 후 호출.")
        self.feature_cols = list(feature_cols)
        self.hyperparams = hyperparams

        self.estimator_ = None
        self.scaler_: Optional[CustomScaler] = None

    # 자식 클래스에서 overide
    def _make_estimator(self):
        raise NotImplementedError

    # ---------------------------------------------------------------- fit
    def fit(self, train_df: pd.DataFrame) -> "_BaseMLModel":
        missing = set(self.feature_cols + [TARGET_COLUMN]) - set(train_df.columns)
        if missing:
            raise ValueError(f"train_df missing columns: {missing}")

        df = train_df[self.feature_cols + [TARGET_COLUMN]].dropna()
        X = df[self.feature_cols].values
        y = df[TARGET_COLUMN].astype(float).values

        if self.use_scaler:
            self.scaler_ = CustomScaler()
            X = self.scaler_.fit_transform(X, self.feature_cols)

        self.estimator_ = self._make_estimator()
        self.estimator_.fit(X, y)
        return self

    # ------------------------------------------------------------ predict
    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        if self.estimator_ is None:
            raise RuntimeError("fit() 호출 후 predict() 사용.")
        missing = set(self.feature_cols) - set(test_df.columns)
        if missing:
            raise ValueError(f"test_df missing columns: {missing}")

        X = test_df[self.feature_cols].values
        if self.use_scaler:
            assert self.scaler_ is not None
            X = self.scaler_.transform(X, self.feature_cols)

        preds = self.estimator_.predict(X)
        preds = np.clip(preds, PRED_FLOOR, None)  # QLIKE 보호
        return pd.Series(preds, index=test_df.index, name=f"{self.name.lower()}_pred")


# ---------------------------------------------------------------------------
# Linear models — scaling 적용
# ---------------------------------------------------------------------------

class RidgeModel(_BaseMLModel):
    name = "Ridge"
    use_scaler = True

    def _make_estimator(self):
        params = dict(alpha=1.0, random_state=RANDOM_STATE)
        params.update(self.hyperparams)
        return Ridge(**params)


class ElasticNetModel(_BaseMLModel):
    name = "ElasticNet"
    use_scaler = True

    def _make_estimator(self):
        params = dict(alpha=1.0, l1_ratio=0.5, max_iter=20000, random_state=RANDOM_STATE)
        params.update(self.hyperparams)
        return ElasticNet(**params)


class HuberModel(_BaseMLModel):
    name = "Huber"
    use_scaler = True

    def _make_estimator(self):
        params = dict(epsilon=1.35, alpha=1e-4, max_iter=2000)
        params.update(self.hyperparams)
        return HuberRegressor(**params)


# ---------------------------------------------------------------------------
# Tree models — raw feature
# ---------------------------------------------------------------------------

class LightGBMModel(_BaseMLModel):
    name = "LightGBM"
    use_scaler = False

    def _make_estimator(self):
        if LGBMRegressor is None:
            raise ImportError("lightgbm 패키지가 필요합니다. pip install lightgbm")
        params = dict(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=RANDOM_STATE,
            objective="regression",
            verbosity=-1,
        )
        params.update(self.hyperparams)
        return LGBMRegressor(**params)


class XGBoostModel(_BaseMLModel):
    name = "XGBoost"
    use_scaler = False

    def _make_estimator(self):
        if XGBRegressor is None:
            raise ImportError("xgboost 패키지가 필요합니다. pip install xgboost")
        params = dict(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=2.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            gamma=0.0,
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            tree_method="hist",
            verbosity=0,
        )
        params.update(self.hyperparams)
        return XGBRegressor(**params)


# ---------------------------------------------------------------------------
# Registry & Factory
# ---------------------------------------------------------------------------

ML_MODEL_REGISTRY = {
    "Ridge":      RidgeModel,
    "ElasticNet": ElasticNetModel,
    "Huber":      HuberModel,
    "LightGBM":   LightGBMModel,
    "XGBoost":    XGBoostModel,
}


def make_ml_model(name: str, feature_cols: List[str], **hyperparams) -> _BaseMLModel:
    """이름과 feature_cols, 하이퍼파라미터로 모델 인스턴스 생성."""
    if name not in ML_MODEL_REGISTRY:
        raise ValueError(f"unknown ML model: {name!r}. choose from {list(ML_MODEL_REGISTRY)}")
    return ML_MODEL_REGISTRY[name](feature_cols=feature_cols, **hyperparams)
