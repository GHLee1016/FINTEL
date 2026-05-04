"""모델 wrapper 모음. 공통 인터페이스: .fit(train_df) → self, .predict(test_df) → pd.Series."""
from .garch import GARCHModel
from .har_rv import HARRVModel
from .ml import (
    RidgeModel,
    ElasticNetModel,
    HuberModel,
    LightGBMModel,
    XGBoostModel,
    ML_MODEL_REGISTRY,
    LINEAR_MODEL_NAMES,
    TREE_MODEL_NAMES,
    ALL_MODEL_NAMES,
    make_ml_model,
)

__all__ = [
    # 금융모형
    "GARCHModel",
    "HARRVModel",
    # ML
    "RidgeModel",
    "ElasticNetModel",
    "HuberModel",
    "LightGBMModel",
    "XGBoostModel",
    "ML_MODEL_REGISTRY",
    "LINEAR_MODEL_NAMES",
    "TREE_MODEL_NAMES",
    "ALL_MODEL_NAMES",
    "make_ml_model",
]
