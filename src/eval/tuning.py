"""
Optuna 하이퍼파라미터 튜닝.

목표: train으로 모델을 fit, valid set의 QLIKE를 최소화하는 best_params 탐색.

탐색 알고리즘: TPE Sampler (Optuna 기본). 시드 고정으로 재현성 보장.
방향: minimize (QLIKE는 낮을수록 좋음).

n_trials 기본값:
- linear (Ridge / ElasticNet / Huber): 15  (탐색 공간 1~2D, 충분한 수)
- tree (LightGBM / XGBoost):           30  (탐색 공간 9D, trial 수가 많을수록 안정)

사용 예:
    best_params, best_value = tune_model(
        "LightGBM", train_df, valid_df, feature_cols,
    )
    final = make_ml_model("LightGBM", feature_cols, **best_params).fit(train_df)

탐색 공간 정의는 _suggest_params 함수 참조.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from .metrics import qlike
from ..models.ml import (
    LINEAR_MODEL_NAMES,
    TREE_MODEL_NAMES,
    RANDOM_STATE,
    make_ml_model,
)

DEFAULT_N_TRIALS_LINEAR = 15
DEFAULT_N_TRIALS_TREE = 30


# ---------------------------------------------------------------------------
# 모델별 search space
# ---------------------------------------------------------------------------

def _suggest_params(model_name: str, trial: optuna.Trial) -> Dict:
    """
    Optuna trial로부터 모델별 하이퍼파라미터 dict를 생성.

    탐색 공간:
    - Ridge      : alpha (1e-3 ~ 100, log-uniform)
    - ElasticNet : alpha (1e-4 ~ 10, log), l1_ratio (0.05 ~ 0.95)
    - Huber      : epsilon (1.1 ~ 2.0), alpha (1e-5 ~ 1, log)
    - LightGBM   : n_estimators (200~1200), learning_rate (0.005~0.08, log),
                   max_depth (3~10), num_leaves (15~255), min_child_samples (5~50),
                   subsample (0.6~1.0), colsample_bytree (0.6~1.0),
                   reg_alpha/reg_lambda (1e-4~10, log)
    - XGBoost    : n_estimators (100~400), learning_rate (0.01~0.05, log),
                   max_depth (3~6), min_child_weight (1.0~6.0),
                   subsample/colsample_bytree (0.7~1.0),
                   reg_alpha/reg_lambda (1e-4~1, log), gamma (0~2)
    """
    if model_name == "Ridge":
        return dict(
            alpha=trial.suggest_float("alpha", 1e-3, 100.0, log=True),
            random_state=RANDOM_STATE,
        )

    if model_name == "ElasticNet":
        return dict(
            alpha=trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.05, 0.95),
            max_iter=20000,
            random_state=RANDOM_STATE,
        )

    if model_name == "Huber":
        return dict(
            epsilon=trial.suggest_float("epsilon", 1.1, 2.0),
            alpha=trial.suggest_float("alpha", 1e-5, 1.0, log=True),
            max_iter=2000,
        )

    if model_name == "LightGBM":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            random_state=RANDOM_STATE,
            objective="regression",
            verbosity=-1,
        )

    if model_name == "XGBoost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 6),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 6.0),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 2.0),
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            tree_method="hist",
            verbosity=0,
        )

    raise ValueError(f"unknown model: {model_name!r}")


# ---------------------------------------------------------------------------
# 튜닝 함수
# ---------------------------------------------------------------------------

def default_n_trials(model_name: str) -> int:
    if model_name in LINEAR_MODEL_NAMES:
        return DEFAULT_N_TRIALS_LINEAR
    if model_name in TREE_MODEL_NAMES:
        return DEFAULT_N_TRIALS_TREE
    raise ValueError(f"unknown model: {model_name!r}")


def tune_model(
    model_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    n_trials: int | None = None,
    seed: int = RANDOM_STATE,
) -> Tuple[Dict, float]:
    """
    Optuna로 model_name의 best hyperparameter 탐색.

    Parameters
    ----------
    model_name : "Ridge" / "ElasticNet" / "Huber" / "LightGBM" / "XGBoost"
    train_df, valid_df : 분할된 train/valid DataFrames (RV_target 컬럼 포함)
    feature_cols : 사용할 feature 컬럼 리스트 (RV_target 제외)
    n_trials : 시도 횟수. None이면 default_n_trials(model_name).
    seed : Optuna sampler 시드 (재현성)

    Returns
    -------
    (best_params, best_value)
        best_params : Optuna가 반환한 파라미터 (재학습 시 그대로 사용 가능)
        best_value  : valid set의 best QLIKE
    """
    if n_trials is None:
        n_trials = default_n_trials(model_name)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(model_name, trial)
        model = make_ml_model(model_name, feature_cols, **params)
        model.fit(train_df)
        y_pred = model.predict(valid_df).values
        y_true = valid_df["RV_target"].values
        return float(qlike(y_true, y_pred))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params, float(study.best_value)
