"""
컬럼 그룹별 scaling — `dataset/scaling_strategy.txt` 정책을 그대로 코드화.

4 그룹:
- log_robust : log1p(clip) 후 RobustScaler  (양수·꼬리 두꺼운 변동성 변수)
- robust     : RobustScaler 단독            (변동률·범위)
- standard   : StandardScaler                (정규분포 가정 가능한 거시·모멘텀·spillover)
- no_scaling : 변환 없음                     (이미 정규화된 값)

타깃 RV_target은 어떤 그룹에도 속하지 않음. 호출 측에서 feature_cols에 절대 포함 안 되게 가드.
정의되지 않은 컬럼은 strict=True 모드에서 명시적으로 에러 (default fallback에 의존하지 않음).

설계 원칙 (3가지 안전장치):
1. spillover 3종 (SP500/Nikkei/KOSPI) 모두 standard 그룹으로 통일 — 의미가 같은 변수는 같은 scaling.
2. weekday_sin/cos를 명시적으로 no_scaling 그룹에 등록 — default fallback에 의존하지 않음.
3. RV_target을 어떤 그룹에도 두지 않고 fit/transform 시 명시적 에러 — 타깃이 feature로 새는 사고 방지.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


# ---------------------------------------------------------------------------
# Scaling 정책 (scaling_strategy.txt 1:1 일치)
# ---------------------------------------------------------------------------

SCALING_CONFIG: Dict[str, Dict] = {
    "log_robust": {
        "type": "RobustScaler",
        "log": True,
        "columns": [
            "RV_d", "RV_w", "RV_m",
            "semivariance", "parkinson_rv",
        ],
    },
    "robust": {
        "type": "RobustScaler",
        "log": False,
        "columns": [
            "hl_range", "log_return",
            "gold_chg", "wti_chg", "corn_chg",
            "epu",
        ],
    },
    "standard": {
        "type": "StandardScaler",
        "log": False,
        "columns": [
            "fx_level", "fx_change",
            "rate_3y", "rate_10y", "rate_spread", "policy_rate",
            "crb_chg", "natgas_chg",
            "momentum_1w", "momentum_1m", "momentum_3m", "momentum_6m",
            "spillover_SP500", "spillover_Nikkei", "spillover_KOSPI",
        ],
    },
    "no_scaling": {
        "type": "NoScaling",
        "log": False,
        "columns": [
            "neg_return",
            "weekday_sin", "weekday_cos",
        ],
    },
}

# 타깃 — feature_cols에 절대 들어가면 안 됨
TARGET_COLUMN = "RV_target"


# ---------------------------------------------------------------------------
# 가드 / 조회 헬퍼
# ---------------------------------------------------------------------------

def _build_column_index() -> Dict[str, tuple]:
    """col_name → (scaler_type, log_flag) 역인덱스. SCALING_CONFIG 변경 시 자동 반영."""
    idx = {}
    for group_cfg in SCALING_CONFIG.values():
        for col in group_cfg["columns"]:
            if col in idx:
                raise ValueError(f"Duplicate column in SCALING_CONFIG: {col!r}")
            idx[col] = (group_cfg["type"], group_cfg.get("log", False))
    return idx


_COLUMN_INDEX = _build_column_index()


def known_columns() -> List[str]:
    """SCALING_CONFIG에 등록된 모든 컬럼 (타깃 제외)."""
    return list(_COLUMN_INDEX.keys())


# ---------------------------------------------------------------------------
# CustomScaler
# ---------------------------------------------------------------------------

class CustomScaler:
    """
    컬럼별로 다른 scaling을 적용하는 transformer.

    train fit → test transform 패턴 (sklearn 스타일).
    walk-forward expanding 시 매 refit마다 새 인스턴스로 fit 권장 (look-ahead 방지).
    """

    def __init__(self, config: Dict[str, Dict] = SCALING_CONFIG, strict: bool = True):
        """
        Parameters
        ----------
        config : SCALING_CONFIG와 동일한 형태의 dict
        strict : True면 SCALING_CONFIG에 없는 컬럼이 입력될 때 ValueError. False면 NoScaling 처리.
        """
        self.config = config
        self.strict = strict

        # 역인덱스 빌드 (custom config 지원)
        self._col_idx: Dict[str, tuple] = {}
        for group_cfg in config.values():
            for col in group_cfg["columns"]:
                self._col_idx[col] = (group_cfg["type"], group_cfg.get("log", False))

        # fit 후 채워짐
        self.scalers_: Dict[str, object] = {}     # col → fitted sklearn scaler (or None)
        self.feature_log_: Dict[str, bool] = {}   # col → log 적용 여부
        self.fitted_columns_: List[str] = []

    # ------------------------------------------------------------------- fit
    def fit_transform(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        if TARGET_COLUMN in feature_names:
            raise ValueError(
                f"feature_names에 타깃 컬럼 {TARGET_COLUMN!r}이 들어있음. "
                "feature_cols에서 제외 후 다시 호출."
            )

        X = np.asarray(X, dtype=float).copy()
        out = np.empty_like(X)
        self.fitted_columns_ = list(feature_names)

        for i, col in enumerate(feature_names):
            scaler_type, apply_log = self._lookup(col)
            self.feature_log_[col] = apply_log

            data_col = X[:, i].reshape(-1, 1)
            if apply_log:
                data_col = np.log1p(np.clip(data_col, 1e-8, None))

            if scaler_type == "RobustScaler":
                sc = RobustScaler()
                out[:, i] = sc.fit_transform(data_col).flatten()
                self.scalers_[col] = sc
            elif scaler_type == "StandardScaler":
                sc = StandardScaler()
                out[:, i] = sc.fit_transform(data_col).flatten()
                self.scalers_[col] = sc
            else:  # NoScaling
                out[:, i] = data_col.flatten()
                self.scalers_[col] = None

        return out

    # --------------------------------------------------------------- transform
    def transform(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        if TARGET_COLUMN in feature_names:
            raise ValueError(
                f"feature_names에 타깃 컬럼 {TARGET_COLUMN!r}이 들어있음."
            )
        if not self.fitted_columns_:
            raise RuntimeError("fit_transform()을 먼저 호출해야 함.")

        X = np.asarray(X, dtype=float).copy()
        out = np.empty_like(X)

        for i, col in enumerate(feature_names):
            apply_log = self.feature_log_.get(col)
            if apply_log is None:
                # 새 컬럼이 train fit에 없었음
                if self.strict:
                    raise ValueError(f"컬럼 {col!r}이 train에서 fit되지 않음.")
                apply_log = False

            data_col = X[:, i].reshape(-1, 1)
            if apply_log:
                data_col = np.log1p(np.clip(data_col, 1e-8, None))

            sc = self.scalers_.get(col)
            if sc is not None:
                out[:, i] = sc.transform(data_col).flatten()
            else:
                out[:, i] = data_col.flatten()

        return out

    # ----------------------------------------------------------------- helper
    def _lookup(self, col: str) -> tuple:
        """col → (scaler_type, log_flag). strict 모드면 미등록 컬럼에 에러."""
        if col in self._col_idx:
            return self._col_idx[col]
        if self.strict:
            raise ValueError(
                f"컬럼 {col!r}이 SCALING_CONFIG에 정의되지 않음. "
                "scaling_strategy.txt에 추가하거나 strict=False로 호출."
            )
        return ("NoScaling", False)

    # --------------------------------------------------------------- pandas 인터페이스
    def fit_transform_df(self, df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """DataFrame 인터페이스 — index/columns 보존."""
        X = df[feature_names].values
        out = self.fit_transform(X, feature_names)
        return pd.DataFrame(out, index=df.index, columns=feature_names)

    def transform_df(self, df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        X = df[feature_names].values
        out = self.transform(X, feature_names)
        return pd.DataFrame(out, index=df.index, columns=feature_names)
