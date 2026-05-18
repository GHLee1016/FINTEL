"""DL용 sliding window 시퀀스 데이터 scaling wrapper.

`src.preprocess.scaler.CustomScaler`를 3D `(N, L, F)` 시퀀스에 적용하는 얇은 래퍼.

핵심 설계:
- `(N, L, F) → (N*L, F)`로 reshape 후 CustomScaler 호출 → 결과를 `(N, L, F)`로 복원.
  RobustScaler/StandardScaler/log1p 모두 column-wise pointwise 연산이라 timestep 차원을
  sample 차원에 합쳐도 통계 결과 동일.
- Default config는 `SCALING_CONFIG_DL` (ML과 단 한 가지 차이: `neg_return`을 robust 그룹으로 이동).
- Look-ahead 방지: scaler는 **train에서만 fit**, valid/test에는 transform만 적용.

사용 예 (가장 흔한 패턴):
    from src.dl import fit_transform_splits

    scaler, X_train_s, X_valid_s, X_test_s = fit_transform_splits(
        X_train, X_valid, X_test, feature_cols
    )

또는 fit과 transform을 따로 호출 (예: tuning loop):
    from src.dl import fit_transform_3d, transform_3d

    scaler, X_train_s = fit_transform_3d(X_train, feature_cols)
    X_valid_s = transform_3d(scaler, X_valid, feature_cols)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..preprocess.scaler import CustomScaler, SCALING_CONFIG_DL


def fit_transform_3d(
    X: np.ndarray,
    feature_cols: List[str],
    config: Dict = SCALING_CONFIG_DL,
    strict: bool = True,
) -> Tuple[CustomScaler, np.ndarray]:
    """3D 시퀀스 데이터에 CustomScaler를 fit하고 transform한 결과를 반환.

    Parameters
    ----------
    X : np.ndarray
        shape `(N, L, F)` 의 시퀀스 텐서. dtype은 float (float32/float64).
    feature_cols : list of str
        길이 F. 각 column의 feature 이름. SCALING_CONFIG에 등록돼 있어야 함.
    config : dict
        SCALING_CONFIG 형식. default는 DL 전용 config (`SCALING_CONFIG_DL`).
    strict : bool
        True면 SCALING_CONFIG에 없는 컬럼 발생 시 ValueError.

    Returns
    -------
    scaler : CustomScaler
        fit 완료된 scaler. valid/test transform에 재사용.
    X_scaled : np.ndarray
        shape `(N, L, F)` — 원본과 동일 dtype 유지.

    Raises
    ------
    ValueError
        X가 3D 아닐 때, feature_cols 길이가 F와 다를 때.
    """
    if X.ndim != 3:
        raise ValueError(f"expected 3D array (N, L, F), got shape {X.shape}")
    N, L, F = X.shape
    if F != len(feature_cols):
        raise ValueError(
            f"feature dim mismatch: X.shape[2]={F}, len(feature_cols)={len(feature_cols)}"
        )

    scaler = CustomScaler(config=config, strict=strict)
    X_2d = X.reshape(-1, F)
    X_scaled_2d = scaler.fit_transform(X_2d, feature_cols)
    X_scaled = X_scaled_2d.reshape(N, L, F).astype(X.dtype, copy=False)
    return scaler, X_scaled


def transform_3d(
    scaler: CustomScaler,
    X: np.ndarray,
    feature_cols: List[str],
) -> np.ndarray:
    """이미 fit된 scaler로 3D 시퀀스를 transform.

    Parameters
    ----------
    scaler : CustomScaler
        train에서 이미 fit 완료된 scaler.
    X : np.ndarray
        shape `(N, L, F)` 의 시퀀스 텐서.
    feature_cols : list of str
        길이 F. 순서가 fit 시점과 일치해야 함.

    Returns
    -------
    X_scaled : np.ndarray
        shape `(N, L, F)` — 원본과 동일 dtype 유지.
    """
    if X.ndim != 3:
        raise ValueError(f"expected 3D array (N, L, F), got shape {X.shape}")
    N, L, F = X.shape
    if F != len(feature_cols):
        raise ValueError(
            f"feature dim mismatch: X.shape[2]={F}, len(feature_cols)={len(feature_cols)}"
        )

    X_2d = X.reshape(-1, F)
    X_scaled_2d = scaler.transform(X_2d, feature_cols)
    return X_scaled_2d.reshape(N, L, F).astype(X.dtype, copy=False)


def fit_transform_splits(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
    feature_cols: List[str],
    config: Dict = SCALING_CONFIG_DL,
    strict: bool = True,
) -> Tuple[CustomScaler, np.ndarray, np.ndarray, np.ndarray]:
    """편의 함수: train에서 fit + valid/test에 transform 한 번에.

    LOOK-AHEAD 방지 강제: scaler는 train으로만 fit, valid/test는 transform만.

    Parameters
    ----------
    X_train, X_valid, X_test : np.ndarray
        모두 shape `(N_*, L, F)` 의 시퀀스 텐서. L, F는 셋이 동일해야 함.
    feature_cols : list of str
        길이 F.
    config : dict
        default `SCALING_CONFIG_DL`.

    Returns
    -------
    scaler : CustomScaler
        train에서 fit된 scaler.
    X_train_s, X_valid_s, X_test_s : np.ndarray
        각각 scaling 적용된 (N_*, L, F) 시퀀스.

    Notes
    -----
    valid 또는 test가 비어있을 수도 있음 (작은 cell에서). 그 경우엔 빈 shape 그대로 반환.
    """
    if X_valid.size > 0 and (X_valid.shape[1:] != X_train.shape[1:]):
        raise ValueError(
            f"valid shape mismatch: {X_valid.shape} vs train {X_train.shape}"
        )
    if X_test.size > 0 and (X_test.shape[1:] != X_train.shape[1:]):
        raise ValueError(
            f"test shape mismatch: {X_test.shape} vs train {X_train.shape}"
        )

    scaler, X_train_s = fit_transform_3d(X_train, feature_cols, config=config, strict=strict)

    if X_valid.size > 0:
        X_valid_s = transform_3d(scaler, X_valid, feature_cols)
    else:
        X_valid_s = X_valid.copy()

    if X_test.size > 0:
        X_test_s = transform_3d(scaler, X_test, feature_cols)
    else:
        X_test_s = X_test.copy()

    return scaler, X_train_s, X_valid_s, X_test_s
