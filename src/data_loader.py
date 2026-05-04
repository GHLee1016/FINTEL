"""
데이터 로더 — CSV 12개를 (regime, country) 키로 읽고, config.SPLITS의 날짜 경계로 분할.

CSV의 `split` 컬럼은 무시한다. 우리 분할은 data_splits.txt 정의를 따른다.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from . import config


REQUIRED_COLUMNS = {"date", "RV_target", "RV_d", "RV_w", "RV_m", "log_return"}


def load_dataset(regime: str, country: str) -> pd.DataFrame:
    """
    CSV 1개 로드. `date`를 DatetimeIndex로 설정.

    Parameters
    ----------
    regime : {"normal", "911", "gfc", "covid"}
    country : {"US", "KR", "JP"}

    Returns
    -------
    pd.DataFrame  — DatetimeIndex, 모든 feature 컬럼 포함, `split` 컬럼은 그대로 두되
    분할 시 사용하지 않음.
    """
    if regime not in config.REGIMES:
        raise ValueError(f"unknown regime: {regime!r}")
    if country not in config.COUNTRIES:
        raise ValueError(f"unknown country: {country!r}")

    path = config.dataset_path(regime, country)
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()

    missing = REQUIRED_COLUMNS - set(df.columns) - {"date"}
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    return df


def split_train_test(
    df: pd.DataFrame,
    regime: str,
    country: str,
    dropna_subset: list[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    config.SPLITS[regime, country]의 날짜 경계로 train / test 분할.

    Parameters
    ----------
    df : pd.DataFrame  — load_dataset()의 반환값
    regime, country : 분할 키
    dropna_subset : 이 컬럼들에 NaN이 있는 행 제거. 기본은
        ["RV_target", "RV_d", "RV_w", "RV_m", "log_return"] (HAR/GARCH 모두 필요).

    Returns
    -------
    (train_df, test_df) — 양쪽 모두 inclusive bound 적용.
    """
    splits = config.SPLITS[(regime, country)]
    train_start, train_end = splits["train"]
    test_start, test_end = splits["test"]

    if dropna_subset is None:
        dropna_subset = ["RV_target", "RV_d", "RV_w", "RV_m", "log_return"]

    df_clean = df.dropna(subset=dropna_subset)

    train = df_clean.loc[train_start:train_end].copy()
    test = df_clean.loc[test_start:test_end].copy()

    return train, test


def load_split(
    regime: str,
    country: str,
    dropna_subset: list[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """`load_dataset` + `split_train_test`를 한 번에 수행하는 헬퍼."""
    df = load_dataset(regime, country)
    return split_train_test(df, regime, country, dropna_subset=dropna_subset)
