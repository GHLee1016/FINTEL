"""
Phase 슬라이서 — test 인덱스를 캘린더 날짜로 마스킹.

config.PHASES[regime]에 정의된 (name, start, end, color) 튜플을 사용.
start/end == None 이면 test 전체 (Full Test).
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd

from .. import config


def get_phase_mask(
    test_index: pd.DatetimeIndex,
    regime: str,
    phase_name: str,
) -> np.ndarray:
    """
    test_index에 대해 phase_name에 속하는 행을 True로 표시한 boolean array.

    regime이 "normal"인 경우 phase_name은 "Full Test"여야 한다.
    """
    if regime not in config.PHASES:
        raise ValueError(f"unknown regime: {regime!r}")

    for name, start, end, _color in config.PHASES[regime]:
        if name != phase_name:
            continue
        if start is None and end is None:
            return np.ones(len(test_index), dtype=bool)
        s = pd.Timestamp(start) if start is not None else test_index.min()
        e = pd.Timestamp(end) if end is not None else test_index.max()
        return (test_index >= s) & (test_index <= e)

    raise ValueError(f"phase {phase_name!r} not defined for regime {regime!r}")


def iter_phases(
    test_df: pd.DataFrame,
    regime: str,
) -> Iterator[Tuple[str, np.ndarray, Optional[str]]]:
    """
    (phase_name, mask, color)를 순서대로 yield.

    Parameters
    ----------
    test_df : DatetimeIndex를 가진 test set
    regime : 구간 키

    Yields
    ------
    (phase_name, boolean mask of len(test_df), color or None)
    """
    if regime not in config.PHASES:
        raise ValueError(f"unknown regime: {regime!r}")

    for name, start, end, color in config.PHASES[regime]:
        if start is None and end is None:
            mask = np.ones(len(test_df), dtype=bool)
        else:
            idx = test_df.index
            s = pd.Timestamp(start) if start is not None else idx.min()
            e = pd.Timestamp(end) if end is not None else idx.max()
            mask = (idx >= s) & (idx <= e)
        yield name, mask, color
