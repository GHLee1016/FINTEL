"""
Feature 셋 정의 (`dataset/feature_tiers.txt` 1:1 일치).

3 tier 누적 구조:
- CORE (10)     : RV/수익률/일중 변동성/요일
- MOMENTUM (14) : Core + 모멘텀 4
- EXTENDED (28) : Momentum + 거시·외생 12 + 시장별 spillover 2

시장별 spillover (자기 시장 제외):
- KR: SP500, Nikkei
- US: KOSPI, Nikkei
- JP: SP500, KOSPI

ML/DL 전용. 금융모형(GARCH/HAR-RV)은 사용하지 않음.
"""

from __future__ import annotations

from typing import List

import pandas as pd


# ---------------------------------------------------------------------------
# Tier 정의
# ---------------------------------------------------------------------------

CORE: List[str] = [
    "RV_d", "RV_w", "RV_m",
    "log_return", "neg_return", "semivariance",
    "parkinson_rv", "hl_range",
    "weekday_sin", "weekday_cos",
]

MOMENTUM_ADD: List[str] = [
    "momentum_1w", "momentum_1m", "momentum_3m", "momentum_6m",
]

EXTENDED_ADD: List[str] = [
    "fx_level", "fx_change",
    "rate_3y", "rate_10y", "rate_spread", "policy_rate",
    "gold_chg", "crb_chg", "wti_chg", "natgas_chg", "corn_chg",
    "epu",
]

# 누적 정의 (편의용)
MOMENTUM: List[str] = CORE + MOMENTUM_ADD              # 14개
EXTENDED: List[str] = CORE + MOMENTUM_ADD + EXTENDED_ADD  # 26개 (시장별 spillover는 SPILLOVER_MAP 통해 추가)


# ---------------------------------------------------------------------------
# 시장별 spillover
# ---------------------------------------------------------------------------

SPILLOVER_MAP = {
    "KR": ["spillover_SP500", "spillover_Nikkei"],
    "US": ["spillover_KOSPI", "spillover_Nikkei"],
    "JP": ["spillover_SP500", "spillover_KOSPI"],
}


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def select_full_coverage_cols(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    """
    `candidates` 중 df에 존재하고 결측치가 전혀 없는 컬럼만 반환 (안전장치).

    Spillover ffill 적용 후 모든 12 CSV에서 0% 결측이라 현재는 사실상 모두 통과.
    향후 신규 변수 추가 시 결측이 발생하면 자동으로 그 셋에서만 제외됨.
    """
    return [c for c in candidates if c in df.columns and df[c].notna().all()]


def get_feature_list(df: pd.DataFrame, market: str, tier: str) -> List[str]:
    """
    (df, market, tier) → 사용할 feature 컬럼 리스트 반환.

    Parameters
    ----------
    df : 로드된 (또는 분할된) DataFrame
    market : "US" / "KR" / "JP"
    tier : "core" / "momentum" / "extended"

    Returns
    -------
    list of column names — df에 실제 존재하고 (extended의 경우) 결측 없는 것들만.
    """
    tier = tier.lower()
    if market not in SPILLOVER_MAP:
        raise ValueError(f"unknown market: {market!r}")

    if tier == "core":
        feats = [c for c in CORE if c in df.columns]
    elif tier == "momentum":
        feats = [c for c in MOMENTUM if c in df.columns]
    elif tier == "extended":
        base = [c for c in (CORE + MOMENTUM_ADD) if c in df.columns]
        ext_candidates = EXTENDED_ADD + SPILLOVER_MAP[market]
        ext_full = select_full_coverage_cols(df, ext_candidates)
        feats = base + ext_full
    else:
        raise ValueError(f"unknown tier: {tier!r}")

    return feats
