"""
프로젝트 전역 설정 — 경로, 구간/국가/phase 메타데이터.

값은 dataset/data_splits.txt 와 dataset/crisis_phases.txt 의 정의를 그대로 옮긴 것.
변경이 필요하면 두 .txt와 이 파일을 함께 갱신할 것.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# 경로 — 로컬(Windows) ↔ Colab 분기
# ---------------------------------------------------------------------------

def _detect_project_root() -> Path:
    """현재 환경에 맞는 Project 루트 경로 반환."""
    # Colab: /content/drive/MyDrive/FINTEL/Project (예시)
    colab_root = os.environ.get("FINTEL_PROJECT_ROOT")
    if colab_root and Path(colab_root).exists():
        return Path(colab_root)

    # 로컬: 이 파일 기준 ../
    here = Path(__file__).resolve().parent
    return here.parent


PROJECT_ROOT: Path = _detect_project_root()
DATASET_DIR: Path = PROJECT_ROOT / "dataset"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"


# ---------------------------------------------------------------------------
# 구간(regime) · 국가(country)
# ---------------------------------------------------------------------------

REGIMES: List[str] = ["normal", "911", "gfc", "covid"]
COUNTRIES: List[str] = ["US", "KR", "JP"]


# ---------------------------------------------------------------------------
# Train/Test 분할 (data_splits.txt 와 1:1 일치)
# (regime, country) -> {"train": (start, end), "test": (start, end)}
# 날짜는 ISO YYYY-MM-DD 문자열. 양 끝 inclusive.
# ---------------------------------------------------------------------------

SPLITS: Dict[Tuple[str, str], Dict[str, Tuple[str, str]]] = {
    # 평시 (normal)
    ("normal", "US"): {"train": ("2000-01-04", "2014-01-07"), "test": ("2014-01-08", "2019-12-31")},
    ("normal", "KR"): {"train": ("2000-01-05", "2013-12-18"), "test": ("2013-12-19", "2019-12-30")},
    ("normal", "JP"): {"train": ("2000-01-05", "2014-01-06"), "test": ("2014-01-07", "2019-12-30")},

    # 9·11
    ("911", "US"): {"train": ("2000-01-03", "2001-08-31"), "test": ("2001-09-04", "2002-03-28")},
    ("911", "KR"): {"train": ("2000-01-04", "2001-08-31"), "test": ("2001-09-03", "2002-03-29")},
    ("911", "JP"): {"train": ("2000-01-04", "2001-08-31"), "test": ("2001-09-03", "2002-03-29")},

    # GFC
    ("gfc", "US"): {"train": ("2000-01-03", "2007-07-31"), "test": ("2007-08-01", "2009-06-30")},
    ("gfc", "KR"): {"train": ("2000-01-04", "2007-07-31"), "test": ("2007-08-01", "2009-06-30")},
    ("gfc", "JP"): {"train": ("2000-01-04", "2007-07-31"), "test": ("2007-08-01", "2009-06-30")},

    # COVID
    ("covid", "US"): {"train": ("2000-01-03", "2019-12-31"), "test": ("2020-01-02", "2020-12-31")},
    ("covid", "KR"): {"train": ("2000-01-04", "2019-12-30"), "test": ("2020-01-02", "2020-12-30")},
    ("covid", "JP"): {"train": ("2000-01-04", "2019-12-30"), "test": ("2020-01-06", "2020-12-30")},
}


# ---------------------------------------------------------------------------
# Phase 정의 (crisis_phases.txt 와 1:1 일치)
# regime -> list of (phase_name, start, end | None, color | None)
#   start/end == None 이면 test 전체를 의미 (Full Test)
# ---------------------------------------------------------------------------

# (name, start, end, color)
PhaseSpec = Tuple[str, str | None, str | None, str | None]

PHASES: Dict[str, List[PhaseSpec]] = {
    "normal": [
        ("Full Test", None, None, None),
    ],
    "911": [
        ("Full Test", None, None, None),
        ("Shock", None, "2001-10-15", "red"),
        ("Recovery", "2001-10-16", None, "green"),
    ],
    "gfc": [
        ("Full Test", None, None, None),
        ("Pre-Lehman", "2007-08-01", "2008-09-14", "blue"),
        ("Lehman Crisis", "2008-09-15", "2009-03-08", "red"),
        ("Recovery", "2009-03-09", "2009-06-30", "green"),
    ],
    "covid": [
        ("Full Test", None, None, None),
        ("Pre-crash", "2020-01-01", "2020-02-23", "blue"),
        ("Crash", "2020-02-24", "2020-03-23", "red"),
        ("Recovery", "2020-03-24", "2020-12-31", "green"),
    ],
}


# ---------------------------------------------------------------------------
# 평가 지표 / 프로토콜 이름 (참조용 상수)
# ---------------------------------------------------------------------------

METRICS: List[str] = ["RMSE", "MAE", "QLIKE", "RMSE_CV"]
PROTOCOLS: List[str] = ["static", "expanding"]


# ---------------------------------------------------------------------------
# 파일명 헬퍼
# ---------------------------------------------------------------------------

def dataset_path(regime: str, country: str) -> Path:
    """`dataset/{regime}_{country}_dataset.csv` 경로 반환."""
    return DATASET_DIR / f"{regime}_{country}_dataset.csv"
