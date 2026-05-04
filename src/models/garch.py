"""
GARCH(1,1) wrapper — Zero-mean variant.

수익률 모형:
    r_t = σ_t · z_t,    z_t ~ N(0, 1)

조건부 분산 재귀:
    σ²_t = ω + α · r²_{t-1} + β · σ²_{t-1}

추정 모수: ω (상수), α (충격반응), β (지속성). QMLE (정규 가정).

예측 (1-step-ahead recursion):
    test 첫 step에서 (last_h_, last_r_)를 train 끝에서 가져와 시작
    이후 각 t에서 σ²_t를 위 재귀식으로 계산하며, r_prev/h_prev를 차례로 갱신
    최종 출력은 sqrt(σ²_t) = σ_t (변동성).

단위:
- 입력: log_return (decimal). 내부에서 ×100으로 % 스케일 변환 후 fit.
- 내부 재귀: σ²_t (% 분산 스케일).
- 최종 출력: σ_t = sqrt(σ²_t) (% 변동성 스케일). RV_target(%, 변동성)과 같은 단위로
  직접 비교 가능.

※ 이전 버전은 σ²_t를 그대로 출력했으나, 본 프로젝트의 RV_target이 변동성(%)
  단위이므로 sqrt를 씌워 단위를 일치시킴. 이로써 RMSE/MAE/QLIKE/RMSE_CV가
  HAR-RV 및 ML 모델과 같은 척도에서 비교됨.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from arch import arch_model


SCALE = 100.0  # log_return → percent


class GARCHModel:
    """공통 인터페이스: .fit(train_df) → self,  .predict(test_df) → pd.Series."""

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        mean: str = "Zero",
        dist: str = "normal",
    ):
        self.p = p
        self.q = q
        self.mean = mean
        self.dist = dist

        # fit() 후 채워짐
        self.omega_: Optional[float] = None
        self.alpha_: Optional[float] = None
        self.beta_: Optional[float] = None
        self.last_h_: Optional[float] = None  # train 마지막 시점의 조건부 분산
        self.last_r_: Optional[float] = None  # train 마지막 시점의 수익률 (% 스케일)
        self.result_ = None  # arch fit 결과 객체

    # ---------------------------------------------------------------- fit
    def fit(self, train_df: pd.DataFrame) -> "GARCHModel":
        """
        log_return 컬럼만 사용. ω, α, β를 추정하고 train 마지막 (h_T, r_T) 보관.
        """
        if "log_return" not in train_df.columns:
            raise ValueError("train_df must have 'log_return' column")

        returns_pct = train_df["log_return"].astype(float).values * SCALE

        model = arch_model(
            returns_pct,
            mean=self.mean,
            vol="GARCH",
            p=self.p,
            q=self.q,
            dist=self.dist,
        )
        result = model.fit(disp="off")
        self.result_ = result

        params = result.params
        # arch는 mean='Zero'일 때 ω를 'omega'로, α를 'alpha[1]', β를 'beta[1]'로 부름
        self.omega_ = float(params["omega"])
        self.alpha_ = float(params[f"alpha[{self.p}]"])
        self.beta_ = float(params[f"beta[{self.q}]"])

        # train 마지막 시점의 조건부 분산과 수익률
        cond_var = result.conditional_volatility ** 2  # 표준편차 → 분산
        self.last_h_ = float(cond_var.iloc[-1] if hasattr(cond_var, "iloc") else cond_var[-1])
        self.last_r_ = float(returns_pct[-1])

        return self

    # ------------------------------------------------------------ predict
    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        """
        Test 기간에 대해 1-step-ahead 변동성 예측 (% 스케일).

        재귀(분산 단위): h_{t+1} = ω + α·r_t² + β·h_t
        출력(변동성 단위): σ_t = sqrt(h_t)

        실제 r_t는 test_df의 log_return을 사용 (관측치 흘려넣기).
        """
        if self.omega_ is None:
            raise RuntimeError("call fit() before predict()")
        if "log_return" not in test_df.columns:
            raise ValueError("test_df must have 'log_return' column")

        returns_pct = test_df["log_return"].astype(float).values * SCALE
        n = len(returns_pct)

        h = np.empty(n, dtype=float)
        h_prev = self.last_h_
        r_prev = self.last_r_

        for t in range(n):
            h[t] = self.omega_ + self.alpha_ * (r_prev ** 2) + self.beta_ * h_prev
            # 다음 step 준비: 오늘 관측 수익률을 r_prev로
            r_prev = returns_pct[t]
            h_prev = h[t]

        # 분산 → 변동성: RV_target(%) 단위와 일치시키기 위함
        # h가 음수가 되는 건 GARCH 정상성 가정상 발생하지 않지만 방어적 처리
        sigma = np.sqrt(np.clip(h, 0.0, None))
        return pd.Series(sigma, index=test_df.index, name="garch_vol")
