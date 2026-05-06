# FINTEL

국가별 주가지수 **실현변동성(Realized Volatility, RV)** 예측력을 비교·분석하는 프로젝트입니다.  
미국(S\&P 500)·한국(KOSPI)·일본(Nikkei 225) 3개 시장을 대상으로, 전통 계량금융 모형부터 머신러닝/딥러닝 및 앙상블까지 **다중 horizon(1일·1주·1개월)** 예측 성능과 **외생적 충격 구간(GFC, COVID, 금리인상기 등)** 에서의 강건성을 체계적으로 평가합니다.

> 성균관대학교 데이터사이언스 캡스톤프로젝트 2026 Spring | TEAM FINTEL

---

## 1) 프로젝트 개요 (Overview)

- **Target**: 5분 고빈도 수익률 기반 일별 RV (필요 시 range-based estimator 대안 사용)
- **Markets**: US / KR / JP (S\&P 500, KOSPI, Nikkei 225)
- **Models**
  - 금융모형(Financial): HAR-RV, GARCH(1,1), EGARCH(1,1)
  - ML: Elastic Net, XGBoost
  - DL: MLP, LSTM, CNN(1D), Transformer
  - Ensemble: Stacking, Boosting, Weighted averaging 등
- **Evaluation**: MSE, MAE, QLIKE + Diebold-Mariano Test, Model Confidence Set

---

## 2) 데이터 (Dataset)

- **기간(예정)**: 2000.01 – 2022.06 (약 25년)
- **출처(예정)**: Oxford-Man (고빈도 기반 RV)
- **Feature tiers**
  - Core set (10개)
  - Momentum set (14개)
  - Extended set (29개)

> 데이터 split, 위기 구간(phase), feature tier 정의는 `dataset/` 아래 텍스트 파일로 관리합니다.

---

## 3) 폴더 구조 (Project Structure)

```text
Project/
├─ dataset/
│  ├─ *.csv                 # (12개 + summary)
│  ├─ data_splits.txt
│  ├─ crisis_phases.txt      # Full Test 포함 (GFC, COVID 등)
│  └─ feature_tiers.txt
├─ src/
│  ├─ __init__.py
│  ├─ config.py              # SPLITS, PHASES dict (txt 1:1 매핑)
│  ├─ data_loader.py         # load_dataset / split_train_test / load_split
│  ├─ models/
│  │  ├─ __init__.py         # GARCHModel, HARRVModel export
│  │  ├─ garch.py            # arch_model + 1-step 재귀 예측
│  │  └─ har_rv.py            # statsmodels OLS HC1
│  └─ eval/
│     ├─ __init__.py
│     ├─ metrics.py          # rmse / mae / qlike / rmse_cv / evaluate
│     ├─ phases.py           # iter_phases / get_phase_mask
│     └─ protocols.py        # run_static / run_expanding
├─ notebooks/
│  └─ 01_financial.ipynb     # 그리드 실행 + 시각화
├─ results/.gitkeep
└─ requirements.txt
```

---

## 4) 설치 (Installation)

> **conda**와 **pip/venv** 둘 다 제공합니다.

### A. conda (Recommended)

```bash
conda create -n fintel python=3.11 -y
conda activate fintel
pip install -r requirements.txt
```

### B. pip + venv

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## 5) 실행 방법 (How to Run)

### 5.1 노트북 실행

```bash
jupyter lab
# or
jupyter notebook
```

- 기본 실행 노트북: `notebooks/01_financial.ipynb`

### 5.2 파이썬 모듈 사용(선택)

프로젝트 구조에 따라 `src/`를 모듈로 실행할 수 있습니다. 예:

```bash
python -m src
```

(필요 시 `src/__main__.py` 또는 CLI 스크립트를 추가해 확장 가능합니다.)

---

## 6) 평가 프로토콜 (Evaluation Protocol)

- **Static split** / **Expanding window** 프로토콜 지원 (예: `src/eval/protocols.py`)
- 위기 구간별 성능 비교를 위한 phase mask 제공 (예: `src/eval/phases.py`)

---

## 7) 팀 (Team)

- 팀리더: 황희석 — 프로젝트 총괄, HAR-RV/GARCH 계열 구현, feature 생성, 해석
- 팀원: 구교현 — 외생적 충격 이벤트 식별, 국가별 해석/보고서, 선행연구
- 팀원: 송준혁 — DL(MLP~Transformer) 구현, 앙상블 설계, 데이터셋 구조화
- 팀원: 이건희 — ML(Elastic Net/XGBoost) 구현 및 평가, 데이터셋 구조화

---

## 8) 참고문헌 (References)

- Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003). *Modeling and forecasting realized volatility*. Econometrica.
- Corsi, F. (2009). *A simple approximate long-memory model of realized volatility*. JFEC.
- Bollerslev, T. (1986). *Generalized autoregressive conditional heteroskedasticity*. JoE.
- Nelson, D. B. (1991). *Conditional heteroskedasticity in asset returns: A new approach*. Econometrica.
- Patton, A. J. (2011). *Volatility forecast comparison using imperfect volatility proxies*. JoE.
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). *The model confidence set*. Econometrica.
- Kim, H. Y., & Won, C. H. (2018). *Forecasting the volatility of stock price index...* ESWA.

(추가 참고문헌은 제안서/보고서에서 확장합니다.)

---

## 9) 면책 조항 (Disclaimer)

본 프로젝트는 **연구/학습 목적**이며, 어떠한 형태로도 **투자 자문(financial advice)** 이 아닙니다.
