# FINTEL 중간발표 결과 시각화

이 폴더는 중간발표의 "분석·모델링 결과" 파트에 사용할 핵심 시각화와 요약 테이블을 생성합니다.

## 목적

기존 결과 파일은 수정하지 않고 읽기만 하며, 발표용 산출물은 이 폴더 내부에 저장합니다.

## 입력 파일

스크립트는 다음 두 구조를 모두 지원합니다.

- `results/`, `dataset/`이 repo root에 있는 구조
- `Project/results/`, `Project/dataset/`이 repo root 아래에 있는 구조

필요 파일:

- `results/financial_results.csv`
- `results/ml_results_core.csv`
- `results/ml_results_momentum.csv`
- `results/ml_results_extended.csv`
- `dataset/dataset_summary.csv`

## 실행 방법

repo root에서 실행합니다.

```bash
python 중간발표/result_visualizations/make_result_visualizations.py
```

## 생성 산출물

### `outputs/`

- `01_experiment_flow.png`: 동일 데이터·동일 지표·동일 구간 비교라는 실험 구조
- `02_best_financial_vs_ml.png`: 전통 금융모형 대비 ML의 성능 비교
- `03_best_model_matrix.png`: 시장·위기별 최적 모델 매트릭스
- `04_feature_tier_effect.png`: Core에서 Extended로 확장했을 때의 개선율
- `05_market_insight_cards.png`: US/KR/JP별 시장 해석 카드

### `summary_tables/`

- `financial_vs_ml_summary.csv`: Best Financial vs Best ML 그래프의 원천 요약표
- `best_model_matrix.csv`: 시장·위기별 best model 요약표
- `feature_tier_effect.csv`: Feature Tier 개선율 요약표

## 발표 메시지

1. 실험 구조: 같은 데이터, 같은 구간, 같은 평가 지표로 공정하게 비교했다.
2. ML vs 금융모형: ML은 전통 금융모형보다 전반적으로 낮은 예측 오차를 보였다.
3. Best Model Matrix: 단일 최적 모델은 없고, 시장과 위기 유형에 따라 적합한 모델이 달라진다.
4. Feature Tier 효과: 변수 추가가 항상 성능 개선으로 이어지지는 않으며, 정보 적합성이 중요하다.
5. 시장별 해석: 성능 차이는 알고리즘 차이뿐 아니라 시장 구조와 충격 반응 차이로 해석할 수 있다.
