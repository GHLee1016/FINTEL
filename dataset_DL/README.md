# dataset_DL/

DL 모델 학습용 sliding window 데이터셋. `notebooks/04_build_dl_dataset.ipynb`로 자동 생성.

## 폴더 구조

```
dataset_DL/
├── feature_columns_US.txt           # 28 features (US, spillover_KOSPI/Nikkei)
├── feature_columns_KR.txt           # 28 features (KR, spillover_SP500/Nikkei)
├── feature_columns_JP.txt           # 28 features (JP, spillover_SP500/KOSPI)
├── L22/   12 cell × 3 file = 36 file
├── L60/   36 file
└── L252/  36 file
```

각 L 폴더에 12개 (regime, country) cell별로 3 파일:
- `{regime}_{country}_meta.csv` — sample_id, prediction_date, split, RV_target
- `{regime}_{country}_X.npy` — `(N, L, 28)` float32 시퀀스 텐서
- `{regime}_{country}_y.npy` — `(N,)` float32 타겟

## Sample 매핑 (핵심)

```
sample_id = i
X[i]            = 원본 CSV의 index i ~ (i+L-1) 행의 features  (shape (L, 28))
y[i]            = 원본 CSV의 index (i+L-1) 행의 RV_target
                  ⚠️ RV_target은 이미 t+1일 RV로 저장돼 있음 (원본 정의 그대로)
prediction_date = 원본 CSV의 index (i+L-1) 행의 date (X[i]의 마지막 timestep)
split           = 원본 CSV의 index (i+L-1) 행의 split
N (총 sample)   = len(원본 CSV) - L + 1
```

**Lookback이 split 경계를 가로지를 수 있음** (예: valid 첫 sample의 일부 timestep이 train 끝부분에 걸침).
과거 정보라 leak 아님.

## Tier(core/momentum/extended) 처리

X.npy에는 항상 시장의 **28 features 모두** 저장. tier는 학습 코드에서 column index로 subset:

```python
feat_names = open('dataset_DL/feature_columns_US.txt').read().splitlines()
core_feats = ['RV_d', 'RV_w', 'RV_m', 'log_return', 'neg_return',
              'semivariance', 'parkinson_rv', 'hl_range', 'weekday_sin', 'weekday_cos']
col_idx = [feat_names.index(f) for f in core_feats]

X = np.load('dataset_DL/L22/normal_US_X.npy')   # (N, 22, 28)
X_core = X[:, :, col_idx]                       # (N, 22, 10)
```

## Scaling

**raw 저장**. 학습 시 train split에서만 fit, valid/test에 transform 적용 (look-ahead 방지).
`src/preprocess/scaler.CustomScaler` 그대로 사용 — `(N*L, F)`로 reshape 후 fit/transform → 원래 shape 복원.

## 재생성

`notebooks/04_build_dl_dataset.ipynb` 한 번 실행. 기존 dataset_DL/ 폴더는 통째로 삭제 후 재생성됨.
원본 `dataset/*.csv`가 바뀌면 다시 실행해야 함.
