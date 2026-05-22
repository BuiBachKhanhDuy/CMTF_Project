# Cross-Modal Temporal Fusion for Vietnamese Stock Forecasting

## 1. Executive Summary

This project builds an end-to-end forecasting pipeline for Vietnamese banking stocks by combining:

- Market time series (OHLCV + engineered technical features + VN-Index macro features)
- Vietnamese financial news embeddings (PhoBERT-based sentence vectors)

The benchmark is centered on Amazon Chronos (`amazon/chronos-t5-small`) with added baselines and fusion variants.

Current experiment set:

1. Chronos Zero-Shot
2. Chronos + CMTF
3. LSTM Baseline
4. LSTM + CMTF
5. Random Forest Baseline
6. Chronos Fine-Tuned

Note: the linear probe path was removed from the codebase because it was not suitable for the final Chronos workflow.

---

## 2. Data and Setup

- Symbols: VCB, BID
- Date range: 2022-01-01 to 2026-03-31
- Frequency: daily
- News sources: CafeF Banking, VnExpress, Vietstock
- Temporal alignment: strict market-close cutoff to avoid leakage
- Targets: forward log returns at 1D, 5D, 20D
- Split protocol: walk-forward train/validation/test with horizon-aware purge

---

## 3. Model Summary

### 3.1 Chronos Zero-Shot

- Uses Chronos directly on close windows
- No task-specific training

### 3.2 Chronos + CMTF

- Cross-modal fusion head on top of Chronos embeddings
- News-conditioned FiLM modulation and gated residual fusion
- Multi-seed ensemble for robustness

### 3.3 LSTM Baseline

- Price-window LSTM regression baseline
- Includes best-checkpoint restore and embedding extraction

### 3.4 LSTM + CMTF

- CMTF applied over LSTM-derived embeddings
- Residual-gated fallback to avoid harmful fusion

### 3.5 Random Forest Baseline

- Tree baseline over engineered market features

### 3.6 Chronos Fine-Tuned

- Trainable head over Chronos features

---

## 4. Evaluation Metrics

Primary metrics:

- MAE
- RMSE
- DA% (directional accuracy)
- Sharpe
- IC (information coefficient)
- Precision / Recall / F1

Composite metric used in model ranking:

$$
\text{CompositeScore} = 0.5\cdot\text{RMSE} + 0.15\cdot\text{MAE} + 0.15\cdot\left(1-\frac{\text{DA\%}}{100}\right) + 0.12\cdot\text{ModalDisagreement} + 0.08\cdot\text{TemporalLag}
$$

Where lower is better.

---

## 5. Latest Results (Averages Across VCB and BID)

### 5.1 Horizon 1D

| Model | MAE | RMSE | DA% | Composite | Sharpe | IC |
|---|---:|---:|---:|---:|---:|---:|
| Chronos Zero-Shot | 0.0133 | 0.0202 | 49.19 | 0.1683 | 0.2883 | -0.0029 |
| Chronos + CMTF | 0.0193 | 0.0309 | 51.88 | 0.2287 | -1.1558 | -0.0569 |
| LSTM Baseline | 0.0122 | 0.0191 | 52.77 | 0.2195 | 0.9877 | 0.1166 |
| LSTM + CMTF | 0.0122 | 0.0190 | 55.28 | 0.2141 | 1.1910 | 0.1322 |
| Random Forest Baseline | 0.0138 | 0.0205 | 52.95 | 0.2294 | 0.3005 | 0.0626 |
| Chronos Fine-Tuned | 0.0122 | 0.0191 | 50.09 | 0.2249 | -0.5063 | -0.0225 |

### 5.2 Horizon 5D

| Model | MAE | RMSE | DA% | Composite | Sharpe | IC |
|---|---:|---:|---:|---:|---:|---:|
| Chronos Zero-Shot | 0.0312 | 0.0499 | 50.84 | 0.1513 | -0.3319 | -0.0052 |
| Chronos + CMTF | 0.0490 | 0.0832 | 52.36 | 0.2517 | 0.4044 | 0.0511 |
| LSTM Baseline | 0.0279 | 0.0451 | 55.57 | 0.1660 | 1.1218 | 0.3082 |
| LSTM + CMTF | 0.0279 | 0.0448 | 57.60 | 0.1614 | 1.3412 | 0.2911 |
| Random Forest Baseline | 0.0298 | 0.0468 | 58.11 | 0.2337 | 1.2380 | 0.1292 |
| Chronos Fine-Tuned | 0.0291 | 0.0467 | 47.47 | 0.2434 | 0.2441 | -0.1080 |

### 5.3 Horizon 20D

| Model | MAE | RMSE | DA% | Composite | Sharpe | IC |
|---|---:|---:|---:|---:|---:|---:|
| Chronos Zero-Shot | 0.0740 | 0.1044 | 46.76 | 0.2232 | -0.3216 | -0.1253 |
| Chronos + CMTF | 0.0652 | 0.0966 | 68.65 | 0.1802 | 1.1937 | 0.4637 |
| LSTM Baseline | 0.0698 | 0.0992 | 55.69 | 0.2560 | 0.3200 | 0.2786 |
| LSTM + CMTF | 0.0704 | 0.1003 | 57.79 | 0.1839 | 0.8131 | 0.2838 |
| Random Forest Baseline | 0.0828 | 0.1153 | 52.89 | 0.2264 | 0.0155 | 0.0146 |
| Chronos Fine-Tuned | 0.0711 | 0.1018 | 47.46 | 0.2577 | 0.0252 | -0.1508 |

Main takeaway at 20D: Chronos + CMTF is the best overall on RMSE, DA%, CompositeScore, Sharpe, and IC.

---

## 6. Implementation Notes

- Composite metric is exported directly in benchmark CSV files.
- CMTF objective is regression-first with directional penalty.
- LSTM + CMTF includes validation-based residual gating to prevent degradation.
- Linear code path is removed from market predictor and benchmark orchestration.

---

## 7. Reproducibility

Key commands:

```powershell
python pipeline.py
python run_model_benchmark.py --stage predict
python run_model_benchmark.py --stage predict --horizons 20
pytest -v
```

Output files:

- `results/chronos_benchmark_1d.csv`
- `results/chronos_benchmark_5d.csv`
- `results/chronos_benchmark_20d.csv`
- `results/figures/*`

---

## 8. References

1. Ansari et al. (2024). Chronos: Learning the Language of Time Series.
2. Perez et al. (2018). FiLM: Visual Reasoning with a General Conditioning Layer.
3. Lim et al. (2021). Temporal Fusion Transformers for Interpretable Multi-Horizon Forecasting.
4. Dong et al. (2021). Attention is Not All You Need: Pure Attention Loses Rank.
5. Nguyen and Nguyen (2020). PhoBERT: Pre-trained Language Models for Vietnamese.