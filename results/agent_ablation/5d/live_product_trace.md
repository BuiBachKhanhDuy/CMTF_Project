# Multi-agent run trace

## Run manifest

```json
{
  "git_sha": "e516b49",
  "eval_mode": false,
  "seed": null,
  "ensemble_seeds": [
    1,
    42,
    123
  ],
  "cmtf_version": "v4",
  "backbone_version": "v3",
  "gate_coverage": 0.25,
  "gate_on_raw_seed": false,
  "news_scope_default": "matched",
  "symbol": "VCB",
  "cutoff": "2025-08-13",
  "horizon": 5
}
```

## Workflow

```text
── STEP 1/9 · orchestrator ──────────────────── 2758.565s
  intent      : PREDICTION
  symbols     : [1 items]
  horizon     : 5
  route       : symbol+horizon supplied by CLI → PREDICTION
```

```text
── STEP 2/9 · market_agent ──────────────────── 0.000s
  vol_20d     : 16.8%
  max_dd      : 4.2%
  trend       : +8.6%
```

```text
── STEP 3/9 · news_agent ──────────────────── 0.002s
  coverage    : 30
  staleness   : 39%
  sentiment   : +0.513
```

```text
── STEP 4/9 · predict_agent ──────────────────── 0.476s
  gate_pred   : +0.01953
  seed_mean   : +0.01953
  seeds       : [3 items]
  source      : frozen_prediction_cache
```

```text
── STEP 5/9 · gate_agent ──────────────────── 0.001s
  tau         : +0.01800
  coverage    : 0.25
  action      : long
  size        : +0.81687
  reason      : |pred|=0.0195 >= tau=0.0180 -> long @ size=+0.82
```

```text
── STEP 6/9 · risk_agent ──────────────────── 0.000s
  action      : long
  vetoed      : False
  veto_reasons: [0 items]
  size        : +0.81687
```

```text
── STEP 7/9 · metalabel_agent ──────────────────── 45.817s
```

```text
── STEP 8/9 · narrator ──────────────────── 33.067s
  chars       : 274
```

```text
── STEP 9/9 · critic_agent ──────────────────── 50.821s
  status      : failed
  findings    : [1 items]
```

## Final answer

Khuyến nghị cho VCB (5 ngày): MUA. Kích thước vị thế: +0.82. |pred|=0.0195 >= tau=0.0180 -> long @ size=+0.82. Điểm vận hành: ~54% độ chính xác hướng ở mức bao phủ ~25% (khoảng tin cậy còn chồng lấn tỷ lệ nền ~53.8% trên một mã — giá trị của hệ thống là việc TỪ CHỐI giao dịch có hiệu chuẩn, không phải độ chính xác). Bối cảnh thị trường: biến động 20 ngày 16.8%, sụt giảm tối đa 4.2%.
