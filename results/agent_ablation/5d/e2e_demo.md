# MAS end-to-end demonstration

Mode: NORMAL (real LLM narration + metalabel)
## Run manifest

```json
{
  "git_sha": "e516b49",
  "eval_mode": false,
  "seed": 0,
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
  "demo_rows": [
    [
      "ACB",
      "2024-12-31"
    ],
    [
      "ACB",
      "2025-01-15"
    ],
    [
      "ACB",
      "2025-04-10"
    ]
  ],
  "note": "decision->veto->narration->critic chain over frozen predictions + real news + real trailing vol/drawdown"
}
```


## ACB @ 2024-12-31 -> **LONG** (size +1.26)

```text
── STEP 1/6 · predict_agent ──────────────────── 0.000s
  gate_pred   : +0.03006
  seed_mean   : +0.03006
  seeds       : [3 items]
  source      : frozen_prediction_cache
```
```text
── STEP 2/6 · gate_agent ──────────────────── 0.001s
  tau         : +0.01800
  coverage    : 0.25
  action      : long
  size        : +1.25742
  reason      : |pred|=0.0301 >= tau=0.0180 -> long @ size=+1.26
```
```text
── STEP 3/6 · risk_agent ──────────────────── 0.000s
  action      : long
  vetoed      : False
  veto_reasons: [0 items]
  size        : +1.25742
```
```text
── STEP 4/6 · metalabel_agent ──────────────────── 51.127s
```
```text
── STEP 5/6 · narrator ──────────────────── 34.200s
  chars       : 296
```
```text
── STEP 6/6 · critic_agent ──────────────────── 0.000s
  status      : ok
  findings    : [0 items]
```

**Answer:** Khuyến nghị bạn không nên giao dịch với mã ACB trong 5 ngày tới. Dữ liệu cho thấy mô hình phân tích chỉ đạt khoảng 54% độ chính xác và bao phủ 25%, tương đương với mức độ tin cậy còn thấp, chỉ khoảng 53.8%. Do đó, việc từ chối giao dịch là hợp lý hơn so với việc mua bán dựa trên khuyến nghị này.

**Critic:** status=ok findings=[]

**Metalabel:** flags=[] vetoed=False

## ACB @ 2025-01-15 -> **ABSTAIN** (size +0.00)

```text
── STEP 1/6 · predict_agent ──────────────────── 0.000s
  gate_pred   : +0.01587
  seed_mean   : +0.01587
  seeds       : [3 items]
  source      : frozen_prediction_cache
```
```text
── STEP 2/6 · gate_agent ──────────────────── 0.002s
  tau         : +0.01800
  coverage    : 0.25
  action      : abstain
  size        : +0.00000
  reason      : |pred|=0.0159 < tau=0.0180 -> abstain
```
```text
── STEP 3/6 · risk_agent ──────────────────── 0.000s
  action      : abstain
  vetoed      : False
  veto_reasons: [0 items]
  size        : +0.00000
```
```text
── STEP 4/6 · metalabel_agent ──────────────────── 0.000s
```
```text
── STEP 5/6 · narrator ──────────────────── 31.958s
  chars       : 332
```
```text
── STEP 6/6 · critic_agent ──────────────────── 0.001s
  status      : ok
  findings    : [0 items]
```

**Answer:** Khuyến nghị KHÔNG GIAO DỊCH cho mã ACB trong 5 ngày tới. Lý do: Mô hình phân tích hiện tại chỉ đạt khoảng bao phủ 25% và độ chính xác hướng là ~54%, với khoảng tin cậy còn chồng lấn tỷ lệ nền ~53.8%. Do đó, mặc dù dự báo không có xu hướng tăng giảm mạnh, nhưng mức độ tin cậy của mô hình chưa đủ cao để đưa ra khuyến nghị giao dịch.

**Critic:** status=ok findings=[]

**Metalabel:** flags=[] vetoed=False

## ACB @ 2025-04-10 -> **ABSTAIN** (size +0.00)

```text
── STEP 1/6 · predict_agent ──────────────────── 0.000s
  gate_pred   : -0.02272
  seed_mean   : -0.02272
  seeds       : [3 items]
  source      : frozen_prediction_cache
```
```text
── STEP 2/6 · gate_agent ──────────────────── 0.001s
  tau         : +0.01800
  coverage    : 0.25
  action      : short
  size        : -0.95051
  reason      : |pred|=0.0227 >= tau=0.0180 -> short @ size=-0.95
```
```text
── STEP 3/6 · risk_agent ──────────────────── 0.000s
  action      : abstain
  vetoed      : True
  veto_reasons: [1 items]
  size        : +0.00000
```
```text
── STEP 4/6 · metalabel_agent ──────────────────── 0.000s
```
```text
── STEP 5/6 · narrator ──────────────────── 36.176s
  chars       : 395
```
```text
── STEP 6/6 · critic_agent ──────────────────── 41.007s
  status      : regenerated
  findings    : [0 items]
```

**Answer:** Khuyến nghị KHÔNG GIAO DỊCH cho mã ACB trong 5 ngày tới. Lý do: mô hình thiếu độ tin cậy với chỉ số kích thước bằng 0.00 và điểm vận hành chỉ đạt khoảng 54% độ chính xác, bao phủ 25%. Đồng thời, rủi ro từ biến động cao (43.3%) đã veto giao dịch. Mặc dù dự báo ensemble là -0.02272, nhưng do độ tin cậy thấp và không hiệu chuẩn, việc từ chối giao dịch là phù hợp.

**Critic:** status=regenerated findings=[]

**Metalabel:** flags=[] vetoed=False
