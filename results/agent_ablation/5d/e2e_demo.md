# MAS end-to-end demonstration

Mode: NORMAL (real LLM narration + metalabel)
## Run manifest

```json
{
  "git_sha": "c5faf5b",
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
  "note": "predict(live forward pass)->gate->horizon_interaction->risk->metalabel->narrator->critic->reasoning chain, real news + real trailing vol/drawdown, real attention/recency-gate explainability (no frozen-cache shortcut)"
}
```


## ACB @ 2024-12-31 -> **ABSTAIN** (size +0.00)

```text
── STEP 1/8 · predict_agent ──────────────────── 48.630s
  gate_pred   : +0.03006
  seed_mean   : +0.03006
  seeds       : [3 items]
  source      : live_inference
```
```text
── STEP 2/8 · gate_agent ──────────────────── 0.001s
  tau         : +0.01800
  coverage    : 0.25
  action      : long
  size        : +1.25742
  reason      : |pred|=0.0301 >= tau=0.0180 -> long @ size=+1.26
```
```text
── STEP 3/8 · horizon_interaction_agent ──────────────────── 67.389s
  agreement   : 2
  multiplier  : +0.60000
  size        : +0.75445
```
```text
── STEP 4/8 · risk_agent ──────────────────── 0.000s
  action      : long
  vetoed      : False
  veto_reasons: [0 items]
  size        : +0.75445
```
```text
── STEP 5/8 · metalabel_agent ──────────────────── 52.405s
  action      : abstain
  flags       : [1 items]
  vetoed      : True
  size        : +0.00000
```
```text
── STEP 6/8 · narrator ──────────────────── 38.248s
  chars       : 363
```
```text
── STEP 7/8 · critic_agent ──────────────────── 0.000s
  status      : ok
  findings    : [0 items]
```
```text
── STEP 8/8 · reasoning_agent ──────────────────── 0.000s
  triggered   : [0 items]
  widened     : False
  size        : +0.00000
```

**Answer:** Khuyến nghị KHÔNG GIAO DỊCH cho mã ACB trong 5 ngày tới. Mô hình phân tích hiện tại thiếu độ tin cậy do chỉ số độ chính xác vận hành thấp (~73%) và khoảng tin cậy còn chồng lấn tỷ lệ nền ~53.1% trên tập kiểm định. Điều này làm giảm đáng kể hiệu suất của hệ thống trong việc đưa ra quyết định giao dịch, nên tránh thực hiện các hoạt động mua/bán tại thời điểm này.

**Critic:** status=ok findings=[]

**Metalabel:** flags=['regulatory_or_policy_action'] vetoed=True

**Attention (top trailing days):** [{'days_before_cutoff': 3, 'weight': 0.0352}, {'days_before_cutoff': 1, 'weight': 0.0351}, {'days_before_cutoff': 5, 'weight': 0.033}]

**Reasoning agent:** triggered=[] widened=False notes=None

## ACB @ 2025-01-15 -> **ABSTAIN** (size +0.00)

```text
── STEP 1/8 · predict_agent ──────────────────── 0.495s
  gate_pred   : +0.01587
  seed_mean   : +0.01587
  seeds       : [3 items]
  source      : live_inference
```
```text
── STEP 2/8 · gate_agent ──────────────────── 0.001s
  tau         : +0.01800
  coverage    : 0.25
  action      : abstain
  size        : +0.00000
  reason      : |pred|=0.0159 < tau=0.0180 -> abstain
```
```text
── STEP 3/8 · horizon_interaction_agent ──────────────────── 0.000s
  agreement   : None
  multiplier  : None
  size        : +0.00000
```
```text
── STEP 4/8 · risk_agent ──────────────────── 0.000s
  action      : abstain
  vetoed      : False
  veto_reasons: [0 items]
  size        : +0.00000
```
```text
── STEP 5/8 · metalabel_agent ──────────────────── 0.000s
  action      : abstain
  flags       : [0 items]
  vetoed      : False
  size        : +0.00000
```
```text
── STEP 6/8 · narrator ──────────────────── 31.357s
  chars       : 336
```
```text
── STEP 7/8 · critic_agent ──────────────────── 0.000s
  status      : ok
  findings    : [0 items]
```
```text
── STEP 8/8 · reasoning_agent ──────────────────── 0.000s
  triggered   : [0 items]
  widened     : False
  size        : +0.00000
```

**Answer:** Khuyến nghị KHÔNG GIAO DỊCH cho mã ACB trong 5 ngày tới. Mô hình phân tích hiện tại thiếu độ tin cậy do chỉ đạt khoảng ~73% độ chính xác với bao phủ ~25%. Điều này có nghĩa là tỷ lệ nền còn chồng lấn ~53.1% trên tập kiểm định, làm giảm đáng kể hiệu chuẩn của hệ thống. Do đó, việc từ chối giao dịch là phù hợp trong tình huống hiện tại.

**Critic:** status=ok findings=[]

**Metalabel:** flags=[] vetoed=False

**Attention (top trailing days):** [{'days_before_cutoff': 1, 'weight': 0.0378}, {'days_before_cutoff': 5, 'weight': 0.0341}, {'days_before_cutoff': 6, 'weight': 0.0326}]

**Reasoning agent:** triggered=[] widened=False notes=None

## ACB @ 2025-04-10 -> **ABSTAIN** (size +0.00)

```text
── STEP 1/8 · predict_agent ──────────────────── 0.344s
  gate_pred   : -0.02272
  seed_mean   : -0.02272
  seeds       : [3 items]
  source      : live_inference
```
```text
── STEP 2/8 · gate_agent ──────────────────── 0.001s
  tau         : +0.01800
  coverage    : 0.25
  action      : short
  size        : -0.95051
  reason      : |pred|=0.0227 >= tau=0.0180 -> short @ size=-0.95
```
```text
── STEP 3/8 · horizon_interaction_agent ──────────────────── 0.681s
  agreement   : 0
  multiplier  : +0.60000
  size        : -0.57031
```
```text
── STEP 4/8 · risk_agent ──────────────────── 0.000s
  action      : abstain
  vetoed      : True
  veto_reasons: [1 items]
  size        : +0.00000
```
```text
── STEP 5/8 · metalabel_agent ──────────────────── 0.000s
  action      : abstain
  flags       : [0 items]
  vetoed      : False
  size        : +0.00000
```
```text
── STEP 6/8 · narrator ──────────────────── 29.742s
  chars       : 269
```
```text
── STEP 7/8 · critic_agent ──────────────────── 0.000s
  status      : ok
  findings    : [0 items]
```
```text
── STEP 8/8 · reasoning_agent ──────────────────── 0.000s
  triggered   : [0 items]
  widened     : False
  size        : +0.00000
```

**Answer:** Khuyến nghị KHÔNG GIAO DỊCH cho mã ACB trong 5 ngày tới. Mô hình phân tích hiện tại thiếu độ tin cậy do chỉ số độ chính xác thấp (~73%) và khoảng tin cậy còn chồng lấn tỷ lệ nền ~53.1% trên tập kiểm định. Biến động cao (43.3%) và rủi ro vetoed cũng là yếu tố cần lưu ý.

**Critic:** status=ok findings=[]

**Metalabel:** flags=[] vetoed=False

**Attention (top trailing days):** [{'days_before_cutoff': 0, 'weight': 0.0428}, {'days_before_cutoff': 3, 'weight': 0.0391}, {'days_before_cutoff': 1, 'weight': 0.0363}]

**Reasoning agent:** triggered=[] widened=False notes=None
