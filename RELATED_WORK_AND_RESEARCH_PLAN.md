# News–Market Fusion: Related Work & Research Plan

Companion to `CMTF_FUSION_FINDINGS.md`. Purpose: situate our placebo-controlled
negative/marginal result (news adds rank-IC but not directional/trading skill; a
shuffled-news placebo matches real news on DA/Sharpe) inside the published
literature, and lay out concrete next steps.

All arXiv IDs below were verified from live arXiv metadata. Pre-2019 foundational
works (StockNet, HAN) are cited from domain knowledge — confirm before quoting
exact numbers.

---

## 1. Why this matters: the field is split, and the split is *methodological*

Two camps report opposite conclusions about news→price prediction, and the
difference is almost entirely **evaluation design**, not model architecture:

**Strong-positive camp** (big, clean gains):
- **Kirtac & Germano (2024)**, *Sentiment trading with LLMs*, Finance Research
  Letters (arXiv 2412.19245). 965k US news articles; OPT sentiment → **74.4%**
  next-day directional accuracy; long–short portfolio **Sharpe 3.05**. Note the
  setup: **cross-sectional long–short** over many names, **daily** horizon, huge
  corpus.
- **STST** (arXiv 2305.03835): +10.4% profit over S&P500 in simulation.

**Marginal / negative camp** (honest, rigorous):
- **Siala et al. (2026)**, *Impact of LLM News Sentiment on Stock Price Movement*,
  ICLR 2026 AFA workshop (arXiv 2602.00086): across LSTM/PatchTST/tPatchGNN/
  TimesNet, sentiment features benefit *"(slightly) some models."*
- **Our CMTF result**: news adds IC only, not DA/Sharpe; placebo-indistinguishable.

**The reconciling insight:** the positive results are almost all
**cross-sectional, relative-ranking, long–short** setups (which is exactly what
rank-IC rewards). The marginal/negative results are **single-name directional**
setups (which DA/Sharpe reward). *This is the same IC-vs-DA decoupling we measured
internally.* Our finding is not anomalous — it is what you get when you evaluate
the metric news *doesn't* help (per-name sign) instead of the one it does
(cross-sectional rank).

> **Implication for our thesis framing:** don't claim "news doesn't work." Claim
> "news provides cross-sectional rank signal that does **not** convert to
> single-name directional/trading skill under a placebo-controlled, base-rate-
> relative evaluation" — and cite the positive camp to show *where* it does work.

---

## 2. Taxonomy of fusion mechanisms (with representative papers)

| Fusion type | Mechanism | Representative work | Relevance to CMTF |
|---|---|---|---|
| Late / decision | combine per-modality outputs | our `late(lstm)`; most sentiment-feature pipelines | easy to gate/veto (our λ-guard lives here) |
| Early / feature | concat text+price, joint encoder | our `early` rows | strong modality dominates / noise injection |
| **Cross-modal attention** | news attends to market ctx | **STONK** (2508.13327); **MulT low-rank** (2007.02038) | **our CMTF family** |
| **Gated cross-attention** | consistency-guided admission | **MSGCA** (2406.06594) | the literature's fix for our instability |
| Graph / relational | cross-stock news propagation | **MGRN** (2107.10941); **CausalStock**, NeurIPS'24 (2411.06391) | denoising + causal relation = our missing pieces |
| Event-structured | structured event tuples, not raw text | Chen et al. (1910.05078); **SER** (2512.19484) | structured inputs beat raw embeddings |
| LLM / agentic | reasoning over multi-source | TradExpert (2411.00782); RETuning (2510.21604) | frontier; out of current scope |

Foundational benchmarks (know these for Related Work):
- **StockNet** — Xu & Cohen, ACL 2018. VAE + attention over tweets+price; the
  **ACL18** dataset. Reports binary accuracy + MCC (not IC/Sharpe).
- **HAN** — Hu et al., *Listening to Chaotic Whispers*, WSDM 2018. Hierarchical
  attention over news; the **KDD17** dataset.

---

## 3. The literature already knows news is mostly noise

This is the strongest external support for our placebo finding — cite it directly:

- **Liu et al. (2020)**, *Attention-Based Noisy Recurrent State Transition*
  (arXiv 2004.01878, Neurocomputing). They **explicitly model a "noisy random
  factor"** separating news effect from noise. A top group found this *necessary*.
- **CausalStock** (arXiv 2411.06391, NeurIPS 2024): premise is *"substantial noise
  exists in the news data"*; solution is an LLM-based **Denoised News Encoder**.
- **RETuning** (arXiv 2510.21604): LLMs *"follow analysts' opinions rather than
  exhibit independent analytical logic"* and *"list summaries without weighing
  adversarial evidence"* — i.e. raw news reasoning is low-signal without structure.

Takeaway: raw projected news embeddings (our `NewsProjector` = linear map) are the
**wrong input**. The consensus is: **denoise / structure the news before fusion.**

---

## 4. The evaluation-rigor literature (our methodology backbone)

Our placebo + block-stability guard + ESS correction align with a mature
"better evaluation, not more models" literature. Cite these to defend the method:

- **Baquero (2026)**, *Bitcoin Price Prediction: Peer-Reviewed Evidence…*
  (arXiv 2606.00071): *"the field's primary need is not more models but better
  evaluation."* Proposes exactly our standards: **walk-forward eval, multi-regime
  holdout, naive-baseline comparison, zero-in-hyperparameter-grid, Diebold-Mariano
  significance.** (We already do naive baseline, DM test, zero-in-λ-grid.)
- **Nguyen & Pham (2026)**, *Reliable Evaluation of LLM Financial Multi-Agent
  Systems* (arXiv 2603.27539): names **five evaluation failures that can reverse
  the sign of reported returns** — look-ahead bias, survivorship bias, backtest
  overfitting, transaction-cost neglect, regime-shift blindness.
- **López de Prado** lineage on **backtest overfitting / deflated Sharpe**
  (Carr & LdP 2014, arXiv 1408.1159; CSCV). Our block-stability guard is a
  home-grown cousin of Combinatorially-Symmetric Cross-Validation.
- **Zhang & Zhang (2026)**, *LLMs for Stock Forecasting: A Hedge-Fund Perspective*
  (arXiv 2605.05211): pitfalls catalogue — sentiment fragility, horizon design,
  evaluation metrics, data leakage, illiquidity premia, limits of predictability.

> Our `metrics.py` (ESS-adjusted, per-series thresholds, base-rate-relative DA,
> DM test, paired bootstrap) and `fusion_selection.py` (block-stability λ-guard)
> are already best-practice by this literature's standards. That is a *strength*
> to foreground, not a footnote.

---

## 5. Techniques that could actually make news help (ranked by expected payoff)

Given our diagnosis (raw embeddings = noise; no relevance filter; fixed recency;
no cross-modal veto in CMTF), the highest-leverage next experiments:

1. **Denoise/structure news before fusion** — *highest payoff.*
   - Structured event representation (SER, 2512.19484) or LLM event extraction
     (CausalStock) instead of raw pooled embeddings.
   - Test: does a structured-event input beat its own shuffled placebo on DA?
2. **Relevance filtering via attention pooling with entity anchors**
   - Self-/cross-/position-aware pooling with stock-name embeddings
     (arXiv 2603.19286 reports −7.11% MAE from this filtering alone).
   - We have recency gating (time) but no relevance gating (content).
3. **News impact-duration modeling** (IDED, arXiv 2409.17419)
   - Replace fixed `recency_gate_k` with a learned decay; horizon-aware.
4. **Gated / consistency-guided fusion** (MSGCA, arXiv 2406.06594)
   - Give CMTF the same validation-time downside veto late fusion has, but
     *architecturally* (a learned gate driven by the market "primary" feature).
5. **Move the eval target to where news works: cross-sectional ranking**
   - Reframe from single-name directional (DA/Sharpe) to **cross-sectional
     long–short rank** (matches the Kirtac & Germano / IC-favorable regime).
   - This directly tests the §1 reconciling hypothesis on our own data.

---

## 6. Concrete research agenda (sequenced)

**Phase A — settle "method vs. signal" on current setup (cheap, 1–2 days)**
- A1. Run `select_by_ic=False` (DA-aware) vs IC/loss selection A/B at 1D and 5D.
      *Question:* can real news beat its shuffled placebo on DA under *any*
      selection objective? If never → tilt toward "no exploitable directional
      signal at these horizons." If yes → "method, not signal."
- A2. Sign-flip / decile-IC diagnostic on saved predictions (read-only): confirm
      CMTF's IC gain lives in the tails while sign flips concentrate near zero.

**Phase B — test the cross-sectional hypothesis (medium, 3–5 days)**
- B1. Re-evaluate the *same* predictions as a cross-sectional long–short book;
      report rank-IC-based portfolio Sharpe. If news helps here but not in per-name
      DA, we have reproduced the field's split on our own data — a clean result.

**Phase C — make news actually help (larger, exploratory)**
- C1. Structured-event or LLM-denoised news input (lever #1) vs raw-embedding
      baseline, both placebo-controlled.
- C2. Entity-anchored relevance pooling (lever #2).
- C3. Learned impact-duration gate (lever #3).
- Each C-experiment must clear the same bar: **beat its own shuffled placebo**
  across seeds *and* validation blocks (our existing protocol).

**Phase D — write-up positioning**
- Frame as: *placebo-controlled, base-rate-relative evaluation of news–market
  fusion*, showing (a) cross-modal attention converts weak news into rank-IC
  overfit, (b) this does **not** survive as single-name directional skill, (c)
  consistent with the noise-modeling (Liu 2020, CausalStock) and evaluation-rigor
  (Baquero 2026, Nguyen 2026) literatures, and (d) — if Phase B confirms — the lift
  reappears only in the cross-sectional ranking regime where the positive camp
  (Kirtac & Germano 2024) operates.

---

## 7. Reading list (priority order)

1. STONK — 2508.13327 (closest architecture to CMTF)
2. MSGCA — 2406.06594 (gated cross-attention = our stability fix)
3. CausalStock — 2411.06391 (news denoising, NeurIPS'24)
4. Siala et al. — 2602.00086 (our marginal result, corroborated)
5. Kirtac & Germano — 2412.19245 (the strong-positive counter-example)
6. Baquero — 2606.00071 (evaluation-standards manifesto)
7. Zhang & Zhang — 2605.05211 (hedge-fund pitfalls review)
8. Liu et al. — 2004.01878 (news-as-noise, explicit noise factor)
9. SER — 2512.19484 (structured event inputs)
10. Nguyen & Pham — 2603.27539 (five sign-reversing eval failures)
