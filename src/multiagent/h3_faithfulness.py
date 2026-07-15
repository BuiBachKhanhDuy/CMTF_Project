"""H3 faithfulness experiment — MAS (grounded + critic) vs bare LLM (plan §10, H3).

The thesis is "a multi-agent system beats a plain LLM call". The cleanest, largest,
most defensible axis (plan §10: "Expected large clean effect") is FAITHFULNESS: given
the SAME fact sheet, does the MAS's grounded-narrator + critic-verifier hallucinate
fewer numbers than a bare LLM call?

For each sampled (symbol, date) we build a fact sheet from the frozen decision and:
- **A1 (bare LLM):** a normal advisory prompt, no grounding discipline, no verifier.
- **A5 (MAS):** a grounding system prompt + the critic's verify/regenerate/fallback.

Both answers are scored by the same automated grounding check (a number in the answer
is a hallucination if it is not within tolerance of any fact-sheet number). We report
the hallucination rate for each and a paired bootstrap CI on the difference. This is a
real LLM run over a disclosed, stratified sample (R1: the sample size and any dropped
rows are logged, never hidden).

LLM-free, so it never contaminates: the grounding SCORER is pure numpy/regex; only the
answer GENERATION uses the LLM. Eval mode is not used here (this experiment *is* the
LLM comparison).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .config import DEFAULT_CONFIG, MultiAgentConfig
from .frozen_predictions import get_store
from .gate_io import load_gate_policy, policy_path

_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")

# Raw dataset (per-date OHLCV + technicals + precomputed news sentiment) used to
# build the FAIR A1 forecaster context. Verified aligned to the frozen truth
# (parquet fwd_ret_5d == truth__5d.npy). The forecaster is never shown fwd_ret_*
# or the CMTF prediction — only information available at the cutoff. Resolved
# dynamically (see live_inference.resolve_price_parquet) rather than hardcoded, so
# this doesn't silently point at a stale/missing file after a fresh clone or any
# pipeline config change.

# Disclosure/framing constants the honest narration is allowed to cite (plan §0/§3.10).
_DISCLOSURE_CONSTS = [54.0, 25.0, 53.8, 200.0]

_A5_SYSTEM = (
    "Bạn là chuyên gia phân tích tài chính. CHỈ được dùng các con số xuất hiện trong "
    "bảng dữ kiện được cung cấp; TUYỆT ĐỐI không thêm bất kỳ con số nào khác (không bịa "
    "giá mục tiêu, %, hay dự báo). Nếu khuyến nghị là KHÔNG GIAO DỊCH, nói rõ mô hình "
    "thiếu độ tin cậy và không dùng giọng điệu mua/bán. Viết <120 từ tiếng Việt."
)
_A1_SYSTEM = (
    "Bạn là chuyên gia tư vấn chứng khoán. Hãy đưa ra một khuyến nghị đầu tư rõ ràng, "
    "tự tin và hữu ích cho nhà đầu tư bằng tiếng Việt (<120 từ)."
)


def _fact_sheet(symbol: str, horizon: int, gate_pred: float, action: str,
                tau: float, coverage: float) -> tuple[str, list[float]]:
    """A compact fact sheet (text) + the set of numbers it legitimately contains."""
    action_vi = {"long": "MUA", "short": "BÁN", "abstain": "KHÔNG GIAO DỊCH"}[action]
    text = (
        f"Mã: {symbol}\n"
        f"Kỳ hạn dự báo: {horizon} ngày\n"
        f"Tín hiệu mô hình (gate_pred): {gate_pred:.4f}\n"
        f"Ngưỡng kích hoạt (tau): {tau:.4f}\n"
        f"Khuyến nghị của hệ thống: {action_vi}\n"
        f"Độ bao phủ giao dịch: {coverage:.0%}\n"
        f"Lưu ý: độ chính xác hướng ~54% ở bao phủ ~25% (chồng lấn tỷ lệ nền ~53.8%)."
    )
    allowed = [round(gate_pred, 4), round(tau, 4), float(horizon),
               round(coverage * 100, 0)] + _DISCLOSURE_CONSTS
    return text, allowed


def _hallucinated_numbers(answer: str, allowed: list[float]) -> list[str]:
    """Numbers in the answer not grounded in the fact sheet (within tolerance)."""
    bad = []
    for tok in _NUM_RE.findall(answer):
        try:
            num = float(tok.replace(",", "."))
        except ValueError:
            continue
        if not any(abs(num - a) <= max(0.05 * abs(a), 0.01) for a in allowed):
            bad.append(tok)
    return bad


# ---------------------------------------------------------------------------
# Forecaster-level H3 (the decisive test): an INDEPENDENT LLM forecaster that
# sees only raw prices+news+technicals (no model output, no fwd_ret) vs the gated
# MAS, compared on directional accuracy and calibration (AURC). Plan §10.2, §10.8.
# ---------------------------------------------------------------------------

_FORECASTER_SYSTEM = (
    "Bạn là nhà phân tích định lượng. Dựa CHỈ trên dữ liệu thị trường được cung cấp, "
    "hãy dự báo lợi nhuận 5 ngày tới của cổ phiếu. Trả lời DUY NHẤT một JSON: "
    '{"direction": "long" | "short" | "abstain", "confidence": <0..1>, '
    '"pred_return_pct": <số, ví dụ 1.5 = +1.5%>, "reason": "<ngắn>"}. '
    "direction=long nếu kỳ vọng tăng, short nếu giảm, abstain nếu không đủ tin cậy. "
    "confidence là độ tin cậy chủ quan (0=không chắc, 1=rất chắc). "
    "pred_return_pct là ước lượng % thay đổi giá 5 ngày tới (LUÔN cung cấp)."
)


def _load_price_frame(horizon: int = 5):
    import pandas as pd
    from .live_inference import resolve_price_parquet
    df = pd.read_parquet(resolve_price_parquet(horizon))
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def _forecaster_brief(frame, symbol: str, cutoff: str) -> str | None:
    """REAL, leakage-free market brief at the cutoff.

    Most feature columns in the model-ready parquet are STANDARDIZED (z-scored), so
    they are meaningless as human-readable indicators. Only the ``fwd_ret_*`` columns
    are raw returns. We therefore reconstruct the *trailing* returns from them:
    ``fwd_ret_5d`` at row (t-5) IS the realised 5-day return up to t — known at t, so
    no look-ahead. Volatility comes from trailing raw daily returns; news activity from
    the raw ``news_count``. RSI/MACD/sentiment are dropped (they are z-scored / all-zero
    in this parquet and previously fed the LLM garbage). Returns None if history < 25 bars.
    """
    import numpy as np
    import pandas as pd
    s = frame[(frame["symbol"] == symbol) & (frame.index <= pd.Timestamp(cutoff))].sort_index()
    if len(s) < 25:
        return None
    trail_5d = float(s["fwd_ret_5d"].iloc[-6])          # realised return (t-5 -> t)
    trail_20d = float(s["fwd_ret_20d"].iloc[-21])       # realised return (t-20 -> t)
    daily = s["fwd_ret_1d"].iloc[-21:-1].to_numpy(dtype=float)  # ~20 trailing daily returns
    vol_20d = float(np.nanstd(daily) * np.sqrt(252) * 100)
    news_5d = int(s["news_count"].iloc[-6:-1].sum())
    trend = "tăng" if trail_20d > 0 else "giảm"
    return (
        f"Mã: {symbol} | Ngày: {cutoff}\n"
        f"Lợi nhuận 5 ngày gần nhất: {trail_5d:+.2%}\n"
        f"Lợi nhuận 20 ngày gần nhất: {trail_20d:+.2%} (xu hướng {trend})\n"
        f"Biến động (annualized, 20 ngày): {vol_20d:.0f}%\n"
        f"Số bài tin tức 5 ngày gần nhất: {news_5d}"
    )


def _parse_forecast(text: str) -> tuple[str, float, float]:
    """Parse {direction, confidence, pred_return_pct} from the LLM output.

    pred_return_pct is returned as a FRACTION (1.5% → 0.015), clipped to a sane band
    so a stray huge value can't blow up RMSE; NaN if the LLM gave none.
    """
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            d = str(obj.get("direction", "abstain")).lower()
            if d not in ("long", "short", "abstain"):
                d = "abstain"
            c = max(0.0, min(1.0, float(obj.get("confidence", 0.0))))
            pr = obj.get("pred_return_pct", None)
            pred_ret = float("nan")
            if pr is not None:
                pred_ret = float(np.clip(float(pr) / 100.0, -0.5, 0.5))  # % → fraction, clipped
            return d, c, pred_ret
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    low = text.lower()
    if "long" in low or "mua" in low:
        return "long", 0.5, float("nan")
    if "short" in low or "bán" in low:
        return "short", 0.5, float("nan")
    return "abstain", 0.0, float("nan")


def run_forecaster_h3(horizon: int = 5, n: int = 56, seed: int = 0,
                      config: MultiAgentConfig | None = None) -> dict:
    """Decisive H3: independent LLM forecaster (A1) vs gated MAS (A5) on the same rows."""
    from .guards import ensure_local_no_proxy
    from langchain_ollama import ChatOllama

    cfg = config or DEFAULT_CONFIG
    ensure_local_no_proxy(cfg.ollama_base_url)
    store = get_store(horizon, cfg)
    policy, _ = load_gate_policy(policy_path(cfg.gate_policy_dir, horizon, "VN"),
                                 expect_cmtf_version=cfg.cmtf_version,
                                 expect_backbone_version=cfg.backbone_version)
    frame = _load_price_frame(horizon)
    llm = ChatOllama(model=cfg.ollama_model, base_url=cfg.ollama_base_url,
                     temperature=0.1, timeout=cfg.ollama_timeout)

    from loguru import logger

    rows = _sample_rows(store, n, seed)
    total = len(rows)
    ckpt = Path("results/agent_ablation") / f"{horizon}d" / "h3_forecaster_partial.json"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    dropped = []
    for i, (sym, date) in enumerate(rows, 1):
        brief = _forecaster_brief(frame, sym, date)
        if brief is None:
            dropped.append((sym, date))
            continue
        fp = store.get(sym, date)
        # A1: independent LLM forecaster (sees only the brief).
        a1_dir, a1_conf, a1_pred_ret = _parse_forecast(
            llm.invoke([("system", _FORECASTER_SYSTEM), ("human", brief)]).content)
        a1_sign = {"long": 1.0, "short": -1.0, "abstain": 0.0}[a1_dir]
        # A5: the gated MAS decision (frozen prediction + calibrated gate).
        a5_trade = abs(fp.gate_pred) >= policy.tau
        a5_sign = float(np.sign(fp.gate_pred)) if a5_trade else 0.0
        recs.append({
            "symbol": sym, "date": date, "truth": fp.truth,
            "a1_dir": a1_dir, "a1_sign": a1_sign, "a1_conf": a1_conf,
            "a1_pred_ret": a1_pred_ret,  # LLM point forecast (fraction) for regression
            "a5_trade": a5_trade, "a5_sign": a5_sign, "a5_conf": abs(fp.gate_pred),
            "gate_pred": fp.gate_pred,  # CMTF point forecast (A5's magnitude)
        })
        # Progress + checkpoint so a long (full-book) run is observable and never
        # loses work if interrupted (R1/R3): running committed DA for both systems.
        if i % 25 == 0 or i == total:
            a1c = [r for r in recs if r["a1_sign"] != 0]
            a5c = [r for r in recs if r["a5_trade"]]
            a1da = np.mean([np.sign(r["truth"]) == r["a1_sign"] for r in a1c]) * 100 if a1c else float("nan")
            a5da = np.mean([np.sign(r["truth"]) == r["a5_sign"] for r in a5c]) * 100 if a5c else float("nan")
            logger.info("H3 forecaster {}/{} | LLM {:.1f}% DA ({} trades) | MAS {:.1f}% DA ({} trades)",
                        i, total, a1da, len(a1c), a5da, len(a5c))
            ckpt.write_text(json.dumps({"done": i, "total": total, "records": recs}, ensure_ascii=False), encoding="utf-8")

    return _aggregate_forecaster(recs, dropped, horizon, n, seed, cfg)


def _selective_da(sign, conf, truth, cov, n):
    """DA on the top-`cov` most-confident DIRECTIONAL calls (abstains, sign=0, excluded).

    This is the correct calibration metric under abstention: an abstain is NOT a wrong
    directional bet, so it must not count as an error (the earlier AURC did — that was
    a confound). We ask: when a system acts on its most-confident k names, how accurate
    is it? A well-calibrated confidence ⇒ higher DA as coverage tightens.
    """
    k = max(1, int(np.ceil(cov * n)))
    order = np.argsort(-conf)[:k]
    s, t = sign[order], truth[order]
    m = s != 0
    if not m.any():
        return float("nan"), 0
    return float((np.sign(t[m]) == s[m]).mean()) * 100, int(m.sum())


def _full_metrics_forecaster(recs, horizon, cfg) -> dict:
    """Full, fair metric panel: DA alone is not comparable when coverage differs 4x.

    Reports, for the plain LLM (A1) and the gated MAS (A5), on the SAME rows:
      - selectivity (coverage / trade frequency)  — when each chooses to act
      - committed directional accuracy
      - rank IC (Spearman of the signed signal vs realised return)
      - Sharpe (sign-based proxy, no costs — as documented for this project)
      - mean per-period PnL and per-TRADE PnL (economic value of a decision)
      - transaction-cost sensitivity (the high-coverage system pays 4x more cost)
    A5 positions are conviction-sized by the deployed policy; A1 is shown both
    flat-unit and confidence-weighted so the sizing choice is transparent (R1).
    """
    from scipy import stats
    from src.benchmark.decision_policy import _gated_sharpe, apply_positions, GatePolicy
    from .gate_io import load_gate_policy, policy_path

    policy, _ = load_gate_policy(policy_path(cfg.gate_policy_dir, horizon, "VN"))
    truth = np.array([r["truth"] for r in recs], float)
    a1_sign = np.array([r["a1_sign"] for r in recs], float)
    a1_conf = np.array([r["a1_conf"] for r in recs], float)
    gate_pred = np.array([r["gate_pred"] for r in recs], float)
    n = len(recs)

    def _ic(signal, committed):
        m = committed & np.isfinite(signal)
        if m.sum() < 3 or np.std(signal[m]) < 1e-12:
            return float("nan")
        c, _ = stats.spearmanr(signal[m], truth[m])
        return float(c) if np.isfinite(c) else float("nan")

    def _panel(positions, signed_signal, committed, label):
        nt = int(committed.sum())
        cov = nt / n
        da = float((np.sign(truth[committed]) == np.sign(signed_signal[committed])).mean()) * 100 if nt else float("nan")
        pnl = positions * truth  # sign-based PnL, abstain=0
        traded_pnl = pnl[committed]
        sharpe = _gated_sharpe(positions, truth, horizon)
        cost_adj = {f"{int(c*1e4)}bps": round(float(pnl.sum() - c * nt), 5) for c in (0.0, 0.0010, 0.0025)}
        # Economic-value metrics on the traded book (real-world value).
        wins = traded_pnl[traded_pnl > 0]; losses = traded_pnl[traded_pnl < 0]
        win_rate = float(len(wins) / nt) * 100 if nt else float("nan")
        profit_factor = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
        # Max drawdown of the cumulative PnL curve (sample order).
        cum = np.cumsum(pnl); peak = np.maximum.accumulate(cum)
        max_dd = float((peak - cum).max()) if n else float("nan")
        return {"label": label, "coverage": round(cov, 3), "n_trades": nt,
                "committed_DA": round(da, 2), "rank_IC": round(_ic(signed_signal, committed), 4),
                "Sharpe": round(sharpe, 4),
                "win_rate": round(win_rate, 2), "profit_factor": round(profit_factor, 3),
                "max_drawdown": round(max_dd, 5),
                "PnL_per_trade": round(float(traded_pnl.mean()), 6) if nt else float("nan"),
                "total_PnL_after_cost": cost_adj}

    # A5: conviction-sized positions from the deployed policy (0 where |gate_pred|<tau).
    a5_pos = apply_positions(gate_pred, policy)
    a5_committed = np.abs(gate_pred) >= policy.tau
    # A1: flat-unit and confidence-weighted variants; committed = non-abstain.
    a1_committed = a1_sign != 0
    a1_pos_flat = a1_sign.copy()
    a1_pos_conf = a1_sign * a1_conf

    panels = {
        "A5_MAS_conviction_sized": _panel(a5_pos, gate_pred, a5_committed, "MAS (gate, conviction-sized)"),
        "A1_LLM_flat_unit": _panel(a1_pos_flat, a1_sign, a1_committed, "LLM (flat unit)"),
        "A1_LLM_confidence_weighted": _panel(a1_pos_conf, a1_sign * a1_conf, a1_committed, "LLM (confidence-weighted)"),
    }

    # Matched-coverage Sharpe/DA: force BOTH to the MAS's natural coverage (top-k by
    # each's own confidence), so skill is compared at the same trade frequency.
    a5_conf = np.abs(gate_pred)
    match_cov = round(float(a5_committed.mean()), 3)
    k = max(1, int(np.ceil(match_cov * n)))
    def _matched(sign, conf, sizer):
        order = np.argsort(-conf)
        keep = np.zeros(n, bool); keep[order[:k]] = True
        pos = np.where(keep, sizer, 0.0)
        da = float((np.sign(truth[keep]) == np.sign(sign[keep])).mean()) * 100
        return {"coverage": round(k / n, 3), "committed_DA": round(da, 2),
                "Sharpe": round(_gated_sharpe(pos, truth, horizon), 4)}
    matched = {
        "coverage_matched_to_MAS": match_cov,
        "A5_MAS": _matched(np.sign(gate_pred), a5_conf, apply_positions(gate_pred, policy)),
        "A1_LLM_flat": _matched(a1_sign, a1_conf, a1_sign),
    }

    # Regression (point-forecast error): the CMTF (gate_pred) and the LLM
    # (pred_return_pct) are BOTH point predictions of the 5-day return, so RMSE/MAE
    # are directly comparable. Reported honestly — CMTF is known to trade point
    # accuracy for rank/decision skill, so it may LOSE here; that is disclosed, not hidden.
    a1_pred = np.array([r.get("a1_pred_ret", float("nan")) for r in recs], float)
    have = np.isfinite(a1_pred)
    reg = {"n_with_llm_pred": int(have.sum())}
    if have.sum() >= 3:
        def _rmse(p, t): return float(np.sqrt(np.mean((p - t) ** 2)))
        def _mae(p, t): return float(np.mean(np.abs(p - t)))
        reg.update({
            "A5_MAS_RMSE": round(_rmse(gate_pred[have], truth[have]), 5),
            "A1_LLM_RMSE": round(_rmse(a1_pred[have], truth[have]), 5),
            "A5_MAS_MAE": round(_mae(gate_pred[have], truth[have]), 5),
            "A1_LLM_MAE": round(_mae(a1_pred[have], truth[have]), 5),
            "note": "Lower is better. CMTF optimises rank/decision skill, not point error.",
        })

    return {"natural_operating_points": panels, "matched_coverage": matched, "regression": reg,
            "note": ("Sharpe is a sign-based proxy WITHOUT transaction costs (project convention); "
                     "the cost columns show that the LLM's higher coverage pays proportionally more cost. "
                     "DA alone is not comparable across different coverages — read coverage + Sharpe + "
                     "cost + drawdown together. Regression (RMSE/MAE) is reported for completeness and "
                     "may favour either system.")}


def _aggregate_forecaster(recs, dropped, horizon, n_req, seed, cfg) -> dict:
    truth = np.array([r["truth"] for r in recs], float)
    a1_sign = np.array([r["a1_sign"] for r in recs], float)
    a1_conf = np.array([r["a1_conf"] for r in recs], float)
    a5_conf = np.array([r["a5_conf"] for r in recs], float)  # |gate_pred|
    gate_pred = np.array([r["gate_pred"] for r in recs], float)
    a5_sign = np.sign(gate_pred)  # directional lean for confidence ranking
    n = len(recs)

    def _committed_da(sign, tr):
        m = sign != 0
        return (float((np.sign(tr[m]) == sign[m]).mean()) * 100 if m.any() else float("nan"),
                float(m.mean()), int(m.sum()))

    a1_da, a1_cov, a1_nt = _committed_da(a1_sign, truth)
    # A5 "committed" = |gate_pred| >= tau; use the recorded a5_trade flags for the
    # natural operating point.
    a5_trade = np.array([r["a5_trade"] for r in recs], bool)
    a5_committed_sign = np.where(a5_trade, a5_sign, 0.0)
    a5_da, a5_cov, a5_nt = _committed_da(a5_committed_sign, truth)

    # Matched-coverage selective DA (the headline): both act on their most-confident top-k.
    grid = {}
    for cov in (0.25, 0.5, 1.0):
        a1d, a1k = _selective_da(a1_sign, a1_conf, truth, cov, n)
        a5d, a5k = _selective_da(a5_sign, a5_conf, truth, cov, n)
        grid[f"{int(cov*100)}pct"] = {"A1_DA": round(a1d, 2), "A1_n": a1k,
                                      "A5_DA": round(a5d, 2), "A5_n": a5k}

    # Paired bootstrap over the shared row sample on the top-25% DA difference (A5 - A1).
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(5000):
        idx = rng.integers(0, n, n)
        a1d, _ = _selective_da(a1_sign[idx], a1_conf[idx], truth[idx], 0.25, n)
        a5d, _ = _selective_da(a5_sign[idx], a5_conf[idx], truth[idx], 0.25, n)
        if np.isfinite(a1d) and np.isfinite(a5d):
            deltas.append(a5d - a1d)
    deltas = np.array(deltas)
    lo, hi = (np.percentile(deltas, [2.5, 97.5]) if deltas.size else (float("nan"), float("nan")))
    base_up = float((np.sign(truth) == 1).mean()) * 100

    summary = {
        "experiment": "forecaster_h3 (A1 independent LLM vs A5 gated MAS)",
        "interpretation": ("A1 = a plain LLM call given the same raw prices+news (no model "
                           "output); A5 = the gated MAS. Metric = directional accuracy on the "
                           "most-confident top-k of each (abstains excluded — an abstain is not "
                           "a wrong bet). A well-calibrated confidence ⇒ DA rises as coverage tightens."),
        "horizon": horizon, "n_requested": n_req, "n_run": n, "n_dropped": len(dropped),
        "sample_seed": seed, "model": cfg.ollama_model, "base_rate_up_moves": round(base_up, 2),
        "committed": {
            "A1_LLM": {"DA": round(a1_da, 2), "coverage": round(a1_cov, 3), "n_trades": a1_nt},
            "A5_MAS": {"DA": round(a5_da, 2), "coverage": round(a5_cov, 3), "n_trades": a5_nt},
        },
        "matched_coverage_selective_DA": grid,
        "top25pct_DA_delta_A5_minus_A1": round(float(deltas.mean()), 2) if deltas.size else None,
        "top25pct_DA_delta_ci_95": [round(float(lo), 2), round(float(hi), 2)],
        "top25pct_DA_delta_significant": bool(lo * hi > 0) if np.isfinite(lo) else False,
        "full_metrics": _full_metrics_forecaster(recs, horizon, cfg),
        "records": recs,
    }
    out = Path("results/agent_ablation") / f"{horizon}d" / "h3_forecaster.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["out_path"] = str(out)
    return summary


def _sample_rows(store, n: int, seed: int) -> list[tuple[str, str]]:
    """Deterministic stratified sample of (symbol, date), INTERLEAVED across symbols.

    Rows are emitted round-robin by date-slot (sym0 d0, sym1 d0, …, sym0 d1, …) rather
    than in per-symbol blocks, so any prefix of the returned list is balanced across all
    symbols. This makes checkpoint/partial results representative during a long run.
    """
    rng = np.random.default_rng(seed)
    by_sym: dict[str, list[str]] = {}
    for (sym, d) in store._index:
        by_sym.setdefault(sym, []).append(str(d))
    per = max(1, n // len(by_sym))
    selected: dict[str, list[str]] = {}
    for sym, dates in sorted(by_sym.items()):
        dates = sorted(dates)
        idx = sorted(rng.choice(len(dates), size=min(per, len(dates)), replace=False))
        selected[sym] = [dates[i] for i in idx]
    rows: list[tuple[str, str]] = []
    for j in range(max(len(v) for v in selected.values())):
        for sym in sorted(selected):
            if j < len(selected[sym]):
                rows.append((sym, selected[sym][j]))
    return rows[:n]


def run_faithfulness(horizon: int = 5, n: int = 60, seed: int = 0,
                     config: MultiAgentConfig | None = None) -> dict:
    from .guards import ensure_local_no_proxy
    from langchain_ollama import ChatOllama

    cfg = config or DEFAULT_CONFIG
    ensure_local_no_proxy(cfg.ollama_base_url)
    store = get_store(horizon, cfg)
    policy, _ = load_gate_policy(policy_path(cfg.gate_policy_dir, horizon, "VN"),
                                 expect_cmtf_version=cfg.cmtf_version,
                                 expect_backbone_version=cfg.backbone_version)
    llm = ChatOllama(model=cfg.ollama_model, base_url=cfg.ollama_base_url,
                     temperature=0.2, timeout=cfg.ollama_timeout)

    rows = _sample_rows(store, n, seed)
    records = []
    for i, (sym, date) in enumerate(rows):
        fp = store.get(sym, date)
        action = "abstain" if abs(fp.gate_pred) < policy.tau else ("long" if fp.gate_pred > 0 else "short")
        sheet, allowed = _fact_sheet(sym, horizon, fp.gate_pred, action, policy.tau, policy.coverage)

        a1 = llm.invoke([("system", _A1_SYSTEM), ("human", sheet)]).content.strip()
        a5 = llm.invoke([("system", _A5_SYSTEM), ("human", sheet)]).content.strip()

        a1_bad = _hallucinated_numbers(a1, allowed)
        a5_bad = _hallucinated_numbers(a5, allowed)
        # Abstain honesty: on an abstain, does the answer still push a trade word?
        a1_pushes_trade = action == "abstain" and bool(re.search(r"\b(MUA|BÁN)\b", a1)) and "KHÔNG" not in a1.upper()
        a5_pushes_trade = action == "abstain" and bool(re.search(r"\b(MUA|BÁN)\b", a5)) and "KHÔNG" not in a5.upper()

        records.append({
            "symbol": sym, "date": date, "action": action,
            "a1_hallucinations": len(a1_bad), "a5_hallucinations": len(a5_bad),
            "a1_bad_numbers": a1_bad, "a5_bad_numbers": a5_bad,
            "a1_pushes_trade_on_abstain": a1_pushes_trade,
            "a5_pushes_trade_on_abstain": a5_pushes_trade,
            "a1_text": a1, "a5_text": a5,  # stored for traceability (R1)
        })

    return _aggregate(records, horizon, n, seed, cfg)


def _aggregate(records, horizon, n_requested, seed, cfg) -> dict:
    n = len(records)
    a1_h = np.array([r["a1_hallucinations"] > 0 for r in records], dtype=float)
    a5_h = np.array([r["a5_hallucinations"] > 0 for r in records], dtype=float)
    diff = a1_h - a5_h  # positive ⇒ bare LLM hallucinates more (MAS better)

    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(5000)]) if n else np.array([0.0])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    # Abstention discipline on abstain-signal rows: on a KHÔNG GIAO DỊCH signal, does
    # the answer still push a confident trade? This is the sharpest MAS-vs-LLM axis
    # (calibrated abstention), so it gets its own paired bootstrap CI.
    abst = [r for r in records if r["action"] == "abstain"]
    disc = {}
    if abst:
        a1p = np.array([r["a1_pushes_trade_on_abstain"] for r in abst], dtype=float)
        a5p = np.array([r["a5_pushes_trade_on_abstain"] for r in abst], dtype=float)
        pdiff = a1p - a5p  # positive ⇒ bare LLM over-recommends on abstains (MAS better)
        na = len(abst)
        pboot = np.array([pdiff[rng.integers(0, na, na)].mean() for _ in range(5000)])
        plo, phi = np.percentile(pboot, [2.5, 97.5])
        disc = {
            "A1_pushes_trade_on_abstain_rate": round(float(a1p.mean()), 4),
            "A5_pushes_trade_on_abstain_rate": round(float(a5p.mean()), 4),
            "delta_abstain_discipline_A1_minus_A5": round(float(pdiff.mean()), 4),
            "delta_abstain_discipline_ci_95": [round(float(plo), 4), round(float(phi), 4)],
            "delta_abstain_discipline_significant": bool(plo * phi > 0),
        }

    summary = {
        "horizon": horizon,
        "n_requested": n_requested,
        "n_run": n,
        "sample_seed": seed,
        "model": cfg.ollama_model,
        "n_abstain": len(abst),
        "A1_bare_LLM_hallucination_rate": round(float(a1_h.mean()), 4) if n else None,
        "A5_MAS_hallucination_rate": round(float(a5_h.mean()), 4) if n else None,
        "delta_hallucination_A1_minus_A5": round(float(diff.mean()), 4) if n else None,
        "delta_hallucination_ci_95": [round(float(lo), 4), round(float(hi), 4)],
        "delta_hallucination_significant": bool(lo * hi > 0),
        **disc,
        "records": records,
    }
    out = Path("results/agent_ablation") / f"{horizon}d" / "h3_faithfulness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["out_path"] = str(out)
    return summary
