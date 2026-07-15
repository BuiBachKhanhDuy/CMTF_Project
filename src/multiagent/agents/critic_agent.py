"""Critic Agent — verify the narration against state (plan §3.11).

The MAS-vs-LLM differentiator: before an answer is returned, check that
1. every number in the answer is grounded in state (no fabrication),
2. the stated action equals the decided ``action`` (no abstain→trade flip),
3. the tone matches the decision (no confident buy/sell wording on an abstain).

On failure it regenerates (bounded, ``critic_max_retries``) with a stricter prompt;
if it still fails it falls back to the deterministic grounded template and marks
``critic_status='failed'``. It NEVER edits numbers into agreement and never hides a
detected problem — a persistent failure is surfaced and logged (R1).
"""

from __future__ import annotations

import re
import time
from typing import Any

from loguru import logger

from ..config import DEFAULT_CONFIG, MultiAgentConfig
from ..state import MultiAgentState
from .narrator_agent import generate_answer, grounded_template

_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
# Vietnamese action words the narrator may emit.
_TRADE_WORDS = ("MUA", "BÁN")


def _allowed_numbers(state: MultiAgentState) -> list[float]:
    """Numeric values the answer is permitted to mention (state-derived)."""
    me = state.get("model_evidence", {})
    vm = state.get("volatility_metrics", {})
    sm = state.get("sentiment_metrics", {})
    vals: list[float] = []
    for v in (me.get("final_pred"), me.get("gate_pred"), state.get("position_scale"),
              state.get("gate_tau"), vm.get("vol_20d"), vm.get("max_drawdown_pct"),
              vm.get("trend_pct"), sm.get("coverage"), state.get("target_horizon_days")):
        if isinstance(v, (int, float)):
            vals.append(float(v))
    # Coverage/CI disclosure constants that legitimately appear in the template.
    gc = state.get("gate_coverage")
    if isinstance(gc, (int, float)):
        vals.append(round(float(gc) * 100, 0))  # coverage as %
    vals += [54.0, 25.0, 53.8, 20.0, 200.0, 1.0, 5.0]  # disclosure/horizon constants
    return vals


def verify_answer(answer: str, state: MultiAgentState) -> list[str]:
    """Return a list of grounding/consistency findings (empty ⇒ verified)."""
    findings: list[str] = []
    if not answer.strip():
        return findings  # empty (eval-mode) narration: nothing to verify

    # 2. action-match / 3. tone
    action = state.get("action", "abstain")
    action_vi = {"long": "MUA", "short": "BÁN", "abstain": "KHÔNG GIAO DỊCH"}.get(action, "")
    if action == "abstain":
        # A confident trade word appearing as a recommendation on an abstain is a flip.
        # (KHÔNG GIAO DỊCH contains neither MUA nor BÁN.)
        for w in _TRADE_WORDS:
            if re.search(rf"\b{w}\b", answer) and "KHÔNG" not in answer.upper():
                findings.append(f"tone: answer uses trade word {w!r} on an abstain")
                break
    elif action_vi and action_vi not in answer.upper():
        findings.append(f"action-match: answer does not state the decided action {action_vi!r}")

    # 1. numbers grounded in state
    allowed = _allowed_numbers(state)
    for tok in _NUM_RE.findall(answer):
        num = float(tok.replace(",", "."))
        if not any(abs(num - a) <= max(0.05 * abs(a), 0.01) for a in allowed):
            findings.append(f"ungrounded number in answer: {tok}")
    return findings


def critic_agent_node(
    state: MultiAgentState,
    config: MultiAgentConfig | None = None,
) -> dict[str, Any]:
    """LangGraph node: verify (and if needed regenerate/fallback) the answer.

    Reads: answer_text, grounded_answer, action, model_evidence, volatility_metrics
    Writes: answer_text (final), critic_status, critic_findings, node_timings
    """
    cfg = config or DEFAULT_CONFIG
    t0 = time.time()

    answer = state.get("answer_text", "")
    template = state.get("grounded_answer") or grounded_template(state)

    findings = verify_answer(answer, state)
    status = "ok"

    if answer.strip() and findings:
        if cfg.evaluation_mode:
            # Eval mode is LLM-free, so we cannot regenerate. We do NOT report "ok"
            # over unverified text (R1): fall back to the deterministic template.
            answer, status = template, "failed"
            logger.warning("Critic | findings in eval mode, cannot regenerate → template fallback; "
                           "findings={}", findings)
        else:
            # Bounded regeneration with a stricter prompt.
            for attempt in range(1, cfg.critic_max_retries + 1):
                regen = generate_answer(state, cfg, strict=True)
                regen_findings = verify_answer(regen, state)
                logger.info("Critic | regen attempt {} → {} findings", attempt, len(regen_findings))
                if not regen_findings:
                    answer, findings, status = regen, [], "regenerated"
                    break
            else:
                # Persistent failure: fall back to the deterministic template, surface it.
                answer, status = template, "failed"
                logger.warning("Critic | verification failed after {} retries → grounded template fallback; "
                               "findings={}", cfg.critic_max_retries, findings)

    elapsed = time.time() - t0
    logger.info("Critic | status={} findings={} | {:.3f}s", status, len(findings), elapsed)
    return {
        "answer_text": answer,
        "critic_status": status,
        "critic_findings": findings,
        "node_timings": {"critic_agent": elapsed},
    }
