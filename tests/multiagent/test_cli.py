"""Regression tests for cmd_h5_reasoning_eval's evaluation_mode wiring.

History: this command originally called `run_reasoning_eval(...)` without a config,
defaulting to `MultiAgentConfig()` (evaluation_mode=False) — which made
`metalabel_agent` attempt a REAL Ollama call for every trade candidate. At the time,
Ollama appeared unreachable in this environment (a raw connectivity check returned
HTTP 403), so the fix then was to force `evaluation_mode=True` unconditionally.

That "Ollama unreachable" conclusion was later found to be a proxy-configuration
testing artifact, not a real limitation (see `src/multiagent/guards.py::
ensure_local_no_proxy`, already called before every real LLM invocation in this
codebase) — and forcing `evaluation_mode=True` unconditionally had its own real cost:
`narrator_agent_node` returns `answer_text=""` in eval mode, so `critic_agent_node`'s
regeneration/failure branch can never fire, meaning `critic_status` is always "ok" and
`reasoning_agent`'s `critic_verification_failed` trigger is permanently untestable.
The command now defaults to `evaluation_mode=False` (real LLM, real critic_status),
with an explicit `--eval-mode` flag for the fast, deterministic fallback.
"""

from src.multiagent import cli


class Args:
    horizon = 1
    n = 10
    seed = 0
    eval_mode = False


def test_h5_reasoning_eval_defaults_to_real_llm(monkeypatch):
    captured = {}

    def fake_run_reasoning_eval(horizon, n, seed, config=None):
        captured["config"] = config
        return {"out_path": "fake.json"}

    monkeypatch.setattr(
        "src.multiagent.h5_reasoning_eval.run_reasoning_eval", fake_run_reasoning_eval,
    )

    cli.cmd_h5_reasoning_eval(Args())

    assert captured["config"] is not None
    assert captured["config"].evaluation_mode is False


def test_h5_reasoning_eval_dash_eval_mode_flag_forces_eval_mode(monkeypatch):
    captured = {}

    def fake_run_reasoning_eval(horizon, n, seed, config=None):
        captured["config"] = config
        return {"out_path": "fake.json"}

    monkeypatch.setattr(
        "src.multiagent.h5_reasoning_eval.run_reasoning_eval", fake_run_reasoning_eval,
    )

    class EvalModeArgs(Args):
        eval_mode = True

    cli.cmd_h5_reasoning_eval(EvalModeArgs())

    assert captured["config"] is not None
    assert captured["config"].evaluation_mode is True
