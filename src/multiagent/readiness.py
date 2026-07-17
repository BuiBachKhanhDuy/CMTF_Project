"""Deployment-readiness checks — "does this horizon have everything it needs?"

Makes "the system can check and use suitable horizons" a real, testable property
instead of an emergent side effect of R1 fail-loud errors first discovered when a
query happens to hit a missing artifact. Deliberately reuses the EXACT loaders the
real request path uses (``load_gate_policy``, ``get_store``,
``live_inference.deploy_checkpoint_paths``) as the single source of truth, so this
module can never drift out of sync with what the graph actually requires — it is a
proactive, deploy-time check, not a re-derived parallel set of rules.

This is deliberately NOT wired into the hot per-request path (``graph.py``): every
node already fails loud (``ArtifactMissingError``/``StalePolicyError``) at the exact
point of use, so adding a stat/glob/JSON-read pass to every single request would be
unnecessary coupling/cost for what is fundamentally a deploy-time concern. Use this
module (or ``python -m src.multiagent check-deploy``) before/after a deployment, or
at a chat entry point's cheap startup check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_CONFIG, MultiAgentConfig
from .gate_io import StalePolicyError, core_cell_for, load_gate_policy, policy_path
from .loaders import ArtifactMissingError

MATCHED_CELL_ID = "8"  # kept in sync with frozen_predictions.MATCHED_CELL_ID (rank_agent's cell)


@dataclass(frozen=True)
class ReadinessReport:
    horizon: int
    gate_policy_ok: bool
    gate_policy_detail: str
    core_predictions_ok: bool
    core_predictions_detail: str
    matched_predictions_ok: bool
    matched_predictions_detail: str
    deploy_checkpoints_ok: bool
    deploy_checkpoints_detail: str

    @property
    def ready(self) -> bool:
        return (
            self.gate_policy_ok
            and self.core_predictions_ok
            and self.matched_predictions_ok
            and self.deploy_checkpoints_ok
        )

    def problems(self) -> list[str]:
        """Human-readable list of what's missing; empty ⇒ ready."""
        checks = (
            (self.gate_policy_ok, self.gate_policy_detail),
            (self.core_predictions_ok, self.core_predictions_detail),
            (self.matched_predictions_ok, self.matched_predictions_detail),
            (self.deploy_checkpoints_ok, self.deploy_checkpoints_detail),
        )
        return [detail for ok, detail in checks if not ok]


def _check_gate_policy(horizon: int, cfg: MultiAgentConfig) -> tuple[bool, str]:
    path = policy_path(cfg.gate_policy_dir, horizon, symbol="VN")
    try:
        load_gate_policy(
            path,
            expect_cmtf_version=cfg.cmtf_version,
            expect_backbone_version=cfg.backbone_version,
        )
        return True, f"gate policy ok ({path})"
    except (ArtifactMissingError, StalePolicyError) as e:
        return False, f"gate policy: {e}"


def _check_predictions(
    horizon: int, cfg: MultiAgentConfig, cell_id: str, label: str, pred_dir: str | Path,
) -> tuple[bool, str]:
    from .frozen_predictions import FrozenPredictionStore

    try:
        FrozenPredictionStore(horizon, cfg, pred_dir=pred_dir, cell_id=cell_id)
        return True, f"{label} predictions ok (cell {cell_id})"
    except ArtifactMissingError as e:
        return False, f"{label} predictions: {e}"


def _check_deploy_checkpoints(horizon: int, cfg: MultiAgentConfig, deploy_dir: str | Path) -> tuple[bool, str]:
    from .live_inference import deploy_checkpoint_paths

    paths = deploy_checkpoint_paths(horizon, deploy_dir=deploy_dir)
    have_seeds = {p.stem.rsplit("_seed", 1)[-1] for p in paths}
    want_seeds = {str(s) for s in cfg.ensemble_seeds}
    missing = want_seeds - have_seeds
    if missing:
        return False, (
            f"deploy checkpoints: missing seed(s) {sorted(missing)} for cmtf_lstm_{horizon}d "
            f"(found {sorted(have_seeds)}) — run `SAVE_DEPLOY_MODEL=1 python run_ablation_registry.py "
            f"--cells 0 --horizons {horizon} --seeds {' '.join(sorted(want_seeds))}`"
        )

    bad_hash = []
    for p in paths:
        meta_path = p.with_suffix(".meta.json")
        if not meta_path.exists():
            bad_hash.append(f"{p.name} (no .meta.json sidecar)")
            continue
        import json
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("horizon") != horizon:
            bad_hash.append(f"{p.name} (meta horizon={meta.get('horizon')} != {horizon})")
    if bad_hash:
        return False, f"deploy checkpoints: inconsistent metadata: {bad_hash}"

    return True, f"deploy checkpoints ok ({len(paths)} seed(s): {sorted(have_seeds)})"


def check_horizon_readiness(
    horizon: int,
    config: MultiAgentConfig | None = None,
    *,
    pred_dir: str | Path = "cache/predictions",
    deploy_dir: str | Path = "cache/deploy_models",
) -> ReadinessReport:
    """Check whether ``horizon`` has everything the graph needs to serve a real
    request: a calibrated gate policy, cached frozen predictions for both the
    single-name (core) and ranking (matched) branches, and live-inference deploy
    checkpoints for out-of-book dates. ``pred_dir``/``deploy_dir`` are overridable
    (tests point them at an isolated tmp_path fixture)."""
    cfg = config or DEFAULT_CONFIG

    gate_ok, gate_detail = _check_gate_policy(horizon, cfg)
    core_ok, core_detail = _check_predictions(horizon, cfg, core_cell_for(horizon), "core", pred_dir)
    matched_ok, matched_detail = _check_predictions(horizon, cfg, MATCHED_CELL_ID, "matched", pred_dir)
    deploy_ok, deploy_detail = _check_deploy_checkpoints(horizon, cfg, deploy_dir)

    return ReadinessReport(
        horizon=horizon,
        gate_policy_ok=gate_ok, gate_policy_detail=gate_detail,
        core_predictions_ok=core_ok, core_predictions_detail=core_detail,
        matched_predictions_ok=matched_ok, matched_predictions_detail=matched_detail,
        deploy_checkpoints_ok=deploy_ok, deploy_checkpoints_detail=deploy_detail,
    )


def check_all_horizons(config: MultiAgentConfig | None = None) -> dict[int, ReadinessReport]:
    return {h: check_horizon_readiness(h, config) for h in (1, 5, 20)}
