"""Live CMTF inference for a symbol, cutoff date, and forecast horizon.

The serving path reuses the training feature pipeline and averages the deployed
ensemble. Dates outside the cached dataset require a fresh market and news build.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_CONFIG, MultiAgentConfig
from .loaders import ArtifactMissingError

_DEPLOY_DIR = Path("cache/deploy_models")


@dataclass(frozen=True)
class LivePrediction:
    symbol: str
    date: str
    horizon: int
    gate_pred: float          # Ensemble mean used by the calibrated gate.
    seed_preds: list[float]
    truth: float | None       # Realized return when it is available.
    n_symbols_in_universe: int
    attn_weights: np.ndarray | None = None    # (S,) mean trailing-day attention.
    recency_gate: np.ndarray | None = None    # (S,) mean trailing-day recency gate.


def deploy_checkpoint_paths(horizon: int, deploy_dir: str | Path = _DEPLOY_DIR) -> list[Path]:
    """Return deployment checkpoints for a horizon.

    ``deploy_dir`` is configurable for readiness checks and tests.
    """
    return sorted(Path(p) for p in glob.glob(str(Path(deploy_dir) / f"cmtf_lstm_{horizon}d_seed*.pt")))


def _deploy_models(horizon: int):
    """Load every cached deploy-seed champion for this horizon (full nn.Module objects)."""
    paths = deploy_checkpoint_paths(horizon)
    if not paths:
        raise ArtifactMissingError(
            f"No deployed champion at {_DEPLOY_DIR}/cmtf_lstm_{horizon}d_seed*.pt — "
            f"run `SAVE_DEPLOY_MODEL=1 python run_ablation_registry.py --cells 0 "
            f"--horizons {horizon} --seeds 1 42 123` to train + persist it."
        )
    return [(p.stem, torch.load(str(p), map_location="cpu", weights_only=False)) for p in paths]


@lru_cache(maxsize=4)
def _pipeline_splits(horizon: int, end: str | None, allow_missing_target: bool = False):
    """Build normalized all-symbol splits for a requested end date.

    ``allow_missing_target`` retains recent rows that lack a future return, which
    is required when serving the latest available date.
    """
    from run_ablation_benchmark import _build_pipeline_config, _extract_and_split
    cfg = _build_pipeline_config(horizon)
    if end is not None:
        cfg = {**cfg, "end": end}
    splits, market_cols = _extract_and_split(cfg, allow_missing_target=allow_missing_target)
    return splits, market_cols


def _prep_test_arrays(cell, splits, market_cols):
    """Reproduce the champion's test-feature prep (news scope + sentiment mode)."""
    from src.benchmark.ablation_runner import _get_news_arrays, _apply_sentiment_mode
    ne_tr, ne_v, ne_te, nm_tr, nm_v, nm_te = _get_news_arrays(cell, splits)
    _mw_tr, _mw_v, mw_te, _ne_tr, _ne_v, ne_te, _cols = _apply_sentiment_mode(
        cell, splits["train"]["market_windows"].copy(), splits["val"]["market_windows"].copy(),
        splits["test"]["market_windows"].copy(), ne_tr, ne_v, ne_te, market_cols)
    return mw_te, ne_te, nm_te


def _locate_row(splits, symbol: str, cutoff: str):
    """Return the test-split row index for (symbol, cutoff), or None if absent."""
    syms = np.asarray(splits["test"]["symbols"])
    days = np.asarray(splits["test"]["times"]).astype("datetime64[D]")
    idx = np.where((syms == symbol) & (days == np.datetime64(cutoff, "D")))[0]
    return int(idx[0]) if len(idx) else None


def resolve_price_parquet(horizon: int = 5, allow_missing_target: bool = False) -> Path:
    """Return the current dataset parquet path for ``horizon``, computed the same
    way ``run_pipeline`` computes its own cache key — never a hardcoded hash.

    Several tools (``chat.py``, ``h3_faithfulness.py``, ``tools/e2e_demo.py``) need
    direct read access to the raw per-date price/technicals table (for trailing
    vol/drawdown, not for the model itself). They used to hardcode one specific
    ``dataset_<hash>.parquet`` filename — but that hash is a content hash of the
    FULL pipeline config, so it silently goes stale on a fresh clone (no cache at
    all) or after any config change (16 different hash-named files already exist
    in ``cache/dataset/`` from routine local runs). Recomputing the hash from the
    same config `run_pipeline` uses guarantees this always points at the one file
    that's actually current, and fails loudly (not silently on a wrong/stale file)
    if it doesn't exist yet.

    ``allow_missing_target=True`` resolves the ``_livewide`` variant instead — the
    standard (default) parquet drops an ENTIRE row when this horizon's own target
    is NaN (correct for training), which also throws away that row's perfectly
    valid ``fwd_ret_1d`` for the last ~horizon days before the pipeline's raw end.
    A real market-data range query (e.g. "analyze March 2026") only needs
    ``fwd_ret_1d``, so that coupling makes recent-but-real days look unavailable.
    Built on demand if missing (unlike the standard variant, which must already
    exist from a real research run) since it reuses already-warm news/embedding
    caches and is comparatively cheap.
    """
    from run_ablation_benchmark import _build_pipeline_config
    from src.pipeline.orchestrator import _config_hash, _DATASET_CACHE_DIR, run_pipeline

    config = _build_pipeline_config(horizon)
    cfg_hash = _config_hash(config)
    suffix = "_livewide" if allow_missing_target else ""
    path = _DATASET_CACHE_DIR / f"dataset_{cfg_hash}{suffix}.parquet"
    if not path.exists():
        if not allow_missing_target:
            raise ArtifactMissingError(
                f"No dataset parquet at {path} for the current pipeline config (horizon="
                f"{horizon}d). Run the pipeline once to build it, e.g.: "
                f"`python run_ablation_registry.py --cells 0 --horizons {horizon} --seeds 1`."
            )
        run_pipeline(config, allow_missing_target=True)
    return path


def predict_live(symbol: str, cutoff: str, horizon: int = 5,
                 config: MultiAgentConfig | None = None, data_end: str | None = None) -> LivePrediction:
    """Real forward-pass prediction for (symbol, cutoff) via the training pipeline.

    Two-tier lookup:
    1. Build splits over the research range (``end=None`` → end=2026-03-31). This HITS
       the cached dataset parquet → fast, and reproduces the frozen cache bit-for-bit.
    2. If the (symbol, cutoff) row is not in that range (a genuinely new/future date),
       rebuild with ``end = data_end or cutoff`` — a fresh OHLCV + news fetch for the
       whole universe (the real realtime latency; the all-scope champion needs
       cross-symbol news). ``data_end`` lets the caller extend the fetch explicitly.
    """
    from src.benchmark.ablation_registry import get_cell
    from .gate_io import core_cell_for

    cfg = config or DEFAULT_CONFIG
    cell = get_cell(core_cell_for(horizon))  # this horizon's validated champion cell

    # Tier 1: cached research range (fast, bit-exact).
    splits, market_cols = _pipeline_splits(horizon, None)
    i = _locate_row(splits, symbol, cutoff)

    # Tier 2: fresh fetch extended to the query date (realtime path).
    # allow_missing_target=True: `cutoff` may be within `horizon` trading days of
    # `end` (or genuinely today), so its forward-return target is legitimately NaN
    # (the future hasn't happened) — this must not exclude the row, only the label.
    if i is None:
        end = data_end or cutoff
        splits, market_cols = _pipeline_splits(horizon, end, allow_missing_target=True)
        i = _locate_row(splits, symbol, cutoff)

    if i is None:
        days = np.asarray(splits["test"]["times"]).astype("datetime64[D]")
        syms = np.asarray(splits["test"]["symbols"])
        avail = sorted(str(d) for d in np.unique(days[syms == symbol]))
        raise ArtifactMissingError(
            f"No pipeline row for symbol={symbol} date={cutoff} at {horizon}d. "
            f"Available for {symbol}: {avail[0] if avail else '(none)'} .. "
            f"{avail[-1] if avail else '(none)'} ({len(avail)} dates). "
            f"cutoff must be >= the OHLCV fetch start and <= data_end (or today)."
        )

    mw_te, ne_te, nm_te = _prep_test_arrays(cell, splits, market_cols)

    seed_preds = []
    attn_by_seed, recency_by_seed = [], []
    for _name, model in _deploy_models(horizon):
        p, attn, recency = model.predict_with_attention(
            mw_te[i:i + 1], ne_te[i:i + 1], nm_te[i:i + 1] if nm_te is not None else None,
        )
        seed_preds.append(float(np.asarray(p).ravel()[0]))
        if attn is not None:
            attn_by_seed.append(attn[0])
        if recency is not None:
            recency_by_seed.append(recency[0])
    gate_pred = float(np.mean(seed_preds))
    attn_weights = np.mean(attn_by_seed, axis=0) if attn_by_seed else None
    recency_gate = np.mean(recency_by_seed, axis=0) if recency_by_seed else None

    # NaN here means the row was kept via allow_missing_target (a genuinely live/
    # current date has no realised return yet) — report as None, never as NaN.
    truth_arr = splits["test"].get("targets")
    truth = float(np.asarray(truth_arr)[i]) if truth_arr is not None else None
    if truth is not None and np.isnan(truth):
        truth = None

    return LivePrediction(
        symbol=symbol, date=str(cutoff), horizon=horizon, gate_pred=gate_pred,
        seed_preds=seed_preds, truth=truth,
        n_symbols_in_universe=int(len(np.unique(np.asarray(splits["test"]["symbols"])))),
        attn_weights=attn_weights, recency_gate=recency_gate,
    )
