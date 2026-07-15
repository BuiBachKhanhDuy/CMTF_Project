"""Live CMTF inference — real forward pass for ANY (symbol, cutoff), incl. today.

This is the product path that makes the system realtime rather than a frozen-cache
replay. It reuses the EXACT training data pipeline (``_extract_and_split`` →
``run_pipeline``) to build features for the requested cutoff, so per-symbol z-score
normalization is fit on train data and applied to the query date identically to
training — eliminating train/serve skew by construction (verified: the loaded
champion reproduces the cached predictions bit-for-bit, max_abs_diff = 0).

The deployed gate was calibrated on the 3-seed ensemble, so we load all cached
deploy seeds and average them, exactly matching ``gate_pred`` in the frozen cache.

Cost note: for a cutoff not already covered by the cached dataset parquet, the
pipeline refetches OHLCV + news and recomputes embeddings for the whole universe
(cross-symbol news is required by the all-scope champion) — this is the real
realtime latency. For cutoffs inside the cached range it is fast (parquet hit).
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
    gate_pred: float          # 3-seed ensemble mean (matches the calibrated gate)
    seed_preds: list[float]
    truth: float | None       # realised return if the date is old enough, else None
    n_symbols_in_universe: int


def _deploy_models(horizon: int):
    """Load every cached deploy-seed champion for this horizon (full nn.Module objects)."""
    paths = sorted(glob.glob(str(_DEPLOY_DIR / f"cmtf_lstm_{horizon}d_seed*.pt")))
    if not paths:
        raise ArtifactMissingError(
            f"No deployed champion at {_DEPLOY_DIR}/cmtf_lstm_{horizon}d_seed*.pt — "
            f"run `SAVE_DEPLOY_MODEL=1 python run_ablation_registry.py --cells 0 "
            f"--horizons {horizon} --seeds 1 42 123` to train + persist it."
        )
    return [(Path(p).stem, torch.load(p, map_location="cpu", weights_only=False)) for p in paths]


@lru_cache(maxsize=4)
def _pipeline_splits(horizon: int, end: str | None, allow_missing_target: bool = False):
    """Build the full (all-symbol) normalized splits for the requested end date.

    Cached per (horizon, end, allow_missing_target): reusing the training pipeline
    guarantees identical normalization. `end=None` uses the config default (the
    research range).

    ``allow_missing_target=True`` (used only by the Tier-2 live fetch in
    ``predict_live``) keeps rows whose forward-return target is NaN — i.e. the most
    recent ~horizon days, which can never have a target because the future hasn't
    happened. Tier 1 (the cached research range) never needs this: every date in
    that range is old enough to have a real target, and keeping the default False
    there preserves the exact bit-for-bit reproduction of the frozen research splits.
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


def resolve_price_parquet(horizon: int = 5) -> Path:
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
    """
    from run_ablation_benchmark import _build_pipeline_config
    from src.pipeline.orchestrator import _config_hash, _DATASET_CACHE_DIR

    config = _build_pipeline_config(horizon)
    cfg_hash = _config_hash(config)
    path = _DATASET_CACHE_DIR / f"dataset_{cfg_hash}.parquet"
    if not path.exists():
        raise ArtifactMissingError(
            f"No dataset parquet at {path} for the current pipeline config (horizon="
            f"{horizon}d). Run the pipeline once to build it, e.g.: "
            f"`python run_ablation_registry.py --cells 0 --horizons {horizon} --seeds 1`."
        )
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

    cfg = config or DEFAULT_CONFIG
    cell = get_cell("0")  # the deployed champion cell (CMTF_CORE, all-scope)

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
    for _name, model in _deploy_models(horizon):
        p = model.predict(mw_te[i:i + 1], ne_te[i:i + 1],
                           nm_te[i:i + 1] if nm_te is not None else None)
        seed_preds.append(float(np.asarray(p).ravel()[0]))
    gate_pred = float(np.mean(seed_preds))

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
    )
