"""Frozen-prediction backend (plan §1.6/§1.7, §3.4).

The CMTF champion has no deployable checkpoint — the ablation registry trains it
on-the-fly and caches only its per-seed *predictions* to ``cache/predictions``.
The redesign therefore runs the decision path over those frozen predictions ("no
retraining in the loop; the signal is unlocked post-hoc by gating frozen
predictions"), which also makes runtime == research byte-for-byte.

This module builds a ``(symbol, date) → row`` index over the frozen predictions of
the pre-registered CORE cell and serves a single name's prediction to
``predict_agent``. It is deterministic and LLM-free.

Honesty (R1): a lookup for a ``(symbol, date)`` not present in the cached book
raises :class:`PredictionNotCachedError` — the backend never invents a prediction
for an out-of-book date. The market-only baseline is not cached per row, so
``news_residual`` is reported as ``None`` (not fabricated); it is wired from the
matched-scope cache in Phase 4.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DEFAULT_CONFIG, MultiAgentConfig
from .gate_io import CORE_CELL_ID, core_cell_for
from .loaders import ArtifactMissingError

PRED_DIR = Path("cache/predictions")


class PredictionNotCachedError(KeyError):
    """Raised when no frozen prediction exists for a requested (symbol, date)."""


@dataclass(frozen=True)
class FrozenPrediction:
    symbol: str
    date: np.datetime64
    horizon: int
    seed_preds: list[float]  # per-seed CMTF final predictions (ensemble members)
    ensemble_pred: float  # mean over seeds (metadata / seed-mean)
    gate_pred: float  # the raw magnitude the gate consumes (per gate_on_raw_seed)
    truth: float  # realised forward return for this row


class FrozenPredictionStore:
    """Serves frozen CMTF predictions keyed by (symbol, date) for one horizon."""

    def __init__(self, horizon: int, config: MultiAgentConfig | None = None,
                 pred_dir: Path | str = PRED_DIR, cell_id: str | None = None):
        self.cfg = config or DEFAULT_CONFIG
        self.horizon = int(horizon)
        self.pred_dir = Path(pred_dir)
        self.cell_id = cell_id if cell_id is not None else core_cell_for(self.horizon)
        self._config_hash = self._resolve_hash(self.cell_id)
        self._symbols, self._days, self._seed_stack, self._truth = self._load()
        # (symbol, date) → row index
        self._index: dict[tuple[str, np.datetime64], int] = {}
        for i, (s, d) in enumerate(zip(self._symbols, self._days)):
            self._index[(str(s), d)] = i

    @staticmethod
    def _resolve_hash(cell_id: str) -> str:
        from src.benchmark.ablation_registry import get_cell
        from src.benchmark.ablation_runner import _config_hash
        return _config_hash(get_cell(cell_id))

    def _load(self):
        h = self.horizon
        sym_f = self.pred_dir / f"test_symbols__{h}d.npy"
        tim_f = self.pred_dir / f"test_times__{h}d.npy"
        tru_f = self.pred_dir / f"truth__{h}d.npy"
        for f in (sym_f, tim_f, tru_f):
            if not f.exists():
                raise ArtifactMissingError(
                    f"Missing frozen-prediction artifact {f} — re-run the registry "
                    f"(cell {self.cell_id}) so the (symbol, date) index is cached."
                )
        symbols = np.load(str(sym_f), allow_pickle=True)
        days = np.asarray(np.load(str(tim_f), allow_pickle=True)).astype("datetime64[D]")
        truth = np.load(str(tru_f)).astype(np.float64)

        # Per-seed test predictions, ordered by the configured ensemble seeds so the
        # mean matches the registry's ensemble and calibration exactly.
        seed_arrays = []
        for seed in self.cfg.ensemble_seeds:
            f = self.pred_dir / f"{self._config_hash}__seed{seed}__{h}d.npy"
            if not f.exists():
                raise ArtifactMissingError(
                    f"Missing frozen seed prediction {f} — re-run the registry "
                    f"(cell {self.cell_id}, seed {seed})."
                )
            seed_arrays.append(np.load(str(f)).astype(np.float64))
        seed_stack = np.stack(seed_arrays, axis=0)  # (n_seeds, n_rows)

        n = len(symbols)
        if not (len(days) == len(truth) == seed_stack.shape[1] == n):
            raise ValueError(
                f"Frozen-prediction length mismatch for {h}d: symbols={n} days={len(days)} "
                f"truth={len(truth)} preds={seed_stack.shape[1]}"
            )
        return symbols, days, seed_stack, truth

    @property
    def symbols(self) -> list[str]:
        return sorted({str(s) for s in self._symbols})

    def get(self, symbol: str, date: str | np.datetime64) -> FrozenPrediction:
        """Return the frozen prediction for (symbol, date), or raise if not cached."""
        d = np.datetime64(date, "D")
        key = (str(symbol), d)
        if key not in self._index:
            raise PredictionNotCachedError(
                f"No frozen prediction for symbol={symbol} date={d} at {self.horizon}d. "
                f"The frozen backend only serves dates in the cached test book "
                f"(no live checkpoint exists). Available dates for {symbol}: "
                f"{self._date_range(symbol)}."
            )
        i = self._index[key]
        seeds = [float(x) for x in self._seed_stack[:, i]]
        ensemble = float(np.mean(seeds))
        gate_pred = seeds[0] if self.cfg.gate_on_raw_seed else ensemble
        return FrozenPrediction(
            symbol=str(symbol), date=d, horizon=self.horizon,
            seed_preds=seeds, ensemble_pred=ensemble, gate_pred=gate_pred,
            truth=float(self._truth[i]),
        )

    def _date_range(self, symbol: str) -> str:
        ds = sorted(d for (s, d) in self._index if s == str(symbol))
        if not ds:
            return "(none)"
        return f"{ds[0]} .. {ds[-1]} ({len(ds)} dates)"


# Matched-scope cell (news_scope='matched') — the only config with genuine
# cross-sectional signal (plan §0), used by the ranking branch.
MATCHED_CELL_ID = "8"

_STORE_CACHE: dict[tuple[int, str], FrozenPredictionStore] = {}


def get_store(horizon: int, config: MultiAgentConfig | None = None,
              cell_id: str | None = None) -> FrozenPredictionStore:
    """Cached accessor for the per-(horizon, cell) frozen-prediction store.

    ``cell_id=None`` (default) resolves to this horizon's validated champion cell
    (``core_cell_for`` — cell 13 for 5D/20D, cell 0 for 1D; see `gate_io.py`).
    """
    resolved_cell_id = cell_id if cell_id is not None else core_cell_for(horizon)
    key = (horizon, resolved_cell_id)
    if key not in _STORE_CACHE:
        _STORE_CACHE[key] = FrozenPredictionStore(horizon, config, cell_id=resolved_cell_id)
    return _STORE_CACHE[key]
