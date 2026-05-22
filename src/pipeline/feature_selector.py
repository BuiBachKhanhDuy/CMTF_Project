"""Stability-based feature selection utilities for the CMTF pipeline.

Implements a lightweight, temporal stability selector inspired by
correlation filtering + sparse linear selection across chronological folds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True, slots=True)
class StabilitySelectionReport:
    """Serializable report for selected features and fold statistics."""

    selected_features: list[str]
    preselected_features: list[str]
    dropped_by_correlation: list[str]
    selection_frequency: dict[str, float]
    folds_used: int

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_features": self.selected_features,
            "preselected_features": self.preselected_features,
            "dropped_by_correlation": self.dropped_by_correlation,
            "selection_frequency": self.selection_frequency,
            "folds_used": self.folds_used,
        }


class StabilityFeatureSelector:
    """Selects features that remain predictive across temporal folds.

    Pipeline:
        1) Correlation-guided preselection (drop highly collinear features)
        2) Expanding-window temporal folds on training range
        3) Sparse linear fit (Lasso) per fold
        4) Keep features whose non-zero frequency exceeds threshold
    """

    def __init__(
        self,
        corr_threshold: float = 0.95,
        lasso_alpha: float = 0.001,
        n_folds: int = 5,
        stability_threshold: float = 0.6,
        min_train_rows: int = 120,
    ) -> None:
        self.corr_threshold = float(corr_threshold)
        self.lasso_alpha = float(lasso_alpha)
        self.n_folds = max(2, int(n_folds))
        self.stability_threshold = float(stability_threshold)
        self.min_train_rows = int(min_train_rows)

    def select(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        train_end: str,
    ) -> StabilitySelectionReport:
        """Run stability-based feature selection on training-period rows only."""
        if not feature_cols:
            raise ValueError("feature_cols is empty")
        if target_col not in df.columns:
            raise ValueError(f"Missing target column: {target_col}")

        work = df.copy()
        work = work.sort_index()
        train_mask = work.index < pd.Timestamp(train_end)
        train = work.loc[train_mask, feature_cols + [target_col]].dropna()

        if len(train) < self.min_train_rows:
            logger.warning(
                "Stability selector skipped: only {} train rows (min={})",
                len(train),
                self.min_train_rows,
            )
            return StabilitySelectionReport(
                selected_features=list(feature_cols),
                preselected_features=list(feature_cols),
                dropped_by_correlation=[],
                selection_frequency={col: 1.0 for col in feature_cols},
                folds_used=0,
            )

        preselected, dropped = self._correlation_preselect(train[feature_cols])
        if not preselected:
            preselected = list(feature_cols)

        x = train[preselected].to_numpy(dtype=np.float64, copy=True)
        y = train[target_col].to_numpy(dtype=np.float64, copy=True)

        fold_ends = self._fold_train_ends(len(train))
        selected_counts = {col: 0 for col in preselected}
        folds_used = 0

        for end_idx in fold_ends:
            x_fold = x[:end_idx]
            y_fold = y[:end_idx]
            if len(x_fold) < self.min_train_rows:
                continue

            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x_fold)

            model = Lasso(alpha=self.lasso_alpha, fit_intercept=True, max_iter=5000)
            model.fit(x_scaled, y_fold)
            coefs = np.asarray(model.coef_, dtype=np.float64)
            non_zero = np.abs(coefs) > 1e-8

            for col, is_selected in zip(preselected, non_zero, strict=True):
                if is_selected:
                    selected_counts[col] += 1
            folds_used += 1

        if folds_used == 0:
            logger.warning("Stability selector had no valid folds; using preselected features")
            return StabilitySelectionReport(
                selected_features=preselected,
                preselected_features=preselected,
                dropped_by_correlation=dropped,
                selection_frequency={col: 1.0 for col in preselected},
                folds_used=0,
            )

        freq = {col: selected_counts[col] / folds_used for col in preselected}
        selected = [col for col in preselected if freq[col] >= self.stability_threshold]

        # Fallback: keep top-K stable features if threshold is too strict.
        if not selected:
            ranked = sorted(preselected, key=lambda col: freq[col], reverse=True)
            k = max(1, min(8, len(ranked)))
            selected = ranked[:k]
            logger.warning(
                "Stability threshold selected no features; using top-{} by frequency",
                k,
            )

        return StabilitySelectionReport(
            selected_features=selected,
            preselected_features=preselected,
            dropped_by_correlation=dropped,
            selection_frequency=freq,
            folds_used=folds_used,
        )

    def save_report(self, report: StabilitySelectionReport, output_path: str | Path) -> Path:
        """Persist selection report as JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        return path

    def _correlation_preselect(self, df_features: pd.DataFrame) -> tuple[list[str], list[str]]:
        corr = df_features.corr().abs()
        keep: list[str] = []
        dropped: list[str] = []

        for col in corr.columns:
            if not keep:
                keep.append(col)
                continue
            too_close = any(float(corr.loc[col, k]) >= self.corr_threshold for k in keep)
            if too_close:
                dropped.append(col)
            else:
                keep.append(col)
        return keep, dropped

    def _fold_train_ends(self, n_rows: int) -> list[int]:
        start = max(self.min_train_rows, int(0.5 * n_rows))
        if start >= n_rows:
            return [n_rows]
        return list(np.linspace(start, n_rows, num=self.n_folds, dtype=int))
