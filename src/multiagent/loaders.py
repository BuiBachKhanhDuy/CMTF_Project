"""Artifact error type shared across the multi-agent loaders.

The CMTF champion has no deployable checkpoint (the registry trains it on-the-fly
and caches only predictions), so the redesign runs the decision path over the
frozen-prediction store (``frozen_predictions.py``) rather than loading model
weights. The former checkpoint loaders were removed with that change (R2); only the
shared missing-artifact error remains, raised by the frozen store and the gate I/O
when a required file is absent (R1: fail loud, never fall back).
"""

from __future__ import annotations


class ArtifactMissingError(FileNotFoundError):
    """Raised when a required artifact (frozen prediction, gate policy) is absent."""
