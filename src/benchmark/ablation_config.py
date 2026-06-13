"""Ablation benchmark configuration and grid generation.

Each cell = one (model × fusion × news_scope × sentiment × toggle) combination.
Invalid cells are never generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Literal


MODELS = ("lstm", "rf", "cnn_lstm")
FUSION_TYPES = ("none", "early", "late", "hybrid")
NEWS_SCOPES = ("none", "matched", "all")
SENTIMENT_MODES = ("none", "scalars", "weighted_emb")


@dataclass(frozen=True)
class AblationConfig:
    """One cell in the ablation grid."""

    model_name: str
    fusion_type: str
    news_scope: str
    sentiment_mode: str
    use_positional_encoding: bool = True
    recency_gate_k: int = 5
    use_news_gate: bool = True
    # CMTF flags (only meaningful for hybrid)
    use_two_stage: bool = True
    use_aux_loss: bool = True
    use_variance_reg: bool = True

    def is_valid(self) -> bool:
        """Check hard constraints for this config."""
        # Fusion requires news
        if self.fusion_type != "none" and self.news_scope == "none":
            return False
        # No-fusion + sentiment still needs news context (sentiment comes from news)
        if self.sentiment_mode != "none" and self.news_scope == "none":
            return False
        # Hybrid requires latent space (not RF)
        if self.fusion_type == "hybrid" and self.model_name == "rf":
            return False
        # Early fusion requires input_dim expansion (not RF)
        if self.fusion_type == "early" and self.model_name == "rf":
            return False
        # Toggle flags only meaningful for hybrid
        if self.fusion_type != "hybrid":
            if not self.use_positional_encoding or not self.use_news_gate or self.recency_gate_k != 5:
                return False
            if not self.use_two_stage or not self.use_aux_loss or not self.use_variance_reg:
                return False
        # No-fusion + no-news + no-sentiment = market-only baseline (valid)
        return True

    @property
    def cell_id(self) -> str:
        """Unique string identifier for caching."""
        parts = [self.model_name, self.fusion_type, self.news_scope, self.sentiment_mode]
        if self.fusion_type == "hybrid":
            parts.append(f"pe={int(self.use_positional_encoding)}")
            parts.append(f"k={self.recency_gate_k}")
            parts.append(f"gate={int(self.use_news_gate)}")
            parts.append(f"ts={int(self.use_two_stage)}")
            parts.append(f"aux={int(self.use_aux_loss)}")
            parts.append(f"vreg={int(self.use_variance_reg)}")
        return "__".join(parts)


def generate_grid(table: str = "all") -> list[AblationConfig]:
    """Generate valid ablation configs grouped by table.

    Tables:
        fusion      — Table 1: Model × FusionType (fixed: news=matched, sentiment=scalars)
        news_scope  — Table 2: Model × NewsScope (fixed: fusion=late, sentiment=scalars)
        sentiment   — Table 3: Model × SentimentMode (fixed: fusion=late, news=matched)
        component   — Table 4: Hybrid-only toggle ablation (pos_enc, gate, recency_k)
        all         — Union of all tables
    """
    configs: list[AblationConfig] = []

    if table in ("fusion", "all"):
        # Table 1: Model × Fusion, fixed news=matched, sentiment=scalars
        for model, fusion in product(MODELS, FUSION_TYPES):
            if fusion == "none":
                cfg = AblationConfig(model_name=model, fusion_type=fusion, news_scope="none", sentiment_mode="none")
            else:
                cfg = AblationConfig(model_name=model, fusion_type=fusion, news_scope="matched", sentiment_mode="scalars")
            if cfg.is_valid():
                configs.append(cfg)

    if table in ("news_scope", "all"):
        # Table 2: Model × NewsScope, fixed fusion=late (universal), sentiment=scalars
        for model, scope in product(MODELS, NEWS_SCOPES):
            if scope == "none":
                cfg = AblationConfig(model_name=model, fusion_type="none", news_scope="none", sentiment_mode="none")
            else:
                cfg = AblationConfig(model_name=model, fusion_type="late", news_scope=scope, sentiment_mode="scalars")
            if cfg.is_valid():
                configs.append(cfg)

    if table in ("sentiment", "all"):
        # Table 3: Model × Sentiment, fixed fusion=late, news=matched
        for model, sent in product(MODELS, SENTIMENT_MODES):
            if sent == "none":
                cfg = AblationConfig(model_name=model, fusion_type="late", news_scope="matched", sentiment_mode=sent)
            else:
                cfg = AblationConfig(model_name=model, fusion_type="late", news_scope="matched", sentiment_mode=sent)
            if cfg.is_valid():
                configs.append(cfg)

    if table in ("component", "all"):
        # Table 4: Component ablation (hybrid only, models with d_model > 0)
        hybrid_models = [m for m in MODELS if m != "rf"]
        toggle_combos = [
            # (pe, gate, k, two_stage, aux_loss, var_reg) — description
            (True,  True,  5, True,  True,  True),   # full CMTF (baseline)
            (False, True,  5, True,  True,  True),   # no positional encoding
            (True,  False, 5, True,  True,  True),   # no news gate
            (True,  True,  3, True,  True,  True),   # smaller recency window
            (True,  True, 10, True,  True,  True),   # larger recency window
            (False, False, 5, True,  True,  True),   # no PE + no gate
            (False, True,  5, False, True,  True),   # pruned CMTF: PE=N + single-stage
            (True,  True,  5, False, True,  True),   # single-stage only
            (True,  True,  5, True,  False, True),   # no aux loss
            (True,  True,  5, True,  True,  False),  # no variance reg
        ]
        for model, (pe, gate, k, ts, aux, vreg) in product(hybrid_models, toggle_combos):
            cfg = AblationConfig(
                model_name=model,
                fusion_type="hybrid",
                news_scope="matched",
                sentiment_mode="scalars",
                use_positional_encoding=pe,
                use_news_gate=gate,
                recency_gate_k=k,
                use_two_stage=ts,
                use_aux_loss=aux,
                use_variance_reg=vreg,
            )
            if cfg.is_valid():
                configs.append(cfg)

    # Deduplicate (frozen dataclass is hashable)
    return list(dict.fromkeys(configs))
