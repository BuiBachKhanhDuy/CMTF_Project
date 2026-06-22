"""Ablation benchmark configuration and grid generation.

Each cell = one (model × fusion × news_scope × sentiment × toggle) combination.
Invalid cells are never generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

MODELS = ("lstm", "lstm_hybrid", "rf", "cnn_lstm", "cnn_lstm_hybrid")
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
    use_two_stage: bool = True
    use_aux_loss: bool = True
    use_variance_reg: bool = True

    def is_valid(self) -> bool:
        """Check hard constraints for this config."""
        hybrid_backbones = {"lstm_hybrid", "cnn_lstm_hybrid"}

        # Fusion requires news
        if self.fusion_type != "none" and self.news_scope == "none":
            return False

        # Sentiment requires news context
        if self.sentiment_mode != "none" and self.news_scope == "none":
            return False

        # RF does not support early/hybrid fusion
        if self.model_name == "rf" and self.fusion_type in ("early", "hybrid"):
            return False

        # Best-state hybrid backbones do not support current EarlyFusionWrapper
        if self.model_name in hybrid_backbones and self.fusion_type == "early":
            return False

        # Toggle flags only meaningful for hybrid fusion
        if self.fusion_type != "hybrid":
            if not self.use_positional_encoding or not self.use_news_gate or self.recency_gate_k != 5:
                return False
            if not self.use_two_stage or not self.use_aux_loss or not self.use_variance_reg:
                return False

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
        for model, fusion in product(MODELS, FUSION_TYPES):
            if fusion == "none":
                cfg = AblationConfig(
                    model_name=model,
                    fusion_type=fusion,
                    news_scope="none",
                    sentiment_mode="none",
                )
            else:
                cfg = AblationConfig(
                    model_name=model,
                    fusion_type=fusion,
                    news_scope="matched",
                    sentiment_mode="scalars",
                )
            if cfg.is_valid():
                configs.append(cfg)

    if table in ("news_scope", "all"):
        for model, scope in product(MODELS, NEWS_SCOPES):
            if scope == "none":
                cfg = AblationConfig(
                    model_name=model,
                    fusion_type="none",
                    news_scope="none",
                    sentiment_mode="none",
                )
            else:
                cfg = AblationConfig(
                    model_name=model,
                    fusion_type="hybrid",
                    news_scope=scope,
                    sentiment_mode="scalars",
                )
            if cfg.is_valid():
                configs.append(cfg)

    if table in ("sentiment", "all"):
        for model, sent in product(MODELS, SENTIMENT_MODES):
            cfg = AblationConfig(
                model_name=model,
                fusion_type="hybrid",
                news_scope="matched",
                sentiment_mode=sent,
            )
            if cfg.is_valid():
                configs.append(cfg)

    if table in ("component", "all"):
        hybrid_models = [m for m in MODELS if m != "rf"]
        toggle_combos = [
            (True, True, 5, True, True, True),
            (False, True, 5, True, True, True),
            (True, False, 5, True, True, True),
            (True, True, 3, True, True, True),
            (True, True, 10, True, True, True),
            (False, False, 5, True, True, True),
            (False, True, 5, False, True, True),
            (True, True, 5, False, True, True),
            (True, True, 5, True, False, True),
            (True, True, 5, True, True, False),
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

    return list(dict.fromkeys(configs))