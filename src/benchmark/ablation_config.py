"""Configurations used by the fusion benchmark and component registry."""

from __future__ import annotations

from dataclasses import dataclass

BACKBONE_MODELS = (
    "lstm",
    "cnn_lstm",
    "gpt4ts",
    "chronos",
)

# Model label used for the cross-modal fusion configuration.
CMTF_MODEL = "cmtf"

MODELS = BACKBONE_MODELS + (CMTF_MODEL,)

FUSION_TYPES = ("none", "early", "late", "cmtf")
NEWS_SCOPES = ("none", "matched", "all")
SENTIMENT_MODES = ("none", "scalars", "weighted_emb")


@dataclass(frozen=True)
class AblationConfig:
    """One cell in the ablation grid."""

    # For baseline rows: model_name is the actual backbone ("lstm", "cnn_lstm", ...)
    # For CMTF rows: model_name should be "cmtf"
    model_name: str

    # One of: "none", "early", "late", "cmtf"
    fusion_type: str

    news_scope: str
    sentiment_mode: str

    # Explicit market backbone used inside CMTF
    # For non-CMTF rows this must be None
    market_encoder_name: str | None = None

    # CMTF output formulation. The predictor returns the selected formulation
    # directly; it does not apply a post-processing blend.
    output_mode: str = "market_plus_fusion"

    # B3 control: if True, permute train/val news rows to destroy the
    # market<->news alignment (shuffled-news placebo). Test set is left intact.
    shuffle_news: bool = False

    # Architecture ablation
    use_cross_attention: bool = True

    # CMTF component toggles
    use_positional_encoding: bool = True
    recency_gate_k: int = 5
    use_news_gate: bool = True
    use_two_stage: bool = True
    use_aux_loss: bool = True
    use_variance_reg: bool = True

    # CMTF fused-feature construction toggles
    use_interaction_prod: bool = True
    use_interaction_diff: bool = True
    use_news_context_prod: bool = True
    use_cosine_sim: bool = True
    use_pooled_news: bool = True

    # CMTF fusion formulation controls
    fusion_style: str = "handcrafted"   # "handcrafted" or "learned"
    market_query_mode: str = "multi"    # "multi", "last", "recent", "global"

    # CMTF tuning knobs
    fusion_market_dim: int = 128
    fusion_hidden_dim: int = 64
    projected_news_dim: int = 128
    n_heads: int = 2
    dropout: float = 0.2
    sign_penalty_weight: float = 0.005
    # Optional differentiable Sharpe surrogate for the fusion loss.
    sharpe_surrogate_weight: float = 0.0
    sharpe_surrogate_k: float = 3.0
    encoder_lr_scale: float = 0.1
    aux_loss_weight: float = 0.3
    stage1_ratio: float = 0.33
    market_epochs: int = 50
    fusion_epochs: int = 40
    market_patience: int = 8
    fusion_patience: int = 8

    # News gate softening
    news_gate_alpha: float = 1.0

    # Variance regularization strength
    variance_reg_coeff: float = 0.001

    # A fixed gate uses ``news_gate_alpha``; a learned gate predicts a
    # per-sample mixing coefficient from the market and news embeddings.
    gate_mode: str = "fixed"

    def is_valid(self) -> bool:
        """Check hard constraints for this config."""

        # Fusion requires news
        if self.fusion_type != "none" and self.news_scope == "none":
            return False

        # Sentiment requires news
        if self.sentiment_mode != "none" and self.news_scope == "none":
            return False

        # CMTF must use cmtf fusion only
        if self.model_name == CMTF_MODEL and self.fusion_type != "cmtf":
            return False

        # Non-CMTF baselines cannot use cmtf fusion
        if self.model_name in BACKBONE_MODELS and self.fusion_type == "cmtf":
            return False

        # market_encoder_name is meaningful only for CMTF
        if self.fusion_type == "cmtf":
            if self.market_encoder_name not in {"gpt4ts", "cnn_lstm", "lstm", "chronos"}:
                return False
        else:
            if self.market_encoder_name is not None:
                return False

        # output_mode is meaningful only for CMTF
        if self.fusion_type == "cmtf":
            if self.output_mode not in {
                "market_plus_fusion", "fusion_plus_news",
                "encoder_residual", "anchored_fusion",
            }:
                return False
        else:
            if self.output_mode != "market_plus_fusion":
                return False

        # shuffle_news is a control only meaningful when news is actually used
        if self.shuffle_news and (self.fusion_type == "none" or self.news_scope == "none"):
            return False

        # Fused-feature toggles are meaningful only for CMTF
        if self.fusion_type != "cmtf":
            if (
                not self.use_interaction_prod
                or not self.use_interaction_diff
                or not self.use_news_context_prod
                or not self.use_cosine_sim
                or not self.use_pooled_news
            ):
                return False

        # fusion_style and market_query_mode are meaningful only for CMTF
        if self.fusion_type == "cmtf":
            if self.fusion_style not in {"handcrafted", "learned"}:
                return False
            if self.market_query_mode not in {"multi", "last", "recent", "global"}:
                return False
        else:
            if self.fusion_style != "handcrafted":
                return False
            if self.market_query_mode != "multi":
                return False

        # CMTF-only toggles must remain default outside cmtf
        if self.fusion_type != "cmtf":
            if not self.use_cross_attention:
                return False
            if (
                not self.use_positional_encoding
                or not self.use_news_gate
                or self.recency_gate_k != 5
            ):
                return False
            if not self.use_two_stage or not self.use_aux_loss or not self.use_variance_reg:
                return False

        # gate_mode is meaningful only for CMTF (the learned gate head lives
        # inside HybridFusionPredictor's news-gate mixing step).
        if self.gate_mode not in {"fixed", "learned"}:
            return False
        if self.fusion_type != "cmtf" and self.gate_mode != "fixed":
            return False

        return True

    @property
    def cell_id(self) -> str:
        """Unique string identifier for caching."""
        parts = [self.model_name, self.fusion_type, self.news_scope, self.sentiment_mode]

        if self.market_encoder_name is not None:
            parts.append(f"enc={self.market_encoder_name}")

        if self.shuffle_news:
            parts.append("shuf=1")

        if self.fusion_type == "cmtf":
            parts.append(f"om={self.output_mode}")
            parts.append(f"fstyle={self.fusion_style}")
            parts.append(f"mq={self.market_query_mode}")
            parts.append(f"xattn={int(self.use_cross_attention)}")
            parts.append(f"pe={int(self.use_positional_encoding)}")
            parts.append(f"k={self.recency_gate_k}")
            parts.append(f"gate={int(self.use_news_gate)}")
            parts.append(f"galpha={self.news_gate_alpha:.2f}")
            if self.gate_mode != "fixed":
                parts.append(f"gm={self.gate_mode}")
            parts.append(f"ts={int(self.use_two_stage)}")
            parts.append(f"aux={int(self.use_aux_loss)}")
            parts.append(f"vreg={int(self.use_variance_reg)}")
            parts.append(f"iprod={int(self.use_interaction_prod)}")
            parts.append(f"idiff={int(self.use_interaction_diff)}")
            parts.append(f"nprod={int(self.use_news_context_prod)}")
            parts.append(f"csim={int(self.use_cosine_sim)}")
            parts.append(f"pnews={int(self.use_pooled_news)}")
            parts.append(f"vcoeff={self.variance_reg_coeff:.4f}")
            parts.append(f"fmd={self.fusion_market_dim}")
            parts.append(f"fhd={self.fusion_hidden_dim}")
            parts.append(f"pnd={self.projected_news_dim}")
            parts.append(f"nh={self.n_heads}")
            parts.append(f"do={self.dropout}")
            parts.append(f"spw={self.sign_penalty_weight}")
            # Include surrogate parameters only when the surrogate is enabled.
            if self.sharpe_surrogate_weight > 0.0:
                parts.append(f"shw={self.sharpe_surrogate_weight}")
                parts.append(f"shk={self.sharpe_surrogate_k}")
            parts.append(f"elr={self.encoder_lr_scale}")
            parts.append(f"auxw={self.aux_loss_weight}")
            parts.append(f"s1r={self.stage1_ratio}")
            parts.append(f"me={self.market_epochs}")
            parts.append(f"fe={self.fusion_epochs}")
            parts.append(f"mp={self.market_patience}")
            parts.append(f"fp={self.fusion_patience}")

        return "__".join(parts)


# Canonical CMTF settings. Registry cells inherit these values and override one
# setting at a time, keeping component comparisons controlled.
CMTF_CORE: dict = dict(
    fusion_type="cmtf",
    news_scope="all",
    sentiment_mode="scalars",
    output_mode="anchored_fusion",
    use_two_stage=False,
    fusion_style="learned",
    market_query_mode="multi",
    use_cross_attention=True,
    use_positional_encoding=False,
    use_news_gate=True,
    recency_gate_k=3,
    use_aux_loss=True,
    use_variance_reg=True,
    use_interaction_prod=False,
    use_interaction_diff=False,
    use_news_context_prod=False,
    use_cosine_sim=False,
    use_pooled_news=False,
    fusion_market_dim=64,
    fusion_hidden_dim=32,
    projected_news_dim=128,
    n_heads=4,
    dropout=0.1,
    sign_penalty_weight=0.01,
    encoder_lr_scale=0.1,
    aux_loss_weight=0.1,
    stage1_ratio=0.33,
    market_epochs=50,
    fusion_epochs=40,
    market_patience=8,
    fusion_patience=8,
    news_gate_alpha=1.0,
    variance_reg_coeff=0.001,
)


def _cmtf(encoder: str, **overrides) -> AblationConfig:
    """A CMTF cell wrapping ``encoder``, derived from CMTF_CORE with overrides."""
    params = {**CMTF_CORE, **overrides}
    return AblationConfig(model_name=CMTF_MODEL, market_encoder_name=encoder, **params)


def _baseline(model: str, fusion: str) -> AblationConfig:
    """Create a market-only, early-fusion, or late-fusion baseline.

    Fusion baselines use the same pooled-news scope and market features as CMTF
    so that the comparison isolates the fusion method.
    """
    news_scope = "matched" if fusion == "none" else "all"
    return AblationConfig(
        model_name=model,
        fusion_type=fusion,
        news_scope=news_scope,
        sentiment_mode="scalars",
    )


def generate_grid(table: str = "all") -> list[AblationConfig]:
    """Generate valid ablation configs for one study table (or all)."""
    cfgs: list[AblationConfig] = []

    # --------------------------------------------------------------
    # Table 1: Fusion comparison (main result + placebo)
    # --------------------------------------------------------------
    if table in ("fusion_comparison", "all"):
        # Trainable encoders support the full fusion ladder.
        for bb in ("lstm", "cnn_lstm"):
            cfgs.append(_baseline(bb, "none"))
            cfgs.append(_baseline(bb, "early"))
            cfgs.append(_baseline(bb, "late"))
            cfgs.append(_cmtf(bb))
        # Frozen foundation backbones: input-concat early fusion is ill-defined,
        # so compare none / late / CMTF only.
        for bb in ("gpt4ts", "chronos"):
            cfgs.append(_baseline(bb, "none"))
            cfgs.append(_baseline(bb, "late"))
            cfgs.append(_cmtf(bb))
        # Placebo: shuffled-news twin of the primary LSTM CMTF cell. If the news
        # lift is genuine, this collapses toward the market-only baseline.
        cfgs.append(_cmtf("lstm", shuffle_news=True))

    valid = [c for c in cfgs if c.is_valid()]
    return list(dict.fromkeys(valid))
