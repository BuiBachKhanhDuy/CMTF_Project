"""Ablation benchmark configuration and grid generation.

Each cell = one (model × fusion × news_scope × sentiment × toggle) combination.
Invalid cells are never generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

BACKBONE_MODELS = ("lstm", "rf", "cnn_lstm")

# Architecture label for the renamed hybrid model
CMTF_MODEL = "cmtf"

# Optional legacy alias if you still have older cached runs / labels elsewhere
LEGACY_HYBRID_MODEL = "hybrid_fusion"

MODELS = BACKBONE_MODELS + (CMTF_MODEL,)

FUSION_TYPES = ("none", "early", "late", "cmtf")
NEWS_SCOPES = ("none", "matched", "all")
SENTIMENT_MODES = ("none", "scalars", "weighted_emb")


@dataclass(frozen=True)
class AblationConfig:
    """One cell in the ablation grid."""

    # For baseline rows: model_name is the actual backbone ("lstm", "rf", "cnn_lstm")
    # For CMTF rows: model_name should be "cmtf"
    model_name: str

    # One of: "none", "early", "late", "cmtf"
    fusion_type: str

    news_scope: str
    sentiment_mode: str

    # Explicit market backbone used inside CMTF
    # For non-CMTF rows this must be None
    market_encoder_name: str | None = None

    # Architecture ablation
    use_cross_attention: bool = True

    # CMTF component toggles
    use_positional_encoding: bool = True
    recency_gate_k: int = 5
    use_news_gate: bool = True
    use_two_stage: bool = True
    use_aux_loss: bool = True
    use_variance_reg: bool = True

    # CMTF tuning knobs
    fusion_market_dim: int = 128
    fusion_hidden_dim: int = 64
    projected_news_dim: int = 128
    n_heads: int = 2
    dropout: float = 0.2
    sign_penalty_weight: float = 0.005
    encoder_lr_scale: float = 0.1
    aux_loss_weight: float = 0.3
    stage1_ratio: float = 0.33
    market_epochs: int = 100
    fusion_epochs: int = 60
    market_patience: int = 15
    fusion_patience: int = 12

    # --- New tuning parameters ---
    # Blending factor for the news gate:
    #   gate_output = attn_out * (news_gate_alpha * gate + (1 - news_gate_alpha))
    #   1.0 = full hard gate (original behaviour)
    #   0.0 = gate completely disabled (identity)
    news_gate_alpha: float = 1.0

    # Coefficient for variance regularization on attended/pooled news.
    # Replaces the previous hardcoded _vr_coeff = 0.001.
    variance_reg_coeff: float = 0.001

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

        # RF does not support early fusion
        if self.model_name == "rf" and self.fusion_type == "early":
            return False

        # market_encoder_name is meaningful only for CMTF
        if self.fusion_type == "cmtf":
            if self.market_encoder_name not in {"lstm", "cnn_lstm"}:
                return False
        else:
            if self.market_encoder_name is not None:
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

        return True

    @property
    def cell_id(self) -> str:
        """Unique string identifier for caching."""
        parts = [self.model_name, self.fusion_type, self.news_scope, self.sentiment_mode]

        if self.market_encoder_name is not None:
            parts.append(f"enc={self.market_encoder_name}")

        if self.fusion_type == "cmtf":
            parts.append(f"xattn={int(self.use_cross_attention)}")
            parts.append(f"pe={int(self.use_positional_encoding)}")
            parts.append(f"k={self.recency_gate_k}")
            parts.append(f"gate={int(self.use_news_gate)}")
            parts.append(f"galpha={self.news_gate_alpha:.2f}")
            parts.append(f"ts={int(self.use_two_stage)}")
            parts.append(f"aux={int(self.use_aux_loss)}")
            parts.append(f"vreg={int(self.use_variance_reg)}")
            parts.append(f"vcoeff={self.variance_reg_coeff:.4f}")
            parts.append(f"fmd={self.fusion_market_dim}")
            parts.append(f"fhd={self.fusion_hidden_dim}")
            parts.append(f"pnd={self.projected_news_dim}")
            parts.append(f"nh={self.n_heads}")
            parts.append(f"do={self.dropout}")
            parts.append(f"spw={self.sign_penalty_weight}")
            parts.append(f"elr={self.encoder_lr_scale}")
            parts.append(f"auxw={self.aux_loss_weight}")
            parts.append(f"s1r={self.stage1_ratio}")
            parts.append(f"me={self.market_epochs}")
            parts.append(f"fe={self.fusion_epochs}")
            parts.append(f"mp={self.market_patience}")
            parts.append(f"fp={self.fusion_patience}")

        return "__".join(parts)
def generate_cmtf_search_grid() -> list[AblationConfig]:
    """Focused CMTF directional-sharpness search.

    Fixed to current best validated CMTF:
        enc=lstm
        xattn=True
        pe=True
        gate=True
        vreg=True
        ts=True
        aux=True
        k=3
        fmd=128
        fhd=64
        projected_news_dim=128
        n_heads=4
        dropout=0.1
        encoder_lr_scale=0.2
        stage1_ratio=0.33
        market_epochs=80
        fusion_epochs=80
        market_patience=12
        fusion_patience=10
        variance_reg_coeff=0.001

    Search axes:
        news_gate_alpha    : soften news gate suppression      (0.2, 0.3, 0.4)
        aux_loss_weight    : aux supervision strength          (0.0, 0.05, 0.1)
        sign_penalty_weight: directional sharpness pressure    (0.005, 0.01, 0.02)

    Total = 3 × 3 × 3 = 27 configs
    """
    configs: list[AblationConfig] = []

    for galpha in (0.2, 0.3, 0.4):
        for auxw in (0.0, 0.05, 0.1):
            for spw in (0.005, 0.01, 0.02):
                cfg = AblationConfig(
                    model_name=CMTF_MODEL,
                    fusion_type="cmtf",
                    news_scope="matched",
                    sentiment_mode="scalars",
                    market_encoder_name="lstm",
                    use_cross_attention=True,
                    use_positional_encoding=True,
                    use_news_gate=True,
                    use_variance_reg=True,
                    use_two_stage=True,
                    use_aux_loss=True,
                    recency_gate_k=3,
                    fusion_market_dim=128,
                    fusion_hidden_dim=64,
                    projected_news_dim=128,
                    n_heads=4,
                    dropout=0.1,
                    sign_penalty_weight=spw,
                    encoder_lr_scale=0.2,
                    aux_loss_weight=auxw,
                    stage1_ratio=0.33,
                    market_epochs=80,
                    fusion_epochs=80,
                    market_patience=12,
                    fusion_patience=10,
                    news_gate_alpha=galpha,
                    variance_reg_coeff=0.001,
                )
                if cfg.is_valid():
                    configs.append(cfg)

    return configs

def generate_grid(table: str = "all") -> list[AblationConfig]:
    """Generate valid ablation configs grouped by final study axis.

    Tables:
        data_ablation
            Representative comparison:
            market-only vs early vs late vs CMTF
        architecture_ablation
            CMTF with cross-attention ON vs OFF
        feature_extractor_ablation
            Backbone / market encoder variation where supported
        cmtf_search
            Small CMTF tuning grid
        all
            Union of final study tables (not including cmtf_search by default)
    """
    configs: list[AblationConfig] = []

    # Use representative backbones to keep tables disjoint and interpretable
    representative_baseline_model = "lstm"
    representative_cmtf_encoder = "lstm"

    if table == "cmtf_search":
        return generate_cmtf_search_grid()

    # ------------------------------------------------------------
    # Table 1: Data ablation
    # ------------------------------------------------------------
    if table in ("data_ablation", "all"):
        data_rows = [
            AblationConfig(
                model_name=representative_baseline_model,
                fusion_type="none",
                news_scope="none",
                sentiment_mode="none",
            ),
            AblationConfig(
                model_name=representative_baseline_model,
                fusion_type="early",
                news_scope="matched",
                sentiment_mode="scalars",
            ),
            AblationConfig(
                model_name=representative_baseline_model,
                fusion_type="late",
                news_scope="matched",
                sentiment_mode="scalars",
            ),
            AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name="lstm",
                use_cross_attention=True,
                use_positional_encoding=True,
                recency_gate_k=3,
                use_news_gate=True,
                use_two_stage=True,
                use_aux_loss=True,
                use_variance_reg=True,

                fusion_market_dim=128,
                fusion_hidden_dim=64,
                projected_news_dim=128,
                n_heads=4,
                dropout=0.1,
                encoder_lr_scale=0.2,
                aux_loss_weight=0.0,
                stage1_ratio=0.33,
                market_epochs=80,
                fusion_epochs=80,
                market_patience=12,
                fusion_patience=10,

                news_gate_alpha=0.3,
                variance_reg_coeff=0.001,
            ),
        ]
        configs.extend([cfg for cfg in data_rows if cfg.is_valid()])

    # ------------------------------------------------------------
    # Table 2: Architecture ablation (CMTF only)
    # ------------------------------------------------------------
    if table in ("architecture_ablation", "all"):
        for xattn in (True, False):
            cfg = AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name="lstm",
                use_cross_attention=xattn,
                use_positional_encoding=True,
                recency_gate_k=3,
                use_news_gate=True,
                use_two_stage=True,
                use_aux_loss=True,
                use_variance_reg=True,

                fusion_market_dim=128,
                fusion_hidden_dim=64,
                projected_news_dim=128,
                n_heads=4,
                dropout=0.1,
                encoder_lr_scale=0.2,
                aux_loss_weight=0.0,
                stage1_ratio=0.33,
                market_epochs=80,
                fusion_epochs=80,
                market_patience=12,
                fusion_patience=10,

                news_gate_alpha=0.3,
                variance_reg_coeff=0.001,
            )
            if cfg.is_valid():
                configs.append(cfg)

    # ------------------------------------------------------------
    # Table 3: Feature extractor ablation
    # ------------------------------------------------------------
    if table in ("feature_extractor_ablation", "all"):
        # Market-only
        for model in BACKBONE_MODELS:
            cfg = AblationConfig(
                model_name=model,
                fusion_type="none",
                news_scope="none",
                sentiment_mode="none",
            )
            if cfg.is_valid():
                configs.append(cfg)

        # Late fusion supports all backbones
        for model in BACKBONE_MODELS:
            cfg = AblationConfig(
                model_name=model,
                fusion_type="late",
                news_scope="matched",
                sentiment_mode="scalars",
            )
            if cfg.is_valid():
                configs.append(cfg)

        # Early fusion only for torch sequence backbones
        for model in ("lstm", "cnn_lstm"):
            cfg = AblationConfig(
                model_name=model,
                fusion_type="early",
                news_scope="matched",
                sentiment_mode="scalars",
            )
            if cfg.is_valid():
                configs.append(cfg)

        # CMTF encoder ablation
        for enc in ("lstm", "cnn_lstm"):
            cfg = AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name=enc,
                use_cross_attention=True,
                use_positional_encoding=True,
                recency_gate_k=3,
                use_news_gate=True,
                use_two_stage=True,
                use_aux_loss=True,
                use_variance_reg=True,

                fusion_market_dim=128,
                fusion_hidden_dim=64,
                projected_news_dim=128,
                n_heads=4,
                dropout=0.1,
                encoder_lr_scale=0.2,
                aux_loss_weight=0.0,
                stage1_ratio=0.33,
                market_epochs=80,
                fusion_epochs=80,
                market_patience=12,
                fusion_patience=10,

                news_gate_alpha=0.3,
                variance_reg_coeff=0.001,
            )
            if cfg.is_valid():
                configs.append(cfg)

    return list(dict.fromkeys(configs))