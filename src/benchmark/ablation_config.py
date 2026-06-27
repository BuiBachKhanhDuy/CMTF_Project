"""Ablation benchmark configuration and grid generation.

Each cell = one (model × fusion × news_scope × sentiment × toggle) combination.
Invalid cells are never generated.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    # Final output formulation for CMTF
    # "market_plus_fusion": market_pred + fusion_delta
    # "fusion_plus_news": fusion_pred + news_residual
    output_mode: str = "market_plus_fusion"

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
    encoder_lr_scale: float = 0.1
    aux_loss_weight: float = 0.3
    stage1_ratio: float = 0.33
    market_epochs: int = 100
    fusion_epochs: int = 60
    market_patience: int = 15
    fusion_patience: int = 12

    # News gate softening
    news_gate_alpha: float = 1.0

    # Variance regularization strength
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

        # output_mode is meaningful only for CMTF
        if self.fusion_type == "cmtf":
            if self.output_mode not in {"market_plus_fusion", "fusion_plus_news"}:
                return False
        else:
            if self.output_mode != "market_plus_fusion":
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

        return True

    @property
    def cell_id(self) -> str:
        """Unique string identifier for caching."""
        parts = [self.model_name, self.fusion_type, self.news_scope, self.sentiment_mode]

        if self.market_encoder_name is not None:
            parts.append(f"enc={self.market_encoder_name}")

        if self.fusion_type == "cmtf":
            parts.append(f"om={self.output_mode}")
            parts.append(f"fstyle={self.fusion_style}")
            parts.append(f"mq={self.market_query_mode}")
            parts.append(f"xattn={int(self.use_cross_attention)}")
            parts.append(f"pe={int(self.use_positional_encoding)}")
            parts.append(f"k={self.recency_gate_k}")
            parts.append(f"gate={int(self.use_news_gate)}")
            parts.append(f"galpha={self.news_gate_alpha:.2f}")
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
            parts.append(f"elr={self.encoder_lr_scale}")
            parts.append(f"auxw={self.aux_loss_weight}")
            parts.append(f"s1r={self.stage1_ratio}")
            parts.append(f"me={self.market_epochs}")
            parts.append(f"fe={self.fusion_epochs}")
            parts.append(f"mp={self.market_patience}")
            parts.append(f"fp={self.fusion_patience}")

        return "__".join(parts)

def generate_cmtf_search_grid() -> list[AblationConfig]:
    """Focused CMTF search around the current best design direction.

    Fixed best direction:
        - output_mode="fusion_plus_news"
        - fusion_style="learned"
        - market_query_mode="multi"
        - use_cross_attention=True
        - use_news_gate=True
        - recency_gate_k=3
        - use_pooled_news=False
        - use_cosine_sim=False
        - use_positional_encoding=False

    Search question:
        Which handcrafted interaction terms still help on top of the learned core?
        - interaction_prod
        - interaction_diff
        - news_context_prod
    """
    configs: list[AblationConfig] = []

    common = dict(
        model_name=CMTF_MODEL,
        fusion_type="cmtf",
        news_scope="matched",
        sentiment_mode="scalars",
        market_encoder_name="lstm",
        output_mode="fusion_plus_news",
        use_cross_attention=True,
        use_positional_encoding=False,
        recency_gate_k=3,
        use_news_gate=True,
        use_two_stage=True,
        use_aux_loss=True,
        use_variance_reg=True,
        use_pooled_news=False,
        use_cosine_sim=False,
        fusion_market_dim=64,
        fusion_hidden_dim=32,
        projected_news_dim=128,
        n_heads=4,
        dropout=0.1,
        sign_penalty_weight=0.005,
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

    search_rows = [
        # --------------------------------------------------------
        # Reduced handcrafted reference under the same scaffold
        # --------------------------------------------------------
        AblationConfig(
            **common,
            fusion_style="handcrafted",
            market_query_mode="multi",
            use_interaction_prod=True,
            use_interaction_diff=True,
            use_news_context_prod=True,
        ),

        # --------------------------------------------------------
        # Learned anchor with NO handcrafted interaction terms
        # --------------------------------------------------------
        AblationConfig(
            **common,
            fusion_style="learned",
            market_query_mode="multi",
            use_interaction_prod=False,
            use_interaction_diff=False,
            use_news_context_prod=False,
        ),

        # --------------------------------------------------------
        # Single handcrafted term on top of learned core
        # --------------------------------------------------------
        AblationConfig(
            **common,
            fusion_style="learned",
            market_query_mode="multi",
            use_interaction_prod=True,
            use_interaction_diff=False,
            use_news_context_prod=False,
        ),
        AblationConfig(
            **common,
            fusion_style="learned",
            market_query_mode="multi",
            use_interaction_prod=False,
            use_interaction_diff=True,
            use_news_context_prod=False,
        ),
        AblationConfig(
            **common,
            fusion_style="learned",
            market_query_mode="multi",
            use_interaction_prod=False,
            use_interaction_diff=False,
            use_news_context_prod=True,
        ),

        # --------------------------------------------------------
        # Pairs of handcrafted terms on top of learned core
        # --------------------------------------------------------
        AblationConfig(
            **common,
            fusion_style="learned",
            market_query_mode="multi",
            use_interaction_prod=True,
            use_interaction_diff=True,
            use_news_context_prod=False,
        ),
        AblationConfig(
            **common,
            fusion_style="learned",
            market_query_mode="multi",
            use_interaction_prod=True,
            use_interaction_diff=False,
            use_news_context_prod=True,
        ),
        AblationConfig(
            **common,
            fusion_style="learned",
            market_query_mode="multi",
            use_interaction_prod=False,
            use_interaction_diff=True,
            use_news_context_prod=True,
        ),

        # --------------------------------------------------------
        # Full handcrafted interaction bundle on learned core
        # --------------------------------------------------------
        AblationConfig(
            **common,
            fusion_style="learned",
            market_query_mode="multi",
            use_interaction_prod=True,
            use_interaction_diff=True,
            use_news_context_prod=True,
        ),
    ]

    configs.extend([cfg for cfg in search_rows if cfg.is_valid()])
    return configs

def generate_cmtf_20d_candidate_search() -> list[AblationConfig]:
    """Three candidate CMTF configs to identify the best representative
    for the final cross-model comparison at 20D.

    Config A — Best DA_skill% at 5D:
        handcrafted, multi, no cross-attention, all features on

    Config B — Best Sharpe at 5D:
        handcrafted, multi, cross-attention on, pooled_news only

    Config C — Best learned at 5D:
        learned, multi, cross-attention on, no handcrafted features

    Winner at 20D on DA_skill% → IC → Sharpe becomes the CMTF row
    in the final data_ablation cross-model comparison.
    """
    common = dict(
        model_name=CMTF_MODEL,
        fusion_type="cmtf",
        news_scope="matched",
        sentiment_mode="scalars",
        market_encoder_name="lstm",
        output_mode="fusion_plus_news",
        use_positional_encoding=True,
        recency_gate_k=3,
        use_news_gate=True,
        use_two_stage=True,
        use_aux_loss=True,
        use_variance_reg=True,
        fusion_market_dim=64,
        fusion_hidden_dim=32,
        projected_news_dim=128,
        n_heads=4,
        dropout=0.1,
        sign_penalty_weight=0.005,
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

    candidates = [
        # Config A: best DA_skill% at 5D
        # no cross-attention, all handcrafted features on
        AblationConfig(
            **common,
            use_cross_attention=False,
            fusion_style="handcrafted",
            market_query_mode="multi",
            use_interaction_prod=True,
            use_interaction_diff=True,
            use_news_context_prod=True,
            use_cosine_sim=True,
            use_pooled_news=True,
        ),

        # Config B: best Sharpe at 5D
        # cross-attention on, pooled_news only, all other features off
        AblationConfig(
            **common,
            use_cross_attention=True,
            fusion_style="handcrafted",
            market_query_mode="multi",
            use_interaction_prod=False,
            use_interaction_diff=False,
            use_news_context_prod=False,
            use_cosine_sim=False,
            use_pooled_news=True,
        ),

        # Config C: best learned candidate at 5D
        # learned, multi-query, cross-attention on, no handcrafted features
        AblationConfig(
            **common,
            use_cross_attention=True,
            fusion_style="learned",
            market_query_mode="multi",
            use_interaction_prod=False,
            use_interaction_diff=False,
            use_news_context_prod=False,
            use_cosine_sim=False,
            use_pooled_news=False,
        ),
    ]

    return [cfg for cfg in candidates if cfg.is_valid()]

def generate_grid(table: str = "all") -> list[AblationConfig]:
    """Generate valid ablation configs grouped by final study axis."""
    configs: list[AblationConfig] = []

    representative_baseline_model = "lstm"

    if table == "cmtf_search":
        return generate_cmtf_search_grid()

    if table == "cmtf_20d_candidate_search":
        return generate_cmtf_20d_candidate_search()
    
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
                output_mode="fusion_plus_news",
                use_cross_attention=True,
                use_positional_encoding=True,
                recency_gate_k=3,
                use_news_gate=True,
                use_two_stage=True,
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
                sign_penalty_weight=0.005,
                encoder_lr_scale=0.2,
                aux_loss_weight=0.0,
                stage1_ratio=0.33,
                market_epochs=80,
                fusion_epochs=80,
                market_patience=12,
                fusion_patience=10,
                news_gate_alpha=0.3,
                variance_reg_coeff=0.001,
                fusion_style="learned",
                market_query_mode="multi",
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
                output_mode="fusion_plus_news",
                use_cross_attention=xattn,
                use_positional_encoding=True,
                recency_gate_k=3,
                use_news_gate=True,
                use_two_stage=True,
                use_aux_loss=True,
                use_variance_reg=True,
                fusion_market_dim=64,
                fusion_hidden_dim=32,
                projected_news_dim=128,
                n_heads=4,
                dropout=0.1,
                sign_penalty_weight=0.005,
                encoder_lr_scale=0.2,
                aux_loss_weight=0.0,
                stage1_ratio=0.33,
                market_epochs=80,
                fusion_epochs=80,
                market_patience=12,
                fusion_patience=10,
                news_gate_alpha=0.3,
                variance_reg_coeff=0.001,
                fusion_style="handcrafted",
                market_query_mode="multi",
            )
            if cfg.is_valid():
                configs.append(cfg)

    # ------------------------------------------------------------
    # Table 3: Feature construction ablation (CMTF only)
    # ------------------------------------------------------------
    if table in ("feature_construction_ablation",):
        feature_rows = [
            AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name="lstm",
                output_mode="fusion_plus_news",
                use_cross_attention=True,
                use_positional_encoding=True,
                recency_gate_k=3,
                use_news_gate=True,
                use_two_stage=True,
                use_aux_loss=True,
                use_variance_reg=True,
                use_interaction_prod=True,
                use_interaction_diff=True,
                use_news_context_prod=True,
                use_cosine_sim=True,
                use_pooled_news=True,
                fusion_market_dim=64,
                fusion_hidden_dim=32,
                projected_news_dim=128,
                n_heads=4,
                dropout=0.1,
                sign_penalty_weight=0.005,
                encoder_lr_scale=0.2,
                aux_loss_weight=0.0,
                stage1_ratio=0.33,
                market_epochs=80,
                fusion_epochs=80,
                market_patience=12,
                fusion_patience=10,
                news_gate_alpha=0.3,
                variance_reg_coeff=0.001,
                fusion_style="handcrafted",
                market_query_mode="multi",
            ),
            AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name="lstm",
                output_mode="fusion_plus_news",
                use_cross_attention=True,
                use_positional_encoding=True,
                recency_gate_k=3,
                use_news_gate=True,
                use_two_stage=True,
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
                sign_penalty_weight=0.005,
                encoder_lr_scale=0.2,
                aux_loss_weight=0.0,
                stage1_ratio=0.33,
                market_epochs=80,
                fusion_epochs=80,
                market_patience=12,
                fusion_patience=10,
                news_gate_alpha=0.3,
                variance_reg_coeff=0.001,
                fusion_style="learned",
                market_query_mode="last",
            ),
            AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name="lstm",
                output_mode="fusion_plus_news",
                use_cross_attention=True,
                use_positional_encoding=True,
                recency_gate_k=3,
                use_news_gate=True,
                use_two_stage=True,
                use_aux_loss=True,
                use_variance_reg=True,
                use_interaction_prod=True,
                use_interaction_diff=True,
                use_news_context_prod=True,
                use_cosine_sim=True,
                use_pooled_news=False,
                fusion_market_dim=64,
                fusion_hidden_dim=32,
                projected_news_dim=128,
                n_heads=4,
                dropout=0.1,
                sign_penalty_weight=0.005,
                encoder_lr_scale=0.2,
                aux_loss_weight=0.0,
                stage1_ratio=0.33,
                market_epochs=80,
                fusion_epochs=80,
                market_patience=12,
                fusion_patience=10,
                news_gate_alpha=0.3,
                variance_reg_coeff=0.001,
                fusion_style="handcrafted",
                market_query_mode="multi",
            ),
        ]
        configs.extend([cfg for cfg in feature_rows if cfg.is_valid()])

    # ------------------------------------------------------------
    # Table 4: News-side ablation (CMTF only)
    # ------------------------------------------------------------
    if table in ("news_ablation",):
        news_rows = [
            AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name="lstm",
                output_mode="fusion_plus_news",
                use_cross_attention=True,
                use_positional_encoding=True,
                recency_gate_k=3,
                use_news_gate=True,
                use_two_stage=True,
                use_aux_loss=True,
                use_variance_reg=True,
                use_interaction_prod=True,
                use_interaction_diff=True,
                use_news_context_prod=True,
                use_cosine_sim=True,
                use_pooled_news=True,
                fusion_market_dim=64,
                fusion_hidden_dim=32,
                projected_news_dim=128,
                n_heads=4,
                dropout=0.1,
                sign_penalty_weight=0.005,
                encoder_lr_scale=0.2,
                aux_loss_weight=0.0,
                stage1_ratio=0.33,
                market_epochs=80,
                fusion_epochs=80,
                market_patience=12,
                fusion_patience=10,
                news_gate_alpha=0.3,
                variance_reg_coeff=0.001,
                fusion_style="handcrafted",
                market_query_mode="multi",
            ),
            AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name="lstm",
                output_mode="fusion_plus_news",
                use_cross_attention=False,
                use_positional_encoding=True,
                recency_gate_k=3,
                use_news_gate=True,
                use_two_stage=True,
                use_aux_loss=True,
                use_variance_reg=True,
                use_interaction_prod=True,
                use_interaction_diff=True,
                use_news_context_prod=True,
                use_cosine_sim=True,
                use_pooled_news=True,
                fusion_market_dim=64,
                fusion_hidden_dim=32,
                projected_news_dim=128,
                n_heads=4,
                dropout=0.1,
                sign_penalty_weight=0.005,
                encoder_lr_scale=0.2,
                aux_loss_weight=0.0,
                stage1_ratio=0.33,
                market_epochs=80,
                fusion_epochs=80,
                market_patience=12,
                fusion_patience=10,
                news_gate_alpha=0.3,
                variance_reg_coeff=0.001,
                fusion_style="handcrafted",
                market_query_mode="multi",
            ),
            AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name="lstm",
                output_mode="fusion_plus_news",
                use_cross_attention=True,
                use_positional_encoding=False,
                recency_gate_k=3,
                use_news_gate=True,
                use_two_stage=True,
                use_aux_loss=True,
                use_variance_reg=True,
                use_interaction_prod=True,
                use_interaction_diff=True,
                use_news_context_prod=True,
                use_cosine_sim=True,
                use_pooled_news=True,
                fusion_market_dim=64,
                fusion_hidden_dim=32,
                projected_news_dim=128,
                n_heads=4,
                dropout=0.1,
                sign_penalty_weight=0.005,
                encoder_lr_scale=0.2,
                aux_loss_weight=0.0,
                stage1_ratio=0.33,
                market_epochs=80,
                fusion_epochs=80,
                market_patience=12,
                fusion_patience=10,
                news_gate_alpha=0.3,
                variance_reg_coeff=0.001,
                fusion_style="handcrafted",
                market_query_mode="multi",
            ),
            AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name="lstm",
                output_mode="fusion_plus_news",
                use_cross_attention=True,
                use_positional_encoding=True,
                recency_gate_k=3,
                use_news_gate=False,
                use_two_stage=True,
                use_aux_loss=True,
                use_variance_reg=True,
                use_interaction_prod=True,
                use_interaction_diff=True,
                use_news_context_prod=True,
                use_cosine_sim=True,
                use_pooled_news=True,
                fusion_market_dim=64,
                fusion_hidden_dim=32,
                projected_news_dim=128,
                n_heads=4,
                dropout=0.1,
                sign_penalty_weight=0.005,
                encoder_lr_scale=0.2,
                aux_loss_weight=0.0,
                stage1_ratio=0.33,
                market_epochs=80,
                fusion_epochs=80,
                market_patience=12,
                fusion_patience=10,
                news_gate_alpha=0.3,
                variance_reg_coeff=0.001,
                fusion_style="handcrafted",
                market_query_mode="multi",
            ),
            AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name="lstm",
                output_mode="fusion_plus_news",
                use_cross_attention=True,
                use_positional_encoding=True,
                recency_gate_k=0,
                use_news_gate=True,
                use_two_stage=True,
                use_aux_loss=True,
                use_variance_reg=True,
                use_interaction_prod=True,
                use_interaction_diff=True,
                use_news_context_prod=True,
                use_cosine_sim=True,
                use_pooled_news=True,
                fusion_market_dim=64,
                fusion_hidden_dim=32,
                projected_news_dim=128,
                n_heads=4,
                dropout=0.1,
                sign_penalty_weight=0.005,
                encoder_lr_scale=0.2,
                aux_loss_weight=0.0,
                stage1_ratio=0.33,
                market_epochs=80,
                fusion_epochs=80,
                market_patience=12,
                fusion_patience=10,
                news_gate_alpha=0.3,
                variance_reg_coeff=0.001,
                fusion_style="handcrafted",
                market_query_mode="multi",
            ),
        ]
        configs.extend([cfg for cfg in news_rows if cfg.is_valid()])

    # ------------------------------------------------------------
    # Table 5: Feature extractor ablation
    # ------------------------------------------------------------
    if table in ("feature_extractor_ablation", "all"):
        for model in BACKBONE_MODELS:
            cfg = AblationConfig(
                model_name=model,
                fusion_type="none",
                news_scope="none",
                sentiment_mode="none",
            )
            if cfg.is_valid():
                configs.append(cfg)

        for model in BACKBONE_MODELS:
            cfg = AblationConfig(
                model_name=model,
                fusion_type="late",
                news_scope="matched",
                sentiment_mode="scalars",
            )
            if cfg.is_valid():
                configs.append(cfg)

        for model in ("lstm", "cnn_lstm"):
            cfg = AblationConfig(
                model_name=model,
                fusion_type="early",
                news_scope="matched",
                sentiment_mode="scalars",
            )
            if cfg.is_valid():
                configs.append(cfg)

        for enc in ("lstm", "cnn_lstm"):
            cfg = AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name=enc,
                output_mode="fusion_plus_news",
                use_cross_attention=True,
                use_positional_encoding=True,
                recency_gate_k=3,
                use_news_gate=True,
                use_two_stage=True,
                use_aux_loss=True,
                use_variance_reg=True,
                use_interaction_prod=True,
                use_interaction_diff=True,
                use_news_context_prod=True,
                use_cosine_sim=True,
                use_pooled_news=True,
                fusion_market_dim=64,
                fusion_hidden_dim=32,
                projected_news_dim=128,
                n_heads=4,
                dropout=0.1,
                sign_penalty_weight=0.005,
                encoder_lr_scale=0.2,
                aux_loss_weight=0.0,
                stage1_ratio=0.33,
                market_epochs=80,
                fusion_epochs=80,
                market_patience=12,
                fusion_patience=10,
                news_gate_alpha=0.3,
                variance_reg_coeff=0.001,
                fusion_style="handcrafted",
                market_query_mode="multi",
            )
            if cfg.is_valid():
                configs.append(cfg)

    # ------------------------------------------------------------
    # Table 6: Mini 5D diagnosis (CMTF only)
    # ------------------------------------------------------------
    if table in ("mini_5d_diagnosis",):
        mini_rows = [
            AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name="lstm",
                output_mode="fusion_plus_news",
                use_cross_attention=True,
                use_positional_encoding=True,
                use_news_gate=True,
                recency_gate_k=3,
                use_two_stage=True,
                use_aux_loss=True,
                use_variance_reg=True,
                use_interaction_prod=True,
                use_interaction_diff=True,
                use_news_context_prod=True,
                use_cosine_sim=True,
                use_pooled_news=True,
                fusion_market_dim=64,
                fusion_hidden_dim=32,
                projected_news_dim=128,
                n_heads=4,
                dropout=0.1,
                sign_penalty_weight=0.005,
                encoder_lr_scale=0.2,
                aux_loss_weight=0.0,
                stage1_ratio=0.33,
                market_epochs=80,
                fusion_epochs=80,
                market_patience=12,
                fusion_patience=10,
                news_gate_alpha=0.3,
                variance_reg_coeff=0.001,
                fusion_style="handcrafted",
                market_query_mode="multi",
            ),
            AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name="lstm",
                output_mode="fusion_plus_news",
                use_cross_attention=False,
                use_positional_encoding=True,
                use_news_gate=True,
                recency_gate_k=3,
                use_two_stage=True,
                use_aux_loss=True,
                use_variance_reg=True,
                use_interaction_prod=True,
                use_interaction_diff=True,
                use_news_context_prod=True,
                useCosine_sim=True,
                use_pooled_news=True,
                fusion_market_dim=64,
                fusion_hidden_dim=32,
                projected_news_dim=128,
                n_heads=4,
                dropout=0.1,
                sign_penalty_weight=0.005,
                encoder_lr_scale=0.2,
                aux_loss_weight=0.0,
                stage1_ratio=0.33,
                market_epochs=80,
                fusion_epochs=80,
                market_patience=12,
                fusion_patience=10,
                news_gate_alpha=0.3,
                variance_reg_coeff=0.001,
                fusion_style="handcrafted",
                market_query_mode="multi",
            ),
            AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name="lstm",
                output_mode="fusion_plus_news",
                use_cross_attention=False,
                use_positional_encoding=False,
                use_news_gate=True,
                recency_gate_k=3,
                use_two_stage=True,
                use_aux_loss=True,
                use_variance_reg=True,
                use_interaction_prod=True,
                use_interaction_diff=True,
                use_news_context_prod=True,
                use_cosine_sim=True,
                use_pooled_news=True,
                fusion_market_dim=64,
                fusion_hidden_dim=32,
                projected_news_dim=128,
                n_heads=4,
                dropout=0.1,
                sign_penalty_weight=0.005,
                encoder_lr_scale=0.2,
                aux_loss_weight=0.0,
                stage1_ratio=0.33,
                market_epochs=80,
                fusion_epochs=80,
                market_patience=12,
                fusion_patience=10,
                news_gate_alpha=0.3,
                variance_reg_coeff=0.001,
                fusion_style="handcrafted",
                market_query_mode="multi",
            ),
        ]
        configs.extend([cfg for cfg in mini_rows if cfg.is_valid()])

    # ------------------------------------------------------------
    # Table 7: Learned vs Handcrafted CMTF
    # ------------------------------------------------------------
    if table in ("learned_cmtf_ablation",):
        learned_rows = [
            AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name="lstm",
                output_mode="fusion_plus_news",
                fusion_style="handcrafted",
                market_query_mode="multi",
                use_cross_attention=True,
                use_positional_encoding=True,
                recency_gate_k=3,
                use_news_gate=True,
                use_two_stage=True,
                use_aux_loss=True,
                use_variance_reg=True,
                use_interaction_prod=True,
                use_interaction_diff=True,
                use_news_context_prod=True,
                use_cosine_sim=True,
                use_pooled_news=True,
                fusion_market_dim=64,
                fusion_hidden_dim=32,
                projected_news_dim=128,
                n_heads=4,
                dropout=0.1,
                sign_penalty_weight=0.005,
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
            AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name="lstm",
                output_mode="fusion_plus_news",
                fusion_style="learned",
                market_query_mode="last",
                use_cross_attention=True,
                use_positional_encoding=True,
                recency_gate_k=3,
                use_news_gate=True,
                use_two_stage=True,
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
                sign_penalty_weight=0.005,
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
            AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name="lstm",
                output_mode="fusion_plus_news",
                fusion_style="learned",
                market_query_mode="last",
                use_cross_attention=False,
                use_positional_encoding=True,
                recency_gate_k=3,
                use_news_gate=True,
                use_two_stage=True,
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
                sign_penalty_weight=0.005,
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
            AblationConfig(
                model_name=CMTF_MODEL,
                fusion_type="cmtf",
                news_scope="matched",
                sentiment_mode="scalars",
                market_encoder_name="lstm",
                output_mode="fusion_plus_news",
                fusion_style="learned",
                market_query_mode="last",
                use_cross_attention=True,
                use_positional_encoding=True,
                recency_gate_k=0,
                use_news_gate=False,
                use_two_stage=True,
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
                sign_penalty_weight=0.005,
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
        configs.extend([cfg for cfg in learned_rows if cfg.is_valid()])
        
        if table == "cmtf_20d_candidate_search":
            return generate_cmtf_20d_candidate_search()

    return list(dict.fromkeys(configs))