"""Ablation benchmark configuration and grid generation.

Two focused study tables (see CMTF_FUSION_FINDINGS.md for the evidence behind
the canonical CMTF design):

    fusion_comparison
        The main result. Backbone x fusion strategy (none / early / late / CMTF)
        across every market backbone, plus a shuffled-news placebo. Proves CMTF
        beats early/late/none and that the lift is genuine news signal.

    component_ablation
        Leave-one-out ablation from the canonical CMTF design (CMTF_CORE) on the
        LSTM backbone. Each row changes exactly ONE thing, so every metric delta
        is attributable to a single component / design choice.

Each cell = one AblationConfig. Invalid cells are never generated.
"""

from __future__ import annotations

from dataclasses import dataclass

BACKBONE_MODELS = (
    "lstm",
    "cnn_lstm",
    "gpt4ts",
    "chronos",
)

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

    # Final output formulation for CMTF (see HybridFusionPredictor for the
    # exact forward-pass formula). NONE of these apply a post-hoc lambda blend
    # or anchor gate — predict() always returns the composed value directly.
    # "anchored_fusion":    fusion_pred + news_residual (DEFAULT). Aux loss is
    #                       anchored to the encoder's own trained scalar
    #                       prediction (keeps the fusion head close to a known-
    #                       good backbone during training); the deployed output
    #                       is the raw fused prediction, no blending.
    # "encoder_residual":   encoder_trained_pred + news_residual (news branch
    #                       learns a fixed-weight additive correction on top of
    #                       the encoder's own scalar output).
    # "fusion_plus_news":   fusion_pred + news_residual (aux loss anchored to
    #                       the fusion model's own market head instead of the
    #                       encoder's). Numerically near-identical to
    #                       anchored_fusion; kept as a separate ablation row.
    # "market_plus_fusion": market_pred + fusion_delta                            (DEPRECATED / harmful)
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
    # LEVER 3 (differentiable Sharpe surrogate). DA/Sharpe are sign/decision
    # objectives but the fused head trains as a point regressor, so it ~optimises
    # IC while barely touching DA/Sharpe. This term optimises a scale-invariant
    # negative-Sharpe surrogate on soft signed positions. DEFAULT 0.0 => legacy
    # objective (cache-identical); raise it (+ sign_penalty_weight) for the sweep.
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
            # Appended only when enabled so legacy cells keep byte-identical
            # cell_ids (and therefore their on-disk prediction caches).
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


# ---------------------------------------------------------------------------
# Canonical CMTF design — single source of truth
# ---------------------------------------------------------------------------
# Every CMTF cell in the study derives from this dict, so all CMTF rows share an
# identical training recipe (enabling L1 encoder-cache sharing within a backbone/
# seed) and every component ablation isolates exactly one change. The values
# encode the validated findings in CMTF_FUSION_FINDINGS.md:
#   * output_mode="anchored_fusion"   -> train the fusion head to predict the
#     full news-using target (fusion_pred + news_residual) and DEPLOY that
#     prediction directly, no post-hoc blend. The aux loss (small weight) keeps
#     the fusion head close to the encoder's own trained scalar during training.
#     NOTE (2026-07-12 correction): earlier revisions of this comment and
#     CMTF_FUSION_FINDINGS.md described a validation-selected lambda blend
#     ("anchor + lambda*(...)") — that mechanism (`fusion_selection.
#     select_additive_lambda`) was NEVER wired into `HybridFusionPredictor`;
#     it was dead code and has been removed. `predict()` always returns the
#     raw fused prediction. Treat any remaining references to a CMTF lambda
#     guard as historical/superseded.
#   * news_gate_alpha=1.0             -> the news gate is load-bearing: only with
#     the full sigmoid gate does news become genuinely DA-positive when anchored
#     (the softened 0.3 gate did not achieve genuine dominance).
#   * use_two_stage=False             -> single-stage frozen-encoder fusion.
#     Preserves the backbone's directional accuracy AND keeps the anchor honest
#     (two-stage's apparent gain was mostly encoder fine-tuning, not news).
#   * fusion_style="learned"          -> minimal [market_latent, attn_out] core;
#     handcrafted interaction terms are a component to ablate, not a default.
#   * news_scope="all"                -> component ablation (seed 42) showed
#     cross-symbol news dominates matched-only news on DA/Sharpe/IC. Adopted as
#     the canonical default; the matched-only variant is now the ablation row.
#     NOTE: pending multi-seed confirmation (single-seed rows were below base
#     rate); revisit if it does not survive additional seeds.
#   * use_positional_encoding=False   -> component ablation (seed 42) showed the
#     news positional encoding hurt every metric; disabled by default. The
#     positional-encoding-on variant is now the ablation row.
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
    """A non-CMTF baseline row (market-only / early / late fusion).

    NEWS-PARITY FIX: early/late now use ``news_scope="all"`` to match CMTF's
    ``CMTF_CORE`` (cross-symbol pooled news). Previously they used
    ``news_scope="matched"`` while CMTF used "all", so "early/late vs CMTF"
    secretly compared different NEWS sets — CMTF's IC/Sharpe edge could reflect
    "more news" rather than "better fusion". With all news-using fusion cells on
    the same pooled news tensor, the comparison isolates the fusion mechanism.
    This is cache-safe: ``news_scope`` is NOT part of the shared market-encoder
    cache key (see ``_encoder_cache_key`` in ablation_runner.py) and the market
    recipe is unchanged, so ``late`` still reloads the exact same frozen market
    encoder CMTF/none already trained — only the small news head is refit.

    APPLES-TO-APPLES FIX (market): all rows use sentiment_mode="scalars" so the
    market feature set / encoder is identical across none/early/late/cmtf (the
    "scalars" branch in ``_apply_sentiment_mode`` is a no-op on the market
    window). ``none`` keeps ``news_scope="matched"`` since ``fusion_type="none"``
    never reads the news tensors (see run_ablation_cell); its news_scope is
    irrelevant to results and left as-is to document the anchor's cache-identity.
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

    # --------------------------------------------------------------
    # Table 2: Component ablation (leave-one-out from CMTF_CORE, LSTM)
    # --------------------------------------------------------------
    if table in ("component_ablation", "all"):
        # Core reference (dedups with the LSTM CMTF cell in fusion_comparison).
        cfgs.append(_cmtf("lstm"))

        # -- Component knock-outs (one change each) --
        cfgs.append(_cmtf("lstm", use_cross_attention=False))     # cross-modal attention
        cfgs.append(_cmtf("lstm", recency_gate_k=0))              # recency gating
        cfgs.append(_cmtf("lstm", use_news_gate=False))           # news gate
        cfgs.append(_cmtf("lstm", use_positional_encoding=True))  # news positional enc
        cfgs.append(_cmtf("lstm", use_aux_loss=False))            # market-aux regulariser
        cfgs.append(_cmtf("lstm", use_variance_reg=False))        # attn-collapse guard

        # -- Cross-modal interaction features (learned core + handcrafted terms) --
        cfgs.append(_cmtf(
            "lstm",
            fusion_style="handcrafted",
            use_interaction_prod=True,
            use_interaction_diff=True,
            use_news_context_prod=True,
            use_cosine_sim=True,
            use_pooled_news=True,
        ))

        # -- Design-choice ablations (establish the research gap) --
        # Core is 'anchored_fusion'; these isolate each alternative formulation.
        cfgs.append(_cmtf("lstm", output_mode="encoder_residual"))    # news-residual only (safe, often news-blind)
        cfgs.append(_cmtf("lstm", output_mode="fusion_plus_news"))    # high-peak, no guard (loses DA anchor)
        cfgs.append(_cmtf("lstm", output_mode="market_plus_fusion"))  # naive re-predict (harmful)
        cfgs.append(_cmtf("lstm", use_two_stage=True))                # end-to-end fine-tune (gain mostly not-news)

        # -- News-side ablations --
        cfgs.append(_cmtf("lstm", news_scope="matched"))              # matched vs cross-symbol news
        cfgs.append(_cmtf("lstm", sentiment_mode="none"))             # sentiment contribution

    valid = [c for c in cfgs if c.is_valid()]
    return list(dict.fromkeys(valid))
