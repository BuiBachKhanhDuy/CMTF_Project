"""ablation_registry.py

Single source-of-truth EXPERIMENT REGISTRY for the "clean component ablation"
study (see the redesign roadmap in session notes). Every cell is declared
purely from :class:`~src.benchmark.ablation_config.AblationConfig` — there are
no hardcoded per-cell scripts. Everything derives from ``CMTF_CORE`` (the
current LSTM champion) via targeted single-field overrides, so every metric
delta in the resulting tables is attributable to exactly one design choice.

Usage::

    from src.benchmark.ablation_registry import ABLATION_CELLS, CELL_NOTES

    cfg = ABLATION_CELLS["0"]
    note = CELL_NOTES["0"]

Groups
------
0   Reference          — core model + shuffled-news placebo anchor.
1   Component knockouts — one CMTF_CORE component disabled/flipped per cell.
2   News side           — news_scope + handcrafted interaction features.
3   Gate sweep          — news_gate_alpha / recency_gate_k grid (core is the
                          implicit centre point of both sweeps; see notes).
4   Output formulation  — reconfirm output_mode / two-stage findings.
5   Learned gate        — gate_mode="learned" vs the fixed-alpha default.

Placebo twins ("<id>p") share every field with their real-news sibling except
``shuffle_news=True``, so ``real_minus_placebo_*`` metrics isolate genuine news
signal from generic decision-layer / architecture effects.
"""

from __future__ import annotations

from src.benchmark.ablation_config import AblationConfig, CMTF_CORE, CMTF_MODEL


def _cmtf(**overrides) -> AblationConfig:
    """A CMTF(lstm) cell derived from CMTF_CORE with targeted overrides.

    The registry only studies the LSTM champion encoder (per the roadmap), so
    ``market_encoder_name`` is always "lstm".
    """
    params = {**CMTF_CORE, **overrides}
    return AblationConfig(model_name=CMTF_MODEL, market_encoder_name="lstm", **params)


# ---------------------------------------------------------------------------
# Group 0 — Reference
# ---------------------------------------------------------------------------
_GROUP_0: dict[str, AblationConfig] = {
    "0": _cmtf(),
    "0p": _cmtf(shuffle_news=True),
}
_NOTES_0: dict[str, str] = {
    "0": "Reference cell. Unmodified CMTF_CORE (LSTM champion). Every other "
         "registry cell's deltas are measured against this anchor.",
    "0p": "Placebo twin of cell 0 (shuffle_news=True). Establishes the noise "
          "floor: if news carries no genuine signal, this should collapse "
          "toward the market-only baseline. Anchors every real-minus-placebo "
          "comparison in the registry.",
}

# ---------------------------------------------------------------------------
# Group 1 — Component knockouts (one CMTF_CORE component changed per cell)
# ---------------------------------------------------------------------------
_GROUP_1: dict[str, AblationConfig] = {
    "1": _cmtf(use_cross_attention=False),
    "2": _cmtf(recency_gate_k=0),
    "3": _cmtf(use_news_gate=False),
    "4": _cmtf(use_aux_loss=False),
    "5": _cmtf(use_variance_reg=False),
    "6": _cmtf(sentiment_mode="none"),
    "7": _cmtf(use_positional_encoding=True),
}
_NOTES_1: dict[str, str] = {
    "1": "Cross-modal attention knockout (use_cross_attention=False). Research "
         "question: does letting market queries attend over news tokens beat "
         "a simpler pooled-news fallback?",
    "2": "Recency gating knockout (recency_gate_k=0, disables the exponential "
         "recency decay entirely). Research question: does down-weighting "
         "stale news tokens by recency matter, or is plain relevance gating "
         "sufficient?",
    "3": "News gate knockout (use_news_gate=False). Research question: does "
         "the learned market-conditioned sigmoid gate on the news branch add "
         "value over ungated attention output?",
    "4": "Auxiliary market-anchor loss knockout (use_aux_loss=False). Research "
         "question: does anchoring training to the encoder's own scalar "
         "prediction help keep the fusion head close to a known-good backbone?",
    "5": "Variance-regularisation knockout (use_variance_reg=False). Research "
         "question: does the attention-collapse guard prevent degenerate "
         "(near-constant) fused predictions?",
    "6": "Sentiment contribution (sentiment_mode='none' strips the scalar "
         "sentiment features). Research question: how much of CMTF's edge is "
         "sentiment vs raw news embeddings?",
    "7": "News positional encoding (use_positional_encoding=True; CMTF_CORE "
         "default is False). Research question: does explicit within-window "
         "recency position embedding help once recency gating already exists, "
         "or is it redundant/harmful?",
}

# ---------------------------------------------------------------------------
# Group 2 — News side
# ---------------------------------------------------------------------------
_GROUP_2: dict[str, AblationConfig] = {
    "8": _cmtf(news_scope="matched"),
    "8p": _cmtf(news_scope="matched", shuffle_news=True),
    "9": _cmtf(
        fusion_style="handcrafted",
        use_interaction_prod=True,
        use_interaction_diff=True,
        use_news_context_prod=True,
        use_cosine_sim=True,
        use_pooled_news=True,
    ),
}
_NOTES_2: dict[str, str] = {
    "8": "Matched-only news scope (news_scope='matched') vs CMTF_CORE's "
         "cross-symbol 'all' scope. Research question: does pooling news "
         "across the whole market beat restricting to each symbol's own "
         "matched news?",
    "8p": "Placebo twin of cell 8 (shuffle_news=True). Isolates whether the "
          "matched-scope result is genuine news signal or a decision-layer "
          "artifact.",
    "9": "Handcrafted cross-modal interaction features (fusion_style="
         "'handcrafted' + all 5 interaction toggles ON) vs CMTF_CORE's "
         "minimal learned core ([market_latent, attn_out] only). Research "
         "question: do explicit interaction/cosine/context-product terms add "
         "signal over letting the fusion head learn its own interactions?",
}

# ---------------------------------------------------------------------------
# Group 3 — Gate sweep (news_gate_alpha and recency_gate_k grids)
# ---------------------------------------------------------------------------
# Core (news_gate_alpha=1.0, recency_gate_k=3) is cell "0" — not duplicated
# here, but included in every gate-sweep ranking/curve as the sweep's centre
# point (see ablation_registry.GATE_SWEEP_CELLS below).
_GROUP_3: dict[str, AblationConfig] = {
    "10": _cmtf(news_gate_alpha=0.3),
    "11": _cmtf(news_gate_alpha=0.5),
    "12": _cmtf(recency_gate_k=1),
    "13": _cmtf(recency_gate_k=5),
}
_NOTES_3: dict[str, str] = {
    "10": "News-gate softening: news_gate_alpha=0.3 (mostly pass-through, "
          "gate barely applied) vs CMTF_CORE's alpha=1.0 (gate fully "
          "applied). Research question: is a softer news gate better?",
    "11": "News-gate softening: news_gate_alpha=0.5 (half-strength gate) vs "
          "CMTF_CORE's alpha=1.0. Midpoint of the alpha sweep.",
    "12": "Tighter recency window: recency_gate_k=1 (very fast decay, only "
          "the most recent news matters) vs CMTF_CORE's k=3.",
    "13": "Wider recency window: recency_gate_k=5 (slower decay, more of the "
          "news history retained) vs CMTF_CORE's k=3.",
}

# ---------------------------------------------------------------------------
# Group 4 — Output formulation
# ---------------------------------------------------------------------------
_GROUP_4: dict[str, AblationConfig] = {
    "14": _cmtf(output_mode="encoder_residual"),
    "15": _cmtf(output_mode="fusion_plus_news"),
    "16": _cmtf(output_mode="market_plus_fusion"),
    "17": _cmtf(use_two_stage=True),
}
_NOTES_4: dict[str, str] = {
    "14": "output_mode='encoder_residual': encoder's trained scalar pred + "
          "news residual, vs CMTF_CORE's 'anchored_fusion'. Reconfirms "
          "whether a fixed-weight additive news correction on top of the "
          "encoder's own head is competitive.",
    "15": "output_mode='fusion_plus_news': fusion_pred + news_residual with "
          "no DA anchor guard. Reconfirms the DA-vs-IC tradeoff documented "
          "in CMTF_FUSION_FINDINGS.md.",
    "16": "output_mode='market_plus_fusion': projects market features instead "
          "of using the encoder's trained head; retained as a negative control.",
    "17": "use_two_stage=True: end-to-end encoder fine-tuning during fusion "
          "training, vs CMTF_CORE's frozen single-stage encoder. Reconfirms "
          "whether the two-stage gain (if any) is genuine news signal or "
          "mostly encoder fine-tuning.",
}

# ---------------------------------------------------------------------------
# Group 5 — Learned gate
# ---------------------------------------------------------------------------
_GROUP_5: dict[str, AblationConfig] = {
    "18": _cmtf(gate_mode="learned"),
    "18p": _cmtf(gate_mode="learned", shuffle_news=True),
}
_NOTES_5: dict[str, str] = {
    "18": "gate_mode='learned': the fixed news_gate_alpha scalar is replaced "
          "by a lightweight trainable head (Linear->GELU->Linear->Sigmoid) "
          "predicting a per-sample mixing coefficient from "
          "[market_emb, pooled_news]. Research question: should the news-"
          "gate mixing strength be learned per-sample rather than fixed?",
    "18p": "Placebo twin of cell 18 (shuffle_news=True). Confirms the "
           "learned gate does not simply learn to fit placebo noise (i.e. "
           "its gain, if any, should collapse here too).",
}

# ---------------------------------------------------------------------------
# Combined registry
# ---------------------------------------------------------------------------
ABLATION_CELLS: dict[str, AblationConfig] = {
    **_GROUP_0,
    **_GROUP_1,
    **_GROUP_2,
    **_GROUP_3,
    **_GROUP_4,
    **_GROUP_5,
}

CELL_NOTES: dict[str, str] = {
    **_NOTES_0,
    **_NOTES_1,
    **_NOTES_2,
    **_NOTES_3,
    **_NOTES_4,
    **_NOTES_5,
}

# Cell -> group label, for report grouping.
CELL_GROUP: dict[str, str] = {}
for _cid in _GROUP_0:
    CELL_GROUP[_cid] = "0 — Reference"
for _cid in _GROUP_1:
    CELL_GROUP[_cid] = "1 — Component knockouts"
for _cid in _GROUP_2:
    CELL_GROUP[_cid] = "2 — News side"
for _cid in _GROUP_3:
    CELL_GROUP[_cid] = "3 — Gate sweep"
for _cid in _GROUP_4:
    CELL_GROUP[_cid] = "4 — Output formulation"
for _cid in _GROUP_5:
    CELL_GROUP[_cid] = "5 — Learned gate"

# real cell id -> placebo twin cell id, for real_minus_placebo comparisons.
PLACEBO_PAIRS: dict[str, str] = {
    "0": "0p",
    "8": "8p",
    "18": "18p",
}

# Cells whose confidence-gate behaviour is the subject of study (Group 3 gate
# sweep + the reference core, which is the sweep's centre point) — these get
# coverage-accuracy diagnostics (coverage_deciles.csv, monotonicity checks).
GATE_SWEEP_CELLS: tuple[str, ...] = ("0", "10", "11", "12", "13")

assert set(ABLATION_CELLS) == set(CELL_NOTES), "Every cell must have a research-question note"


def all_cell_ids() -> list[str]:
    """Return every registered cell id in declaration order."""
    return list(ABLATION_CELLS.keys())


def get_cell(cell_id: str) -> AblationConfig:
    """Look up one registry cell by id (e.g. '0', '0p', '18')."""
    try:
        return ABLATION_CELLS[cell_id]
    except KeyError as e:
        raise KeyError(
            f"Unknown ablation cell id={cell_id!r}. Known ids: {all_cell_ids()}"
        ) from e
