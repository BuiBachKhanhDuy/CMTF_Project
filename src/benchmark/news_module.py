"""news_modules.py

All news-side neural components, shared across fusion strategies.

Ownership model
---------------
- NewsProjector is an nn.Module.  It must be owned by exactly one
  fusion wrapper and registered in that wrapper's parameter tree.
  Never share the same instance across two wrappers.
- AttentionPoolingNewsEncoder and NewsBranchPredictor are internal
  building blocks; callers should not instantiate them directly.

Design decisions
----------------
- Random projection removed.  A trained Linear + LayerNorm replaces it.
- Mean pooling removed.  A single-query attention pooling layer replaces it.
  When all news slots are masked the attention collapses to the null token,
  so the branch naturally suppresses itself without a manual scalar gate.
- learned_alpha removed.  Branch confidence is implicit in the attention
  weights and the MLP output magnitude, both of which are learned.
- Null news embedding: an nn.Parameter used when every slot in a window
  is masked.  This keeps "no news" in-distribution for the MLP.
- Mask ownership: _as_bool_mask is always called on the *projected*
  embedding, never on the raw one, eliminating the mask / representation
  mismatch from the previous design.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STANDARD_NEWS_DIM = 128


def _to_numpy_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _as_bool_mask(
    news_mask: np.ndarray | torch.Tensor | None,
    projected_news: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Return boolean mask where True means 'masked / invalid / no news'.

    Explicit masks are required. We do not infer missingness from projected
    embeddings because projection destroys raw zero-vector semantics.
    """
    if news_mask is None:
        raise ValueError(
            "news_mask is required. Do not infer mask from projected embeddings."
        )

    if isinstance(news_mask, np.ndarray):
        return news_mask.astype(bool)

    return news_mask.to(dtype=torch.bool)


# ---------------------------------------------------------------------------
# Trainable news projector
# ---------------------------------------------------------------------------

class NewsProjector(nn.Module):
    """Trainable projection from raw news embeddings to a standard dimension.

    Replaces the previous fixed random-matrix projection.  Because this is an
    nn.Module it must be registered inside the owning fusion wrapper so that
    its parameters are included in the optimizer's parameter group.

    The projection is deliberately shallow (Linear + LayerNorm) so that it
    does not consume expressive capacity that should live in the fusion head.
    Alignment between semantic directions in the raw embedding space and the
    projected axes is preserved through training rather than lost to a random
    rotation.

    Parameters
    ----------
    input_dim:
        Dimensionality of the raw news embeddings (e.g. 768 for BERT-base).
    output_dim:
        Target dimensionality after projection.
    dropout:
        Applied after LayerNorm to regularise the projection during training.
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = STANDARD_NEWS_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=True)
        self.norm = nn.LayerNorm(self.output_dim)
        self.drop = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, news_embs: torch.Tensor) -> torch.Tensor:
        """Project a batch of news sequences.

        Parameters
        ----------
        news_embs:
            Shape (B, S, input_dim).

        Returns
        -------
        Tensor of shape (B, S, output_dim).
        """
        if news_embs.ndim != 3 or news_embs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected (B, S, {self.input_dim}), got {tuple(news_embs.shape)}"
            )
        return self.drop(self.norm(self.proj(news_embs)))

    def transform_numpy(self, news_embs: np.ndarray) -> np.ndarray:
        """Numpy convenience wrapper for inference-time use.
    
        Runs in eval mode with no_grad; output is detached numpy float32.
        """
        arr = torch.as_tensor(
            news_embs,
            dtype=torch.float32,
            device=next(self.parameters()).device,
        )
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                out = self.forward(arr)
            return out.cpu().numpy().astype(np.float32)
        finally:
            self.train(was_training)

    def ensure_projected(self, news_embs: np.ndarray) -> np.ndarray:
        """Accept either raw (input_dim) or already-projected (output_dim) numpy arrays.

        Returns float32 numpy of shape (N, S, output_dim).
        """
        arr = np.asarray(news_embs, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected (N, S, D), got {arr.shape}")
        if arr.shape[-1] == self.output_dim:
            return arr
        if arr.shape[-1] == self.input_dim:
            return self.transform_numpy(arr)
        raise ValueError(
            f"Last dim must be {self.input_dim} (raw) or {self.output_dim} (projected), "
            f"got {arr.shape[-1]}"
        )


# ---------------------------------------------------------------------------
# Attention pooling
# ---------------------------------------------------------------------------

class AttentionPoolingNewsEncoder(nn.Module):
    """Sequence-aware news encoder using single-query attention pooling.

    Architecture
    ------------
    1. Per-position news MLP: projects each news token to fusion_dim.
    2. Optional positional embedding added to each token.
    3. Single learnable query vector attends over the sequence.
    4. Padded positions are excluded from the softmax.
    5. When *all* positions are masked the null_token absorbs the full
       attention weight, producing a deterministic "no news" representation
       rather than a NaN or a biased mean.

    The attention weight distribution (not stored here) can be extracted
    by callers for interpretability.

    Parameters
    ----------
    news_dim:
        Dimensionality of projected news tokens (output_dim of NewsProjector).
    fusion_dim:
        Internal dimension after the per-position MLP.
    seq_len:
        Maximum sequence length; used for positional embeddings.
    use_positional_encoding:
        Whether to add learned positional embeddings before attention.
    dropout:
        Dropout on the per-position MLP and on the query projection.
    """

    def __init__(
        self,
        news_dim: int = STANDARD_NEWS_DIM,
        fusion_dim: int = 128,
        seq_len: int = 30,
        use_positional_encoding: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.news_dim = news_dim
        self.fusion_dim = fusion_dim
        self.seq_len = seq_len

        # Per-position MLP (shared weights across positions)
        self.token_mlp = nn.Sequential(
            nn.Linear(news_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )

        # Learned positional embeddings
        self.pos_enc: nn.Embedding | None = None
        if use_positional_encoding:
            self.pos_enc = nn.Embedding(seq_len, fusion_dim)
            nn.init.normal_(self.pos_enc.weight, std=0.02)

        # Single learned query for attention pooling
        self.query = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        nn.init.normal_(self.query, std=0.02)

        self.query_proj = nn.Linear(fusion_dim, fusion_dim, bias=False)
        self.key_proj = nn.Linear(fusion_dim, fusion_dim, bias=False)

        # Null token: "no news" in-distribution representation
        self.null_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        nn.init.normal_(self.null_token, std=0.01)

        self._scale = float(fusion_dim) ** -0.5

    def forward(
        self,
        news_proj: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of news sequences into a single pooled vector.

        Parameters
        ----------
        news_proj:
            Projected news tokens, shape (B, S, news_dim).
        pad_mask:
            Boolean mask, shape (B, S). True = masked / invalid slot.

        Returns
        -------
        pooled:
            Shape (B, fusion_dim).
        attn_weights:
            Shape (B, S+1).  Last position corresponds to the null token.
            Useful for interpretability.
        """
        B, S, _ = news_proj.shape
        device = news_proj.device

        # Encode each position
        tokens = self.token_mlp(news_proj)  # (B, S, fusion_dim)

        if self.pos_enc is not None:
            pos = torch.arange(S, device=device).clamp(max=self.pos_enc.num_embeddings - 1)
            tokens = tokens + self.pos_enc(pos).unsqueeze(0)

        # Prepend null token so attention always has at least one valid slot
        null = self.null_token.expand(B, -1, -1)  # (B, 1, fusion_dim)
        tokens_with_null = torch.cat([null, tokens], dim=1)  # (B, S+1, fusion_dim)

        # Extend mask: null token is never masked
        null_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        mask_with_null = torch.cat([null_mask, pad_mask], dim=1)  # (B, S+1)

        # Attention: query (1, fusion_dim) vs keys (B, S+1, fusion_dim)
        q = self.query_proj(self.query).expand(B, -1, -1)  # (B, 1, fusion_dim)
        k = self.key_proj(tokens_with_null)  # (B, S+1, fusion_dim)

        scores = torch.bmm(q, k.transpose(1, 2)) * self._scale  # (B, 1, S+1)
        scores = scores.squeeze(1)  # (B, S+1)

        # Mask out invalid positions with -inf before softmax
        scores = scores.masked_fill(mask_with_null, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # (B, S+1)

        # Weighted sum
        pooled = torch.bmm(attn_weights.unsqueeze(1), tokens_with_null).squeeze(1)  # (B, fusion_dim)

        return pooled, attn_weights


# ---------------------------------------------------------------------------
# News branch predictor (used by LateFusionWrapper)
# ---------------------------------------------------------------------------

class NewsBranchPredictor(nn.Module):
    """Market-conditioned news residual branch.

    Architecture
    ------------
    1. AttentionPoolingNewsEncoder pools the news sequence into a single
       fusion_dim vector.  The null token handles the zero-news case.
    2. The pooled news vector is concatenated with the market context scalar.
    3. A 3-layer MLP predicts the residual (targets - market_pred).

    Key differences from the previous design
    -----------------------------------------
    - Attention pooling replaces mean pooling: temporal/ordering signal
      is preserved and the model can learn to focus on recent or extreme news.
    - learned_alpha removed: branch output magnitude is governed by the
      MLP weights and the attention collapse on no-news inputs rather than
      a post-hoc scalar that had scale-mismatch problems.
    - null_token in AttentionPoolingNewsEncoder keeps "no news" inputs
      in-distribution without relying on a zero vector.

    Parameters
    ----------
    news_dim:
        Dimensionality of projected news tokens.
    fusion_dim:
        Internal dim for the attention encoder and MLP.
    seq_len:
        Maximum news sequence length.
    dropout:
        Dropout rate for both the encoder and MLP.
    use_positional_encoding:
        Passed through to AttentionPoolingNewsEncoder.
    """

    def __init__(
        self,
        news_dim: int = STANDARD_NEWS_DIM,
        fusion_dim: int = 128,
        seq_len: int = 30,
        dropout: float = 0.2,
        use_positional_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.news_dim = news_dim
        self.fusion_dim = fusion_dim

        self.attn_encoder = AttentionPoolingNewsEncoder(
            news_dim=news_dim,
            fusion_dim=fusion_dim,
            seq_len=seq_len,
            use_positional_encoding=use_positional_encoding,
            dropout=dropout,
        )

        # MLP: pooled news (fusion_dim) + market context scalar (1) → residual scalar
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim + 1, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_dim // 2, 1),
        )

        nn.init.xavier_uniform_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Last attention weights, stored for interpretability
        self.last_attn_weights: torch.Tensor | None = None

    def forward(
        self,
        news_proj: torch.Tensor,
        pad_mask: torch.Tensor,
        market_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the news residual.

        Parameters
        ----------
        news_proj:
            Projected news tokens, shape (B, S, news_dim).
        pad_mask:
            Boolean mask, shape (B, S). True = invalid slot.
        market_pred:
            Optional scalar market prediction, shape (B,).
            When None, zeros are used as market context.

        Returns
        -------
        Scalar residual prediction, shape (B,).
        """
        pooled, attn_weights = self.attn_encoder(news_proj, pad_mask)
        self.last_attn_weights = attn_weights.detach()

        if market_pred is None:
            ctx = torch.zeros(pooled.shape[0], 1, device=pooled.device, dtype=pooled.dtype)
        else:
            ctx = market_pred.unsqueeze(-1)

        fused = torch.cat([pooled, ctx], dim=-1)
        return self.mlp(fused).squeeze(-1)