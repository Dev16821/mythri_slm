"""
Compact Transformer Encoder for MYTHRI SLM.

A from-scratch implementation (not using HuggingFace Transformer
internals) to have full control over architecture and to keep
the model small and inspectable.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class PositionalEncoding(nn.Module):
    """Learned positional embeddings — simpler and as effective as sinusoidal
    for short sequences (max 128 tokens)."""

    def __init__(self, hidden_dim: int, max_seq_len: int = 128, dropout: float = 0.1, initializer_range: float = 0.02):
        super().__init__()
        self.initializer_range = initializer_range
        self.embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.initializer_range)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_dim)
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.embedding(positions)
        x = x + pos_emb
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention with pre-norm residual."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1, initializer_range: float = 0.02):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.initializer_range = initializer_range

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self._init_weights()

    def _init_weights(self, residual_scale: Optional[float] = None):
        """Consistent Gaussian initialization with zero biases."""
        std = self.initializer_range
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=std)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        out_std = std * (residual_scale if residual_scale is not None else 1.0)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=out_std)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_dim)
        attention_mask: (batch, seq_len) — 1 for valid, 0 for padding
        """
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, S, S)

        if attention_mask is not None:
            # mask: (B, S) -> (B, 1, 1, S)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # (B, H, S, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, S, D)

        return self.out_proj(context)


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1, initializer_range: float = 0.02):
        super().__init__()
        self.initializer_range = initializer_range
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self, residual_scale: Optional[float] = None):
        """Consistent Gaussian initialization with zero biases."""
        std = self.initializer_range
        nn.init.normal_(self.fc1.weight, mean=0.0, std=std)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        fc2_std = std * (residual_scale if residual_scale is not None else 1.0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=fc2_std)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        initializer_range: float = 0.02,
        residual_scale: Optional[float] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout, initializer_range)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout, initializer_range)
        self._init_weights(initializer_range=initializer_range, residual_scale=residual_scale)

    def _init_weights(self, initializer_range: float = 0.02, residual_scale: Optional[float] = None):
        self.attn._init_weights(residual_scale=residual_scale)
        self.ffn._init_weights(residual_scale=residual_scale)
        nn.init.ones_(self.norm1.weight)
        nn.init.zeros_(self.norm1.bias)
        nn.init.ones_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm residual connections
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Full transformer encoder stack.

    This is the shared backbone for the cognitive SLM.
    All task-specific heads branch off the pooled output of this encoder.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        initializer_range: float = 0.02,
        pretrained_embeddings: Optional[str] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.initializer_range = initializer_range

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout, initializer_range)
        self.layer_norm = nn.LayerNorm(hidden_dim)


        residual_scale = 1.0 / math.sqrt(2 * num_layers)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                initializer_range=initializer_range,
                residual_scale=residual_scale,
            )
            for _ in range(num_layers)
        ])

        # Apply unified Gaussian initialization across all encoder submodules
        self.apply(self._init_weights)
        # Apply scaled initialization to residual projections
        self._init_residual_projections()

        if pretrained_embeddings is not None:
            self.load_pretrained_embeddings(pretrained_embeddings)

    def load_pretrained_embeddings(self, model_name: str = "distilbert-base-uncased") -> None:
        """
        Initializes the token embedding weights from a pretrained DistilBERT model,
        transferring learned lexical representations into the encoder.
        If hidden_dim matches (e.g. 768), copies weights directly.
        If hidden_dim differs (e.g. 512), uses PCA low-rank projection to align the
        primary semantic components into target dimensions.
        """
        from transformers import AutoModel
        pretrained = AutoModel.from_pretrained(model_name)
        pretrained_emb = pretrained.get_input_embeddings().weight.data.clone().cpu()

        if pretrained_emb.shape[0] != self.token_embedding.num_embeddings:
            raise ValueError(
                f"Vocab size mismatch: model has {self.token_embedding.num_embeddings}, "
                f"pretrained has {pretrained_emb.shape[0]}"
            )

        src_dim = pretrained_emb.shape[1]
        target_dim = self.token_embedding.embedding_dim

        if src_dim == target_dim:
            self.token_embedding.weight.data.copy_(pretrained_emb.to(self.token_embedding.weight.device))
        else:
            # Low-rank PCA projection to preserve primary semantic axes
            mean = pretrained_emb.mean(dim=0, keepdim=True)
            centered = pretrained_emb - mean
            _, _, v = torch.pca_lowrank(centered, q=target_dim)
            projected = torch.matmul(centered, v)
            # Scale to match target initializer range
            target_std = self.initializer_range
            projected = projected / projected.std() * target_std
            self.token_embedding.weight.data.copy_(projected.to(self.token_embedding.weight.device))

        if self.token_embedding.padding_idx is not None:
            self.token_embedding.weight.data[self.token_embedding.padding_idx].zero_()

    def _init_weights(self, module: nn.Module):
        """
        Unified initialization strategy across the entire transformer:
        - nn.Linear: Normal(mean=0.0, std=initializer_range), zero biases
        - nn.Embedding: Normal(mean=0.0, std=initializer_range), zero padding token
        - nn.LayerNorm: weight=1.0, bias=0.0
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_residual_projections(self):
        """
        Scale residual projection weights (out_proj and fc2) by 1 / sqrt(2 * num_layers)
        to prevent activation variance from compounding with depth.
        """
        residual_scale = 1.0 / math.sqrt(2 * self.num_layers)
        for block in self.blocks:
            nn.init.normal_(
                block.attn.out_proj.weight,
                mean=0.0,
                std=self.initializer_range * residual_scale,
            )
            nn.init.normal_(
                block.ffn.fc2.weight,
                mean=0.0,
                std=self.initializer_range * residual_scale,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        input_ids: (batch, seq_len) — token IDs
        attention_mask: (batch, seq_len) — 1 for valid, 0 for padding
        return_hidden: if True, also return per-token hidden states

        Returns:
            pooled: (batch, hidden_dim) — pooled representation
            token_hidden: (batch, seq_len, hidden_dim) if return_hidden
        """
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.layer_norm(x)

        # CLS pooling (first token) + mean pooling
        cls_hidden = x[:, 0, :]  # (batch, hidden_dim)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            mean_hidden = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            mean_hidden = x.mean(dim=1)

        pooled = torch.cat([cls_hidden, mean_hidden], dim=-1)  # (batch, 2*hidden_dim)

        if return_hidden:
            return pooled, x
        return pooled

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PretrainedDistilBertEncoder(nn.Module):
    """
    Pretrained DistilBERT backbone encoder for MYTHRI SLM.

    Leverages pretrained representations from DistilBERT (distilbert-base-uncased)
    rather than training a 36.7M parameter transformer from scratch on small datasets.
    Provides CLS + Mean pooling to create rich, contextualized representations for
    downstream multi-task heads.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        dropout: float = 0.1,
        freeze: bool = False,
        target_dim: Optional[int] = None,
    ):
        super().__init__()
        from transformers import DistilBertModel
        self.model_name = model_name
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.base_hidden_dim = self.distilbert.config.hidden_size  # 768

        self.target_dim = target_dim
        if target_dim is not None and target_dim != self.base_hidden_dim:
            self.proj = nn.Linear(self.base_hidden_dim, target_dim)
            self.hidden_dim = target_dim
        else:
            self.proj = None
            self.hidden_dim = self.base_hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.pooled_dim = 2 * self.hidden_dim

        if freeze:
            for param in self.distilbert.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # (B, S, base_hidden_dim)
        x = self.dropout(x)

        if self.proj is not None:
            x_proj = self.proj(x)
        else:
            x_proj = x

        # CLS pooling (first token)
        cls_hidden = x_proj[:, 0, :]  # (B, hidden_dim)

        # Masked mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            mean_hidden = (x_proj * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            mean_hidden = x_proj.mean(dim=1)

        pooled = torch.cat([cls_hidden, mean_hidden], dim=-1)  # (B, 2 * hidden_dim)

        if return_hidden:
            return pooled, x_proj
        return pooled

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)