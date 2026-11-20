import torch
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from tfsformer.configuration import TFSpecFormerConfig
from tfsformer.output import AttentionOutput


class RMSNorm(nn.Module):
    """T5-style RMSNorm: no mean subtraction and no bias."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * normalized).to(hidden_states.dtype)


if RMSNorm not in ALL_LAYERNORM_LAYERS:
    ALL_LAYERNORM_LAYERS.append(RMSNorm)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RoPE(nn.Module):
    """Rotary embedding applied to query and key. Used on the time axis only."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dim must be even, found {dim}.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = position_ids.to(device=query.device, dtype=torch.float32)
        freqs = torch.einsum("bs,d->bsd", pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=query.dtype).unsqueeze(1)
        sin = emb.sin().to(dtype=query.dtype).unsqueeze(1)
        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin
        return query, key


class MemoryAttention(nn.Module):
    """
    Bidirectional attention with a shared memory vector.

    ``V = K + RMSNorm(memory)``. ``memory`` has shape ``(num_heads, d_kv)`` and is
    broadcast over the batch and the sequence. There is no KV cache and no token id.
    A sigmoid gate, taken from the query projection, multiplies the attention output
    before ``o_proj``.
    """

    def __init__(self, config: TFSpecFormerConfig, use_rope: bool = True) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_kv
        self.inner_dim = self.num_heads * self.head_dim
        self.dropout_rate = config.dropout_rate
        self.gate = config.attn_output_gate
        self.qk_norm = config.qk_norm
        self.use_rope = use_rope
        self.scale = self.head_dim ** -0.5
        self.config = config

        if self.gate == "headwise":
            q_features = self.inner_dim + self.num_heads
        elif self.gate == "elementwise":
            q_features = self.inner_dim * 2
        else:
            q_features = self.inner_dim

        self.q = nn.Linear(config.d_model, q_features, bias=False)
        self.k = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, config.d_model, bias=False)
        self.memory = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.m_norm = RMSNorm(self.head_dim, eps=config.layer_norm_epsilon)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.layer_norm_epsilon)
            self.k_norm = RMSNorm(self.head_dim, eps=config.layer_norm_epsilon)
        if use_rope:
            self.rope = RoPE(self.head_dim, base=config.rope_theta)

    def _project(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch, seq_len, _ = hidden_states.shape
        query = self.q(hidden_states)
        key = rearrange(self.k(hidden_states), "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim)
        gate = None
        if self.gate == "headwise":
            query = query.view(batch, seq_len, self.num_heads, self.head_dim + 1)
            query, gate = query.split([self.head_dim, 1], dim=-1)
            query = query.permute(0, 2, 1, 3)
            gate = gate.permute(0, 2, 1, 3)
        elif self.gate == "elementwise":
            query = query.view(batch, seq_len, self.num_heads, self.head_dim * 2)
            query, gate = query.split([self.head_dim, self.head_dim], dim=-1)
            query = query.permute(0, 2, 1, 3)
            gate = gate.permute(0, 2, 1, 3)
        else:
            query = rearrange(query, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim)

        memory = self.m_norm(self.memory).to(dtype=key.dtype)
        value = key + memory[None, :, None, :]
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        return query, key, value, gate

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        eager: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dropout = self.dropout_rate if self.training else 0.0
        if eager:
            scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
            scores = scores + mask
            weights = torch.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
            weights = nn.functional.dropout(weights, p=dropout, training=self.training)
            return torch.matmul(weights, value), weights

        output = nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=dropout,
            scale=self.scale,
        )
        return output, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        query, key, value, gate = self._project(hidden_states)
        if self.use_rope:
            if position_ids is None:
                raise ValueError("position_ids are required when use_rope=True.")
            query, key = self.rope(query, key, position_ids)

        implementation = getattr(self.config, "_attn_implementation", None) or "sdpa"
        eager = output_attentions or implementation == "eager"
        attended, weights = self._attend(query, key, value, mask, eager=eager)
        # A fully padded group has no legal keys. Zero that row instead of letting NaN spread.
        attended = torch.where(torch.isfinite(attended), attended, torch.zeros_like(attended))
        if weights is not None:
            weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
        if gate is not None:
            attended = attended * torch.sigmoid(gate)
        hidden = self.o(rearrange(attended, "b h s d -> b s (h d)"))
        return AttentionOutput(hidden_states=hidden, attn_weights=weights if output_attentions else None)


class TimeSelfAttention(nn.Module):
    def __init__(self, config: TFSpecFormerConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.attn = MemoryAttention(config, use_rope=True)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        attended = self.attn(
            self.norm(hidden_states),
            mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        hidden = hidden_states + self.dropout(attended.hidden_states)
        return AttentionOutput(hidden_states=hidden, attn_weights=attended.attn_weights)


class GroupSelfAttention(nn.Module):
    """Attention across variates that share a time index. No RoPE: channels have no order."""

    def __init__(self, config: TFSpecFormerConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.attn = MemoryAttention(config, use_rope=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        hidden_states = rearrange(hidden_states, "batch time dim -> time batch dim")
        attended = self.attn(
            self.norm(hidden_states),
            mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attended.hidden_states)
        hidden_states = rearrange(hidden_states, "time batch dim -> batch time dim")
        return AttentionOutput(hidden_states=hidden_states, attn_weights=attended.attn_weights)


class FeedForward(nn.Module):
    def __init__(self, config: TFSpecFormerConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.act = ACT2FN[config.dense_act_fn]
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = self.down(self.dropout(self.act(self.up(self.norm(hidden_states)))))
        return hidden_states + self.dropout(hidden)


class ResidualBlock(nn.Module):
    """MLP with a linear skip, used to embed patches and to emit quantiles."""

    def __init__(self, in_dim: int, h_dim: int, out_dim: int, dropout_p: float, act_fn_name: str = "gelu") -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.dropout = nn.Dropout(dropout_p)
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.output_layer(self.dropout(self.act(self.hidden_layer(x))))
        return hidden + self.residual_layer(x)
