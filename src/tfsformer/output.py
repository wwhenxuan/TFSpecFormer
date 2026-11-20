from dataclasses import dataclass

import torch
from transformers.utils import ModelOutput


@dataclass
class AttentionOutput(ModelOutput):
    hidden_states: torch.Tensor | None = None
    attn_weights: torch.Tensor | None = None


@dataclass
class TFSpecFormerEncoderBlockOutput(ModelOutput):
    hidden_states: torch.Tensor | None = None
    time_attn_weights: torch.Tensor | None = None
    group_attn_weights: torch.Tensor | None = None


@dataclass
class TFSpecFormerEncoderOutput(ModelOutput):
    last_hidden_state: torch.Tensor | None = None
    time_attn_weights: tuple[torch.Tensor, ...] | None = None
    group_attn_weights: tuple[torch.Tensor, ...] | None = None


@dataclass
class TFSpecFormerOutput(ModelOutput):
    loss: torch.Tensor | None = None
    quantile_preds: torch.Tensor | None = None
    time_attn_weights: tuple[torch.Tensor, ...] | None = None
    group_attn_weights: tuple[torch.Tensor, ...] | None = None
