import torch
from einops import rearrange
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from tfsformer.configuration import TFSpecFormerConfig
from tfsformer.layers import (
    FeedForward,
    GroupSelfAttention,
    MemoryAttention,
    RMSNorm,
    ResidualBlock,
    TimeSelfAttention,
)
from tfsformer.output import (
    TFSpecFormerEncoderBlockOutput,
    TFSpecFormerEncoderOutput,
    TFSpecFormerOutput,
)
from tfsformer.utils import InstanceNorm, content_mask, pad_left


class TFSpecFormerEncoderBlock(nn.Module):
    def __init__(self, config: TFSpecFormerConfig) -> None:
        super().__init__()
        self.time_attn = TimeSelfAttention(config)
        self.group_attn = GroupSelfAttention(config)
        self.ffn = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        group_time_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> TFSpecFormerEncoderBlockOutput:
        time_out = self.time_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        group_out = self.group_attn(
            time_out.hidden_states,
            attention_mask=group_time_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.ffn(group_out.hidden_states)
        return TFSpecFormerEncoderBlockOutput(
            hidden_states=hidden_states,
            time_attn_weights=time_out.attn_weights,
            group_attn_weights=group_out.attn_weights,
        )


class TFSpecFormerEncoder(nn.Module):
    def __init__(self, config: TFSpecFormerConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            TFSpecFormerEncoderBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    @staticmethod
    def time_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """``(batch, seq)`` keep-mask -> additive ``(batch, 1, 1, seq)``."""
        mask = attention_mask[:, None, None, :].to(dtype=dtype)
        return (1.0 - mask) * torch.finfo(dtype).min

    @staticmethod
    def group_mask(
        group_ids: torch.Tensor, attention_mask: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Additive mask of shape ``(seq, 1, batch, batch)``.

        A query variate may read a key variate at the same patch only when they share
        a ``group_id`` and the key patch is valid.
        """
        same_group = (group_ids[:, None] == group_ids[None, :]).to(
            dtype=attention_mask.dtype
        )
        allowed = torch.einsum("qb,bt->qbt", same_group, attention_mask)
        allowed = rearrange(allowed, "q b t -> t 1 q b").to(dtype=dtype)
        return (1.0 - allowed) * torch.finfo(dtype).min

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        group_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> TFSpecFormerEncoderOutput:
        _, seq_len, _ = inputs_embeds.shape
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
        time_mask = self.time_mask(attention_mask, inputs_embeds.dtype)
        group_time_mask = self.group_mask(
            group_ids, attention_mask.to(dtype=inputs_embeds.dtype), inputs_embeds.dtype
        )

        time_weights = []
        group_weights = []
        hidden = self.dropout(inputs_embeds)
        for block in self.blocks:
            block_out = block(
                hidden,
                position_ids=position_ids,
                attention_mask=time_mask,
                group_time_mask=group_time_mask,
                output_attentions=output_attentions,
            )
            hidden = block_out.hidden_states
            if output_attentions:
                time_weights.append(block_out.time_attn_weights)
                group_weights.append(block_out.group_attn_weights)
        hidden = self.dropout(self.final_norm(hidden))
        return TFSpecFormerEncoderOutput(
            last_hidden_state=hidden,
            time_attn_weights=tuple(time_weights) if output_attentions else None,
            group_attn_weights=tuple(group_weights) if output_attentions else None,
        )


class TFSpecFormerModel(PreTrainedModel):
    config_class = TFSpecFormerConfig
    _supports_sdpa = True

    def __init__(self, config: TFSpecFormerConfig) -> None:
        super().__init__(config)
        self.config = config
        forecasting = config.forecasting
        if forecasting.input_patch_size != forecasting.output_patch_size:
            raise ValueError("input_patch_size and output_patch_size must be equal.")
        if forecasting.input_patch_stride != forecasting.input_patch_size:
            raise ValueError("continuous masks assume a non-overlapping patch stride.")

        self.forecasting = forecasting
        self.patch_size = forecasting.input_patch_size
        self.input_patch_embedding = ResidualBlock(
            in_dim=self.patch_size * 3,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            dropout_p=config.dropout_rate,
            act_fn_name=config.dense_act_fn,
        )
        self.instance_norm = InstanceNorm(use_arcsinh=forecasting.use_arcsinh)
        self.encoder = TFSpecFormerEncoder(config)
        self.num_quantiles = len(forecasting.quantiles)
        self.register_buffer(
            "quantiles",
            torch.tensor(forecasting.quantiles, dtype=torch.float32),
            persistent=False,
        )
        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=self.num_quantiles * self.patch_size,
            dropout_p=config.dropout_rate,
            act_fn_name=config.dense_act_fn,
        )
        self.post_init()

    def _init_weights(self, module) -> None:
        factor = self.config.initializer_factor
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight, mean=0.0, std=factor * (module.weight.shape[-1] ** -0.5)
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, MemoryAttention):
            nn.init.normal_(module.memory, mean=0.0, std=factor)

    def _compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pinball loss on hidden steps only: mean over time, sum over quantiles, mean over series."""
        target = torch.nan_to_num(target, nan=0.0).unsqueeze(1).to(dtype=pred.dtype)
        quantiles = self.quantiles.to(device=pred.device, dtype=pred.dtype).view(
            1, -1, 1
        )
        pinball = 2 * torch.abs(
            (target - pred) * ((target <= pred).to(pred.dtype) - quantiles)
        )
        weights = loss_mask.to(dtype=pred.dtype)
        denom = weights.sum(dim=-1).clamp(min=1).unsqueeze(1)
        per_series = ((pinball * weights.unsqueeze(1)).sum(dim=-1) / denom).sum(dim=-1)
        valid = weights.sum(dim=-1) > 0
        if not bool(valid.any()):
            return pred.sum() * 0
        return per_series[valid].mean()

    def forward(
        self,
        values: torch.Tensor,
        input_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> TFSpecFormerOutput:
        if values.ndim != 2:
            raise ValueError(
                f"values must have shape (batch, time), found {tuple(values.shape)}."
            )
        values = values.to(device=self.device, dtype=torch.float32)
        batch, length = values.shape
        observed = torch.isfinite(values)
        if input_mask is None:
            visible = observed
        else:
            if input_mask.shape != values.shape:
                raise ValueError(
                    f"input_mask must have shape {tuple(values.shape)}, found {tuple(input_mask.shape)}."
                )
            visible = observed & input_mask.to(device=self.device).bool()
        tokens = content_mask(observed)

        max_len = (
            self.forecasting.context_length
            + self.forecasting.max_output_patches * self.patch_size
        )
        if length > max_len:
            values = values[:, -max_len:]
            observed = observed[:, -max_len:]
            visible = visible[:, -max_len:]
            tokens = tokens[:, -max_len:]
            length = max_len

        pad = (self.patch_size - length % self.patch_size) % self.patch_size
        values = pad_left(values, pad, float("nan"))
        observed = pad_left(observed, pad, False)
        visible = pad_left(visible, pad, False)
        tokens = pad_left(tokens, pad, False)
        length = values.shape[-1]

        shown, loc_scale = self.instance_norm(
            values.masked_fill(~visible, float("nan"))
        )
        target, _ = self.instance_norm(
            values.masked_fill(~observed, float("nan")), loc_scale
        )
        shown = torch.nan_to_num(shown, nan=0.0)
        presence = visible.float()

        num_patches = length // self.patch_size
        patched_value = shown.unfold(-1, self.patch_size, self.patch_size)
        patched_presence = presence.unfold(-1, self.patch_size, self.patch_size)
        attention_mask = (
            tokens.float().unfold(-1, self.patch_size, self.patch_size).sum(-1) > 0
        )

        time_index = torch.arange(length, device=values.device, dtype=torch.float32)
        time_index = time_index / float(self.forecasting.time_encoding_scale)
        time_enc = time_index.view(1, num_patches, self.patch_size).expand(
            batch, -1, -1
        )
        embed_in = torch.cat((time_enc, patched_value, patched_presence), dim=-1)
        embed_in = embed_in.to(
            dtype=self.input_patch_embedding.hidden_layer.weight.dtype
        )

        if group_ids is None:
            group_ids = torch.arange(batch, device=values.device)
        else:
            group_ids = group_ids.to(device=values.device, dtype=torch.long)
            if group_ids.shape != (batch,):
                raise ValueError(
                    f"group_ids must have shape ({batch},), found {tuple(group_ids.shape)}."
                )

        encoded = self.encoder(
            self.input_patch_embedding(embed_in),
            group_ids=group_ids,
            attention_mask=attention_mask.float(),
            output_attentions=output_attentions,
        )
        quantile_preds = self.output_patch_embedding(encoded.last_hidden_state)
        quantile_preds = rearrange(
            quantile_preds,
            "b s (q p) -> b q (s p)",
            q=self.num_quantiles,
            p=self.patch_size,
        )

        loss_mask = (observed & ~visible).float()
        loss = (
            self._compute_loss(quantile_preds, target, loss_mask)
            if bool(loss_mask.any())
            else None
        )

        loc, scale = loc_scale
        with torch.no_grad():
            restored = self.instance_norm.inverse(
                quantile_preds.detach(), (loc.unsqueeze(1), scale.unsqueeze(1))
            )
        if pad:
            restored = restored[..., pad:]
        return TFSpecFormerOutput(
            loss=loss,
            quantile_preds=restored,
            time_attn_weights=encoded.time_attn_weights,
            group_attn_weights=encoded.group_attn_weights,
        )

    @torch.no_grad()
    def forecast(
        self,
        context: torch.Tensor,
        prediction_length: int,
        group_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Append mask steps after ``context`` and return quantiles of shape ``(batch, Q, prediction_length)``."""
        if context.ndim != 2:
            raise ValueError(
                f"context must have shape (batch, time), found {tuple(context.shape)}."
            )
        if prediction_length < 1:
            raise ValueError("prediction_length must be positive.")
        max_horizon = self.forecasting.max_output_patches * self.patch_size
        if prediction_length > max_horizon:
            raise ValueError(
                f"prediction_length {prediction_length} exceeds the maximum horizon {max_horizon}."
            )

        context = context.to(dtype=torch.float32)
        if context.shape[-1] > self.forecasting.context_length:
            context = context[..., -self.forecasting.context_length :]
        horizon = (
            (prediction_length + self.patch_size - 1) // self.patch_size
        ) * self.patch_size
        future = torch.full(
            (context.shape[0], horizon),
            float("nan"),
            dtype=context.dtype,
            device=context.device,
        )
        history = context.shape[-1]
        values = torch.cat((context, future), dim=-1)
        input_mask = torch.zeros(
            values.shape, dtype=torch.float32, device=values.device
        )
        input_mask[:, :history] = torch.isfinite(context).float()
        preds = self.forward(
            values, input_mask=input_mask, group_ids=group_ids
        ).quantile_preds
        return preds[:, :, history : history + prediction_length]
