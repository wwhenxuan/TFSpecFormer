import math
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from tfsformer.configuration import MASK_MODES


class Patch(nn.Module):
    """Non-overlapping unfold. The caller aligns the length to ``patch_size`` first."""

    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        if length % self.patch_size != 0:
            pad = self.patch_size - (length % self.patch_size)
            filler = torch.full((*x.shape[:-1], pad), torch.nan, dtype=x.dtype, device=x.device)
            x = torch.cat((filler, x), dim=-1)
        return x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)


class InstanceNorm(nn.Module):
    """Per-series normalization. NaNs are ignored, and arcsinh is optional."""

    def __init__(self, eps: float = 1e-5, use_arcsinh: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.use_arcsinh = use_arcsinh

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.float()
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale == 0, torch.full_like(scale, self.eps), scale)
        else:
            loc, scale = loc_scale
        scaled = (x - loc) / scale
        if self.use_arcsinh:
            scaled = torch.arcsinh(scaled)
        return scaled.to(orig_dtype), (loc, scale)

    def inverse(self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        loc, scale = loc_scale
        if self.use_arcsinh:
            x = torch.sinh(x)
        return (x * scale + loc).to(orig_dtype)


def pad_left(tensor: torch.Tensor, pad: int, value) -> torch.Tensor:
    if pad <= 0:
        return tensor
    filler = torch.full((*tensor.shape[:-1], pad), value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((filler, tensor), dim=-1)


def left_pad_and_cat(tensors: Sequence[torch.Tensor], fill_value) -> torch.Tensor:
    """Left-pad a list of ``(rows, time)`` tensors to a common length and concatenate rows."""
    max_len = max(tensor.shape[-1] for tensor in tensors)
    padded = []
    for tensor in tensors:
        gap = max_len - tensor.shape[-1]
        if gap:
            filler = torch.full((*tensor.shape[:-1], gap), fill_value, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat((filler, tensor), dim=-1)
        padded.append(tensor)
    return torch.cat(padded, dim=0)


def content_mask(observed: torch.Tensor) -> torch.Tensor:
    """
    Mark real token positions.

    Leading missing steps are padding. Everything from the first observation to the
    end of the row is content, including a trailing forecast mask.
    """
    length = observed.shape[-1]
    positions = torch.arange(length, device=observed.device)
    first = torch.where(observed, positions, length).amin(dim=-1)
    return positions.unsqueeze(0) >= first.unsqueeze(1)


def as_series(target) -> torch.Tensor:
    """Return a float tensor of shape ``(channels, time)``."""
    if isinstance(target, torch.Tensor):
        series = target.detach().float().cpu()
    else:
        series = torch.tensor(np.asarray(target), dtype=torch.float32)
    if series.ndim == 1:
        series = series.unsqueeze(0)
    if series.ndim != 2:
        raise ValueError(f"target must have shape (T,) or (C, T), found {tuple(series.shape)}.")
    return series


def _place_spans(
    num_patches: int,
    n_hidden: int,
    min_span: int,
    max_span: int,
    max_spans: int,
    rng: np.random.Generator,
) -> np.ndarray:
    hidden = np.zeros(num_patches, dtype=bool)
    remaining = n_hidden
    placed = 0
    attempts = 0
    while remaining > 0 and placed < max_spans and attempts < 64:
        attempts += 1
        length = int(rng.integers(min_span, max_span + 1))
        length = min(length, remaining)
        if length < min_span:
            break
        starts = []
        for start in range(num_patches - length + 1):
            left = max(0, start - 1)
            right = min(num_patches, start + length + 1)
            if not hidden[left:right].any():
                starts.append(start)
        if not starts:
            break
        start = int(rng.choice(starts))
        hidden[start : start + length] = True
        remaining -= length
        placed += 1
    return hidden


def sample_patch_mask(
    num_channels: int,
    num_patches: int,
    mode: str,
    *,
    mask_ratio: float = 0.25,
    min_span_patches: int = 1,
    max_span_patches: int = 4,
    max_mask_spans: int = 4,
    future_patches: int = 1,
    min_visible_patches: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Return a boolean mask of shape ``(channels, patches)``.

    ``True`` means the patch is hidden from the model. Spans stay contiguous and do
    not touch, so two spans cannot merge into a longer one.
    """
    if mode not in MASK_MODES:
        raise ValueError(f"Unknown mask mode {mode!r}.")
    if num_channels < 1 or num_patches < 1:
        raise ValueError("num_channels and num_patches must be positive.")
    rng = rng or np.random.default_rng()
    hidden = np.zeros((num_channels, num_patches), dtype=bool)
    max_hidden = max(num_patches - min_visible_patches, 0)

    if mode == "future":
        n_hidden = min(max(future_patches, 0), max_hidden)
        if n_hidden:
            hidden[:, -n_hidden:] = True
        return hidden

    n_hidden = int(round(mask_ratio * num_patches))
    n_hidden = min(max(n_hidden, 1), max_hidden)
    if n_hidden < 1:
        return hidden
    min_span = min(min_span_patches, n_hidden)
    max_span = max(min_span, min(max_span_patches, n_hidden))

    if mode == "group":
        span = _place_spans(num_patches, n_hidden, min_span, max_span, max_mask_spans, rng)
        hidden[:] = span
        return hidden

    if mode == "random":
        for channel in range(num_channels):
            hidden[channel] = _place_spans(num_patches, n_hidden, min_span, max_span, max_mask_spans, rng)
        return hidden

    base = _place_spans(num_patches, n_hidden, min_span, max_span, max_mask_spans, rng)
    shift = max(1, num_patches // 2)
    rolled = np.roll(base, shift)
    if np.array_equal(base, rolled):
        hidden_idx = np.flatnonzero(base)
        visible_idx = np.flatnonzero(~base)
        if len(hidden_idx) and len(visible_idx):
            rolled = base.copy()
            rolled[hidden_idx[0]] = False
            rolled[visible_idx[0]] = True
    hidden[0::2] = base
    hidden[1::2] = rolled
    return hidden


def hidden_runs(row: np.ndarray) -> list[int]:
    """Lengths of contiguous hidden blocks in one patch row."""
    runs = []
    length = 0
    for value in row:
        if value:
            length += 1
        elif length:
            runs.append(length)
            length = 0
    if length:
        runs.append(length)
    return runs


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def plot_confidence_prediction(
    context: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    quantiles: Sequence[float] | None = None,
    axis: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot one series: observed context, future target, median, and the outer quantile band."""
    context = np.asarray(context, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    created = axis is None
    if created:
        fig, axis = plt.subplots(figsize=(8, 2.6), dpi=100)
    else:
        fig = axis.figure

    ctx_time = np.arange(len(context))
    fut_time = np.arange(len(context), len(context) + len(target))
    axis.plot(ctx_time, context, color="royalblue", label="context")
    if len(target):
        axis.plot(fut_time, target, color="0.45", label="target")
    if prediction.size:
        if quantiles is not None and 0.5 in list(quantiles):
            median = list(quantiles).index(0.5)
        else:
            median = prediction.shape[0] // 2
        axis.plot(fut_time, prediction[median], color="darkorange", linestyle="--", label="median")
        axis.fill_between(fut_time, prediction[0], prediction[-1], color="darkorange", alpha=0.2, label="interval")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left", fontsize=8)
    if created:
        fig.tight_layout()
    return fig


def plot_forecast_grid(
    rows: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    quantiles: Sequence[float] | None = None,
) -> plt.Figure:
    count = max(len(rows), 1)
    fig, axes = plt.subplots(count, 1, figsize=(8, 2.4 * count), dpi=100, squeeze=False)
    for axis, row in zip(axes[:, 0], rows):
        plot_confidence_prediction(row[0], row[1], row[2], quantiles=quantiles, axis=axis)
    fig.tight_layout()
    return fig


def min_visible_patches(min_past: int, patch_size: int) -> int:
    return int(math.ceil(min_past / patch_size))
