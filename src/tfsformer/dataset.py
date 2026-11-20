import os
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset

from tfsformer.configuration import (
    MASK_MODES,
    DatasetMode,
    TFSpecFormerForecastingConfig,
    TFSpecFormerTrainingConfig,
)
from tfsformer.utils import (
    as_series,
    left_pad_and_cat,
    min_visible_patches,
    sample_patch_mask,
)


def list_data_dirs(path: str) -> list[str]:
    """Return ``path`` when it is a saved dataset, otherwise its subdirectories."""
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return []
    if os.path.isfile(os.path.join(path, "dataset_info.json")) or os.path.isfile(
        os.path.join(path, "dataset_dict.json")
    ):
        return [path]
    return [
        os.path.join(path, name)
        for name in sorted(os.listdir(path))
        if os.path.isdir(os.path.join(path, name))
    ]


def load_spectrum_dataset(path: str):
    """Load a Hugging Face dataset that stores each spectrum in a ``target`` column."""
    from datasets import DatasetDict, load_from_disk

    loaded = load_from_disk(path)
    if isinstance(loaded, DatasetDict):
        split = "train" if "train" in loaded else next(iter(loaded))
        loaded = loaded[split]
    if "target" not in getattr(loaded, "column_names", []):
        raise ValueError(f"{path} has no 'target' column.")
    return loaded


class TFSpecFormerDataset(IterableDataset):
    """
    Windows of spectrum variates with a continuous mask.

    Each item is a dict with ``target`` of shape ``(T,)`` or ``(C, T)``. Channels of
    one spectrum share a ``group_id``. A batch is padded on the left to the longest
    window and counts variate rows toward ``batch_size``.
    """

    def __init__(
        self,
        inputs: Sequence,
        forecasting_config: TFSpecFormerForecastingConfig,
        training_config: TFSpecFormerTrainingConfig,
        mode: str | DatasetMode = DatasetMode.TRAIN,
        seed: int = 0,
        forced_mode: str | None = None,
    ) -> None:
        super().__init__()
        if not hasattr(inputs, "__len__") or not hasattr(inputs, "__getitem__"):
            raise TypeError("inputs must support len() and indexing.")
        self.inputs = inputs
        self.forecasting = forecasting_config
        self.training = training_config
        self.mode = DatasetMode(mode)
        self.forced_mode = forced_mode
        self.rng = np.random.default_rng(seed)
        self.patch_size = forecasting_config.input_patch_size
        self.batch_size = (
            training_config.batch_size
            if self.mode == DatasetMode.TRAIN
            else training_config.test_batch_size
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self.mode == DatasetMode.TRAIN:
            while True:
                yield self.sample_batch()
        else:
            yield from self._iter_ordered()

    def _series_at(self, index: int) -> torch.Tensor:
        item = self.inputs[index]
        target = item["target"] if isinstance(item, dict) else item
        return as_series(target)

    def _choose_mode(self) -> str:
        if self.forced_mode is not None:
            return self.forced_mode
        if self.mode != DatasetMode.TRAIN:
            return "future"
        probs = self.training.mask_mode_probs
        keys = list(MASK_MODES)
        weights = np.array([probs[key] for key in keys], dtype=np.float64)
        weights = weights / weights.sum()
        return str(self.rng.choice(keys, p=weights))

    def _sample_window(
        self, series: torch.Tensor, mode: str
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        channels, length = series.shape
        patch = self.patch_size
        min_past = self.forecasting.min_past
        visible_floor = min_visible_patches(min_past, patch)

        if mode == "future":
            if self.mode == DatasetMode.TRAIN:
                n_out = int(
                    self.rng.integers(1, self.forecasting.max_output_patches + 1)
                )
            else:
                n_out = min(
                    self.training.eval_output_patches,
                    self.forecasting.max_output_patches,
                )
            horizon = n_out * patch
            if length < min_past + horizon:
                n_out = (length - min_past) // patch
                if n_out < 1:
                    return None
                horizon = n_out * patch
            hist_len = (
                min(self.forecasting.context_length, length - horizon) // patch
            ) * patch
            if hist_len < min_past:
                return None
            total = hist_len + horizon
            if self.mode == DatasetMode.TRAIN:
                start = int(self.rng.integers(0, length - total + 1))
            else:
                start = length - total
            window = series[:, start : start + total].clone()
            hidden = sample_patch_mask(
                channels,
                total // patch,
                "future",
                future_patches=n_out,
                min_visible_patches=hist_len // patch,
                rng=self.rng,
            )
            return window, self._mask_from_patches(hidden)

        max_patches = min(length // patch, self.forecasting.context_length // patch)
        if max_patches <= visible_floor:
            return None
        if self.mode == DatasetMode.TRAIN:
            num_patches = int(self.rng.integers(visible_floor + 1, max_patches + 1))
            total = num_patches * patch
            start = int(self.rng.integers(0, length - total + 1))
        else:
            num_patches = max_patches
            total = num_patches * patch
            start = length - total
        window = series[:, start : start + total].clone()
        hidden = sample_patch_mask(
            channels,
            num_patches,
            mode,
            mask_ratio=self.training.mask_ratio,
            min_span_patches=self.training.min_span_patches,
            max_span_patches=self.training.max_span_patches,
            max_mask_spans=self.training.max_mask_spans,
            min_visible_patches=visible_floor,
            rng=self.rng,
        )
        return window, self._mask_from_patches(hidden)

    def _mask_from_patches(self, hidden: np.ndarray) -> torch.Tensor:
        visible = ~np.repeat(hidden, self.patch_size, axis=1)
        return torch.from_numpy(visible.astype(np.float32))

    def _pack(
        self,
        values: list[torch.Tensor],
        masks: list[torch.Tensor],
        groups: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            "values": left_pad_and_cat(values, float("nan")),
            "input_mask": left_pad_and_cat(masks, 0.0),
            "group_ids": torch.cat(groups, dim=0),
        }

    def _append_group(
        self, buckets, series: torch.Tensor, mode: str, group_id: int
    ) -> bool:
        sampled = self._sample_window(series, mode)
        if sampled is None:
            return False
        values, mask = sampled
        buckets[0].append(values)
        buckets[1].append(mask)
        buckets[2].append(torch.full((values.shape[0],), group_id, dtype=torch.long))
        return True

    def sample_batch(self) -> dict[str, torch.Tensor]:
        if len(self.inputs) == 0:
            raise RuntimeError("Cannot sample a batch from an empty dataset.")
        values, masks, groups = [], [], []
        rows = 0
        group_id = 0
        failures = 0
        while rows < self.batch_size and failures < 64:
            series = self._series_at(int(self.rng.integers(0, len(self.inputs))))
            before = len(values)
            if not self._append_group(
                (values, masks, groups), series, self._choose_mode(), group_id
            ):
                failures += 1
                continue
            added = values[-1].shape[0]
            if before and rows + added > self.batch_size:
                values.pop()
                masks.pop()
                groups.pop()
                failures += 1
                continue
            rows += added
            group_id += 1
            failures = 0
        if not values:
            raise RuntimeError(
                "Could not build a batch. Series are shorter than min_past plus one masked patch."
            )
        return self._pack(values, masks, groups)

    def _iter_ordered(self) -> Iterator[dict[str, torch.Tensor]]:
        values, masks, groups = [], [], []
        rows = 0
        group_id = 0
        for index in range(len(self.inputs)):
            series = self._series_at(index)
            if values and rows + series.shape[0] > self.batch_size:
                yield self._pack(values, masks, groups)
                values, masks, groups = [], [], []
                rows = 0
            if not self._append_group(
                (values, masks, groups), series, "future", group_id
            ):
                continue
            rows += values[-1].shape[0]
            group_id += 1
        if values:
            yield self._pack(values, masks, groups)
