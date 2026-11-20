import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.support import tiny_training
from tfsformer.configuration import TFSpecFormerForecastingConfig
from tfsformer.dataset import TFSpecFormerDataset
from tfsformer.utils import hidden_runs, sample_patch_mask


def forecasting() -> TFSpecFormerForecastingConfig:
    return TFSpecFormerForecastingConfig(
        context_length=64,
        input_patch_size=16,
        input_patch_stride=16,
        output_patch_size=16,
        max_output_patches=2,
        quantiles=[0.1, 0.5, 0.9],
        min_past=32,
    )


class MaskSamplerTest(unittest.TestCase):
    def test_future_mask_is_a_shared_suffix(self):
        mask = sample_patch_mask(
            3,
            10,
            "future",
            future_patches=4,
            min_visible_patches=2,
            rng=np.random.default_rng(0),
        )
        self.assertTrue(np.all(mask[:, -4:]))
        self.assertFalse(np.any(mask[:, :-4]))
        self.assertTrue(np.all(mask == mask[:1]))

    def test_group_mask_is_shared_and_contiguous(self):
        mask = sample_patch_mask(
            4,
            16,
            "group",
            mask_ratio=0.25,
            min_span_patches=1,
            max_span_patches=4,
            max_mask_spans=4,
            min_visible_patches=2,
            rng=np.random.default_rng(0),
        )
        self.assertTrue(np.all(mask == mask[:1]))
        runs = hidden_runs(mask[0])
        self.assertGreater(len(runs), 0)
        self.assertLessEqual(len(runs), 4)
        self.assertTrue(all(1 <= run <= 4 for run in runs))
        self.assertGreaterEqual((~mask[0]).sum(), 2)

    def test_random_masks_can_differ_and_staggered_masks_do_differ(self):
        differed = False
        for seed in range(8):
            mask = sample_patch_mask(
                4,
                16,
                "random",
                mask_ratio=0.25,
                min_span_patches=1,
                max_span_patches=4,
                max_mask_spans=4,
                min_visible_patches=2,
                rng=np.random.default_rng(seed),
            )
            for row in mask:
                runs = hidden_runs(row)
                self.assertLessEqual(len(runs), 4)
                self.assertTrue(all(1 <= run <= 4 for run in runs))
                self.assertGreaterEqual((~row).sum(), 2)
            differed = differed or not np.all(mask == mask[:1])
        self.assertTrue(differed)

        staggered = sample_patch_mask(
            4,
            16,
            "staggered",
            mask_ratio=0.25,
            min_span_patches=1,
            max_span_patches=4,
            max_mask_spans=4,
            min_visible_patches=2,
            rng=np.random.default_rng(0),
        )
        self.assertFalse(np.array_equal(staggered[0], staggered[1]))
        self.assertTrue(np.array_equal(staggered[0], staggered[2]))


class DatasetMaskTest(unittest.TestCase):
    def setUp(self):
        self.series = torch.randn(4, 160)
        self.forecasting = forecasting()

    def _batch(self, mode, batch_size=4):
        training = tiny_training(batch_size=batch_size, test_batch_size=batch_size)
        dataset = TFSpecFormerDataset(
            [{"target": self.series}],
            self.forecasting,
            training,
            mode="train",
            seed=0,
            forced_mode=mode,
        )
        return dataset.sample_batch()

    def test_future_validation_window_keeps_a_visible_prefix(self):
        training = tiny_training()
        dataset = TFSpecFormerDataset(
            [{"target": self.series}],
            self.forecasting,
            training,
            mode="validation",
            seed=0,
        )
        batch = dataset.sample_batch()
        mask = batch["input_mask"]
        self.assertEqual(tuple(batch["values"].shape), tuple(mask.shape))
        self.assertTrue(torch.all(mask[:, :-32] == 1))
        self.assertTrue(torch.all(mask[:, -32:] == 0))
        self.assertTrue(torch.equal(mask[0], mask[1]))
        self.assertGreaterEqual(int(mask[0].sum()), 32)
        self.assertEqual(batch["group_ids"].tolist(), [0, 0, 0, 0])

    def test_group_and_staggered_batches_follow_their_patterns(self):
        group = self._batch("group")
        self.assertTrue(torch.equal(group["input_mask"][0], group["input_mask"][-1]))
        self.assertGreaterEqual(int(group["input_mask"][0].sum()), 32)
        self.assertTrue(torch.any(group["input_mask"][0] == 0))

        staggered = self._batch("staggered")
        self.assertFalse(
            torch.equal(staggered["input_mask"][0], staggered["input_mask"][1])
        )
        self.assertGreaterEqual(int(staggered["input_mask"].sum(dim=-1).min()), 32)

    def test_two_draws_get_different_group_ids(self):
        batch = self._batch("future", batch_size=8)
        self.assertEqual(batch["group_ids"].tolist(), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_univariate_target_is_one_channel(self):
        training = tiny_training(batch_size=1, test_batch_size=1)
        dataset = TFSpecFormerDataset(
            [{"target": torch.randn(160)}],
            self.forecasting,
            training,
            mode="validation",
            seed=0,
        )
        batch = dataset.sample_batch()
        self.assertEqual(batch["values"].shape[0], 1)
        self.assertEqual(batch["group_ids"].tolist(), [0])


if __name__ == "__main__":
    unittest.main()
