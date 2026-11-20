import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.support import tiny_config
from tfsformer.modeling import TFSpecFormerEncoder, TFSpecFormerModel


class ModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = TFSpecFormerModel(tiny_config()).eval()

    def test_group_mask_blocks_other_groups(self):
        group_ids = torch.tensor([0, 0, 1, 1])
        attention_mask = torch.ones(4, 3)
        mask = TFSpecFormerEncoder.group_mask(group_ids, attention_mask, torch.float32)
        self.assertEqual(tuple(mask.shape), (3, 1, 4, 4))
        self.assertEqual(mask[0, 0, 0, 1].item(), 0.0)
        self.assertLess(mask[0, 0, 0, 2].item(), -1e30)
        attention_mask[2, 1] = 0
        mask = TFSpecFormerEncoder.group_mask(group_ids, attention_mask, torch.float32)
        self.assertLess(mask[1, 0, 3, 2].item(), -1e30)

    def test_forward_loss_shape_and_mask_token_is_visible_to_attention(self):
        values = torch.randn(4, 16)
        input_mask = torch.ones(4, 16)
        input_mask[:, -4:] = 0
        group_ids = torch.tensor([0, 0, 1, 1])
        output = self.model(values, input_mask, group_ids, output_attentions=True)
        self.assertEqual(tuple(output.quantile_preds.shape), (4, 3, 16))
        self.assertTrue(torch.isfinite(output.loss))
        self_weight = output.time_attn_weights[-1][0, 0, -1, -1]
        cross_group = output.group_attn_weights[-1][:, :, 0, 2]
        self.assertGreater(self_weight.item(), 0.0)
        self.assertTrue(torch.all(cross_group < 1e-5))

        self.model.train()
        supervised = self.model(values, input_mask, group_ids)
        supervised.loss.backward()
        self.assertIsNotNone(self.model.encoder.blocks[0].time_attn.attn.memory.grad)

    def test_loss_ignores_visible_targets_and_uses_masked_targets(self):
        pred = torch.randn(2, 3, 8)
        target = torch.randn(2, 8)
        mask = torch.zeros(2, 8)
        mask[:, -3:] = 1
        baseline = self.model._compute_loss(pred, target, mask)
        visible = target.clone()
        visible[:, :5] += 10
        self.assertTrue(
            torch.equal(baseline, self.model._compute_loss(pred, visible, mask))
        )
        hidden = target.clone()
        hidden[:, -1] += 4
        self.assertFalse(
            torch.allclose(baseline, self.model._compute_loss(pred, hidden, mask))
        )

    def test_left_padded_groups_stay_finite(self):
        long = torch.randn(2, 16)
        short = torch.randn(2, 8)
        values = torch.cat(
            (long, torch.cat((torch.full((2, 8), float("nan")), short), dim=-1)), dim=0
        )
        input_mask = torch.ones(4, 16)
        input_mask[:, -4:] = 0
        input_mask[2:, :8] = 0
        output = self.model(values, input_mask, torch.tensor([0, 0, 1, 1]))
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(torch.isfinite(output.quantile_preds).all())

    def test_forecast_returns_the_requested_horizon(self):
        context = torch.randn(2, 30)
        prediction = self.model.forecast(
            context, prediction_length=6, group_ids=torch.tensor([0, 0])
        )
        self.assertEqual(tuple(prediction.shape), (2, 3, 6))
        self.assertTrue(torch.isfinite(prediction).all())
        with self.assertRaises(ValueError):
            self.model.forecast(context, prediction_length=64)

    def test_pretrained_folder_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.model.save_pretrained(temp_dir)
            self.assertTrue(os.path.isfile(os.path.join(temp_dir, "config.json")))
            self.assertTrue(os.path.isfile(os.path.join(temp_dir, "model.safetensors")))
            loaded = TFSpecFormerModel.from_pretrained(temp_dir)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.allclose(value, loaded.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
