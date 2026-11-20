import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfsformer.configuration import (
    MASK_MODES,
    MODEL_SIZES,
    TFSpecFormerConfig,
    TFSpecFormerTrainingConfig,
)


class ConfigTest(unittest.TestCase):
    def test_sizes_match_the_scaling_ladder(self):
        for name, spec in MODEL_SIZES.items():
            config = TFSpecFormerConfig.from_size(name)
            for key, value in spec.items():
                self.assertEqual(getattr(config, key), value)
            self.assertEqual(config.d_model, config.num_heads * config.d_kv)

    def test_default_constructor_is_base(self):
        config = TFSpecFormerConfig()
        self.assertEqual(config.d_model, MODEL_SIZES["base"]["d_model"])
        self.assertEqual(config.num_layers, MODEL_SIZES["base"]["num_layers"])

    def test_hidden_width_must_match_heads(self):
        with self.assertRaises(ValueError):
            TFSpecFormerConfig(d_model=32, num_heads=4, d_kv=16)

    def test_forecasting_config_round_trips_through_json(self):
        config = TFSpecFormerConfig.from_size(
            "tiny",
            forecasting_config={"context_length": 64, "min_past": 16},
        )
        restored = TFSpecFormerConfig.from_dict(json.loads(config.to_json_string()))
        self.assertEqual(restored.d_model, 256)
        self.assertEqual(restored.forecasting.context_length, 64)
        self.assertEqual(restored.forecasting.min_past, 16)
        self.assertEqual(restored.forecasting.input_patch_size, 16)
        self.assertEqual(restored.forecasting.max_output_patches, 16)

    def test_mask_probabilities_cover_every_mode(self):
        config = TFSpecFormerTrainingConfig()
        self.assertEqual(set(config.mask_mode_probs), set(MASK_MODES))
        self.assertAlmostEqual(sum(config.mask_mode_probs.values()), 1.0)
        self.assertEqual(config.mask_ratio, 0.25)
        self.assertEqual(config.eval_output_patches, 4)
        self.assertEqual(config.report_to, "swanlab")
        with self.assertRaises(ValueError):
            TFSpecFormerTrainingConfig(mask_mode_probs={"future": 1.0})


if __name__ == "__main__":
    unittest.main()
