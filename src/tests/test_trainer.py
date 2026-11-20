import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from accelerate import Accelerator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.support import tiny_config, tiny_training
from tfsformer.trainer import TFSpecFormerPreTrainer


class TrainerSmokeTest(unittest.TestCase):
    def test_one_step_saves_optimizer_state_and_pretrained_files(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as temp_dir:
            training = tiny_training(checkpoint_path=temp_dir, batch_size=2, test_batch_size=2)
            inputs = [{"target": torch.randn(2, 64)}]
            accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=1)
            trainer = TFSpecFormerPreTrainer(
                accelerator,
                tiny_config(),
                training,
                train_inputs=inputs,
                eval_inputs=inputs,
            )
            trainer.run()
            pretrained = list(Path(temp_dir).glob("**/pretrained/config.json"))
            weights = list(Path(temp_dir).glob("**/pretrained/model.safetensors"))
            plots = list(Path(temp_dir).glob("**/plots/step-1.png"))
            self.assertTrue(pretrained)
            self.assertTrue(weights)
            self.assertTrue(plots)
            self.assertTrue(any(path.name == "optimizer.bin" or path.suffix == ".safetensors" for path in Path(temp_dir).rglob("*") if "pretrained" not in path.parts))


if __name__ == "__main__":
    unittest.main()
