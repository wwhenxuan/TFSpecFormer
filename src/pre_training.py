import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from accelerate import Accelerator

from tfsformer.configuration import TFSpecFormerConfig, TFSpecFormerTrainingConfig
from tfsformer.trainer import TFSpecFormerPreTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain TFSpecFormer")
    parser.add_argument("--size", default="base", choices=["tiny", "small", "base", "large"])
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--model-params-only", action="store_true")
    parser.add_argument("--num-training-steps", type=int, default=None)
    parser.add_argument("--num-test-steps", type=int, default=None)
    args = parser.parse_args()

    training_config = TFSpecFormerTrainingConfig()
    if args.data_path is not None:
        training_config.data_path = args.data_path
    if args.checkpoint_path is not None:
        training_config.checkpoint_path = args.checkpoint_path
    if args.num_training_steps is not None:
        training_config.num_training_steps = args.num_training_steps
    if args.num_test_steps is not None:
        training_config.num_test_steps = args.num_test_steps

    model_config = TFSpecFormerConfig.from_size(args.size)
    accelerator = Accelerator(
        mixed_precision=training_config.mixed_precision,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        split_batches=True,
    )
    trainer = TFSpecFormerPreTrainer(
        accelerator,
        model_config,
        training_config,
        resume_from_checkpoint=args.resume,
        model_params_only=args.model_params_only,
    )
    trainer.run()


if __name__ == "__main__":
    main()
