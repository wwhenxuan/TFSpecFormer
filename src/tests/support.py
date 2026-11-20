import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfsformer.configuration import TFSpecFormerConfig, TFSpecFormerTrainingConfig


def tiny_config(**kwargs) -> TFSpecFormerConfig:
    forecasting = {
        "context_length": 32,
        "input_patch_size": 4,
        "input_patch_stride": 4,
        "output_patch_size": 4,
        "max_output_patches": 4,
        "quantiles": [0.1, 0.5, 0.9],
        "use_arcsinh": True,
        "min_past": 8,
    }
    forecasting.update(kwargs.pop("forecasting_config", {}))
    params = {
        "d_model": 32,
        "d_kv": 8,
        "d_ff": 64,
        "num_layers": 2,
        "num_heads": 4,
        "dropout_rate": 0.0,
        "qk_norm": False,
        "forecasting_config": forecasting,
    }
    params.update(kwargs)
    return TFSpecFormerConfig(**params)


def tiny_training(**kwargs) -> TFSpecFormerTrainingConfig:
    params = {
        "batch_size": 4,
        "test_batch_size": 4,
        "num_workers": 0,
        "num_training_steps": 1,
        "num_test_steps": 1,
        "num_warmup_steps": 0,
        "mask_ratio": 0.5,
        "min_span_patches": 1,
        "max_span_patches": 2,
        "max_mask_spans": 2,
        "eval_output_patches": 2,
        "report_to": "none",
    }
    params.update(kwargs)
    return TFSpecFormerTrainingConfig(**params)
