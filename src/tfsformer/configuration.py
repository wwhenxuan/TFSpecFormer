from enum import Enum
from typing import Dict, List, Literal, Optional

from transformers.configuration_utils import PretrainedConfig


class DatasetMode(str, Enum):
    """Dataset split names shared by the loader and the trainer."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


MASK_MODES = ("future", "random", "group", "staggered")

# Width and depth presets used to check scaling. d_model == num_heads * d_kv.
MODEL_SIZES = {
    "tiny": {"d_model": 256, "d_kv": 64, "d_ff": 768, "num_layers": 4, "num_heads": 4},
    "small": {
        "d_model": 384,
        "d_kv": 64,
        "d_ff": 1152,
        "num_layers": 6,
        "num_heads": 6,
    },
    "base": {"d_model": 512, "d_kv": 64, "d_ff": 1536, "num_layers": 6, "num_heads": 8},
    "large": {
        "d_model": 768,
        "d_kv": 64,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
}


class TFSpecFormerForecastingConfig(object):
    """Patch, context, and quantile settings for continuous masked pretraining."""

    def __init__(
        self,
        context_length: int = 1024,
        input_patch_size: int = 16,
        input_patch_stride: int = 16,
        output_patch_size: int = 16,
        max_output_patches: int = 16,
        quantiles: List[float] = None,
        use_arcsinh: bool = True,
        min_past: int = 32,
        time_encoding_scale: Optional[int] = None,
    ) -> None:
        self.context_length = context_length
        self.input_patch_size = input_patch_size
        self.input_patch_stride = input_patch_stride
        self.output_patch_size = output_patch_size
        self.max_output_patches = max_output_patches
        self.quantiles = (
            list(quantiles)
            if quantiles is not None
            else [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        )
        self.use_arcsinh = use_arcsinh
        self.min_past = min_past
        self.time_encoding_scale = (
            context_length if time_encoding_scale is None else time_encoding_scale
        )

    def to_dict(self) -> Dict:
        return {
            "context_length": self.context_length,
            "input_patch_size": self.input_patch_size,
            "input_patch_stride": self.input_patch_stride,
            "output_patch_size": self.output_patch_size,
            "max_output_patches": self.max_output_patches,
            "quantiles": list(self.quantiles),
            "use_arcsinh": self.use_arcsinh,
            "min_past": self.min_past,
            "time_encoding_scale": self.time_encoding_scale,
        }


class TFSpecFormerConfig(PretrainedConfig):
    """
    Architecture config for TFSpecFormer.

    ``forecasting_config`` is stored as a plain dict so ``save_pretrained`` writes
    valid JSON. Use :attr:`forecasting` when the object form is easier.
    """

    model_type = "tfsformer"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "head_dim": "d_kv",
    }

    def __init__(
        self,
        d_model: int = 512,
        d_kv: int = 64,
        d_ff: int = 1536,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        attn_output_gate: Optional[str] = "headwise",
        qk_norm: bool = False,
        layer_norm_epsilon: float = 1e-6,
        initializer_factor: float = 0.05,
        feed_forward_proj: str = "gelu",
        rope_theta: float = 10000.0,
        pad_token_id: int = 0,
        attn_implementation: Literal["eager", "sdpa"] | None = None,
        forecasting_config: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        if d_model != num_heads * d_kv:
            raise ValueError(
                f"d_model ({d_model}) must equal num_heads * d_kv ({num_heads * d_kv})."
            )
        if attn_output_gate not in ("headwise", "elementwise", "none"):
            raise ValueError(f"attn_output_gate {attn_output_gate} is not supported.")
        if "gated" in feed_forward_proj:
            raise ValueError("gated feed-forward activations are not supported.")

        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attn_output_gate = attn_output_gate
        self.qk_norm = qk_norm
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.dense_act_fn = feed_forward_proj
        self.is_gated_act = False
        self.rope_theta = rope_theta

        if isinstance(forecasting_config, TFSpecFormerForecastingConfig):
            forecasting_config = forecasting_config.to_dict()
        elif forecasting_config is None:
            forecasting_config = TFSpecFormerForecastingConfig().to_dict()
        else:
            forecasting_config = TFSpecFormerForecastingConfig(
                **forecasting_config
            ).to_dict()
        self.forecasting_config = forecasting_config

        attn_implementation = attn_implementation or "sdpa"
        if attn_implementation not in ("eager", "sdpa"):
            raise ValueError(
                f"attn_implementation {attn_implementation} is not supported."
            )

        kwargs.pop("is_encoder_decoder", None)
        kwargs.pop("eos_token_id", None)
        super().__init__(
            pad_token_id=pad_token_id,
            is_encoder_decoder=False,
            attn_implementation=attn_implementation,
            **kwargs,
        )

    @classmethod
    def from_size(cls, size: str, **kwargs) -> "TFSpecFormerConfig":
        if size not in MODEL_SIZES:
            raise ValueError(
                f"Unknown size {size!r}. Expected one of {list(MODEL_SIZES)}."
            )
        params = dict(MODEL_SIZES[size])
        params.update(kwargs)
        return cls(**params)

    @property
    def forecasting(self) -> TFSpecFormerForecastingConfig:
        config = self.forecasting_config
        if isinstance(config, TFSpecFormerForecastingConfig):
            return config
        return TFSpecFormerForecastingConfig(**config)


class TFSpecFormerTrainingConfig(object):
    """Optimization, masking, logging, and data-path settings."""

    def __init__(
        self,
        batch_size: int = 4096,
        shuffle: bool = True,
        test_batch_size: int = 1024,
        num_workers: int = 0,
        mixed_precision: str = "fp16",
        gradient_accumulation_steps: int = 1,
        optimizer: str = "AdamW",
        learning_rate: float = 4e-4,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        num_training_steps: int = 500000,
        num_test_steps: int = 500,
        num_warmup_steps: int = 1280,
        anneal_strategy: str = "cos",
        mask_mode_probs: Optional[Dict[str, float]] = None,
        mask_ratio: float = 0.25,
        min_span_patches: int = 1,
        max_span_patches: int = 4,
        max_mask_spans: int = 4,
        eval_output_patches: int = 4,
        report_to: str = "swanlab",
        swanlab_project: str = "TFSpecFormer",
        swanlab_mode: str = "cloud",
        checkpoint_path: str = "./checkpoints",
        data_path: str = "data/Large-Spectrum-Prediction-Dataset-split/train",
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.num_warmup_steps = num_warmup_steps
        self.anneal_strategy = anneal_strategy
        self.num_training_steps = num_training_steps
        self.num_test_steps = num_test_steps
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.train_data_class = "all"
        self.eval_context_lengths = [128, 192, 256, 336, 512]

        probs = dict(
            mask_mode_probs
            or {"future": 0.4, "random": 0.2, "group": 0.2, "staggered": 0.2}
        )
        missing = set(MASK_MODES) - set(probs)
        unknown = set(probs) - set(MASK_MODES)
        if missing or unknown:
            raise ValueError(
                f"mask_mode_probs must contain exactly {MASK_MODES}, found {sorted(probs)}."
            )
        if any(value < 0 for value in probs.values()) or sum(probs.values()) <= 0:
            raise ValueError(
                "mask_mode_probs values must be non-negative and sum to a positive number."
            )
        self.mask_mode_probs = probs
        self.mask_ratio = mask_ratio
        self.min_span_patches = min_span_patches
        self.max_span_patches = max_span_patches
        self.max_mask_spans = max_mask_spans
        self.eval_output_patches = eval_output_patches
        self.report_to = report_to
        self.swanlab_project = swanlab_project
        self.swanlab_mode = swanlab_mode

    def to_dict(self) -> Dict:
        return dict(self.__dict__)
