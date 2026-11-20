import os
from datetime import datetime

import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from tfsformer.configuration import DatasetMode, TFSpecFormerConfig, TFSpecFormerTrainingConfig
from tfsformer.dataset import TFSpecFormerDataset, list_data_dirs, load_spectrum_dataset
from tfsformer.modeling import TFSpecFormerModel
from tfsformer.utils import count_parameters, plot_forecast_grid

MAX_EVAL_BATCHES = 8


def identity_collate(batch):
    return batch[0]


class TFSpecFormerPreTrainer(object):
    """Accelerate pretrainer: quantile loss, periodic forecast plots, dual checkpoints."""

    def __init__(
        self,
        accelerator: Accelerator,
        model_config: TFSpecFormerConfig,
        training_config: TFSpecFormerTrainingConfig,
        resume_from_checkpoint: str | None = None,
        model_params_only: bool = False,
        train_inputs=None,
        eval_inputs=None,
    ) -> None:
        self.accelerator = accelerator
        self.model_config = model_config
        self.training_config = training_config
        self.forecasting = model_config.forecasting
        self.resume_from_checkpoint = resume_from_checkpoint
        self.model_params_only = model_params_only
        self.train_inputs = train_inputs
        self.eval_inputs = eval_inputs
        self.num_training_steps = training_config.num_training_steps
        self.num_test_steps = training_config.num_test_steps
        self._swanlab = None

        self.model = TFSpecFormerModel(model_config)
        if accelerator.is_main_process:
            total, trainable = count_parameters(self.model)
            accelerator.print(f"Parameters: {trainable:,} trainable / {total:,} total")
        self.optimizer, self.scheduler = self._build_optimizer()
        self._prepare_checkpoint_dir()

    def _prepare_checkpoint_dir(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        holder = [stamp] if self.accelerator.is_main_process else [None]
        if self.accelerator.num_processes > 1:
            broadcast_object_list(holder, from_process=0)
        self.run_name = f"tfsformer-{holder[0]}"
        self.checkpoint_dir = os.path.join(self.training_config.checkpoint_path, self.run_name)
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()

    def _build_optimizer(self):
        config = self.training_config
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
        )
        total = max(config.num_training_steps, 1)
        warmup = min(max(config.num_warmup_steps / total, 1e-4), 0.9)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total,
            pct_start=warmup,
            anneal_strategy=config.anneal_strategy,
        )
        return optimizer, scheduler

    def _resolve_inputs(self, path: str, override):
        if override is not None:
            return override
        source = self._source_for_process(path)
        self.accelerator.print(f"Loading {source}")
        return load_spectrum_dataset(source)

    def _source_for_process(self, path: str) -> str:
        directories = list_data_dirs(path)
        if not directories:
            raise FileNotFoundError(f"No dataset found at {path}.")
        return directories[self.accelerator.process_index % len(directories)]

    def _eval_path(self) -> str:
        data_path = self.training_config.data_path.rstrip(os.sep)
        if os.path.basename(data_path) == "train":
            sibling = os.path.join(os.path.dirname(data_path), "test")
            if list_data_dirs(sibling):
                return sibling
        return data_path

    def _build_loaders(self):
        train_inputs = self._resolve_inputs(self.training_config.data_path, self.train_inputs)
        if self.eval_inputs is not None:
            eval_inputs = self.eval_inputs
        elif self.train_inputs is not None:
            eval_inputs = train_inputs
        else:
            eval_inputs = self._resolve_inputs(self._eval_path(), None)
        train_dataset = TFSpecFormerDataset(
            train_inputs,
            self.forecasting,
            self.training_config,
            mode=DatasetMode.TRAIN,
            seed=self.accelerator.process_index,
        )
        eval_dataset = TFSpecFormerDataset(
            eval_inputs,
            self.forecasting,
            self.training_config,
            mode=DatasetMode.VALIDATION,
            seed=10_000 + self.accelerator.process_index,
        )
        workers = self.training_config.num_workers
        train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=identity_collate, num_workers=workers)
        eval_loader = DataLoader(eval_dataset, batch_size=1, collate_fn=identity_collate, num_workers=0)
        return train_loader, eval_loader

    def _setup_logging(self) -> None:
        if self.training_config.report_to != "swanlab" or not self.accelerator.is_main_process:
            return
        import swanlab

        swanlab.init(
            project=self.training_config.swanlab_project,
            experiment_name=self.run_name,
            config={"model": self.model_config.to_dict(), "training": self.training_config.to_dict()},
            mode=self.training_config.swanlab_mode,
        )
        self._swanlab = swanlab

    def _log(self, step: int, metrics: dict) -> None:
        if self._swanlab is None or not metrics:
            return
        self._swanlab.log(metrics, step=step)

    def _load_model_weights(self, path: str) -> None:
        from safetensors.torch import load_file

        candidates = [
            os.path.join(path, "pretrained", "model.safetensors"),
            os.path.join(path, "model.safetensors"),
        ]
        weight_path = next(candidate for candidate in candidates if os.path.isfile(candidate))
        self.model.load_state_dict(load_file(weight_path), strict=False)
        self.accelerator.print(f"Loaded weights from {weight_path}")

    def _train_step(self, batch) -> torch.Tensor:
        self.model.train()
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(batch["values"], batch["input_mask"], batch["group_ids"])
            if outputs.loss is None:
                raise RuntimeError("Training batch has no masked targets to supervise.")
            self.accelerator.backward(outputs.loss)
            self.optimizer.step()
            self.scheduler.step()
        return outputs.loss.detach()

    @torch.no_grad()
    def _eval_step(self):
        self.model.eval()
        device = self.accelerator.device
        total = torch.zeros((), device=device)
        count = torch.zeros((), device=device)
        preview = None
        for index, batch in enumerate(self.eval_loader):
            if index >= MAX_EVAL_BATCHES:
                break
            outputs = self.model(batch["values"], batch["input_mask"], batch["group_ids"])
            if outputs.loss is None:
                continue
            total = total + outputs.loss.detach()
            count = count + 1
            if preview is None:
                preview = (batch, outputs.quantile_preds.detach())
        total = self.accelerator.reduce(total, reduction="sum")
        count = self.accelerator.reduce(count, reduction="sum")
        self.model.train()
        return (total / count.clamp(min=1)).item(), preview

    def _visualize(self, step: int, preview) -> str | None:
        batch, preds = preview
        values = batch["values"].detach().float().cpu().numpy()
        mask = batch["input_mask"].detach().float().cpu().numpy()
        preds = preds.detach().float().cpu().numpy()
        rows = []
        for index in range(min(4, values.shape[0])):
            observed = values[index] == values[index]
            visible = (mask[index] > 0) & observed
            hidden = (mask[index] <= 0) & observed
            if not hidden.any():
                continue
            rows.append((values[index, visible], values[index, hidden], preds[index][:, hidden]))
        if not rows:
            return None
        figure = plot_forecast_grid(rows, quantiles=self.forecasting.quantiles)
        folder = os.path.join(self.checkpoint_dir, "plots")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"step-{step}.png")
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        return path

    def save_checkpoint(self, step: int, train_loss: float, eval_loss: float) -> None:
        folder = os.path.join(self.checkpoint_dir, f"checkpoint-{step}-{train_loss:.5f}-{eval_loss:.5f}")
        os.makedirs(folder, exist_ok=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(folder, safe_serialization=True)
        if self.accelerator.is_main_process:
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.save_pretrained(os.path.join(folder, "pretrained"))
            self.accelerator.print(f"Saved checkpoint to {folder}")
        self.accelerator.wait_for_everyone()

    def run(self) -> None:
        if self.resume_from_checkpoint and self.model_params_only:
            self._load_model_weights(self.resume_from_checkpoint)

        train_loader, eval_loader = self._build_loaders()
        self.model, self.optimizer, self.scheduler, train_loader, eval_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, train_loader, eval_loader
        )
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        if self.resume_from_checkpoint and not self.model_params_only:
            self.accelerator.load_state(self.resume_from_checkpoint)

        self._setup_logging()
        train_iter = iter(self.train_loader)
        progress = tqdm(range(1, self.num_training_steps + 1), disable=not self.accelerator.is_main_process)
        for step in progress:
            loss = self._train_step(next(train_iter))
            loss_value = self.accelerator.gather(loss).mean().item()
            learning_rate = self.scheduler.get_last_lr()[0]
            if self.accelerator.is_main_process:
                progress.set_postfix(loss=f"{loss_value:.4f}", lr=f"{learning_rate:.2e}")
            self._log(step, {"train/loss": loss_value, "train/learning_rate": learning_rate})

            if step % self.num_test_steps == 0 or step == self.num_training_steps:
                eval_loss, preview = self._eval_step()
                metrics = {"eval/loss": eval_loss}
                if self.accelerator.is_main_process and preview is not None:
                    figure_path = self._visualize(step, preview)
                    if figure_path is not None and self._swanlab is not None:
                        metrics["eval/forecast"] = self._swanlab.Image(figure_path)
                self._log(step, metrics)
                self.save_checkpoint(step, loss_value, eval_loss)

        if self._swanlab is not None:
            self._swanlab.finish()
