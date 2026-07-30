import copy
import json
import os
import pathlib
import random
from datetime import timedelta

import hydra
import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    broadcast_object_list,
)
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.semantic_lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.semantic_diffusion_transformer_hybrid_image_policy import (
    SemanticDiffusionTransformerHybridImagePolicy,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainSemanticDiffusionTransformerHybridWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch", "train_batch_in_epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: SemanticDiffusionTransformerHybridImagePolicy = (
            hydra.utils.instantiate(cfg.policy)
        )
        checkpoint_path = cfg.task.get("ckpt_path")
        if checkpoint_path:
            self._initialize_from_ema_checkpoint(str(checkpoint_path))

        self.ema_model = (
            copy.deepcopy(self.model) if cfg.training.use_ema else None
        )
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        self.global_step = 0
        self.epoch = 0
        self.train_batch_in_epoch = 0

    def _initialize_from_ema_checkpoint(self, path: str) -> None:
        try:
            payload = torch.load(
                path, map_location="cpu", weights_only=False
            )
        except TypeError:
            # PyTorch < 2.0 does not expose weights_only.
            payload = torch.load(path, map_location="cpu")
        try:
            state = payload["state_dicts"]["ema_model"]
        except KeyError as error:
            raise KeyError(f"EMA model missing from checkpoint: {path}") from error
        # LinearNormalizer is populated after the target dataset is built.
        # Load the pretrained network here and set an equivalent normalizer in run().
        network_state = {
            key: value for key, value in state.items()
            if not key.startswith("normalizer.")
        }
        incompatible = self.model.load_state_dict(network_state, strict=False)
        allowed_prefixes = (
            "semantic_pool_score.",
            "semantic_heads.",
            "direct_boundary_heads.",
            "bifurcation_calibrator.",
            "subtask_class_weight",
            "stage_class_weight",
            "boundary_pos_weight",
        )
        illegal_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_prefixes)
        ]
        if illegal_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "checkpoint base mismatch: "
                f"missing={illegal_missing}, unexpected={incompatible.unexpected_keys}"
            )
        print(
            f"initialized EMA base from {path}; "
            f"new auxiliary keys={len(incompatible.missing_keys)}"
        )

    def _set_auxiliary_weights(self, cfg: OmegaConf) -> None:
        metadata = cfg.task.get("semantic_metadata")
        if not metadata or not self.model.semantic_enabled:
            return
        values = json.loads(
            pathlib.Path(str(metadata.loss_weights_path)).read_text()
        )["splits"][str(metadata.split)]
        kwargs = {
            "subtask_class_weight": values["subtask_class_weight"],
            "stage_class_weight": values["stage_class_weight"],
            "boundary_pos_weight": [
                values["boundary_subtask_future_h8_pos_weight"],
                values["boundary_stage_future_h8_pos_weight"],
                values["boundary_any_future_h8_pos_weight"],
            ],
        }
        self.model.set_auxiliary_loss_weights(**kwargs)
        if self.ema_model is not None:
            self.ema_model.set_auxiliary_loss_weights(**kwargs)

    @staticmethod
    def _scalar_log(losses: dict[str, torch.Tensor]) -> dict[str, float]:
        return {
            name: float(value.detach().mean().cpu())
            for name, value in losses.items()
            if torch.is_tensor(value) and value.numel() == 1
        }

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        timeout = InitProcessGroupKwargs(timeout=timedelta(hours=3))
        ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            log_with="wandb",
            split_batches=True,
            kwargs_handlers=[timeout, ddp],
        )
        if cfg.training.debug:
            cfg.logging.mode = "disabled"

        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        project = wandb_cfg.pop("project")
        if wandb_cfg.get("tags") is not None:
            wandb_cfg["tags"] = [str(tag) for tag in wandb_cfg["tags"]]
        accelerator.init_trackers(
            project_name=project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg},
        )

        if accelerator.is_main_process:
            shared_output = self.output_dir
        else:
            shared_output = None
        values = [shared_output]
        broadcast_object_list(values)
        self._output_dir = values[0]

        if cfg.training.resume:
            latest = self.get_checkpoint_path()
            if latest.is_file():
                self.load_checkpoint(path=latest, include_keys=self.include_keys)
                print(f"resumed checkpoint: {latest} at step={self.global_step}")

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)
        self._set_auxiliary_weights(cfg)
        dataloader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.epoch)
        train_generator = torch.Generator()
        train_generator.manual_seed(int(cfg.training.seed) + self.epoch)
        train_dataloader = DataLoader(
            dataset, generator=train_generator, **dataloader_cfg
        )
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(
            val_dataset,
            **OmegaConf.to_container(cfg.val_dataloader, resolve=True),
        )

        total_steps = int(cfg.training.total_optimizer_steps)
        checkpoint_steps = {
            int(value) for value in cfg.training.get("checkpoint_steps", [])
        }
        scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=int(cfg.training.lr_warmup_steps),
            num_training_steps=total_steps,
            last_epoch=self.global_step - 1,
        )
        ema: EMAModel | None = None
        if self.ema_model is not None:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

            ema.optimization_step = self.global_step
        train_dataloader, val_dataloader, self.model, self.optimizer, scheduler = (
            accelerator.prepare(
                train_dataloader,
                val_dataloader,
                self.model,
                self.optimizer,
                scheduler,
            )
        )
        device = accelerator.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        if cfg.training.debug:
            cfg.training.total_optimizer_steps = min(
                int(cfg.training.total_optimizer_steps), 20
            )
            cfg.training.validation_every_steps = 10
            cfg.training.checkpoint_every_steps = 10

        # Skip only the consumed part of the first resumed epoch. Keep the
        # full loader intact so all later epochs traverse the full dataset.
        if self.train_batch_in_epoch > 0:
            resumed_dataloader = accelerator.skip_first_batches(
                train_dataloader, self.train_batch_in_epoch
            )
            train_iterator = iter(resumed_dataloader)
        else:
            train_iterator = iter(train_dataloader)
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as logger:
            while self.global_step < int(cfg.training.total_optimizer_steps):
                self.model.train()
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    self.epoch += 1
                    self.train_batch_in_epoch = 0
                    if hasattr(dataset, "set_epoch"):
                        dataset.set_epoch(self.epoch)
                    train_generator.manual_seed(int(cfg.training.seed) + self.epoch)
                    train_iterator = iter(train_dataloader)
                    batch = next(train_iterator)
                batch = dict_apply(
                    batch, lambda x: x.to(device, non_blocking=True)
                )

                aux_scale = min(
                    1.0,
                    (self.global_step + 1)
                    / max(1, int(cfg.training.aux_warmup_steps)),
                )
                unwrapped = accelerator.unwrap_model(self.model)
                run_bifurcation = (
                    unwrapped.bifurcation_enabled
                    and self.global_step
                    % int(cfg.training.bifurcation_every_steps) == 0
                )
                losses = self.model(
                    batch,
                    auxiliary_scale=aux_scale,
                    run_bifurcation=run_bifurcation,
                )
                total = losses["loss_total"]
                if (
                    run_bifurcation
                    and bool(cfg.training.compensate_bifurcation_frequency)
                ):
                    interval = int(cfg.training.bifurcation_every_steps)
                    total = total + (interval - 1) * losses["loss_bifurcation"]
                    losses["loss_bifurcation_frequency_scaled"] = (
                        interval * losses["loss_bifurcation"]
                    )

                self.optimizer.zero_grad(set_to_none=True)
                accelerator.backward(total)
                if cfg.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        float(cfg.training.max_grad_norm),
                    )
                self.optimizer.step()
                scheduler.step()
                if ema is not None:
                    ema.step(accelerator.unwrap_model(self.model))

                step_log = self._scalar_log(losses)
                step_log.update(
                    global_step=self.global_step,
                    epoch=self.epoch,
                    aux_scale=aux_scale,
                    lr=float(scheduler.get_last_lr()[0]),
                )
                accelerator.log(step_log, step=self.global_step)
                if accelerator.is_main_process:
                    logger.log(step_log)

                self.global_step += 1
                self.train_batch_in_epoch += 1
                if (
                    self.global_step
                    % int(cfg.training.validation_every_steps) == 0
                ):
                    self._validate(
                        accelerator, val_dataloader, cfg, logger
                    )
                checkpoint_interval = int(cfg.training.checkpoint_every_steps)
                if (
                    (
                        checkpoint_interval > 0
                        and self.global_step % checkpoint_interval == 0
                    )
                    or self.global_step in checkpoint_steps
                ):
                    self._save(accelerator)
        self._save(accelerator)
        accelerator.end_training()

    def _validate(self, accelerator, dataloader, cfg, logger):
        self.model.eval()
        sums: dict[str, list[torch.Tensor]] = {}
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader):
                if (
                    cfg.training.max_val_steps is not None
                    and batch_index >= int(cfg.training.max_val_steps)
                ):
                    break
                batch = dict_apply(
                    batch, lambda x: x.to(accelerator.device, non_blocking=True)
                )
                losses = self.model(batch)
                unwrapped = accelerator.unwrap_model(self.model)
                if (
                    unwrapped.bifurcation_enabled
                    and batch_index
                    < int(cfg.training.max_bifurcation_val_batches)
                ):
                    if "obs_feature" in batch:
                        probe_size = min(
                            unwrapped.probe_batch_size_per_device,
                            batch["obs_feature"].shape[0],
                        )
                        prediction = unwrapped.predict_bifurcation_features(
                            batch["obs_feature"][:probe_size]
                        )
                    else:
                        probe_size = min(
                            unwrapped.probe_batch_size_per_device,
                            next(iter(batch["obs"].values())).shape[0],
                        )
                        probe_obs = {
                            key: value[:probe_size]
                            for key, value in batch["obs"].items()
                        }
                        prediction = unwrapped.predict_bifurcation(probe_obs)
                    valid = batch["boundary_target"]["valid"][:probe_size].bool()
                    label = batch["boundary_target"]["any_future_h8"][:probe_size]
                    if torch.any(valid):
                        probability = prediction["probability"][valid]
                        losses["boundary_b_eval_bce"] = (
                            torch.nn.functional.binary_cross_entropy(
                                probability, label[valid].to(probability.dtype)
                            )
                        )
                        losses["boundary_b_eval_accuracy"] = (
                            (probability >= 0.5) == (label[valid] >= 0.5)
                        ).float().mean()
                for name, value in losses.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        gathered = accelerator.gather_for_metrics(
                            value.detach().reshape(1)
                        )
                        sums.setdefault(name, []).append(gathered.cpu())
        result = {
            f"val_{name}": float(torch.cat(values).mean())
            for name, values in sums.items() if values
        }
        result["global_step"] = self.global_step
        accelerator.log(result, step=self.global_step)
        if accelerator.is_main_process:
            logger.log(result)
        self.model.train()

    def _save(self, accelerator):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            checkpoint_dir = pathlib.Path(self.output_dir) / "checkpoints"
            step_path = checkpoint_dir / f"step_{self.global_step:06d}.ckpt"
            wrapped = self.model
            self.model = accelerator.unwrap_model(self.model)
            try:
                if not step_path.is_file():
                    self.save_checkpoint(path=step_path, use_thread=False)
                latest_path = checkpoint_dir / "latest.ckpt"
                temporary_link = checkpoint_dir / (
                    f".latest.{os.getpid()}.tmp"
                )
                if temporary_link.exists() or temporary_link.is_symlink():
                    temporary_link.unlink()
                temporary_link.symlink_to(step_path.name)
                os.replace(temporary_link, latest_path)
                print(f"saved checkpoint: {step_path}")
            finally:
                self.model = wrapped
        accelerator.wait_for_everyone()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent / "config"),
    config_name="train_semantic_diffusion_transformer_bs192",
)
def main(cfg):
    workspace = TrainSemanticDiffusionTransformerHybridWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
