"""Trainer for the separate Diffusion Policy + Jump architecture."""

from __future__ import annotations

if __name__ == "__main__":
    import sys

    _ROOT = str(__import__("pathlib").Path(__file__).parent.parent.parent)
    sys.path.append(_ROOT)
    __import__("os").chdir(_ROOT)

import copy
from datetime import timedelta
import hashlib
import json
import math
import os
import pathlib
import random
from typing import Any

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs, broadcast_object_list
import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import tqdm

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.dataset.jump_mvp_chunk_dataset import (
    JumpMVPChunkDataset,
    jump_chunk_collate,
)
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.common.normalizer_config import (
    linear_normalizer_from_spec,
)
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.jump_diffusion_transformer_hybrid_image_policy import (
    JumpDiffusionTransformerHybridImagePolicy,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _sha256(path: str | pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TrainJumpDiffusionTransformerHybridWorkspace(BaseWorkspace):
    """Independent-chunk trainer with only diffusion and boundary losses."""

    architecture_id = "dp_jump"
    include_keys = (
        "global_step",
        "epoch",
        "batch_in_epoch",
        "ema_optimization_step",
    )

    @staticmethod
    def _validate_train_rollout_contract(cfg: OmegaConf) -> None:
        """Reject configurations that reintroduce the mismatched v3 path."""
        if str(cfg.task.get("input_path")) != "cached":
            raise ValueError("DP + Jump v3 requires the cached training feature path")
        if int(cfg.task.history_feature_variant) != 0:
            raise ValueError(
                "DP + Jump rollout parity requires center-crop feature variant 0")
        if not bool(cfg.policy.eval_fixed_crop):
            raise ValueError(
                "DP + Jump rollout parity requires eval_fixed_crop=True")
        if bool(cfg.policy.get("pred_action_steps_only", False)):
            raise ValueError("DP + Jump requires the full vanilla DP horizon")
        if int(cfg.n_obs_steps) >= int(cfg.horizon):
            raise ValueError(
                "n_obs_steps must leave executable Jump dynamics edges")
        if int(cfg.n_action_steps) > (
                int(cfg.horizon) - int(cfg.n_obs_steps) + 1):
            raise ValueError(
                "n_action_steps exceeds the executable full-horizon suffix")
        for name in ("train_dataset", "validation_dataset"):
            dataset = cfg.task[name]
            if int(dataset.n_obs_steps) != int(cfg.n_obs_steps):
                raise ValueError(f"{name}.n_obs_steps must match the policy")
            if int(dataset.cache_action_n_obs_steps) != int(
                    cfg.task.cache_action_n_obs_steps):
                raise ValueError(
                    f"{name}.cache_action_n_obs_steps drifted from the cache")
        runner = cfg.task.get("env_runner")
        if runner is None:
            raise ValueError("DP + Jump v3 requires an executed-segment runner")
        if not bool(runner.get("return_executed_observations", False)):
            raise ValueError(
                "DP + Jump v3 runner must return every executed observation")
        if bool(runner.get("past_action", False)):
            raise ValueError("DP + Jump v3 does not use the legacy past_action path")
        if bool(runner.get("abs_action", False)):
            raise ValueError("DP + Jump training and rollout require abs_action=False")

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        self._validate_train_rollout_contract(cfg)
        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.model: JumpDiffusionTransformerHybridImagePolicy = (
            hydra.utils.instantiate(cfg.policy))
        if not isinstance(self.model, JumpDiffusionTransformerHybridImagePolicy):
            raise TypeError("policy target is not the DP + Jump class")
        self.ema_model: JumpDiffusionTransformerHybridImagePolicy | None = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        self.lr_scheduler = None
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.ema_optimization_step = 0

    def _load_initial_checkpoint(self, path: str, state_key: str) -> None:
        with open(path, "rb") as stream:
            payload = torch.load(
                stream, pickle_module=dill, map_location="cpu")
        try:
            state_dict = payload["state_dicts"][state_key]
        except KeyError as exc:
            raise KeyError(
                f"checkpoint has no state_dicts[{state_key!r}]") from exc
        self.model.load_pretrained_backbone(state_dict)
        if self.ema_model is not None:
            self.ema_model.load_pretrained_backbone(state_dict)

    @staticmethod
    def _loader(dataset, cfg: OmegaConf, *, training: bool) -> DataLoader:
        worker_count = int(cfg.num_workers)
        return DataLoader(
            dataset,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            drop_last=bool(cfg.get("drop_last", training)),
            collate_fn=jump_chunk_collate,
            num_workers=worker_count,
            pin_memory=bool(cfg.pin_memory),
            persistent_workers=(
                bool(cfg.persistent_workers) and worker_count > 0),
            multiprocessing_context="spawn" if worker_count > 0 else None,
        )

    def _write_manifest(self, cfg: OmegaConf) -> None:
        resolved = OmegaConf.to_container(cfg, resolve=True)
        config_text = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
        manifest = {
            "architecture_id": self.architecture_id,
            "workspace_target": cfg._target_,
            "policy_target": cfg.policy._target_,
            "history_input_schema": self.model.history_input_schema,
            "history_time_alignment": self.model.history_time_alignment,
            "jump_token_alignment": self.model.jump_token_alignment,
            "vanilla_transformer_blob": (
                "2948be7064d8ceb4b2fe88e83c313c68efeed6a8"),
            "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
            "base_checkpoint": str(cfg.task.ckpt_path),
            "base_checkpoint_sha256": _sha256(cfg.task.ckpt_path),
            "split_manifest": str(cfg.task.split_artifact),
            "split_manifest_sha256": _sha256(cfg.task.split_artifact),
            "edge_dt_seconds": float(cfg.policy.jump_edge_dt_seconds),
            "horizon": int(cfg.horizon),
            "n_action_steps": int(cfg.n_action_steps),
            "n_obs_steps": int(cfg.n_obs_steps),
            "jump_action_start": int(self.model.jump_action_start),
            "jump_token_count": int(self.model.jump_token_count),
            "jump_edge_count": int(self.model.jump_edge_count),
            "cache_action_n_obs_steps": int(
                cfg.task.cache_action_n_obs_steps),
            "history_feature_variant": int(
                cfg.task.history_feature_variant),
            "rollout_returns_executed_observations": bool(
                cfg.task.env_runner.return_executed_observations),
        }
        path = pathlib.Path(self.output_dir) / "architecture_manifest.json"
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
        if not rows:
            return {}
        keys = set.intersection(*(set(row) for row in rows))
        return {
            key: float(np.mean([row[key] for row in rows]))
            for key in sorted(keys)
        }

    def _validate(
            self,
            *,
            accelerator: Accelerator,
            policy: JumpDiffusionTransformerHybridImagePolicy,
            loader,
            max_steps: int | None,
        ) -> dict[str, float]:
        policy.eval()
        rows = []
        with torch.no_grad():
            for step, batch in enumerate(loader):
                batch = _to_device(batch, accelerator.device)
                policy.compute_loss(batch)
                rows.append(dict(policy.last_loss_metrics))
                if max_steps is not None and step + 1 >= int(max_steps):
                    break
        local = self._mean_metrics(rows)
        result = {}
        for key, value in local.items():
            tensor = torch.tensor(value, device=accelerator.device)
            result[f"val/{key}"] = float(
                accelerator.reduce(tensor, reduction="mean").item())
        return result

    def run(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        if str(cfg.get("architecture_id")) != self.architecture_id:
            raise ValueError("resolved config architecture_id is not dp_jump")
        ddp = DistributedDataParallelKwargs(find_unused_parameters=False)
        timeout = InitProcessGroupKwargs(timeout=timedelta(hours=3))
        accelerator = Accelerator(
            log_with="wandb",
            gradient_accumulation_steps=int(
                cfg.training.gradient_accumulate_every),
            kwargs_handlers=[ddp, timeout],
        )
        logging_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        project = logging_cfg.pop("project")
        accelerator.init_trackers(
            project_name=project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": logging_cfg},
        )
        output = self.output_dir if accelerator.is_main_process else None
        output_list = [output]
        broadcast_object_list(output_list)
        self._output_dir = output_list[0]

        train_dataset = hydra.utils.instantiate(cfg.task.train_dataset)
        validation_dataset = hydra.utils.instantiate(
            cfg.task.validation_dataset)
        if not isinstance(train_dataset, JumpMVPChunkDataset) \
                or not isinstance(validation_dataset, JumpMVPChunkDataset):
            raise TypeError("task datasets must use JumpMVPChunkDataset")
        observed = {
            "train": len(train_dataset.store.episode_ids),
            "validation": len(validation_dataset.store.episode_ids),
        }
        expected = {
            "train": int(cfg.task.expected_train_episodes),
            "validation": int(cfg.task.expected_validation_episodes),
        }
        if observed != expected:
            raise ValueError(f"split/cache intersection drifted: {observed} != {expected}")

        explicit_resume = cfg.training.get("resume_checkpoint")
        if explicit_resume:
            path = pathlib.Path(str(explicit_resume)).expanduser().resolve(strict=True)
            self.load_checkpoint(path=path)
        elif cfg.training.resume and self.get_checkpoint_path().is_file():
            self.load_checkpoint(path=self.get_checkpoint_path())
        else:
            self._load_initial_checkpoint(
                str(cfg.task.ckpt_path), str(cfg.task.ckpt_state_key))

        normalizer = linear_normalizer_from_spec(cfg.task.cache_normalizer)
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)
        if accelerator.is_main_process:
            self._write_manifest(cfg)
        accelerator.wait_for_everyone()

        train_loader = self._loader(
            train_dataset, cfg.dataloader, training=True)
        validation_loader = self._loader(
            validation_dataset, cfg.val_dataloader, training=False)
        steps_per_epoch = math.ceil(
            len(train_loader) / int(cfg.training.gradient_accumulate_every))
        total_steps = steps_per_epoch * int(cfg.training.num_epochs)
        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=int(cfg.training.lr_warmup_steps),
            num_training_steps=total_steps,
            last_epoch=self.global_step - 1,
        )
        train_loader, validation_loader, self.model, self.optimizer, \
            self.lr_scheduler = accelerator.prepare(
                train_loader,
                validation_loader,
                self.model,
                self.optimizer,
                self.lr_scheduler,
            )
        if self.ema_model is not None:
            self.ema_model.to(accelerator.device)
        ema = None
        if self.ema_model is not None:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
            ema.optimization_step = self.ema_optimization_step

        policy = accelerator.unwrap_model(self.model)
        audit = policy.trainable_parameter_audit()
        accelerator.print(
            f"DP + Jump parameters: trainable="
            f"{audit['trainable_parameter_count']:,}, "
            f"frozen={audit['frozen_parameter_count']:,}")
        topk = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        self.optimizer.zero_grad()

        if bool(cfg.training.get("validation_only", False)):
            validation_policy = self.ema_model or policy
            metrics = self._validate(
                accelerator=accelerator,
                policy=validation_policy,
                loader=validation_loader,
                max_steps=cfg.training.max_val_steps,
            )
            accelerator.log(metrics, step=self.global_step)
            accelerator.end_training()
            return

        with JsonLogger(log_path) as logger:
            stop = False
            while self.epoch < int(cfg.training.num_epochs):
                train_dataset.set_epoch(self.epoch)
                self.model.train()
                progress = min(
                    self.global_step / max(total_steps - 1, 1), 1.0)
                policy.set_training_progress(progress)
                if self.ema_model is not None:
                    self.ema_model.set_training_progress(progress)
                iterator = tqdm.tqdm(
                    train_loader,
                    disable=not accelerator.is_local_main_process,
                    desc=f"DP + Jump epoch {self.epoch}",
                    mininterval=float(cfg.training.tqdm_interval_sec),
                )
                for batch_index, batch in enumerate(iterator):
                    if batch_index < self.batch_in_epoch:
                        continue
                    progress = min(
                        self.global_step / max(total_steps - 1, 1), 1.0)
                    policy.set_training_progress(progress)
                    if self.ema_model is not None:
                        self.ema_model.set_training_progress(progress)
                    batch = _to_device(batch, accelerator.device)
                    with accelerator.accumulate(self.model):
                        loss = self.model(batch)
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            if cfg.training.max_grad_norm is not None:
                                accelerator.clip_grad_norm_(
                                    self.model.parameters(),
                                    float(cfg.training.max_grad_norm),
                                )
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad()
                            if ema is not None:
                                ema.step(accelerator.unwrap_model(self.model))
                                self.ema_optimization_step = ema.optimization_step
                    self.global_step += 1
                    self.batch_in_epoch = batch_index + 1
                    metrics = {
                        f"train/{key}": value
                        for key, value in policy.last_loss_metrics.items()
                    }
                    metrics.update({
                        "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "train/progress": progress,
                        "epoch": self.epoch,
                        "global_step": self.global_step,
                    })
                    accelerator.log(metrics, step=self.global_step)
                    if accelerator.is_main_process:
                        logger.log(metrics)
                    iterator.set_postfix(loss=float(loss.detach()))
                    maximum = cfg.training.get("max_train_steps")
                    if maximum is not None and self.global_step >= int(maximum):
                        stop = True
                        break
                if stop:
                    break

                self.epoch += 1
                self.batch_in_epoch = 0
                epoch_metrics = {}
                if bool(cfg.training.validation_enabled) \
                        and self.epoch % int(cfg.training.val_every) == 0:
                    validation_dataset.set_epoch(self.epoch)
                    validation_policy = self.ema_model or policy
                    epoch_metrics = self._validate(
                        accelerator=accelerator,
                        policy=validation_policy,
                        loader=validation_loader,
                        max_steps=cfg.training.max_val_steps,
                    )
                    accelerator.log(epoch_metrics, step=self.global_step)
                    if accelerator.is_main_process:
                        logger.log({
                            **epoch_metrics,
                            "epoch": self.epoch,
                            "global_step": self.global_step,
                        })
                if self.epoch % int(cfg.training.checkpoint_every) == 0:
                    if accelerator.is_main_process:
                        wrapped = self.model
                        self.model = accelerator.unwrap_model(self.model)
                        self.save_checkpoint()
                        if "val/loss_total" in epoch_metrics:
                            path = topk.get_ckpt_path({
                                "epoch": self.epoch,
                                "val_loss_total": epoch_metrics["val/loss_total"],
                            })
                            if path is not None:
                                self.save_checkpoint(path=path)
                        self.model = wrapped
                    accelerator.wait_for_everyone()
        accelerator.end_training()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name="train_jump_diffusion_transformer_hybrid_workspace",
)
def main(cfg):
    TrainJumpDiffusionTransformerHybridWorkspace(cfg).run()


if __name__ == "__main__":
    main()
