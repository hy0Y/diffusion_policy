if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import json
import copy
from datetime import timedelta
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
from diffusion_policy.dataset.p1_mvp_group_dataset import (
    P1MVPCachedGroupDataset,
    P1MVPRawGroupDataset,
    p1_group_collate,
)
from diffusion_policy.dataset.p1_mvp_sampler import (
    AnnotationCandidateStore,
    CATEGORY_QUOTA,
    CandidateStore,
    EpisodeBundleBatchSampler,
    EpisodeBundleSampler,
    partition_epoch_for_ddp,
)
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.p1_diffusion_transformer_hybrid_image_policy import (
    P1DiffusionTransformerHybridImagePolicy,
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


class TrainP1MVPWorkspace(BaseWorkspace):
    """Episode-complete trainer shared by Baseline, Oracle-Gate-Symbol, and Main."""

    include_keys = (
        "global_step",
        "epoch",
        "batch_in_epoch",
        "episodes_seen",
        "ema_optimization_step",
    )

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        seed = int(cfg.training.seed)
        torch.random.default_generator.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: P1DiffusionTransformerHybridImagePolicy = (
            hydra.utils.instantiate(cfg.policy))
        self.ema_model: P1DiffusionTransformerHybridImagePolicy | None = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        self.lr_scheduler = None
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.episodes_seen = 0
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

    def _make_datasets(self, cfg: OmegaConf):
        include_history = cfg.policy.p1_mode != "baseline"
        if cfg.task.input_path == "raw":
            base_dataset = hydra.utils.instantiate(cfg.task.base_dataset)
            train_dataset = P1MVPRawGroupDataset(
                base_dataset,
                cfg.task.split_artifact,
                split="train",
                history_feature_variant=cfg.task.history_feature_variant,
                include_history=include_history,
                task=cfg.task.source_task,
                annotation_root=cfg.task.annotation_root,
                cache_root=cfg.task.cache_root,
                canonical_sha256=cfg.task.split_manifest_sha256,
            )
            validation_dataset = P1MVPRawGroupDataset(
                base_dataset,
                cfg.task.split_artifact,
                split="validation",
                history_feature_variant=cfg.task.history_feature_variant,
                include_history=include_history,
                task=cfg.task.source_task,
                annotation_root=cfg.task.annotation_root,
                cache_root=cfg.task.cache_root,
                canonical_sha256=cfg.task.split_manifest_sha256,
            )
        elif cfg.task.input_path == "cached":
            if not include_history:
                raise ValueError(
                    "P1-Baseline-DiT-Unfreeze must use raw observations so its observation "
                    "encoder is fine-tuned")
            train_dataset = P1MVPCachedGroupDataset(
                cfg.task.split_artifact,
                split="train",
                feature_variant=cfg.task.history_feature_variant,
                include_history=True,
                task=cfg.task.source_task,
                annotation_root=cfg.task.annotation_root,
                cache_root=cfg.task.cache_root,
                canonical_sha256=cfg.task.split_manifest_sha256,
            )
            validation_dataset = P1MVPCachedGroupDataset(
                cfg.task.split_artifact,
                split="validation",
                feature_variant=cfg.task.history_feature_variant,
                include_history=True,
                task=cfg.task.source_task,
                annotation_root=cfg.task.annotation_root,
                cache_root=cfg.task.cache_root,
                canonical_sha256=cfg.task.split_manifest_sha256,
            )
            base_dataset = train_dataset
        else:
            raise ValueError("task.input_path must be raw or cached")
        return base_dataset, train_dataset, validation_dataset

    @staticmethod
    def _make_candidate_store(
            *, split: str, split_artifact: str, candidate_artifact: str | None,
            task: str, annotation_root: str, cache_root: str,
            canonical_sha256: str | None):
        if candidate_artifact:
            store = CandidateStore(candidate_artifact)
            if split != "train":
                raise ValueError("NPZ candidate artifact is currently train-only")
            return store
        return AnnotationCandidateStore(
            split_artifact,
            split=split,
            task=task,
            annotation_root=annotation_root,
            cache_root=cache_root,
            canonical_sha256=canonical_sha256,
        )

    @staticmethod
    def _rank_schedule(global_schedule, accelerator: Accelerator, episodes_per_rank: int):
        if accelerator.num_processes == 1:
            return global_schedule
        return partition_epoch_for_ddp(
            global_schedule,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            episodes_per_rank=episodes_per_rank,
        )

    @staticmethod
    def _loader(dataset, schedule, cfg: OmegaConf, *, start_batch: int = 0):
        num_workers = int(cfg.num_workers)
        return DataLoader(
            dataset,
            batch_sampler=EpisodeBundleBatchSampler(
                schedule, start_batch=start_batch),
            collate_fn=p1_group_collate,
            num_workers=num_workers,
            pin_memory=bool(cfg.pin_memory),
            persistent_workers=(
                bool(cfg.persistent_workers) and num_workers > 0),
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )

    @staticmethod
    def _step_seed(base_seed: int, epoch: int, batch_index: int, rank: int) -> int:
        return int(base_seed + epoch * 1_000_003 + batch_index * 1009 + rank)

    @staticmethod
    def _seed_step(seed: int) -> None:
        torch.random.default_generator.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    @staticmethod
    def _reduce_metrics(
            accelerator: Accelerator,
            metrics: dict[str, float],
            *,
            local_weight: Tensor,
            global_weight: Tensor,
        ) -> dict[str, float]:
        result: dict[str, float] = {}
        denominator = global_weight.clamp_min(1)
        for key, value in metrics.items():
            scalar = torch.as_tensor(
                value, device=accelerator.device, dtype=torch.float64)
            if key.endswith("_count"):
                reduced = accelerator.reduce(scalar, reduction="sum")
            else:
                reduced = accelerator.reduce(
                    scalar * local_weight.to(dtype=torch.float64),
                    reduction="sum",
                ) / denominator.to(dtype=torch.float64)
            result[key] = float(reduced.item())
        return result

    @staticmethod
    def _ddp_loss_scale(
            local_valid_groups: Tensor,
            global_valid_groups: Tensor,
            world_size: int,
        ) -> Tensor:
        """Compensate DDP's rank mean to obtain a real-group global mean."""
        if world_size < 1:
            raise ValueError("world_size must be positive")
        if bool(torch.any(local_valid_groups < 0)):
            raise ValueError("local_valid_groups must be non-negative")
        if bool(torch.any(global_valid_groups <= 0)):
            raise ValueError("global_valid_groups must be positive")
        return local_valid_groups * world_size / global_valid_groups

    def _save_training_checkpoint(
            self, accelerator: Accelerator, *, path: str | pathlib.Path | None = None):
        wrapped_model = self.model
        self.model = accelerator.unwrap_model(self.model)
        try:
            return self.save_checkpoint(path=path, use_thread=False)
        finally:
            self.model = wrapped_model

    @staticmethod
    def _platform_signal_writer():
        if not os.environ.get("SEM_BIF_RUN_DIR"):
            return None
        from sem_bif_experiment_runtime import SignalWriter
        return SignalWriter.from_environment()

    def _publish_platform_metrics(
            self,
            metrics: dict[str, float],
            *,
            training_step: int,
            epoch: float,
        ) -> None:
        writer = self._platform_signal_writer()
        if writer is None:
            return
        definitions = writer.definitions()
        for source_key, value in metrics.items():
            signal_id = source_key.split("/", maxsplit=1)[-1]
            definition = definitions.get(signal_id)
            if definition is None or definition.signal_type != "metric":
                continue
            writer.metric(
                signal_id,
                value,
                training_step,
                epoch=epoch,
                context={"source_key": source_key},
            )

    def _register_platform_checkpoint(
            self,
            path: str | pathlib.Path,
            *,
            aliases: list[str],
            metrics: dict[str, float],
            checkpoint_id: str | None = None,
        ) -> None:
        if not os.environ.get("SEM_BIF_RUN_DIR"):
            return
        from sem_bif_experiment_runtime import CheckpointRegistry
        registry = CheckpointRegistry.from_environment()
        resolved_id = checkpoint_id or f"step_{self.global_step:08d}"
        registry.register(
            pathlib.Path(path),
            resolved_id,
            self.global_step,
            aliases=aliases,
            metrics=metrics,
            metadata={
                "p1_mode": self.cfg.policy.p1_mode,
            },
        )

    @staticmethod
    def _platform_probe_row(
            row: dict[str, float | int]) -> dict[str, float | int]:
        payload = dict(row)
        payload.setdefault("value", payload["probability"])
        return payload

    def _publish_platform_probe(
            self,
            rows: list[dict[str, float | int]],
        ) -> None:
        writer = self._platform_signal_writer()
        if writer is None or not rows:
            return
        definition = writer.definitions().get("boundary_probability_trace")
        if definition is None or definition.signal_type != "probe":
            return
        artifact_dir = pathlib.Path(self.output_dir) / "platform_probes"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        source = artifact_dir / (
            f"boundary_probability_step_{self.global_step:08d}.jsonl")
        ordered = sorted(
            rows,
            key=lambda row: (row["episode_id"], row["physical_time"]),
        )
        with source.open("w", encoding="utf-8") as handle:
            for row in ordered:
                payload = self._platform_probe_row(row)
                handle.write(
                    json.dumps(payload, sort_keys=True, separators=(",", ":"))
                    + "\n")
        writer.publish_probe(
            "boundary_probability_trace",
            source,
            f"boundary_probability_step_{self.global_step:08d}",
            self.global_step,
            rows=len(ordered),
            scope={
                "p1_mode": self.cfg.policy.p1_mode,
                "validation_seed": int(self.cfg.validation.seed),
            },
            artifact_format="jsonl",
            content_type="application/x-ndjson",
        )




    def _validation(
            self,
            *,
            accelerator: Accelerator,
            policy: P1DiffusionTransformerHybridImagePolicy,
            dataset,
            sampler: EpisodeBundleSampler,
            cfg: OmegaConf,
        ) -> tuple[dict[str, float], list[dict[str, float | int]]]:
        schedule = sampler.sample_epoch(epoch=0)
        rank_schedule = self._rank_schedule(
            schedule, accelerator, cfg.sampling.episodes_per_rank)
        loader = self._loader(dataset, rank_schedule, cfg.val_dataloader)
        totals: dict[str, float] = {}
        local_probe_rows: list[dict[str, float | int]] = []
        count = 0.0
        policy.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if cfg.training.max_val_steps is not None \
                        and batch_index >= cfg.training.max_val_steps:
                    break
                self._seed_step(self._step_seed(
                    cfg.validation.seed, 0, batch_index,
                    accelerator.process_index))
                batch = _to_device(batch, accelerator.device)
                policy.compute_loss(batch)
                local_probe_rows.extend(policy.last_probe_rows)
                real_groups = float(batch["episode_valid"].sum().item())
                for key, value in policy.last_loss_metrics.items():
                    if not key.endswith("_count"):
                        totals[key] = (
                            totals.get(key, 0.0) + float(value) * real_groups)
                count += real_groups
        packed = torch.tensor(
            [count] + [totals.get(key, 0.0) for key in sorted(totals)],
            device=accelerator.device,
            dtype=torch.float64,
        )
        packed = accelerator.reduce(packed, reduction="sum")
        keys = sorted(totals)
        global_count = max(int(packed[0].item()), 1)
        metrics = {
            f"val/{key}": float(packed[index + 1].item() / global_count)
            for index, key in enumerate(keys)
        }
        if accelerator.num_processes > 1:
            gathered: list[list[dict[str, float | int]]] = [
                [] for _ in range(accelerator.num_processes)
            ]
            torch.distributed.all_gather_object(
                gathered, local_probe_rows)
            probe_rows = [
                row
                for rank_rows in gathered
                for row in rank_rows
            ]
        else:
            probe_rows = local_probe_rows
        return metrics, probe_rows

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        # Keep W&B local files with this Hydra run even outside expctl.
        # Under expctl, self.output_dir is the platform Run directory.
        wandb_dir = pathlib.Path(self.output_dir) / "tracking" / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(wandb_dir)
        configured_groups = int(cfg.sampling.groups_per_episode)
        required_groups = sum(CATEGORY_QUOTA.values())
        if configured_groups != required_groups:
            raise ValueError(
                f"sampling.groups_per_episode={configured_groups} but the locked "
                f"quota produces {required_groups}")
        if int(cfg.training.gradient_accumulate_every) != 1:
            raise ValueError(
                "P1 MVP currently requires gradient_accumulate_every=1: "
                "checkpoints do not serialize partially accumulated gradients")
        ddp = DistributedDataParallelKwargs(find_unused_parameters=False)
        timeout = InitProcessGroupKwargs(timeout=timedelta(hours=3))
        accelerator = Accelerator(
            log_with="wandb",
            gradient_accumulation_steps=cfg.training.gradient_accumulate_every,
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

        base_dataset, train_dataset, validation_dataset = self._make_datasets(cfg)
        normalizer = base_dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        train_store = self._make_candidate_store(
            split="train",
            split_artifact=cfg.task.split_artifact,
            candidate_artifact=cfg.task.train_candidate_artifact,
            task=cfg.task.source_task,
            annotation_root=cfg.task.annotation_root,
            cache_root=cfg.task.cache_root,
            canonical_sha256=cfg.task.split_manifest_sha256,
        )
        validation_store = self._make_candidate_store(
            split="validation",
            split_artifact=cfg.task.split_artifact,
            candidate_artifact=None,
            task=cfg.task.source_task,
            annotation_root=cfg.task.annotation_root,
            cache_root=cfg.task.cache_root,
            canonical_sha256=cfg.task.split_manifest_sha256,
        )
        expected_counts = {
            "train": int(cfg.task.expected_train_episodes),
            "validation": int(cfg.task.expected_validation_episodes),
        }
        observed_counts = {
            "train": len(train_store.episode_ids),
            "validation": len(validation_store.episode_ids),
        }
        if observed_counts != expected_counts:
            raise ValueError(
                f"P1 MVP split/cache intersection drifted: {observed_counts} != "
                f"{expected_counts}")
        train_sampler = EpisodeBundleSampler(
            train_store,
            seed=cfg.training.seed,
            natural_done_policy=cfg.sampling.natural_done_policy,
            episodes_per_batch=cfg.sampling.episodes_per_rank,
        )
        validation_sampler = EpisodeBundleSampler(
            validation_store,
            seed=cfg.validation.seed,
            natural_done_policy=cfg.sampling.natural_done_policy,
            episodes_per_batch=cfg.sampling.episodes_per_rank,
        )
        steps_per_epoch = len(self._rank_schedule(
            train_sampler.sample_epoch(epoch=0),
            accelerator,
            cfg.sampling.episodes_per_rank,
        ).batches)
        total_steps = math.ceil(
            steps_per_epoch * cfg.training.num_epochs
            / cfg.training.gradient_accumulate_every)
        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=total_steps,
            last_epoch=self.global_step - 1,
        )

        resume_path = self.get_checkpoint_path()
        if cfg.training.resume and resume_path.is_file():
            accelerator.print(f"Resuming P1 MVP from {resume_path}")
            self.load_checkpoint(path=resume_path)
        else:
            accelerator.print(
                f"Initializing P1 MVP from {cfg.task.ckpt_path}"
                f"[{cfg.task.ckpt_state_key}]")
            self._load_initial_checkpoint(
                cfg.task.ckpt_path, cfg.task.ckpt_state_key)

        ema: EMAModel | None = None
        if self.ema_model is not None:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
            ema.optimization_step = self.ema_optimization_step

        self.model, self.optimizer, self.lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler)
        if self.ema_model is not None:
            self.ema_model.to(accelerator.device)

        audit = accelerator.unwrap_model(self.model).trainable_parameter_audit()
        accelerator.print(
            f"P1 parameter audit: mode={audit['mode']} "
            f"trainable={audit['trainable_parameter_count']:,} "
            f"frozen={audit['frozen_parameter_count']:,}")
        total_episode_budget = len(train_store.episode_ids) * cfg.training.num_epochs
        topk = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        self.optimizer.zero_grad()

        if bool(cfg.training.get("validation_only", False)):
            with JsonLogger(log_path) as json_logger:
                validation_policy = accelerator.unwrap_model(self.model)
                if self.ema_model is not None:
                    validation_policy = self.ema_model
                validation_metrics, probe_rows = self._validation(
                    accelerator=accelerator,
                    policy=validation_policy,
                    dataset=validation_dataset,
                    sampler=validation_sampler,
                    cfg=cfg,
                )
                accelerator.log(validation_metrics, step=self.global_step)
                if accelerator.is_main_process:
                    json_logger.log({
                        **validation_metrics,
                        "epoch": self.epoch,
                        "global_step": self.global_step,
                    })
                    self._publish_platform_metrics(
                        validation_metrics,
                        training_step=self.global_step,
                        epoch=float(self.epoch),
                    )
                    self._publish_platform_probe(probe_rows)
            accelerator.end_training()
            return

        with JsonLogger(log_path) as json_logger:
            stop_after_step = False
            while self.epoch < cfg.training.num_epochs:
                global_schedule = train_sampler.sample_epoch(epoch=self.epoch)
                rank_schedule = self._rank_schedule(
                    global_schedule,
                    accelerator,
                    cfg.sampling.episodes_per_rank,
                )
                loader = self._loader(
                    train_dataset,
                    rank_schedule,
                    cfg.dataloader,
                    start_batch=self.batch_in_epoch,
                )
                self.model.train()
                iterator = tqdm.tqdm(
                    loader,
                    disable=not accelerator.is_local_main_process,
                    desc=f"P1 epoch {self.epoch}",
                    mininterval=cfg.training.tqdm_interval_sec,
                )
                start_batch = self.batch_in_epoch
                for offset, batch in enumerate(iterator):
                    batch_index = start_batch + offset
                    self._seed_step(self._step_seed(
                        cfg.training.seed,
                        self.epoch,
                        batch_index,
                        accelerator.process_index,
                    ))
                    batch = _to_device(batch, accelerator.device)
                    local_real_groups = batch["episode_valid"].sum(
                        dtype=torch.float32)
                    global_real_groups = accelerator.reduce(
                        local_real_groups, reduction="sum")
                    if int(global_real_groups.item()) % configured_groups != 0:
                        raise RuntimeError(
                            "global valid-group count is not episode-complete")
                    global_real_episodes = global_real_groups / configured_groups
                    progress = min(
                        self.episodes_seen / max(total_episode_budget, 1), 1.0)
                    policy = accelerator.unwrap_model(self.model)
                    policy.set_training_progress(progress)
                    if self.ema_model is not None:
                        self.ema_model.set_training_progress(progress)

                    with accelerator.accumulate(self.model):
                        loss = self.model(batch)
                        loss_scale = self._ddp_loss_scale(
                            local_real_groups,
                            global_real_groups,
                            accelerator.num_processes,
                        )
                        accelerator.backward(loss * loss_scale)
                        if accelerator.sync_gradients:
                            if cfg.training.max_grad_norm is not None:
                                accelerator.clip_grad_norm_(
                                    self.model.parameters(),
                                    cfg.training.max_grad_norm,
                                )
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad()
                            if ema is not None:
                                ema.step(accelerator.unwrap_model(self.model))
                                self.ema_optimization_step = ema.optimization_step

                    self.episodes_seen += int(global_real_episodes.item())
                    self.batch_in_epoch = batch_index + 1
                    self.global_step += 1
                    reduced = self._reduce_metrics(
                        accelerator,
                        policy.last_loss_metrics,
                        local_weight=local_real_groups,
                        global_weight=global_real_groups,
                    )
                    metrics = {
                        f"train/{key}": value
                        for key, value in reduced.items()
                    }
                    metrics.update({
                        "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "train/episodes_seen": self.episodes_seen,
                        "train/progress": progress,
                        "epoch": self.epoch,
                        "global_step": self.global_step,
                    })
                    accelerator.log(metrics, step=self.global_step)
                    if accelerator.is_main_process:
                        json_logger.log(metrics)
                        self._publish_platform_metrics(
                            metrics,
                            training_step=self.global_step,
                            epoch=float(self.epoch),
                        )
                    iterator.set_postfix(loss=float(loss.detach()))

                    max_train_steps = cfg.training.get("max_train_steps")
                    if cfg.training.checkpoint_every_steps is not None \
                            and self.global_step % cfg.training.checkpoint_every_steps == 0:
                        if accelerator.is_main_process:
                            checkpoint_path = self._save_training_checkpoint(accelerator)
                            aliases = (
                                ["last"]
                                if max_train_steps is not None
                                and self.global_step >= int(max_train_steps)
                                else []
                            )
                            self._register_platform_checkpoint(
                                checkpoint_path,
                                aliases=aliases,
                                metrics=reduced,
                            )
                        accelerator.wait_for_everyone()

                    if max_train_steps is not None \
                            and self.global_step >= int(max_train_steps):
                        stop_after_step = True
                        break

                if stop_after_step:
                    break

                self.epoch += 1
                self.batch_in_epoch = 0
                epoch_metrics: dict[str, float] = {}
                if self.epoch % cfg.training.val_every == 0:
                    validation_policy = accelerator.unwrap_model(self.model)
                    if self.ema_model is not None:
                        validation_policy = self.ema_model
                    validation_metrics, probe_rows = self._validation(
                        accelerator=accelerator,
                        policy=validation_policy,
                        dataset=validation_dataset,
                        sampler=validation_sampler,
                        cfg=cfg,
                    )
                    epoch_metrics.update(validation_metrics)
                    accelerator.log(epoch_metrics, step=self.global_step)
                    if accelerator.is_main_process:
                        json_logger.log({
                            **epoch_metrics,
                            "epoch": self.epoch,
                            "global_step": self.global_step,
                        })
                        self._publish_platform_metrics(
                            validation_metrics,
                            training_step=self.global_step,
                            epoch=float(self.epoch),
                        )
                        self._publish_platform_probe(probe_rows)

                if self.epoch % cfg.training.checkpoint_every == 0:
                    if accelerator.is_main_process:
                        checkpoint_path = self._save_training_checkpoint(accelerator)
                        aliases = ["last"] if self.epoch >= cfg.training.num_epochs else []
                        self._register_platform_checkpoint(
                            checkpoint_path,
                            aliases=aliases,
                            metrics=epoch_metrics,
                        )
                        if "val/loss_total" in epoch_metrics:
                            path = topk.get_ckpt_path({
                                "epoch": self.epoch,
                                "val_loss_total": epoch_metrics["val/loss_total"],
                            })
                            if path is not None:
                                self._save_training_checkpoint(
                                    accelerator, path=path)
                    accelerator.wait_for_everyone()

        accelerator.end_training()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name="train_p1_mvp",
)
def main(cfg):
    TrainP1MVPWorkspace(cfg).run()


if __name__ == "__main__":
    main()
