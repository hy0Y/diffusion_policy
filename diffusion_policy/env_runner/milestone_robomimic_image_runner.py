"""RoboCasa image runner with policy-independent physical milestone traces.

The environment construction and rollout loop are derived from
``diffusion_policy/env_runner/robomimic_image_runner.py`` at the provenance
recorded below. This additive runner exists because the original constructor
does not expose an inner-wrapper injection seam.
"""

from __future__ import annotations

import collections
import math
import os
import pathlib
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import dill
import gymnasium as gym
import numpy as np
import torch
import tqdm
import wandb
import wandb.sdk.data_types.video as wv
from omegaconf import OmegaConf

import robocasa  # noqa: F401 - registers RoboCasa Gym environments
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.robomimic.milestone_seeded_robomimic_image_wrapper import (
    MilestoneSeededRobomimicImageWrapper,
)
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env_runner.executed_segment import collate_executed_segments
from diffusion_policy.env_runner.robomimic_image_runner import (
    summarize_rollout_outcome,
)
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.milestone_multistep_wrapper import (
    MilestoneMultiStepWrapper,
)
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from robocasa_milestones.serialization import (
    write_episode_artifacts,
    write_manifest,
)


COPIED_FROM_PATH = "diffusion_policy/env_runner/robomimic_image_runner.py"
COPIED_FROM_GIT_SHA = "11ab9e573138921beefc2f78c06a9c7d0ec881e2"
COPIED_FROM_BLOB_SHA = "c4d40b56d1abfd353398841fb91a83aadfffb114"
MILESTONE_RUNNER_SCHEMA_VERSION = "milestone_robomimic_image_runner_v2"


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def create_env(split: str, env_name: str, seed: int | None = None):
    return gym.make(f"robocasa/{env_name}", split=split, seed=seed)


class MilestoneRobomimicImageRunner(BaseImageRunner):
    """Common evaluator-aware runner for vanilla DP and DP + Jump."""

    def __init__(
        self,
        output_dir,
        dataset_path,
        shape_meta: dict,
        n_train=10,
        n_train_vis=3,
        train_start_idx=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        render_obs_key="agentview_image",
        fps=10,
        crf=22,
        past_action=False,
        return_executed_observations=False,
        abs_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
        env_kwargs=None,
        milestone_evaluation: Mapping[str, Any] | None = None,
    ):
        super().__init__(output_dir)
        del train_start_idx, n_train_vis, dataset_path
        if n_envs is None:
            n_envs = n_train + n_test
        if past_action and return_executed_observations:
            raise ValueError(
                "past_action and executed-segment history are mutually exclusive"
            )
        if int(n_train) != 0:
            raise ValueError("training envs are not supported by this runner")

        self.env_kwargs = (
            OmegaConf.to_container(env_kwargs, resolve=True)
            if env_kwargs is not None
            else {}
        )
        if "env_name" not in self.env_kwargs or "split" not in self.env_kwargs:
            raise ValueError("env_kwargs must contain env_name and split")
        env_name = str(self.env_kwargs["env_name"])
        reset_mode = self.env_kwargs.get("reset_mode", "generated_seed")
        if reset_mode not in {"generated_seed", "demonstration_state"}:
            raise ValueError(f"unsupported rollout reset mode: {reset_mode}")
        episode_root = self.env_kwargs.get("episode_root")
        initial_state_index = int(self.env_kwargs.get("initial_state_index", 0))
        if reset_mode == "demonstration_state" and not episode_root:
            raise ValueError("demonstration_state rollout requires episode_root")

        self.milestone_evaluation = validate_milestone_evaluation_contract(
            milestone_evaluation,
            env_name=env_name,
            max_steps=max_steps,
        )

        rotation_transformer = None
        if abs_action:
            rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // int(fps), 1)

        def build_wrapped_env(robocasa_env):
            return MilestoneMultiStepWrapper(
                VideoRecordingWrapper(
                    MilestoneSeededRobomimicImageWrapper(
                        env=robocasa_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                        milestone_evaluation=self.milestone_evaluation,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                return_executed_observations=return_executed_observations,
            )

        def env_fn(env_i):
            env_seed = self.env_kwargs.get("seed")
            if env_seed is not None:
                env_seed = int(env_seed) + int(env_i)
            robocasa_env = create_env(
                split=str(self.env_kwargs["split"]),
                env_name=env_name,
                seed=env_seed,
            )
            return build_wrapped_env(robocasa_env)

        def dummy_env_fn():
            robocasa_env = create_env(
                split=str(self.env_kwargs["split"]),
                env_name=env_name,
                seed=self.env_kwargs.get("seed"),
            )
            return build_wrapped_env(robocasa_env)

        env_fns = [partial(env_fn, env_i) for env_i in range(int(n_envs))]
        env_seeds: list[int] = []
        env_prefixes: list[str] = []
        env_init_fn_dills: list[bytes] = []
        for i in range(int(n_test)):
            seed = int(test_start_seed) + i
            enable_render = i < int(n_test_vis)

            def init_fn(
                env,
                seed=seed,
                reset_mode=reset_mode,
                episode_root=episode_root,
                initial_state_index=initial_state_index,
                enable_render=enable_render,
            ):
                assert isinstance(env, MilestoneMultiStepWrapper)
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    env.env.file_path = str(filename)
                assert isinstance(
                    env.env.env, MilestoneSeededRobomimicImageWrapper
                )
                if reset_mode == "demonstration_state":
                    env.env.env.set_reset_snapshot(
                        episode_root, state_index=initial_state_index
                    )
                else:
                    env.env.env.set_reset_seed(seed)

            env_seeds.append(seed)
            env_prefixes.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        self.env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixes
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.past_action = bool(past_action)
        self.return_executed_observations = bool(return_executed_observations)
        self.max_steps = int(max_steps)
        self.rotation_transformer = rotation_transformer
        self.abs_action = bool(abs_action)
        self.tqdm_interval_sec = float(tqdm_interval_sec)
        self.reset_mode = reset_mode
        self.episode_root = episode_root
        self.initial_state_index = initial_state_index
        self.last_rollout_outcomes: list[dict[str, Any]] = []
        self.last_milestone_outcomes: list[dict[str, Any]] = []
        self.last_milestone_manifest: dict[str, Any] | None = None

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        env = self.env
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_milestone_outcomes: list[dict[str, Any] | None] = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            global_slice = slice(start, end)
            active_count = end - start
            local_slice = slice(0, active_count)
            init_fns = self.env_init_fn_dills[global_slice]
            padding = n_envs - len(init_fns)
            if padding > 0:
                init_fns.extend([self.env_init_fn_dills[0]] * padding)
            env.call_each(
                "run_dill_function", args_list=[(value,) for value in init_fns]
            )

            obs = env.reset()
            past_action = None
            executed_segment = None
            policy.reset()
            env_name = self.env_kwargs["env_name"]
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {env_name}Image {chunk_idx + 1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )
            done = False
            while not done:
                np_obs_dict = dict(obs)
                if executed_segment is not None:
                    np_obs_dict.update(executed_segment)
                if self.past_action and past_action is not None:
                    np_obs_dict["past_action"] = past_action[
                        :, -(self.n_obs_steps - 1) :
                    ].astype(np.float32)
                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda value: torch.from_numpy(value).to(device=device),
                )
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                np_action_dict = dict_apply(
                    action_dict, lambda value: value.detach().to("cpu").numpy()
                )
                action = np_action_dict["action"]
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("policy produced NaN or Inf action")
                env_action = (
                    self.undo_transform_action(action) if self.abs_action else action
                )
                obs, reward, env_done, info = env.step(env_action)
                done = np.all(env_done) or np.all(
                    [this_info["success"][0] for this_info in info]
                )
                if self.return_executed_observations:
                    executed_segment = collate_executed_segments(info, action)
                past_action = action
                pbar.update(action.shape[1])
            pbar.close()

            round_rewards = env.call("get_attr", "reward")
            termination_metadata = [
                summarize_rollout_outcome(rewards, self.max_steps)
                for rewards in round_rewards
            ]
            round_milestones = env.call_each(
                "finalize_milestone_episode",
                args_list=[(metadata,) for metadata in termination_metadata],
            )
            all_video_paths[global_slice] = env.render()[local_slice]
            all_rewards[global_slice] = round_rewards[local_slice]
            all_milestone_outcomes[global_slice] = round_milestones[local_slice]

        self.last_rollout_outcomes = [
            summarize_rollout_outcome(all_rewards[i], self.max_steps)
            for i in range(n_inits)
        ]
        if any(outcome is None for outcome in all_milestone_outcomes):
            raise RuntimeError("an active rollout did not produce a milestone outcome")
        typed_outcomes = [
            outcome for outcome in all_milestone_outcomes if outcome is not None
        ]
        self.last_milestone_outcomes = [
            dict(outcome["summary"]) for outcome in typed_outcomes
        ]

        episode_artifacts = []
        for episode_index, outcome in enumerate(typed_outcomes):
            episode_artifacts.append(
                write_episode_artifacts(
                    self.output_dir,
                    episode_index=episode_index,
                    summary=outcome["summary"],
                    trace=outcome["trace"],
                )
            )
        self.last_milestone_manifest = write_manifest(
            self.output_dir,
            episodes=episode_artifacts,
            source_provenance={
                "runner_schema_version": MILESTONE_RUNNER_SCHEMA_VERSION,
                "copied_from_path": COPIED_FROM_PATH,
                "copied_from_git_sha": COPIED_FROM_GIT_SHA,
                "copied_from_blob_sha": COPIED_FROM_BLOB_SHA,
            },
        )

        # All completed episodes were finalized above, so this reset cannot
        # erase an uncollected trace.
        _ = env.reset()
        _ = env.call_each(
            "finalize_milestone_episode",
            args_list=[
                ({"termination_reason": "post_run_video_buffer_clear"},)
                for _ in range(n_envs)
            ],
        )
        return self._build_log_data(all_rewards, all_video_paths)

    def _build_log_data(self, all_rewards, all_video_paths):
        n_inits = len(all_rewards)
        max_rewards = collections.defaultdict(list)
        log_data: dict[str, Any] = {
            "evaluation/reset_mode": self.reset_mode,
            "evaluation/environment_seeds": (
                list(self.env_seeds)
                if self.reset_mode == "generated_seed"
                else []
            ),
            "evaluation/explicit_seed_reset": self.reset_mode
            == "generated_seed",
        }
        if self.reset_mode == "demonstration_state":
            log_data["evaluation/episode_root"] = str(self.episode_root)
            log_data["evaluation/initial_state_index"] = self.initial_state_index

        success_rate = sum(
            np.max(all_rewards[i]) > 0 for i in range(n_inits)
        ) / n_inits
        for i in range(n_inits):
            result_id = (
                self.env_seeds[i]
                if self.reset_mode == "generated_seed"
                else f"{Path(self.episode_root).name}_{i}"
            )
            prefix = self.env_prefixs[i]
            max_reward = float(np.max(all_rewards[i]))
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f"sim_max_reward_{result_id}"] = max_reward
            video_path = all_video_paths[i]
            if video_path is not None:
                log_data[prefix + f"sim_video_{result_id}"] = wandb.Video(
                    video_path, format="mp4"
                )

            summary = self.last_milestone_outcomes[i]
            episode_key = f"milestone/episode_{i:06d}"
            log_data[f"{episode_key}/normalized_valid_progress"] = float(
                summary["normalized_valid_progress"]
            )
            log_data[f"{episode_key}/normalized_linear_prefix"] = float(
                summary["normalized_linear_prefix"]
            )
            log_data[f"{episode_key}/prefix_auc_fixed_horizon"] = float(
                summary["prefix_auc_fixed_horizon"]
            )
            log_data[f"{episode_key}/invalid_event_count"] = int(
                summary["invalid_event_count"]
            )
            log_data[f"{episode_key}/official_success_contract_disagreement"] = bool(
                summary["official_success_contract_disagreement"]
            )
            lever = summary["spec_final_metadata"]
            log_data[f"{episode_key}/max_lever_position_tracked_slot"] = lever[
                "max_lever_position_tracked_slot"
            ]
            log_data[f"{episode_key}/max_lever_position_any_slot"] = float(
                lever["max_lever_position_any_slot"]
            )
            log_data[f"{episode_key}/lever_threshold_shortfall_tracked_slot"] = (
                lever["lever_threshold_shortfall_tracked_slot"]
            )
            log_data[f"{episode_key}/lever_threshold_reached_tracked_slot"] = bool(
                lever["lever_threshold_reached_tracked_slot"]
            )
            log_data[f"{episode_key}/first_lever_threshold_step_tracked_slot"] = (
                lever["first_lever_threshold_step_tracked_slot"]
            )

        env_name = self.env_kwargs["env_name"]
        log_data[f"success_rate/{env_name}"] = float(success_rate)
        for prefix, values in max_rewards.items():
            log_data[prefix + "mean_score"] = float(np.mean(values))

        summaries = self.last_milestone_outcomes
        tracked_lever_positions = [
            summary["spec_final_metadata"]["max_lever_position_tracked_slot"]
            for summary in summaries
            if summary["spec_final_metadata"][
                "max_lever_position_tracked_slot"
            ]
            is not None
        ]
        lever_shortfalls = [
            summary["spec_final_metadata"][
                "lever_threshold_shortfall_tracked_slot"
            ]
            for summary in summaries
            if summary["spec_final_metadata"][
                "lever_threshold_shortfall_tracked_slot"
            ]
            is not None
        ]
        log_data.update(
            {
                "milestone/schema_version": MILESTONE_RUNNER_SCHEMA_VERSION,
                "milestone/spec_id": summaries[0]["task_spec_id"],
                "milestone/spec_version": summaries[0]["task_spec_version"],
                "milestone/mean_normalized_valid_progress": float(
                    np.mean(
                        [summary["normalized_valid_progress"] for summary in summaries]
                    )
                ),
                "milestone/mean_normalized_linear_prefix": float(
                    np.mean(
                        [summary["normalized_linear_prefix"] for summary in summaries]
                    )
                ),
                "milestone/mean_prefix_auc_fixed_horizon": float(
                    np.mean(
                        [summary["prefix_auc_fixed_horizon"] for summary in summaries]
                    )
                ),
                "milestone/official_success_disagreement_rate": float(
                    np.mean(
                        [
                            summary["official_success_contract_disagreement"]
                            for summary in summaries
                        ]
                    )
                ),
                "milestone/mean_max_lever_position_tracked_slot": (
                    float(np.mean(tracked_lever_positions))
                    if tracked_lever_positions
                    else None
                ),
                "milestone/mean_lever_threshold_shortfall_tracked_slot": (
                    float(np.mean(lever_shortfalls))
                    if lever_shortfalls
                    else None
                ),
                "milestone/lever_threshold_reach_rate": float(
                    np.mean(
                        [
                            summary["spec_final_metadata"][
                                "lever_threshold_reached_tracked_slot"
                            ]
                            for summary in summaries
                        ]
                    )
                ),
                "milestone/trace_manifest_path": self.last_milestone_manifest[
                    "path"
                ],
                "milestone/trace_manifest_sha256": self.last_milestone_manifest[
                    "sha256"
                ],
                "milestone/trace_manifest_episode_count": int(
                    self.last_milestone_manifest["episode_count"]
                ),
            }
        )
        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            action = action.reshape(-1, 2, 10)
        rotation_dimension = action.shape[-1] - 4
        position = action[..., :3]
        rotation = action[..., 3 : 3 + rotation_dimension]
        gripper = action[..., [-1]]
        rotation = self.rotation_transformer.inverse(rotation)
        untransformed = np.concatenate([position, rotation, gripper], axis=-1)
        if raw_shape[-1] == 20:
            untransformed = untransformed.reshape(*raw_shape[:-1], 14)
        return untransformed

    def close(self):
        if not isinstance(self.env, SyncVectorEnv):
            return
        for env in self.env.envs:
            chain = [env]
            while hasattr(env, "env"):
                env = env.env
                chain = [env] + chain
            for wrapped in chain:
                if hasattr(wrapped, "close"):
                    wrapped.close()


def _plain_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {
            str(key): _plain_mapping(item)
            if isinstance(item, Mapping)
            else item
            for key, item in value.items()
        }
    try:
        return _plain_mapping(OmegaConf.to_container(value, resolve=True))
    except Exception as error:
        raise TypeError("milestone_evaluation must be mapping-like") from error


def validate_milestone_evaluation_contract(
    value: Mapping[str, Any] | None,
    *,
    env_name: str,
    max_steps: int,
) -> dict[str, Any]:
    """Fail before env creation when the evaluator contract is incomplete."""
    config = _plain_mapping(value or {})
    if not bool(config.get("enabled", False)):
        raise ValueError(
            "MilestoneRobomimicImageRunner requires milestone evaluation enabled"
        )
    if not bool(config.get("required", False)):
        raise ValueError(
            "MilestoneRobomimicImageRunner requires fail-closed evaluation"
        )
    required_keys = (
        "spec_target",
        "expected_task_name",
        "expected_spec_id",
        "expected_spec_version",
        "fixed_horizon",
        "environment_source_revision",
    )
    missing = [key for key in required_keys if config.get(key) is None]
    if missing:
        raise ValueError(f"milestone_evaluation lacks required keys: {missing}")
    expected_task_name = str(config["expected_task_name"])
    if str(env_name) != expected_task_name:
        raise ValueError(
            "milestone task contract does not match rollout environment: "
            f"{expected_task_name!r} != {str(env_name)!r}"
        )
    configured_horizon = int(config["fixed_horizon"])
    if configured_horizon != int(max_steps):
        raise ValueError(
            "milestone fixed_horizon must equal runner max_steps: "
            f"{configured_horizon} != {max_steps}"
        )
    config["fixed_horizon"] = configured_horizon
    return config
