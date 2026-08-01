import os
import h5py
import numpy as np
import random
import json
import math
from copy import deepcopy
from contextlib import contextmanager
from collections import OrderedDict
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.pytorch_util import dict_apply
import robomimic.utils.torch_utils as TorchUtils
from tqdm import tqdm
import robomimic.utils.tensor_utils as TensorUtils
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
import copy

import robomimic.utils.lang_utils as LangUtils
from robomimic.macros import LANG_EMB_KEY

import torch.utils.data
import torch
from typing import Dict, List


from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY


from robocasa.utils.groot_utils.groot_dataset import LeRobotSingleDataset, LE_ROBOT_MODALITY_FILENAME, ModalityConfig, LE_ROBOT_EPISODE_FILENAME, LeRobotMixtureDataset
import pathlib

def get_modality_keys(dataset_path: pathlib.Path) -> dict[str, list[str]]:
    """
    Get the modality keys from the dataset path.
    Returns a dictionary with modality types as keys and their corresponding modality keys as values,
    maintaining the order: video, state, action, annotation
    """
    modality_path = dataset_path / LE_ROBOT_MODALITY_FILENAME
    with open(modality_path, "r") as f:
        modality_meta = json.load(f)

    # Initialize dictionary with ordered keys
    modality_dict = {}
    for key in modality_meta.keys():
        modality_dict[key] = []
        for modality in modality_meta[key]:
            modality_dict[key].append(f"{key}.{modality}")
    return modality_dict

class LerobotDataset(LeRobotSingleDataset, BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            filter_key=None,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            lang_encoder=None,
            del_lang_encoder_after_init=True,
        ):

        assert n_obs_steps and n_obs_steps > 0
        self.abs_action = abs_action
        assert not self.abs_action, "abs_action is not supported in LerobotDataset"
        dataset_path = pathlib.Path(dataset_path)
        delta_indices = list(range(-n_obs_steps+1, horizon - n_obs_steps + 1))
        delta_indices_obs = list(range(-n_obs_steps+1, 1))
        assert len(delta_indices_obs) == n_obs_steps, \
            f"delta_indices_obs length {len(delta_indices_obs)} != n_obs_steps {n_obs_steps}"   
        modality_keys_dict = get_modality_keys(dataset_path)
        video_modality_keys = modality_keys_dict["video"]
        language_modality_keys = modality_keys_dict["annotation"]
        state_modality_keys = modality_keys_dict["state"]
        action_modality_keys = modality_keys_dict["action"]
        state_modality_keys = [key for key in state_modality_keys if key != "state.dummy_tensor"]
        modality_configs = {
            "video": ModalityConfig(
                delta_indices=delta_indices_obs,
                modality_keys=video_modality_keys,  # we will include all video modalities
            ),
            "state": ModalityConfig(
                delta_indices=delta_indices_obs,
                modality_keys=state_modality_keys,
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=action_modality_keys,
            ),
        }

        LeRobotSingleDataset.__init__(
            self,
            dataset_path=dataset_path,
            filter_key=filter_key,
            embodiment_tag="oxe_droid",
            modality_configs=modality_configs,
        )
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths
        rgb_keys = dict()
        lowdim_keys = dict()
        obs_shape_meta = copy.deepcopy(shape_meta['obs'])
        self.lang_emb = obs_shape_meta.pop('lang_emb', None)
        if self.lang_emb is not None:
            assert language_modality_keys, "Language modality keys should not be empty if lang_emb is defined"
            self._lang_encoder = lang_encoder
            self._get_lang_embeddings()
            if del_lang_encoder_after_init:
                del self._lang_encoder
                self._lang_encoder = None
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys[key] = attr["lerobot_keys"]
            elif type == 'low_dim':
                lowdim_keys[key] = attr["lerobot_keys"]
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.action_info = self.shape_meta['action']
        self.lerobot_action_keys = self.action_info['lerobot_keys']
        self.action_size = self.action_info['shape'][0]
    
    def _get_lang_embeddings(self):
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        if self._lang_encoder is None:
            self._lang_encoder = LangUtils.LangEncoder(
                    device=device,
            )
        self._demo_id_to_demo_lang_emb = {}
    
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]
        id2remark = {e["episode_index"]: e["tasks"][0] for e in episode_metadata}
        
        for ep_batch in tqdm(np.array_split(self.trajectory_ids, int(math.ceil(len(self.trajectory_ids) / 64)))):
            # get language embedding
            lang_batch = [id2remark[ep] for ep in ep_batch]
            emb_batch = self._lang_encoder.get_lang_emb(lang_batch)
            emb_batch = TensorUtils.to_numpy(emb_batch)
            for batch_idx, ep in enumerate(ep_batch):
                self._demo_id_to_demo_lang_emb[ep] = emb_batch[batch_idx]
            
    

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # threadpool_limits(1)

        # super call to get data
        data = LeRobotSingleDataset.__getitem__(self, idx)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key, lerobot_keys in self.rgb_keys.items():
            assert len(lerobot_keys) == 1, f"multiple lerobot keys for {key} not supported"
            lerobot_key = lerobot_keys[0]
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[lerobot_key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
        for key, lerobot_keys in self.lowdim_keys.items():
            assert len(lerobot_keys) == 1, f"multiple lerobot keys for {key} not supported"
            lerobot_key = lerobot_keys[0]
            obs_dict[key] = data[lerobot_key][T_slice].astype(np.float32)

        if self.lang_emb is not None:
            trajectory_id, _ = self.all_steps[idx]
            lang_emb = self._demo_id_to_demo_lang_emb[trajectory_id]
            obs_dict[LANG_EMB_KEY] = np.tile(
                lang_emb,
                (self.n_obs_steps, 1)
            ).astype(np.float32)
        
        action_concat = []

        for lr_key in self.lerobot_action_keys:
            if lr_key in data:
                action_concat.append(data[lr_key])
            else:
                raise ValueError(f"Key {lr_key} not found in data")
        

        action_concat = np.concatenate(action_concat, axis=-1)
        assert action_concat.shape[-1] == self.action_size, \
            f"action_concat shape mismatch: {action_concat.shape[-1]} != {self.action_size}"
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action_concat.astype(np.float32))
        }

        return torch_data
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        # Almost same as robomimic_replay_image_dataset.py
        normalizer = LinearNormalizer()
        assert not self.abs_action, "normalizer for abs_action is not supported in LerobotDataset"


        scale = np.ones((self.action_size), dtype=np.float32)
        offset = np.zeros((self.action_size), dtype=np.float32)
        normalizer['action'] = SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict={}, #stat
        )

        # obs
        for key, lerobot_keys in self.lowdim_keys.items():
            assert len(lerobot_keys) == 1, f"multiple lerobot keys for {key} not supported"
            lerobot_key = lerobot_keys[0]
            # strip "state." prefix
            lerobot_key = lerobot_key.replace("state.", "")
            stat = self._metadata.statistics.state[lerobot_key].model_dump()
            for k, v in stat.items():
                if type(v) is np.ndarray:
                    stat[k] = v.astype(np.float32)

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('sin'):
                # sin is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('cos'):
                # sin is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer
        
        # lang_emb
        if self.lang_emb is not None:
            dim = int(np.prod(self.lang_emb["shape"]))  
            scale  = np.ones((dim,), dtype=np.float32)  
            offset = np.zeros((dim,), dtype=np.float32) 
            normalizer[LANG_EMB_KEY] = SingleFieldLinearNormalizer.create_manual(
                scale=scale,
                offset=offset,
                input_stats_dict={}, #stat
            )

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

class LerobotCotrainingDataset(LeRobotMixtureDataset, BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_paths: List[str] | None = None,
            dataset_soup=None,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0, # validation not implemented yet,
            ds_weights=None,
            ds_weights_alpha=0.40,
            metadata_config: dict = {
            "percentile_mixing_method": "weighted_average",
        } 
        ):
        # exactly one of dataset_paths or dataset_soup must be defined
        assert (dataset_paths == None) + (dataset_soup == None) == 1

        if dataset_soup is not None:
            dataset_soup_list = copy.deepcopy(DATASET_SOUP_REGISTRY[dataset_soup])
        else:
            dataset_soup_list = [
                {"path": ds_path, "filter_key": None}
                for ds_path in dataset_paths
            ]

        for i in range(len(dataset_soup_list)):
            ds_path = dataset_soup_list[i]["path"]
            if not os.path.isabs(ds_path):
                # hack: fill in robocasa base dataset path
                from robocasa.macros import DATASET_BASE_PATH
                dataset_soup_list[i]["path"] = os.path.join(DATASET_BASE_PATH, ds_path)
            
            dataset_soup_list[i]["ds_weight"] = dataset_soup_list[i].get("ds_weight", None)

        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        lang_encoder = LangUtils.LangEncoder(device=device)
        datasets = [
            LerobotDataset(
                shape_meta=shape_meta,
                dataset_path=ds_meta["path"],
                filter_key=ds_meta["filter_key"],
                horizon=horizon,
                pad_after=pad_after,
                pad_before=pad_before,
                n_obs_steps=n_obs_steps,
                abs_action=abs_action,
                rotation_rep=rotation_rep,
                use_legacy_normalizer=use_legacy_normalizer,
                use_cache=use_cache,
                seed=seed,
                val_ratio=val_ratio,
                lang_encoder=lang_encoder,
                del_lang_encoder_after_init=False,
            ) for ds_meta in dataset_soup_list
        ]
        del lang_encoder
        self.abs_action = abs_action
        assert not self.abs_action, "abs_action is not supported in LerobotCotrainingDataset"
        assert ds_weights is None or len(ds_weights) == len(datasets), \
            f"ds_weights length {len(ds_weights)} != datasets length {len(datasets)}"
        
        if ds_weights is None and all(ds_meta["ds_weight"] is not None for ds_meta in dataset_soup_list):
            ds_weights = [ds_meta["ds_weight"] for ds_meta in dataset_soup_list]
        
        if not ds_weights:
            ds_weights = np.array([np.power(len(dataset), ds_weights_alpha) for dataset in datasets])
            # the groot dataloader requires that at least one dataset has weight 1.0
            ds_weights = ds_weights / ds_weights[0]
        print("dataset weights:", ds_weights)
        
        dataset_mixture = list(zip(datasets, ds_weights))
        # set balance_dataset_weights to False, since we are calculating weights ourselves
        LeRobotMixtureDataset.__init__(self,  data_mixture=dataset_mixture, mode="train",  balance_dataset_weights=False, balance_trajectory_weights=False, metadata_config=metadata_config)
        rgb_keys = dict()
        lowdim_keys = dict()
        obs_shape_meta = copy.deepcopy(shape_meta['obs'])
        self.lang_emb = obs_shape_meta.pop('lang_emb', None)
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys[key] = attr["lerobot_keys"]
            elif type == 'low_dim':
                lowdim_keys[key] = attr["lerobot_keys"]
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.action_info = self.shape_meta['action']
        self.lerobot_action_keys = self.action_info['lerobot_keys']
        self.action_size = self.action_info['shape'][0]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset, trajectory_name, step = self.sample_step(idx)
        global_ds_index = self.to_global_index(dataset, trajectory_name, step)
        return dataset.__getitem__(global_ds_index)

    def to_global_index(self, dataset, trajectory_id: int, base_index: int) -> int:
        """Convert (trajectory_id, base_index) → global index for a given dataset"""
        traj_idx = dataset.get_trajectory_index(trajectory_id) 
        g_idx = int(dataset.start_indices[traj_idx] + base_index)
        # # TODO: remove
        # assert g_idx == dataset.all_steps.index((trajectory_id, base_index)), \
        #     f"g_idx {g_idx} != dataset.all_steps.index({trajectory_id}, {base_index})"
        return g_idx
    
    def __len__(self):
        return np.sum(self.dataset_lengths)

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        # Almost same as robomimic_replay_image_dataset.py
        normalizer = LinearNormalizer()
        assert not self.abs_action, "normalizer for abs_action is not supported in LerobotDataset"

        # tag should be same for all datasets
        tag = self.datasets[0].tag
        # TODO, look into how these vals are affected in original code
        all_stats = self.merged_metadata[tag].statistics

        scale = np.ones((self.action_size), dtype=np.float32)
        offset = np.zeros((self.action_size), dtype=np.float32)
        normalizer['action'] = SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict={}, #stat
        )


        for key, lerobot_keys in self.lowdim_keys.items():
            assert len(lerobot_keys) == 1, f"multiple lerobot keys for {key} not supported"
            lerobot_key = lerobot_keys[0]
            # strip "state." prefix
            lerobot_key = lerobot_key.replace("state.", "")
            stat = all_stats.state[lerobot_key].model_dump()
            for k, v in stat.items():
                if type(v) is np.ndarray:
                    stat[k] = v.astype(np.float32)

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('sin'):
                # sin is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('cos'):
                # sin is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer
        # lang_emb
        if self.lang_emb is not None:
            dim = int(np.prod(self.lang_emb["shape"]))  
            scale  = np.ones((dim,), dtype=np.float32)  
            offset = np.zeros((dim,), dtype=np.float32) 
            normalizer[LANG_EMB_KEY] = SingleFieldLinearNormalizer.create_manual(
                scale=scale,
                offset=offset,
                input_stats_dict={}, #stat
            )

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer



# =============================================================================
# Semantic bifurcation extension. The copied classes above remain unchanged.
# =============================================================================
from functools import lru_cache
import pyarrow.parquet as pq


class LerobotSemanticDataset(LerobotDataset):
    """LerobotDataset plus episode-indexed semantic/boundary sidecar targets."""

    def __init__(
        self,
        *args,
        semantic_gt_root: str,
        semantic_split: str,
        task_name: str,
        allowed_episode_ids: List[int] | None = None,
        future_offset: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.semantic_gt_root = pathlib.Path(semantic_gt_root)
        self.semantic_split = semantic_split
        self.semantic_task_name = task_name
        self.future_offset = int(future_offset)
        if self.future_offset != 8:
            raise ValueError("GT v1.0 currently supports future_offset=8")

        if allowed_episode_ids is not None:
            allowed = np.asarray(sorted(int(x) for x in allowed_episode_ids))
            available = set(int(x) for x in self._trajectory_ids.tolist())
            missing = sorted(set(allowed.tolist()) - available)
            if missing:
                subset = (
                    set(int(x) for x in self.subset_demos)
                    if self.subset_demos is not None
                    else None
                )
                unexpected = missing if subset is None else sorted(set(missing) & subset)
                if unexpected:
                    raise ValueError(
                        f"missing dataset episodes for {task_name}: {unexpected[:10]}"
                    )
                print(
                    f"Skipping {len(missing)} manifest episodes excluded by "
                    f"the dataset filter for {task_name}: {missing[:10]}"
                )
            keep = np.isin(self._trajectory_ids, allowed)
            self._trajectory_ids = self._trajectory_ids[keep]
            self._trajectory_lengths = self._trajectory_lengths[keep]
            self._all_steps = self._get_all_steps()
            self.start_indices = (
                np.cumsum(self.trajectory_lengths) - self.trajectory_lengths
            )
        if len(self._trajectory_ids) == 0:
            raise ValueError(f"empty semantic dataset: {task_name}")

        sidecar_dir = self.semantic_gt_root / semantic_split / task_name
        if not sidecar_dir.is_dir():
            raise FileNotFoundError(f"semantic sidecar directory missing: {sidecar_dir}")
        self._sidecar_dir = sidecar_dir

    @lru_cache(maxsize=8)
    def _load_semantic_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        path = self._sidecar_dir / f"episode_{int(episode_index):06d}.parquet"
        columns = [
            "frame_index",
            "subtask_local_id",
            "semantic_skill_local_id",
            "stage_local_id",
            "future_h8_valid",
            "boundary_subtask_future_h8",
            "boundary_stage_future_h8",
            "boundary_any_future_h8",
        ]
        table = pq.read_table(path, columns=columns)
        result = {
            name: table[name].to_numpy(zero_copy_only=False)
            for name in columns
        }
        expected = np.arange(table.num_rows)
        if not np.array_equal(result["frame_index"], expected):
            raise ValueError(f"invalid sidecar frame_index: {path}")
        return result

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        torch_data = super().__getitem__(idx)
        trajectory_id, base_index = self.all_steps[idx]
        sidecar = self._load_semantic_episode(int(trajectory_id))
        future_index = int(base_index) + self.future_offset
        valid = bool(sidecar["future_h8_valid"][base_index])
        if valid:
            subtask = int(sidecar["subtask_local_id"][future_index])
            skill = int(sidecar["semantic_skill_local_id"][future_index])
            stage = int(sidecar["stage_local_id"][future_index])
        else:
            subtask = skill = stage = -100

        torch_data["semantic_target"] = {
            "subtask": torch.tensor(subtask, dtype=torch.long),
            "semantic_skill": torch.tensor(skill, dtype=torch.long),
            "stage": torch.tensor(stage, dtype=torch.long),
            "valid": torch.tensor(valid, dtype=torch.bool),
        }
        torch_data["boundary_target"] = {
            "subtask_future_h8": torch.tensor(
                bool(sidecar["boundary_subtask_future_h8"][base_index]),
                dtype=torch.float32,
            ),
            "stage_future_h8": torch.tensor(
                bool(sidecar["boundary_stage_future_h8"][base_index]),
                dtype=torch.float32,
            ),
            "any_future_h8": torch.tensor(
                bool(sidecar["boundary_any_future_h8"][base_index]),
                dtype=torch.float32,
            ),
            "valid": torch.tensor(valid, dtype=torch.bool),
        }
        torch_data["sample_index"] = {
            "episode_index": torch.tensor(int(trajectory_id), dtype=torch.long),
            "frame_index": torch.tensor(int(base_index), dtype=torch.long),
        }
        return torch_data


class LerobotP1CotrainingDataset(
    LerobotCotrainingDataset, BaseImageDataset
):
    """Target-task mixture with fixed episode-level train/validation manifests."""

    def __init__(
        self,
        shape_meta: dict,
        dataset_soup: str,
        semantic_gt_root: str,
        split_manifest_path: str,
        semantic_split: str = "composite_seen",
        future_offset: int = 8,
        split_role: str = "train",
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: int | None = None,
        abs_action: bool = False,
        rotation_rep: str = "rotation_6d",
        use_legacy_normalizer: bool = False,
        use_cache: bool = False,
        seed: int = 42,
        val_ratio: float = 0.02,
        ds_weights=None,
        ds_weights_alpha: float = 0.40,
        metadata_config: dict = {
            "percentile_mixing_method": "weighted_average",
        },
        single_dataset_class=None,
        single_dataset_kwargs: dict | None = None,
    ):
        if split_role not in {"train", "validation"}:
            raise ValueError(f"invalid split_role: {split_role}")
        manifest = json.loads(pathlib.Path(split_manifest_path).read_text())
        if manifest.get("task_set") != semantic_split:
            raise ValueError(
                f"P1 manifest task_set must be {semantic_split}: "
                f"{manifest.get('task_set')!r}"
            )
        mixture_mode = "train" if split_role == "train" else "val"
        dataset_soup_list = copy.deepcopy(DATASET_SOUP_REGISTRY[dataset_soup])
        for ds_meta in dataset_soup_list:
            ds_path = ds_meta["path"]
            if not os.path.isabs(ds_path):
                from robocasa.macros import DATASET_BASE_PATH
                ds_meta["path"] = os.path.join(DATASET_BASE_PATH, ds_path)
            ds_meta["ds_weight"] = ds_meta.get("ds_weight", None)

        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        lang_encoder = LangUtils.LangEncoder(device=device)
        datasets = []
        if single_dataset_class is None:
            single_dataset_class = LerobotSemanticDataset
        single_dataset_kwargs = dict(single_dataset_kwargs or {})
        for ds_meta in dataset_soup_list:
            task = ds_meta["task"]
            task_entry = manifest["tasks"].get(task)
            if task_entry is None:
                raise KeyError(f"task missing from split manifest: {task}")
            datasets.append(
                single_dataset_class(
                    shape_meta=shape_meta,
                    dataset_path=ds_meta["path"],
                    filter_key=ds_meta["filter_key"],
                    horizon=horizon,
                    pad_after=pad_after,
                    pad_before=pad_before,
                    n_obs_steps=n_obs_steps,
                    abs_action=abs_action,
                    rotation_rep=rotation_rep,
                    use_legacy_normalizer=use_legacy_normalizer,
                    use_cache=use_cache,
                    seed=seed,
                    val_ratio=val_ratio,
                    lang_encoder=lang_encoder,
                    del_lang_encoder_after_init=False,
                    semantic_gt_root=semantic_gt_root,
                    semantic_split=semantic_split,
                    task_name=task,
                    allowed_episode_ids=task_entry[split_role],
                    future_offset=future_offset,
                    **single_dataset_kwargs,
                )
            )
        del lang_encoder
        for dataset in datasets:
            if hasattr(dataset, "set_split_role"):
                dataset.set_split_role(mixture_mode)

        self.abs_action = abs_action
        if self.abs_action:
            raise ValueError("abs_action is not supported")
        if ds_weights is None and all(
            ds_meta["ds_weight"] is not None for ds_meta in dataset_soup_list
        ):
            ds_weights = [ds_meta["ds_weight"] for ds_meta in dataset_soup_list]
        if ds_weights is None:
            ds_weights = np.asarray(
                [np.power(len(dataset), ds_weights_alpha) for dataset in datasets]
            )
            ds_weights = ds_weights / ds_weights[0]

        LeRobotMixtureDataset.__init__(
            self,
            data_mixture=list(zip(datasets, ds_weights)),
            mode=mixture_mode,
            balance_dataset_weights=False,
            balance_trajectory_weights=False,
            seed=seed,
            metadata_config=metadata_config,
        )
        self.rgb_keys = datasets[0].rgb_keys
        self.lowdim_keys = datasets[0].lowdim_keys
        self.lang_emb = datasets[0].lang_emb
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.action_info = shape_meta["action"]
        self.lerobot_action_keys = self.action_info["lerobot_keys"]
        self.action_size = self.action_info["shape"][0]
        self.split_role = split_role

        self._validation_init = dict(
            shape_meta=shape_meta,
            dataset_soup=dataset_soup,
            semantic_gt_root=semantic_gt_root,
            semantic_split=semantic_split,
            split_manifest_path=split_manifest_path,
            future_offset=future_offset,
            split_role="validation",
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            n_obs_steps=n_obs_steps,
            abs_action=abs_action,
            rotation_rep=rotation_rep,
            use_legacy_normalizer=use_legacy_normalizer,
            use_cache=use_cache,
            seed=seed,
            val_ratio=val_ratio,
            ds_weights=ds_weights,
            ds_weights_alpha=ds_weights_alpha,
            metadata_config=metadata_config,
            single_dataset_class=single_dataset_class,
            single_dataset_kwargs=single_dataset_kwargs,
        )
        self._sequential_index = []
        if split_role == "validation":
            for dataset_index, dataset in enumerate(self.datasets):
                self._sequential_index.extend(
                    (dataset_index, local_index)
                    for local_index in range(len(dataset))
                )

    def __len__(self):
        if self.split_role == "validation":
            return len(self._sequential_index)
        return super().__len__()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.split_role == "validation":
            dataset_index, local_index = self._sequential_index[idx]
            return self.datasets[dataset_index][local_index]
        dataset, trajectory_id, step = self.sample_step(idx)
        global_index = self.to_global_index(dataset, trajectory_id, step)
        return dataset[global_index]

    def get_validation_dataset(self) -> "LerobotP1CotrainingDataset":
        if self.split_role == "validation":
            return self
        return LerobotP1CotrainingDataset(**self._validation_init)
