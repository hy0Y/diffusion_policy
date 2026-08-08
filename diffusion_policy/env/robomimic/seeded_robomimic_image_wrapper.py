"""RoboCasa image wrapper with explicit one-shot reset sources.

The upstream wrapper exposes a seed label but its active reset path calls
``env.reset()`` without forwarding that seed.  This wrapper keeps the upstream
observation processing intact and only changes the reset contract. It also
supports restoring a LeRobot episode snapshot before a rollout.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import robosuite

from diffusion_policy.env.robomimic.robomimic_image_wrapper import (
    RobomimicImageWrapper,
)


class SeededRobomimicImageWrapper(RobomimicImageWrapper):
    """Apply a pending seed or episode snapshot to one subsequent reset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_reset_seed = None
        self._pending_reset_snapshot = None
        self.last_reset_seed = None
        self.last_reset_snapshot = None

    def set_reset_seed(self, seed):
        """Set the seed consumed by the next :meth:`reset` call."""
        self._pending_reset_snapshot = None
        self._pending_reset_seed = None if seed is None else int(seed)

    def set_reset_snapshot(self, episode_root, state_index=0):
        """Set a LeRobot episode snapshot consumed by the next reset."""
        self._pending_reset_seed = None
        self._pending_reset_snapshot = (str(episode_root), int(state_index))

    @staticmethod
    def load_reset_snapshot(episode_root, state_index=0):
        """Load and validate one episode's model, metadata, and MuJoCo state."""
        root = Path(episode_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"episode_root is not a directory: {root}")
        model_path = root / "model.xml.gz"
        metadata_path = root / "ep_meta.json"
        states_path = root / "states.npz"
        for path in (model_path, metadata_path, states_path):
            if not path.is_file():
                raise FileNotFoundError(f"episode snapshot file is missing: {path}")

        with gzip.open(model_path, "rt", encoding="utf-8") as handle:
            model_xml = handle.read()
        if not model_xml.strip():
            raise ValueError(f"episode model XML is empty: {model_path}")
        ep_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(ep_meta, dict):
            raise ValueError(f"episode metadata must be a JSON object: {metadata_path}")
        with np.load(states_path, allow_pickle=False) as archive:
            if "states" not in archive.files:
                raise ValueError(f"states.npz does not contain 'states': {states_path}")
            states = archive["states"]
        if states.ndim != 2 or states.shape[0] == 0:
            raise ValueError(
                f"episode states must have shape [T, D] with T > 0: {states.shape}")
        index = int(state_index)
        if index < 0 or index >= states.shape[0]:
            raise IndexError(
                f"initial state index out of range: index={index}, T={states.shape[0]}")
        state = np.asarray(states[index]).copy()
        if not np.all(np.isfinite(state)):
            raise ValueError("episode initial simulator state contains NaN or Inf")
        return {
            "model": model_xml,
            "ep_meta": json.dumps(ep_meta),
            "states": state,
        }

    def _reset_to_snapshot(self, episode_root, state_index):
        """Restore the raw RoboCasa env and return its Gym observation."""
        state = self.load_reset_snapshot(episode_root, state_index)
        gym_env = self.env.unwrapped
        raw_env = gym_env.env
        ep_meta = json.loads(state["ep_meta"])
        if hasattr(raw_env, "set_attrs_from_ep_meta"):
            raw_env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(raw_env, "set_ep_meta"):
            raw_env.set_ep_meta(ep_meta)

        # Reset through Gymnasium once so OrderEnforcing permits the subsequent
        # policy step. The sampled state is immediately replaced below.
        self.env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            model_xml = postprocess_model_xml(state["model"])
        else:
            model_xml = raw_env.edit_model_xml(state["model"])
        raw_env.reset_from_xml_string(model_xml)
        raw_env.sim.reset()
        raw_env.sim.set_state_from_flattened(state["states"])
        raw_env.sim.forward()
        if hasattr(raw_env, "update_sites"):
            raw_env.update_sites()
        if hasattr(raw_env, "update_state"):
            raw_env.update_state()
        raw_obs = raw_env._get_observations(force_update=True)
        return gym_env.get_observation(raw_obs)

    def reset(self):
        seed = self._pending_reset_seed
        snapshot = self._pending_reset_snapshot
        self._pending_reset_seed = None
        self._pending_reset_snapshot = None

        if snapshot is None:
            raw_obs, _ = self.env.reset(seed=seed)
            self.last_reset_seed = seed
            self.last_reset_snapshot = None
        else:
            raw_obs = self._reset_to_snapshot(*snapshot)
            self.last_reset_seed = None
            self.last_reset_snapshot = {
                "episode_root": snapshot[0],
                "state_index": snapshot[1],
            }
        self.lang = raw_obs["annotation.human.task_description"]
        self.lang_emb = self.lang_encoder.get_lang_emb(self.lang).numpy()
        return self.get_observation(raw_obs)
