import gym
from gym import spaces
import numpy as np

from diffusion_policy.env_runner.executed_segment import (
    collate_executed_segments,
)
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper


class SegmentEnvironment(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict({
            "state": spaces.Box(
                low=-100, high=100, shape=(1,), dtype=np.float32),
            "lang_emb": spaces.Box(
                low=-100, high=100, shape=(3,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32)
        self.time = 0

    def _observation(self):
        return {
            "state": np.asarray([self.time], dtype=np.float32),
            "lang_emb": np.asarray([1, 2, 3], dtype=np.float32),
        }

    def reset(self):
        self.time = 0
        return self._observation()

    def step(self, action):
        self.time += 1
        return self._observation(), 0.0, False, {"success": False}


def test_multistep_wrapper_returns_full_segment_and_configured_recent_window():
    wrapper = MultiStepWrapper(
        SegmentEnvironment(),
        n_obs_steps=3,
        n_action_steps=5,
        return_executed_observations=True,
    )
    initial = wrapper.reset()
    np.testing.assert_array_equal(initial["state"][:, 0], [0, 0, 0])
    recent, _, _, info = wrapper.step(
        np.zeros((5, 1), dtype=np.float32))
    np.testing.assert_array_equal(recent["state"][:, 0], [3, 4, 5])
    np.testing.assert_array_equal(
        info["executed_observations"]["state"][:, 0], [1, 2, 3, 4, 5])
    assert int(info["executed_length"]) == 5


def test_single_action_segment_uses_the_same_contract():
    wrapper = MultiStepWrapper(
        SegmentEnvironment(),
        n_obs_steps=4,
        n_action_steps=1,
        return_executed_observations=True,
    )
    wrapper.reset()
    recent, _, _, info = wrapper.step(
        np.zeros((1, 1), dtype=np.float32))
    np.testing.assert_array_equal(recent["state"][:, 0], [0, 0, 0, 1])
    np.testing.assert_array_equal(
        info["executed_observations"]["state"][:, 0], [1])
    assert int(info["executed_length"]) == 1


def test_collate_executed_segments_pads_variable_lengths():
    infos = [
        {
            "executed_length": 2,
            "executed_observations": {
                "state": np.asarray([[1], [2]], dtype=np.float32),
                "lang_emb": np.asarray(
                    [[1, 2, 3], [1, 2, 3]], dtype=np.float32),
            },
        },
        {
            "executed_length": 4,
            "executed_observations": {
                "state": np.asarray([[3], [4], [5], [6]], dtype=np.float32),
                "lang_emb": np.asarray(
                    [[1, 2, 3]] * 4, dtype=np.float32),
            },
        },
    ]
    proposed = np.arange(10, dtype=np.float32).reshape(2, 5, 1)
    batch = collate_executed_segments(infos, proposed)
    assert batch["executed_obs"]["state"].shape == (2, 4, 1)
    assert batch["executed_action"].shape == (2, 4, 1)
    np.testing.assert_array_equal(batch["executed_valid_mask"], [
        [True, True, False, False],
        [True, True, True, True],
    ])
    np.testing.assert_array_equal(
        batch["executed_action"][0, :2], proposed[0, :2])
    np.testing.assert_array_equal(
        batch["executed_obs"]["state"][1, :, 0], [3, 4, 5, 6])
