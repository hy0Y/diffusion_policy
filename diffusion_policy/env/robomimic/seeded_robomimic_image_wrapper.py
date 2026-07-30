"""RoboCasa image wrapper with an explicit one-shot reset seed.

The upstream wrapper exposes a seed label but its active reset path calls
``env.reset()`` without forwarding that seed.  This wrapper keeps the upstream
observation processing intact and only changes the reset contract.
"""

from diffusion_policy.env.robomimic.robomimic_image_wrapper import (
    RobomimicImageWrapper,
)


class SeededRobomimicImageWrapper(RobomimicImageWrapper):
    """Apply a pending environment seed to exactly one subsequent reset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_reset_seed = None
        self.last_reset_seed = None

    def set_reset_seed(self, seed):
        """Set the seed consumed by the next :meth:`reset` call."""
        self._pending_reset_seed = None if seed is None else int(seed)

    def reset(self):
        seed = self._pending_reset_seed
        self._pending_reset_seed = None

        raw_obs, _ = self.env.reset(seed=seed)
        self.last_reset_seed = seed
        self.lang = raw_obs["annotation.human.task_description"]
        self.lang_emb = self.lang_encoder.get_lang_emb(self.lang).numpy()
        return self.get_observation(raw_obs)
