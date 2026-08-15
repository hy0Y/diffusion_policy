"""RoboCasa runner that reports every observation produced by an action segment."""

from diffusion_policy.env_runner.robomimic_image_runner import (
    RobomimicImageRunner,
)


class JumpRobomimicImageRunner(RobomimicImageRunner):
    """Configure the generic runner for the DP + Jump history contract."""

    def __init__(
            self,
            *args,
            return_executed_observations: bool = True,
            past_action: bool = False,
            abs_action: bool = False,
            **kwargs,
        ):
        if not return_executed_observations:
            raise ValueError(
                "Jump rollout requires complete executed observation segments")
        if past_action:
            raise ValueError("Jump rollout does not use the legacy past_action path")
        if abs_action:
            raise ValueError("DP + Jump training uses abs_action=False")
        super().__init__(
            *args,
            return_executed_observations=True,
            past_action=False,
            abs_action=False,
            **kwargs,
        )
