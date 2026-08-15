"""Read-only RoboCasa state access used by task milestone specifications."""

from __future__ import annotations

from typing import Any

import numpy as np


def resolve_raw_task_env(env: Any) -> Any:
    """Resolve the RoboCasa task env from the Gym registration wrapper."""
    gym_env = getattr(env, "unwrapped", env)
    raw_env = getattr(gym_env, "env", None)
    if raw_env is None:
        raise TypeError("could not resolve raw RoboCasa task env")
    return raw_env


class GetToastedBreadStateReader:
    """Public-state reader; methods do not call task ``_check_success``."""

    def validate(self, raw_env: Any) -> None:
        required = ("toaster", "dining_counter", "objects", "obj_body_id", "sim")
        missing = [name for name in required if not hasattr(raw_env, name)]
        if missing:
            raise TypeError(f"GetToastedBread raw env lacks: {missing}")

    def slot_states(self, raw_env: Any) -> dict[int, dict[str, Any]]:
        state = raw_env.toaster.get_state(raw_env)
        if not isinstance(state, dict) or not state:
            raise RuntimeError("toaster returned no slot-pair state")
        return {int(slot): dict(values) for slot, values in state.items()}

    def slot_contacts(self, raw_env: Any, slot_pairs: tuple[int, ...]) -> dict[int, bool]:
        return {
            int(slot): bool(
                raw_env.toaster.check_slot_contact(raw_env, "obj", slot_pair=int(slot))
            )
            for slot in slot_pairs
        }

    def bread_grasped(self, raw_env: Any) -> bool:
        from robocasa.utils import object_utils as object_utils

        return bool(object_utils.check_obj_grasped(raw_env, "obj"))

    def bread_in_plate(self, raw_env: Any) -> bool:
        from robocasa.utils import object_utils as object_utils

        return bool(
            object_utils.check_obj_in_receptacle(raw_env, "obj", "plate")
        )

    def gripper_bread_far(self, raw_env: Any) -> bool:
        from robocasa.utils import object_utils as object_utils

        return bool(object_utils.gripper_obj_far(raw_env, "obj"))

    def plate_navigation_target(self, raw_env: Any) -> tuple[np.ndarray, np.ndarray]:
        from robocasa.utils import env_utils as env_utils

        position, orientation = env_utils.compute_robot_base_placement_pose(
            raw_env, raw_env.dining_counter, ref_object="plate"
        )
        return np.asarray(position, dtype=float), np.asarray(orientation, dtype=float)

    def robot_base_pose(self, raw_env: Any) -> tuple[np.ndarray, np.ndarray]:
        from robosuite.utils import transform_utils as transform_utils

        robot_id = raw_env.sim.model.body_name2id("mobilebase0_base")
        position = np.asarray(raw_env.sim.data.body_xpos[robot_id], dtype=float)
        rotation_matrix = np.asarray(
            raw_env.sim.data.body_xmat[robot_id], dtype=float
        ).reshape((3, 3))
        orientation = np.asarray(
            transform_utils.mat2euler(rotation_matrix), dtype=float
        )
        return position, orientation
