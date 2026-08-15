"""Utilities for transporting variable-length executed rollout segments."""

from __future__ import annotations

import numpy as np


def collate_executed_segments(infos, proposed_actions):
    """Pad per-environment executed observations/actions to variable ``M``."""
    actions = np.asarray(proposed_actions)
    if actions.ndim < 3 or actions.shape[0] != len(infos):
        raise ValueError("proposed_actions must have shape [B,M,...]")
    lengths = np.asarray([
        int(info["executed_length"]) for info in infos
    ], dtype=np.int64)
    if np.any(lengths < 0) or np.any(lengths > actions.shape[1]):
        raise ValueError("executed_length lies outside the proposed segment")
    maximum = int(lengths.max(initial=0))
    valid_mask = np.arange(maximum)[None] < lengths[:, None]
    padded_actions = np.zeros(
        (actions.shape[0], maximum) + actions.shape[2:], dtype=actions.dtype)
    for batch_index, length in enumerate(lengths.tolist()):
        padded_actions[batch_index, :length] = actions[batch_index, :length]

    first = infos[0]["executed_observations"]
    if not isinstance(first, dict):
        raise TypeError("executed observations must be a dictionary")
    padded_observations = {}
    for key in first:
        example = np.asarray(first[key])
        target = np.zeros(
            (len(infos), maximum) + example.shape[1:], dtype=example.dtype)
        for batch_index, (info, length) in enumerate(
                zip(infos, lengths.tolist())):
            source = np.asarray(info["executed_observations"][key])
            if source.shape[0] != length or source.shape[1:] != example.shape[1:]:
                raise ValueError(
                    f"executed observation {key!r} disagrees with its length")
            target[batch_index, :length] = source
        padded_observations[key] = target
    return {
        "executed_obs": padded_observations,
        "executed_action": padded_actions,
        "executed_valid_mask": valid_mask,
    }
