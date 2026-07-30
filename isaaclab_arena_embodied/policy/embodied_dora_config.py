# Copyright (c) 2026, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from isaaclab_arena.policy.policy_base import PolicyCfg


@dataclass
class EmbodiedDoraPolicyCfg(PolicyCfg):
    """Config for :class:`EmbodiedDoraPolicy` (embodied-rs ``dora-policy`` client)."""

    policy_device: str = "cpu"
    """Device for returned action tensors."""

    embodiment_adapter: str = "smolvla_libero"
    """T1 adapter id. Currently only ``smolvla_libero`` is implemented."""

    open_loop_horizon: int = 50
    """Steps to replay from each ActionChunk before re-querying policy."""

    action_dim: int = 7
    """Env action dim after T1 (Franka IK: 6 EE + grip)."""

    unnorm_key: str | None = "default"
    """Forwarded on ObservationWire; must match package stats key."""

    # --- dora-policy process ---
    dora_policy_bin: str | None = None
    """Path to ``dora-policy`` binary. Required when ``spawn_policy`` is True."""

    model_dir: str | None = None
    """LeRobot / HF package directory (only used when spawning dora-policy)."""

    dora_device: str = "cuda"
    """Backend for spawned ``dora-policy`` (flex|metal|cuda|…).

    Named ``dora_device`` (not ``device``) so it does not clash with Arena's
    shared ``--device`` CLI flag (default ``cuda:0``).
    """

    inference_steps: int = 0
    """FM steps for SmolVLA; 0 = package default."""

    spawn_policy: bool = True
    """If True, spawn ``dora-policy --stdio-loop``. If False, attach via pipes is N/A —
    use an already-spawned process by setting ``dora_policy_bin`` with spawn and
    managing lifecycle externally is preferred; for attach-only use
    ``attach_command``."""

    attach_command: list[str] | None = None
    """Optional full argv to spawn instead of building from bin/model_dir.
    Example: ``[\"/path/dora-policy\", \"--model-dir\", \"…\", \"--stdio-loop\"]``."""

    ready_timeout_s: float = 600.0
    """Seconds to wait for ``STDIO_LOOP_READY`` on stderr after spawn."""

    predict_timeout_s: float = 120.0
    """Seconds to wait for one ActionChunk JSON line."""

    ee_action_scale: float = 1.0
    """Scale EE delta dims before clip (Arena IK scale may need 2.0)."""

    ee_pos_clip: float = 0.10
    ee_rot_clip: float = 0.5
    invert_gripper: bool = False
    binarize_gripper: bool = True
