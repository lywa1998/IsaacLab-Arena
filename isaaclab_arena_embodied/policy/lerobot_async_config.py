# Copyright (c) 2026, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

from isaaclab_arena.policy.policy_base import PolicyCfg


@dataclass
class EmbodiedLerobotAsyncPolicyCfg(PolicyCfg):
    """Arena client for LeRobot ``async_inference.policy_server`` (gRPC).

    **Eval / diagnosis only** — production remains Dora-native (see DESIGN.md).
    Unnorm runs inside LeRobot server postprocessor (same package stats as dora-policy).
    """

    policy_device: str = "cpu"
    """Device for returned action tensors on the Arena side."""

    embodiment_adapter: str = "smolvla_libero"
    open_loop_horizon: int = 50
    action_dim: int = 7

    # --- LeRobot policy_server connection ---
    server_host: str = "127.0.0.1"
    server_port: int = 8080
    lerobot_policy_type: str = "smolvla"
    """LeRobot policy type registered on the server (smolvla, act, …).

    Named ``lerobot_policy_type`` so it does not clash with Arena's
    ``--policy_type`` class-path CLI flag.
    """

    pretrained_name_or_path: str | None = None
    """HF id or local package dir (smolvla_libero). Required."""

    server_policy_device: str = "cuda"
    """Device string sent in RemotePolicyConfig (where server loads weights)."""

    actions_per_chunk: int = 50
    ready_timeout_s: float = 600.0
    get_actions_timeout_s: float = 120.0

    # Optional: Arena spawns ``python -m lerobot.async_inference.policy_server``
    spawn_server: bool = False
    server_python: str | None = None
    """Python interpreter that has lerobot+libero. Default: sys.executable."""

    server_extra_args: list[str] = field(default_factory=list)
    """Extra argv for policy_server (e.g. ``--fps=30``)."""

    # --- T1 action / state (same knobs as EmbodiedDoraPolicy) ---
    ee_action_scale: float = 2.0
    ee_pos_clip: float = 0.10
    ee_rot_clip: float = 0.5
    invert_gripper: bool = False
    binarize_gripper: bool = True
    invert_ee_pos: bool = False
    invert_ee_rot: bool = False
    flip_hw_180: bool = False
    state_ablation: str = "none"
    align_axis_angle: bool = True
    align_eef_frame: bool = True
    action_unit: str = "libero_osc"
    diag_log_chunks: int = 3
