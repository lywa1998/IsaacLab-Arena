# Copyright (c) 2026, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""LIBERO-like tabletop lift for smolvla_libero closed-loop debugging.

Not a bit-exact MuJoCo LIBERO suite — an Isaac Arena scene that is **close enough**
for adapter/policy plumbing tests:

* Franka + relative EE (``franka_ik``) like LIBERO ``control_mode=relative``
* Dual cameras (agentview + wrist) at 256² (see FrankaCameraCfg)
* Success = **lift height only** (no yaw-90 as in ``cube_goal_pose``)
* Language instruction default: ``pick up the cube``
* Workspace / init joints aligned with existing cube_goal layout (known working)

Use with EmbodiedDoraPolicy or EmbodiedLerobotAsyncPolicy + smolvla_libero weights.
True LIBERO SR still comes from LeRobot MuJoCo ``lerobot-eval``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class LiberoLikeLiftEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure LIBERO-like lift environment.

    ``enable_cameras`` is declared on the child (not only via base) so CLI
    reconstruction works on Arena trees whose ``ArenaEnvironmentCfg`` is still
    an empty marker (university deploy).
    """

    enable_cameras: bool = False
    object: str = "dex_cube"
    background: str = "table"
    embodiment: str = "franka_ik"
    teleop_device: str | None = None
    # Success: object height in meters (table-top cube ~0.05–0.2 depending on spawn).
    # Require a modest lift above initial z without orientation constraint.
    success_z_min: float = 0.18
    success_z_max: float = 1.0
    episode_length_s: float = 20.0
    language_instruction: str = "pick up the cube"


@register_environment
class LiberoLikeLiftEnvironment(ArenaEnvironmentFactory[LiberoLikeLiftEnvironmentCfg]):
    """Tabletop cube lift, LIBERO-shaped for VLA closed-loop smoke tests."""

    name = "libero_like_lift"
    _legacy_argparse_cfg_type = LiberoLikeLiftEnvironmentCfg

    def build(self, cfg: LiberoLikeLiftEnvironmentCfg) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.goal_pose_task import GoalPoseTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name(cfg.background)()
        light = self.asset_registry.get_asset_by_name("light")()
        obj = self.asset_registry.get_asset_by_name(cfg.object)()
        # Cube in front of Franka, on table height band used by cube_goal_pose.
        obj.set_initial_pose(
            Pose(
                position_xyz=(0.1, 0.0, 0.12),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(-0.4, 0.0, 0.0),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )
        # Same ready pose as cube_goal (known IK workspace); LIBERO mean eef_z~0.76 is
        # different frame, but this keeps closed-loop reachable.
        embodiment.set_initial_joint_pose(
            initial_joint_pose=[0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400]
        )

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
            teleop_device.pos_sensitivity = 0.25
            teleop_device.rot_sensitivity = 0.5
        else:
            teleop_device = None

        scene = Scene(assets=[background, light, obj])

        # Height-only success — closer to "lift/pick" than cube_goal_pose (height + yaw90).
        task = GoalPoseTask(
            obj,
            episode_length_s=cfg.episode_length_s,
            target_z_range=(cfg.success_z_min, cfg.success_z_max),
        )
        task.task_description = cfg.language_instruction

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
        )
