# Copyright (c) 2026, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena PolicyBase → LeRobot async_inference policy_server (gRPC).

Chunk scheduling stays on the Arena / world side (same as EmbodiedDoraPolicy).
LeRobot server owns model load + preprocessor MEAN_STD + unnorm postprocessor.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_embodied.adapters.smolvla_libero import (
    SmolVlaLiberoAdapter,
    smolvla_libero_lerobot_features,
)
from isaaclab_arena_embodied.policy.lerobot_async_client import LerobotAsyncClient
from isaaclab_arena_embodied.policy.lerobot_async_config import EmbodiedLerobotAsyncPolicyCfg


def _resolve_adapter(name: str, cfg: EmbodiedLerobotAsyncPolicyCfg) -> SmolVlaLiberoAdapter:
    if name != "smolvla_libero":
        raise ValueError(f"unknown embodiment_adapter={name!r}; supported: smolvla_libero")
    return SmolVlaLiberoAdapter(
        ee_action_scale=cfg.ee_action_scale,
        ee_pos_clip=cfg.ee_pos_clip,
        ee_rot_clip=cfg.ee_rot_clip,
        invert_gripper=cfg.invert_gripper,
        binarize_gripper=cfg.binarize_gripper,
        invert_ee_pos=cfg.invert_ee_pos,
        invert_ee_rot=cfg.invert_ee_rot,
        flip_hw_180=cfg.flip_hw_180,
        state_ablation=cfg.state_ablation,
        align_axis_angle=cfg.align_axis_angle,
        align_eef_frame=cfg.align_eef_frame,
        action_unit=cfg.action_unit,
    )


@register_policy
class EmbodiedLerobotAsyncPolicy(PolicyBase[EmbodiedLerobotAsyncPolicyCfg]):
    """Remote closed-loop: T1 adapter + LeRobot gRPC async server + chunk replay."""

    name = "embodied_lerobot_async"

    def __init__(self, config: EmbodiedLerobotAsyncPolicyCfg) -> None:
        super().__init__(config)
        self.device = config.policy_device
        self._open_loop_horizon = int(config.open_loop_horizon)
        self._action_dim = int(config.action_dim)
        self._adapter = _resolve_adapter(config.embodiment_adapter, config)
        self.task_description: str | None = None
        self._diag_log_chunks = max(0, int(config.diag_log_chunks))
        self._diag_logged = 0
        self._server_proc: subprocess.Popen[str] | None = None

        if not config.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path is required (local smolvla_libero dir or HF id)")

        print(
            f"[EmbodiedLerobotAsyncPolicy] server={config.server_host}:{config.server_port} "
            f"lerobot_policy_type={config.lerobot_policy_type} path={config.pretrained_name_or_path} "
            f"server_device={config.server_policy_device} flip={config.flip_hw_180} "
            f"action_unit={config.action_unit} align_eef_frame={config.align_eef_frame} "
            f"align_aa={config.align_axis_angle} invert_ee_pos={config.invert_ee_pos} "
            f"state_ablation={config.state_ablation}"
        )

        if config.spawn_server:
            self._spawn_server(config)

        features = smolvla_libero_lerobot_features(image_size=self._adapter.image_size)
        self._client = LerobotAsyncClient(
            server_address=f"{config.server_host}:{config.server_port}",
            policy_type=config.lerobot_policy_type,
            pretrained_name_or_path=config.pretrained_name_or_path,
            policy_device=config.server_policy_device,
            actions_per_chunk=config.actions_per_chunk,
            lerobot_features=features,
            rename_map={},
            ready_timeout_s=config.ready_timeout_s,
            get_actions_timeout_s=config.get_actions_timeout_s,
        )
        self._client.start()

        self._cached_action_chunks: list[np.ndarray | None] | None = None
        self._next_chunk_steps: list[int] | None = None

    @property
    def is_remote(self) -> bool:
        return True

    def _spawn_server(self, config: EmbodiedLerobotAsyncPolicyCfg) -> None:
        py = config.server_python or sys.executable
        argv = [
            py,
            "-m",
            "lerobot.async_inference.policy_server",
            f"--host={config.server_host}",
            f"--port={config.server_port}",
            *list(config.server_extra_args),
        ]
        env = os.environ.copy()
        print(f"[EmbodiedLerobotAsyncPolicy] spawning: {' '.join(argv)}")
        self._server_proc = subprocess.Popen(
            argv,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Give the process a moment to bind the port before Ready().
        time.sleep(1.5)
        if self._server_proc.poll() is not None:
            err = ""
            if self._server_proc.stderr is not None:
                err = self._server_proc.stderr.read() or ""
            raise RuntimeError(f"policy_server exited early code={self._server_proc.returncode}: {err[:800]}")

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        assert self.task_description, (
            "EmbodiedLerobotAsyncPolicy requires a language instruction "
            "(--language_instruction or set_task_description)."
        )
        assert self._client is not None

        num_envs = int(env.unwrapped.num_envs)
        self._maybe_init_per_env_state(num_envs)

        actions: list[np.ndarray] = []
        for env_id in range(num_envs):
            assert self._cached_action_chunks is not None and self._next_chunk_steps is not None
            exhausted = (
                self._cached_action_chunks[env_id] is None
                or self._next_chunk_steps[env_id] >= self._open_loop_horizon
            )
            if exhausted:
                self._cached_action_chunks[env_id] = self._fetch_env_chunk(observation, env_id)
                self._next_chunk_steps[env_id] = 0
            step_i = self._next_chunk_steps[env_id]
            row = self._cached_action_chunks[env_id][step_i]
            actions.append(row)
            self._next_chunk_steps[env_id] = step_i + 1

        batch = np.stack(actions, axis=0)
        return torch.from_numpy(batch).to(dtype=torch.float32, device=self.device)

    def _fetch_env_chunk(self, observation: dict[str, Any], env_id: int) -> np.ndarray:
        assert self._client is not None
        extracted = self._adapter.extract(observation, env_id)
        raw = self._adapter.to_lerobot_async_raw_obs(
            extracted,
            instruction=self.task_description or "",
        )
        if self._diag_logged < self._diag_log_chunks:
            diag = self._adapter.diagnose_state(extracted.state)
            print(
                f"[EmbodiedLerobotAsyncPolicy][diag] env={env_id} chunk={self._diag_logged} "
                f"ablation={diag['state_ablation']} "
                f"max|z|raw={diag['max_abs_z_raw']:.2f} max|z|wire={diag['max_abs_z_wire']:.2f}"
            )
        chunk = self._client.predict_chunk(raw, must_go=True)
        if chunk.shape[1] < self._action_dim:
            raise RuntimeError(f"action_dim {chunk.shape[1]} < expected {self._action_dim}")
        env_chunk = self._adapter.chunk_to_env(chunk[:, : self._action_dim])
        h = min(self._open_loop_horizon, env_chunk.shape[0])
        out = env_chunk[:h].astype(np.float32, copy=True)
        if self._diag_logged < self._diag_log_chunks:
            print(
                f"[EmbodiedLerobotAsyncPolicy][diag] env={env_id} chunk={self._diag_logged} "
                f"action_first={_fmt_vec(out[0].tolist())} "
                f"action_mean_xyz={_fmt_vec(out[:, :3].mean(axis=0).tolist())}"
            )
            self._diag_logged += 1
        return out

    def _maybe_init_per_env_state(self, num_envs: int) -> None:
        if self._cached_action_chunks is None:
            self._cached_action_chunks = [None] * num_envs
            self._next_chunk_steps = [0] * num_envs
            return
        assert len(self._cached_action_chunks) == num_envs

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._cached_action_chunks is None or self._next_chunk_steps is None:
            return
        ids = range(len(self._cached_action_chunks)) if env_ids is None else env_ids.reshape(-1).tolist()
        for env_id in ids:
            i = int(env_id)
            self._cached_action_chunks[i] = None
            self._next_chunk_steps[i] = 0

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None  # type: ignore[assignment]
        if self._server_proc is not None:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
            self._server_proc = None

    def shutdown_remote(self, kill_server: bool = False) -> None:
        """Arena remote cleanup. If we spawned the server, always stop it."""
        _ = kill_server
        self.close()


def _fmt_vec(values: list[float] | np.ndarray, nd: int = 3) -> str:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{float(v):.{nd}f}" for v in arr) + "]"
