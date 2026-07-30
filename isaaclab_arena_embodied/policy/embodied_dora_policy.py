# Copyright (c) 2026, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena PolicyBase that calls embodied-rs ``dora-policy`` over stdio JSONL."""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_embodied.adapters.smolvla_libero import SmolVlaLiberoAdapter
from isaaclab_arena_embodied.policy.dora_stdio_client import (
    DoraStdioClient,
    action_chunk_wire_to_numpy,
    build_dora_policy_argv,
)
from isaaclab_arena_embodied.policy.embodied_dora_config import EmbodiedDoraPolicyCfg


def _resolve_adapter(name: str, cfg: EmbodiedDoraPolicyCfg) -> SmolVlaLiberoAdapter:
    if name != "smolvla_libero":
        raise ValueError(f"unknown embodiment_adapter={name!r}; supported: smolvla_libero")
    return SmolVlaLiberoAdapter(
        ee_action_scale=cfg.ee_action_scale,
        ee_pos_clip=cfg.ee_pos_clip,
        ee_rot_clip=cfg.ee_rot_clip,
        invert_gripper=cfg.invert_gripper,
        binarize_gripper=cfg.binarize_gripper,
    )


@register_policy
class EmbodiedDoraPolicy(PolicyBase[EmbodiedDoraPolicyCfg]):
    """Remote closed-loop policy: T1 adapter + dora-policy stdio + chunk replay.

    Chunk scheduling lives here (Arena / world side), not in ``dora-chunk-exec``.
    Unnorm is performed only inside ``dora-policy`` (PolicyEngine).
    """

    name = "embodied_dora"

    def __init__(self, config: EmbodiedDoraPolicyCfg) -> None:
        super().__init__(config)
        self.device = config.policy_device
        self._open_loop_horizon = int(config.open_loop_horizon)
        self._action_dim = int(config.action_dim)
        self._unnorm_key = config.unnorm_key
        self._adapter = _resolve_adapter(config.embodiment_adapter, config)
        self.task_description: str | None = None

        self._client: DoraStdioClient | None = None
        if config.spawn_policy or config.attach_command is not None:
            argv = self._build_argv(config)
            self._client = DoraStdioClient(
                argv=argv,
                ready_timeout_s=config.ready_timeout_s,
                predict_timeout_s=config.predict_timeout_s,
            )
            self._client.start()
            try:
                meta = self._client.meta()
                print(
                    f"[EmbodiedDoraPolicy] meta framework={meta.get('framework')} "
                    f"horizon={meta.get('action_chunk_size')} action_dim={meta.get('action_dim')}"
                )
                if meta.get("action_chunk_size"):
                    # Prefer server horizon when smaller than config (still clip).
                    server_h = int(meta["action_chunk_size"])
                    if server_h < self._open_loop_horizon:
                        print(
                            f"[EmbodiedDoraPolicy] clamping open_loop_horizon "
                            f"{self._open_loop_horizon} -> {server_h} (server)"
                        )
                        self._open_loop_horizon = server_h
            except Exception as exc:  # noqa: BLE001
                print(f"[EmbodiedDoraPolicy] meta() failed (continuing): {exc}")

        self._cached_action_chunks: list[np.ndarray | None] | None = None
        self._next_chunk_steps: list[int] | None = None

    @property
    def is_remote(self) -> bool:
        return True

    def _build_argv(self, config: EmbodiedDoraPolicyCfg) -> list[str]:
        if config.attach_command:
            return list(config.attach_command)
        if not config.dora_policy_bin:
            raise ValueError("dora_policy_bin is required when spawn_policy=True")
        if not config.model_dir:
            raise ValueError("model_dir is required when spawning dora-policy (or use synthetic bin flags)")
        return build_dora_policy_argv(
            dora_policy_bin=config.dora_policy_bin,
            model_dir=config.model_dir,
            device=config.device,
            unnorm_key=config.unnorm_key,
            inference_steps=config.inference_steps,
        )

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        assert self.task_description, (
            "EmbodiedDoraPolicy requires a language instruction "
            "(--language_instruction or task description via set_task_description)."
        )
        assert self._client is not None, "stdio client not started"

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

        batch = np.stack(actions, axis=0)  # (N, action_dim)
        return torch.from_numpy(batch).to(dtype=torch.float32, device=self.device)

    def _fetch_env_chunk(self, observation: dict[str, Any], env_id: int) -> np.ndarray:
        assert self._client is not None
        extracted = self._adapter.extract(observation, env_id)
        wire = self._adapter.to_observation_wire(
            extracted,
            instruction=self.task_description or "",
            unnorm_key=self._unnorm_key,
            timestamp_ns=time.time_ns(),
        )
        chunk_wire = self._client.predict(wire)
        chunk = action_chunk_wire_to_numpy(chunk_wire)
        if chunk.shape[1] < self._action_dim:
            raise RuntimeError(f"action_dim {chunk.shape[1]} < expected {self._action_dim}")
        # T1: env action space only (already unnormed by server).
        env_chunk = self._adapter.chunk_to_env(chunk[:, : self._action_dim])
        h = min(self._open_loop_horizon, env_chunk.shape[0])
        return env_chunk[:h].astype(np.float32, copy=True)

    def _maybe_init_per_env_state(self, num_envs: int) -> None:
        if self._cached_action_chunks is None:
            self._cached_action_chunks = [None] * num_envs
            self._next_chunk_steps = [0] * num_envs
            return
        assert len(self._cached_action_chunks) == num_envs, (
            f"num_envs changed mid-rollout ({len(self._cached_action_chunks)} -> {num_envs})"
        )

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
            self._client = None
