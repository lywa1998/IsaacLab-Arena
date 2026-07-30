# Copyright (c) 2026, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Policy unit tests with a fake stdio client (no real dora-policy binary)."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

pytest.importorskip("isaaclab")  # full Arena stack; adapter tests do not need it

from isaaclab_arena.assets.registries import PolicyRegistry
from isaaclab_arena_embodied.policy.embodied_dora_config import EmbodiedDoraPolicyCfg
from isaaclab_arena_embodied.policy.embodied_dora_policy import EmbodiedDoraPolicy


def _fake_env(num_envs: int = 1):
    return types.SimpleNamespace(unwrapped=types.SimpleNamespace(num_envs=num_envs))


def _fake_observation(num_envs: int = 1) -> dict:
    return {
        "camera_obs": {
            "external_camera_rgb": torch.zeros((num_envs, 64, 64, 3), dtype=torch.uint8),
            "wrist_camera_rgb": torch.zeros((num_envs, 64, 64, 3), dtype=torch.uint8),
        },
        "policy": {
            "eef_pos": torch.zeros((num_envs, 3), dtype=torch.float32),
            "eef_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32).expand(num_envs, -1).clone(),
            "gripper_pos": torch.zeros((num_envs, 1), dtype=torch.float32),
        },
    }


class _FakeClient:
    def __init__(self, horizon: int = 4, action_dim: int = 7):
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_predict = 0

    def start(self) -> None:
        return None

    def meta(self) -> dict:
        return {
            "framework": "SmolVLA",
            "action_chunk_size": self.horizon,
            "action_dim": self.action_dim,
        }

    def predict(self, observation_wire: dict) -> dict:
        self.n_predict += 1
        assert "images" in observation_wire and "state" in observation_wire
        h, d = self.horizon, self.action_dim
        values = np.zeros((h, d), dtype=np.float32)
        for t in range(h):
            values[t, 0] = float(t)  # distinguishable rows
            values[t, 6] = 0.5  # open gripper
        return {
            "values": values.reshape(-1).tolist(),
            "batch": 1,
            "horizon": h,
            "action_dim": d,
            "space": "relative_ee",
            "normalized": False,
            "timestamp_ns": 0,
            "error": None,
        }

    def close(self) -> None:
        return None


def test_policy_registers_as_embodied_dora():
    # Import of EmbodiedDoraPolicy module triggers @register_policy (no full asset load).
    assert PolicyRegistry().is_registered("embodied_dora", ensure_loaded=False)
    # Avoid get_policy() here — it may ensure_assets_registered() and pull Isaac packages.
    reg = PolicyRegistry()
    assert reg._components["embodied_dora"] is EmbodiedDoraPolicy


def test_chunk_replay_advances(monkeypatch):
    fake = _FakeClient(horizon=3)

    def _fake_init(self, config):
        # Bypass real spawn; wire fake client + adapter like real __init__.
        from isaaclab_arena.policy.policy_base import PolicyBase
        from isaaclab_arena_embodied.adapters.smolvla_libero import SmolVlaLiberoAdapter

        PolicyBase.__init__(self, config)
        self.device = "cpu"
        self._open_loop_horizon = 3
        self._action_dim = 7
        self._unnorm_key = "default"
        self._adapter = SmolVlaLiberoAdapter()
        self.task_description = "pick cube"
        self._client = fake
        self._cached_action_chunks = None
        self._next_chunk_steps = None

    monkeypatch.setattr(EmbodiedDoraPolicy, "__init__", _fake_init)
    policy = EmbodiedDoraPolicy(EmbodiedDoraPolicyCfg(spawn_policy=False))
    env = _fake_env(1)
    obs = _fake_observation(1)

    a0 = policy.get_action(env, obs).cpu().numpy()
    a1 = policy.get_action(env, obs).cpu().numpy()
    a2 = policy.get_action(env, obs).cpu().numpy()
    assert fake.n_predict == 1
    assert a0[0, 0] == 0.0
    assert a1[0, 0] == 1.0
    assert a2[0, 0] == 2.0

    # Next step refetches
    a3 = policy.get_action(env, obs).cpu().numpy()
    assert fake.n_predict == 2
    assert a3[0, 0] == 0.0

    policy.reset()
    _ = policy.get_action(env, obs)
    assert fake.n_predict == 3
