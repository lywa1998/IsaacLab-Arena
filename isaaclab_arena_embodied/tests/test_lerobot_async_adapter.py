# Copyright (c) 2026, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LeRobot async raw-obs packing (no gRPC / no server)."""

from __future__ import annotations

import numpy as np
import torch

from isaaclab_arena_embodied.adapters.smolvla_libero import (
    CAMERA1_KEY,
    CAMERA2_KEY,
    CAMERA3_KEY,
    STATE_DIM,
    STATE_NAMES,
    SmolVlaLiberoAdapter,
    smolvla_libero_lerobot_features,
)


def _fake_obs() -> dict:
    return {
        "camera_obs": {
            "agentview_cam_rgb": torch.full((1, 64, 64, 3), 10, dtype=torch.uint8),
            "wrist_cam_rgb": torch.full((1, 64, 64, 3), 200, dtype=torch.uint8),
        },
        "policy": {
            "eef_pos": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
            "eef_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
            "gripper_pos": torch.tensor([[0.04, -0.04]], dtype=torch.float32),
        },
    }


def test_lerobot_features_layout():
    feats = smolvla_libero_lerobot_features(image_size=32)
    assert feats["observation.state"]["shape"] == [STATE_DIM]
    assert feats["observation.state"]["names"] == list(STATE_NAMES)
    assert f"observation.images.{CAMERA1_KEY}" in feats
    assert f"observation.images.{CAMERA3_KEY}" in feats


def test_to_lerobot_async_raw_obs():
    adapter = SmolVlaLiberoAdapter(image_size=32, flip_hw_180=False)
    ex = adapter.extract(_fake_obs(), 0)
    raw = adapter.to_lerobot_async_raw_obs(ex, instruction="pick up the cube")
    assert raw["task"] == "pick up the cube"
    for name in STATE_NAMES:
        assert name in raw
    assert raw[CAMERA1_KEY].shape == (32, 32, 3)
    assert raw[CAMERA2_KEY].shape == (32, 32, 3)
    assert raw[CAMERA3_KEY].shape == (32, 32, 3)
    assert raw[CAMERA3_KEY].dtype == np.uint8
    assert int(raw[CAMERA1_KEY].mean()) < 50
    assert int(raw[CAMERA2_KEY].mean()) > 150


def test_async_raw_obs_respects_state_ablation_mean():
    from isaaclab_arena_embodied.adapters.smolvla_libero import LIBERO_STATE_MEAN

    adapter = SmolVlaLiberoAdapter(image_size=16, state_ablation="mean")
    ex = adapter.extract(_fake_obs(), 0)
    raw = adapter.to_lerobot_async_raw_obs(ex, instruction="x")
    state = np.array([raw[n] for n in STATE_NAMES], dtype=np.float32)
    np.testing.assert_allclose(state, LIBERO_STATE_MEAN, atol=1e-5)
