# Copyright (c) 2026, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for T1 smolvla_libero adapter (no Isaac / no dora-policy)."""

from __future__ import annotations

import numpy as np
import torch

from isaaclab_arena_embodied.adapters.smolvla_libero import (
    ACTION_DIM,
    LIBERO_STATE_MEAN,
    STATE_DIM,
    SmolVlaLiberoAdapter,
    align_axis_angle_to_libero,
    apply_state_ablation,
    flip_hw_180,
    quat_xyzw_to_axis_angle,
    state_z_scores,
)


def _fake_observation(num_envs: int = 1) -> dict:
    return {
        "camera_obs": {
            "external_camera_rgb": torch.zeros((num_envs, 128, 160, 3), dtype=torch.uint8),
            "wrist_camera_rgb": torch.full((num_envs, 128, 160, 3), 40, dtype=torch.uint8),
        },
        "policy": {
            "eef_pos": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32).expand(num_envs, -1).clone(),
            "eef_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32).expand(num_envs, -1).clone(),
            "gripper_pos": torch.tensor([[0.04]], dtype=torch.float32).expand(num_envs, -1).clone(),
            "joint_pos": torch.zeros((num_envs, 7), dtype=torch.float32),
        },
    }


def test_state_layout_dim8():
    adapter = SmolVlaLiberoAdapter()
    ex = adapter.extract(_fake_observation(), 0)
    assert ex.state.shape == (STATE_DIM,)
    np.testing.assert_allclose(ex.state[:3], [0.1, 0.2, 0.3], atol=1e-5)
    # identity quat → zero axis-angle
    np.testing.assert_allclose(ex.state[3:6], 0.0, atol=1e-5)
    assert ex.state[6] == ex.state[7]  # gripper padded to 2


def test_observation_wire_cameras_and_flip():
    adapter = SmolVlaLiberoAdapter(image_size=64, flip_hw_180=True)
    # Distinct pattern so flip is observable.
    obs = _fake_observation()
    img = np.zeros((128, 160, 3), dtype=np.uint8)
    img[0, 0, :] = 255
    obs["camera_obs"]["external_camera_rgb"][0] = torch.from_numpy(img)
    ex = adapter.extract(obs, 0)
    wire = adapter.to_observation_wire(ex, instruction="pick the cube", unnorm_key="default")
    assert wire["instruction"] == "pick the cube"
    assert len(wire["images"]) == 3
    assert wire["images"][0]["key"] == "camera1"
    assert wire["images"][1]["key"] == "camera2"
    assert wire["images"][2]["key"] == "camera3"
    assert wire["images"][0]["c"] == 3
    assert wire["images"][0]["h"] == 64
    assert wire["images"][0]["w"] == 64
    assert len(wire["images"][0]["data"]) == 3 * 64 * 64
    assert wire["state"] is not None and len(wire["state"]) == STATE_DIM
    # camera3 is zeros
    assert max(wire["images"][2]["data"]) == 0.0


def test_arena_camera_keys_preferred():
    adapter = SmolVlaLiberoAdapter(image_size=32, flip_hw_180=False)
    obs = {
        "camera_obs": {
            "agentview_cam_rgb": torch.full((1, 32, 32, 3), 10, dtype=torch.uint8),
            "wrist_cam_rgb": torch.full((1, 32, 32, 3), 200, dtype=torch.uint8),
        },
        "policy": {
            "eef_pos": torch.zeros((1, 3), dtype=torch.float32),
            "eef_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
            "gripper_pos": torch.zeros((1, 1), dtype=torch.float32),
        },
    }
    ex = adapter.extract(obs, 0)
    assert int(ex.agentview_hwc.mean()) < 50
    assert int(ex.wrist_hwc.mean()) > 150

def test_flip_hw_180():
    img = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    flipped = flip_hw_180(img)
    np.testing.assert_array_equal(flipped[0, 0], img[-1, -1])


def test_action_row_to_env_dims():
    adapter = SmolVlaLiberoAdapter(binarize_gripper=True)
    row = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5], dtype=np.float32)
    env_a = adapter.action_row_to_env(row)
    assert env_a.shape == (ACTION_DIM,)
    assert env_a[6] == -1.0  # closed


def test_invert_ee_pos_negates_xyz():
    adapter = SmolVlaLiberoAdapter(ee_action_scale=1.0, invert_ee_pos=True, binarize_gripper=False)
    row = np.array([0.05, -0.02, 0.03, 0.0, 0.0, 0.0, 0.1], dtype=np.float32)
    env_a = adapter.action_row_to_env(row)
    np.testing.assert_allclose(env_a[:3], [-0.05, 0.02, -0.03], atol=1e-5)


def test_chunk_to_env():
    adapter = SmolVlaLiberoAdapter()
    chunk = np.zeros((10, 7), dtype=np.float32)
    chunk[:, 6] = 0.5
    out = adapter.chunk_to_env(chunk)
    assert out.shape == (10, 7)
    assert np.all(out[:, 6] == 1.0)


def test_quat_identity():
    aa = quat_xyzw_to_axis_angle(np.array([0, 0, 0, 1], dtype=np.float32))
    np.testing.assert_allclose(aa, 0.0, atol=1e-5)


def test_quat2aa_matches_lerobot_no_w_flip():
    """LeRobot keeps w sign; w≈0- yields angle slightly above π (positive hemisphere)."""
    # Double-cover of near-π rotation around +x
    q_pos_w = np.array([0.9997, 0.0, 0.0, 0.0023], dtype=np.float32)
    q_neg_w = -q_pos_w
    aa_pos = quat_xyzw_to_axis_angle(q_pos_w)
    aa_neg = quat_xyzw_to_axis_angle(q_neg_w)
    # Opposite double-cover → opposite axis-angle hemisphere
    assert aa_pos[0] * aa_neg[0] < 0
    # With align, both land near LIBERO mean hemisphere (+π)
    al_pos = align_axis_angle_to_libero(aa_pos)
    al_neg = align_axis_angle_to_libero(aa_neg)
    assert al_pos[0] > 2.5 and al_neg[0] > 2.5


def test_align_axis_angle_near_pi_to_libero_mean():
    raw = np.array([-3.136, -0.008, -0.078], dtype=np.float32)
    aligned = align_axis_angle_to_libero(raw)
    assert aligned[0] > 0
    z = state_z_scores(np.concatenate([np.zeros(3), aligned, np.zeros(2)]))
    assert abs(z[3]) < 1.0  # was ~18σ before flip


def test_identity_quat_state_is_ood_on_axis_angle():
    """Arena identity orientation → axis_angle≈0 vs LIBERO mean≈π → still OOD on angle.

    Identity is a true pose mismatch (not double-cover); align does not invent π.
    """
    adapter = SmolVlaLiberoAdapter(align_axis_angle=True)
    ex = adapter.extract(_fake_observation(), 0)
    z = state_z_scores(ex.state)
    # dim 3 (axis_angle[0]): mean ~2.97, std ~0.34 → z ≈ -8.6 for zero aa
    assert z[3] < -5.0


def test_extract_aligns_near_pi_quat():
    """Near-π quat that raw-converts to −π is flipped into +π hemisphere."""
    adapter = SmolVlaLiberoAdapter(align_axis_angle=True)
    # Build quat for aa ≈ −π on x (w small positive, x negative)
    angle = 3.136
    half = angle / 2.0
    q = np.array([-np.sin(half), 0.0, 0.0, np.cos(half)], dtype=np.float32)
    obs = _fake_observation()
    obs["policy"]["eef_quat"] = torch.tensor([q.tolist()], dtype=torch.float32)
    ex = adapter.extract(obs, 0)
    assert ex.axis_angle[0] > 2.5
    z = state_z_scores(ex.state)
    assert abs(z[3]) < 2.0


def test_state_ablation_zero_and_mean():
    raw = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04, 0.04], dtype=np.float32)
    z0 = apply_state_ablation(raw, "zero")
    np.testing.assert_allclose(z0, 0.0)
    zm = apply_state_ablation(raw, "mean")
    np.testing.assert_allclose(zm, LIBERO_STATE_MEAN, atol=1e-5)
    # mean → near-zero z after MEAN_STD
    z_scores = state_z_scores(zm)
    np.testing.assert_allclose(z_scores, 0.0, atol=1e-4)


def test_observation_wire_respects_state_ablation_mean():
    adapter = SmolVlaLiberoAdapter(image_size=32, state_ablation="mean")
    ex = adapter.extract(_fake_observation(), 0)
    wire = adapter.to_observation_wire(ex, instruction="pick")
    np.testing.assert_allclose(wire["state"], LIBERO_STATE_MEAN.tolist(), atol=1e-5)
    diag = adapter.diagnose_state(ex.state)
    assert diag["max_abs_z_wire"] < 1e-3
    assert diag["max_abs_z_raw"] > 5.0
