# Copyright (c) 2026, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""T1 adapter: Arena Franka obs ↔ embodied-rs policy-contract (smolvla_libero).

Aligned with embodied-rs ``deploy/profiles/smolvla_libero.yaml``:

* state: eef_pos(3) + axis_angle(3) + gripper_qpos(2)  → dim 8
* images: camera1=agentview, camera2=wrist, camera3=empty @ 256²;
  camera1/2 get 180° H+W flip; RGB f32 CHW in ~[0, 1]
* action: 7-D relative EE + grip (no second unnorm here)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Defaults match smolvla_libero profile.
IMAGE_SIZE = 256
STATE_DIM = 8
ACTION_DIM = 7
CAMERA1_KEY = "camera1"
CAMERA2_KEY = "camera2"
CAMERA3_KEY = "camera3"


@dataclass(frozen=True)
class SmolVlaExtracted:
    """Per-env tensors after T1 extract (before wire pack)."""

    agentview_hwc: np.ndarray  # uint8 HWC
    wrist_hwc: np.ndarray  # uint8 HWC
    state: np.ndarray  # (8,) float32
    eef_pos: np.ndarray
    axis_angle: np.ndarray
    gripper: np.ndarray


class SmolVlaLiberoAdapter:
    """World-side T1 only — never unnorms actions (PolicyEngine does that)."""

    name = "smolvla_libero"
    action_dim = ACTION_DIM
    state_dim = STATE_DIM
    image_size = IMAGE_SIZE

    arena_camera_obs_group = "camera_obs"
    arena_policy_obs_group = "policy"

    # Arena camera key candidates → agentview / wrist.
    # Prefer exact Arena cube_goal names first (`agentview_cam_rgb`, `wrist_cam_rgb`).
    agentview_keys = (
        "agentview_cam_rgb",
        "external_camera_rgb",
        "robot_pov_cam_rgb",
        "agentview_rgb",
        "agentview_image",
    )
    wrist_keys = (
        "wrist_cam_rgb",
        "wrist_camera_rgb",
        "eye_in_hand_rgb",
        "robot0_eye_in_hand_image",
    )

    def __init__(
        self,
        *,
        image_size: int = IMAGE_SIZE,
        # Arena FrankaIK DifferentialIK uses scale=0.5 → default 2.0 cancels to ~model units.
        ee_action_scale: float = 2.0,
        ee_pos_clip: float = 0.10,
        ee_rot_clip: float = 0.5,
        invert_gripper: bool = False,
        binarize_gripper: bool = True,
        gripper_deadzone: float = 0.0,
        # LIBERO raw cameras need 180° flip; Arena cameras are usually already upright.
        # Default False for Arena closed-loop; set True only when matching real LIBERO env raw.
        flip_hw_180: bool = False,
    ) -> None:
        self.image_size = int(image_size)
        self.ee_action_scale = float(ee_action_scale)
        self.ee_pos_clip = float(ee_pos_clip)
        self.ee_rot_clip = float(ee_rot_clip)
        self.invert_gripper = bool(invert_gripper)
        self.binarize_gripper = bool(binarize_gripper)
        self.gripper_deadzone = float(gripper_deadzone)
        self.flip_hw_180 = bool(flip_hw_180)

    # ------------------------------------------------------------------ extract

    def extract(self, observation: dict[str, Any], env_id: int) -> SmolVlaExtracted:
        proprio = observation[self.arena_policy_obs_group]
        eef_pos = self._vec(proprio, "eef_pos", env_id, default=np.zeros(3, dtype=np.float32))
        if "eef_axis_angle" in proprio:
            axis_angle = self._vec(proprio, "eef_axis_angle", env_id, default=np.zeros(3, dtype=np.float32))
        elif "eef_quat" in proprio:
            axis_angle = quat_xyzw_to_axis_angle(self._vec(proprio, "eef_quat", env_id, default=None))
        else:
            axis_angle = np.zeros(3, dtype=np.float32)

        if "gripper_pos" in proprio:
            grip = to_numpy(proprio["gripper_pos"][env_id]).astype(np.float32).reshape(-1)
        elif "gripper_qpos" in proprio:
            grip = to_numpy(proprio["gripper_qpos"][env_id]).astype(np.float32).reshape(-1)
        else:
            grip = np.zeros(2, dtype=np.float32)
        if grip.size == 1:
            grip = np.array([float(grip[0]), float(grip[0])], dtype=np.float32)
        elif grip.size >= 2:
            grip = grip[:2].astype(np.float32)
        else:
            grip = np.zeros(2, dtype=np.float32)

        state = np.concatenate([eef_pos[:3], axis_angle[:3], grip]).astype(np.float32)
        assert state.shape == (STATE_DIM,), f"state shape {state.shape}"

        agent = self._pick_camera(observation, env_id, self.agentview_keys)
        wrist = self._pick_camera(observation, env_id, self.wrist_keys)
        return SmolVlaExtracted(
            agentview_hwc=agent,
            wrist_hwc=wrist,
            state=state,
            eef_pos=eef_pos[:3].astype(np.float32),
            axis_angle=axis_angle[:3].astype(np.float32),
            gripper=grip,
        )

    def to_observation_wire(
        self,
        extracted: SmolVlaExtracted,
        *,
        instruction: str,
        unnorm_key: str | None = "default",
        timestamp_ns: int = 0,
    ) -> dict[str, Any]:
        """Build ObservationWire JSON-serializable dict for dora-policy."""
        a = resize_hwc(extracted.agentview_hwc, self.image_size)
        w = resize_hwc(extracted.wrist_hwc, self.image_size)
        if self.flip_hw_180:
            a = flip_hw_180(a)
            w = flip_hw_180(w)
        img1 = hwc_uint8_to_chw_f32(a)
        img2 = hwc_uint8_to_chw_f32(w)
        img3 = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
        return {
            "timestamp_ns": int(timestamp_ns),
            "instruction": instruction,
            "images": [
                image_wire(CAMERA1_KEY, img1),
                image_wire(CAMERA2_KEY, img2),
                image_wire(CAMERA3_KEY, img3),
            ],
            "state": extracted.state.astype(np.float32).tolist(),
            "unnorm_key": unnorm_key,
            "vision_features": None,
            "lang_features": None,
        }

    def action_row_to_env(self, action_7d: np.ndarray) -> np.ndarray:
        """Map one unnormed 7-D row to Arena ``franka_ik`` action space."""
        a = np.asarray(action_7d, dtype=np.float32).reshape(-1)
        if a.size < ACTION_DIM:
            raise ValueError(f"action dim {a.size} < {ACTION_DIM}")
        out = a[:ACTION_DIM].copy()
        out[0:3] = np.clip(out[0:3] * self.ee_action_scale, -self.ee_pos_clip, self.ee_pos_clip)
        out[3:6] = np.clip(out[3:6] * self.ee_action_scale, -self.ee_rot_clip, self.ee_rot_clip)
        g = float(out[6])
        if self.invert_gripper:
            g = -g
        if self.binarize_gripper:
            g = -1.0 if g < -self.gripper_deadzone else 1.0
        out[6] = g
        return out

    def chunk_to_env(self, chunk_th_d: np.ndarray) -> np.ndarray:
        """``[H, D]`` model chunk → ``[H, 7]`` env actions."""
        chunk = np.asarray(chunk_th_d, dtype=np.float32)
        if chunk.ndim != 2:
            raise ValueError(f"expected [H, D], got {chunk.shape}")
        return np.stack([self.action_row_to_env(row) for row in chunk], axis=0)

    # ------------------------------------------------------------------ helpers

    def _pick_camera(self, observation: dict[str, Any], env_id: int, keys: tuple[str, ...]) -> np.ndarray:
        cam_group = observation.get(self.arena_camera_obs_group)
        if cam_group is not None:
            for key in keys:
                if key in cam_group:
                    return as_hwc_uint8(to_numpy(cam_group[key][env_id]))
            for _k, v in cam_group.items():
                try:
                    img = to_numpy(v[env_id])
                    if getattr(img, "ndim", 0) >= 3:
                        return as_hwc_uint8(img)
                except Exception:  # noqa: BLE001
                    continue
        return np.full((self.image_size, self.image_size, 3), 128, dtype=np.uint8)

    def _vec(
        self,
        proprio: dict[str, Any],
        key: str,
        env_id: int,
        *,
        default: np.ndarray | None,
    ) -> np.ndarray:
        if key not in proprio:
            if default is None:
                raise KeyError(key)
            return np.asarray(default, dtype=np.float32).reshape(-1)
        return to_numpy(proprio[key][env_id]).astype(np.float32).reshape(-1)


def image_wire(key: str, chw: np.ndarray) -> dict[str, Any]:
    c, h, w = (int(x) for x in chw.shape)
    return {
        "key": key,
        "c": c,
        "h": h,
        "w": w,
        "data": chw.astype(np.float32).reshape(-1).tolist(),
    }


def to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def as_hwc_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        f = img.astype(np.float32)
        if f.max() <= 1.0 + 1e-3:
            f = f * 255.0
        img = np.clip(f, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def resize_hwc(img: np.ndarray, size: int) -> np.ndarray:
    img = as_hwc_uint8(img)
    if img.shape[0] == size and img.shape[1] == size:
        return img
    try:
        from PIL import Image
    except ImportError:
        # Nearest-neighbor fallback without PIL.
        ys = (np.linspace(0, img.shape[0] - 1, size)).astype(np.int64)
        xs = (np.linspace(0, img.shape[1] - 1, size)).astype(np.int64)
        return img[ys][:, xs]
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR), dtype=np.uint8)


def flip_hw_180(img: np.ndarray) -> np.ndarray:
    """180° rotation = flip H and W (LiberoProcessorStep)."""
    return np.ascontiguousarray(img[::-1, ::-1, ...])


def hwc_uint8_to_chw_f32(img: np.ndarray) -> np.ndarray:
    hwc = as_hwc_uint8(img).astype(np.float32) / 255.0
    return np.transpose(hwc, (2, 0, 1)).astype(np.float32)


def quat_xyzw_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Quaternion xyzw → axis-angle (3,)."""
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    if q.size < 4:
        return np.zeros(3, dtype=np.float32)
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # Normalize
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n < 1e-8:
        return np.zeros(3, dtype=np.float32)
    x, y, z, w = x / n, y / n, z / n, w / n
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    angle = 2.0 * float(np.arccos(np.clip(w, -1.0, 1.0)))
    s = (1.0 - w * w) ** 0.5
    if s < 1e-6:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([x / s, y / s, z / s], dtype=np.float32)
    return (axis * angle).astype(np.float32)
