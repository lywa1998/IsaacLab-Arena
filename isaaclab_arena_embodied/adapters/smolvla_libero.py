# Copyright (c) 2026, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""T1 adapter: Arena Franka obs ↔ embodied-rs policy-contract (smolvla_libero).

Aligned with embodied-rs ``deploy/profiles/smolvla_libero.yaml`` and LeRobot
``LiberoProcessorStep`` / OSC_POSE training domain:

* state: eef_pos(3) + axis_angle(3) + gripper_qpos(2)  → dim 8
* images: camera1=agentview, camera2=wrist, camera3=empty @ 256²;
  optional 180° H+W flip (default off for Arena upright cams); RGB f32 CHW ~[0, 1]
* action: 7-D relative EE + grip (unnorm only in PolicyEngine / LeRobot server)
* domain: ``align_eef_frame`` + ``action_unit=libero_osc`` map Arena env-frame
  proprio / DiffIK into LIBERO training stats + OSC-normalized action units
* diagnosis: ``state_ablation`` in {none, zero, mean}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

# Defaults match smolvla_libero profile.
IMAGE_SIZE = 256
STATE_DIM = 8
ACTION_DIM = 7
CAMERA1_KEY = "camera1"
CAMERA2_KEY = "camera2"
CAMERA3_KEY = "camera3"
# Flat state names for LeRobot async_inference raw obs / build_dataset_frame.
STATE_NAMES = tuple(f"s{i}" for i in range(STATE_DIM))

# From models/lerobot/smolvla_libero policy_preprocessor normalizer
# (observation.state.mean / .std). Used for OOD z-score logs and state_ablation=mean.
LIBERO_STATE_MEAN = np.array(
    [
        -0.04651878,
        0.03440907,
        0.7645525,
        2.9722095,
        -0.22046979,
        -0.1255794,
        0.02691425,
        -0.02719078,
    ],
    dtype=np.float32,
)
LIBERO_STATE_STD = np.array(
    [
        0.10494395,
        0.1517662,
        0.37851673,
        0.34427372,
        0.90694684,
        0.32539192,
        0.0141759,
        0.01405889,
    ],
    dtype=np.float32,
)

# Ablation modes for wire state (diagnosis R2).
# - none: raw Arena proprio (optionally eef-offset)
# - zero: all-zero vector (still OOD after MEAN_STD for axis-angle)
# - mean: package training mean → after MEAN_STD ≈ 0 (true proprio neutralize)
STATE_ABLATION_NONE = "none"
STATE_ABLATION_ZERO = "zero"
STATE_ABLATION_MEAN = "mean"
STATE_ABLATION_MODES = (STATE_ABLATION_NONE, STATE_ABLATION_ZERO, STATE_ABLATION_MEAN)

# --- Training-domain action units (LeRobot LIBERO / robosuite OSC_POSE) ---
# Unnormalized actions live in ~Box(-1, 1) controller units; OSC multiplies by
# output_max (pos ≈ 0.05 m, rot ≈ 0.5 rad). Arena FrankaIK DiffIK multiplies
# env action by scale=0.5, so env_action = a_ctrl * (output_max / ik_scale).
ACTION_UNIT_LEGACY = "legacy"
ACTION_UNIT_LIBERO_OSC = "libero_osc"
ACTION_UNIT_MODES = (ACTION_UNIT_LEGACY, ACTION_UNIT_LIBERO_OSC)

ARENA_DIFFIK_SCALE = 0.5
LIBERO_OSC_POS_OUTPUT_MAX = 0.05  # meters at |a|=1
LIBERO_OSC_ROT_OUTPUT_MAX = 0.5  # radians at |a|=1
LIBERO_OSC_POS_TO_ENV = LIBERO_OSC_POS_OUTPUT_MAX / ARENA_DIFFIK_SCALE  # 0.1
LIBERO_OSC_ROT_TO_ENV = LIBERO_OSC_ROT_OUTPUT_MAX / ARENA_DIFFIK_SCALE  # 1.0

# Arena ee_frame_pos = target_pos_w - env_origins (not LIBERO/robosuite world).
# Calibrated on libero_like_lift / cube_goal ready pose (dora diag first frame)
# so wire eef ≈ LIBERO training mean when the arm is at that ready configuration.
_ARENA_READY_EEF_APPROX = np.array([0.057, -0.024, 0.250], dtype=np.float32)
DEFAULT_EEF_POS_OFFSET = (LIBERO_STATE_MEAN[:3] - _ARENA_READY_EEF_APPROX).astype(np.float32)


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
        # legacy only: Arena DiffIK scale=0.5 → 2.0 cancelled model units (often saturates).
        ee_action_scale: float = 2.0,
        ee_pos_clip: float = 0.10,
        ee_rot_clip: float = 0.5,
        invert_gripper: bool = False,
        binarize_gripper: bool = True,
        gripper_deadzone: float = 0.0,
        # Diagnosis: negate EE position deltas (xyz) before clip — action-frame A/B.
        invert_ee_pos: bool = False,
        # Diagnosis: negate EE rotation deltas (rx,ry,rz) before clip.
        invert_ee_rot: bool = False,
        # LIBERO raw cameras need 180° flip; Arena cameras are usually already upright.
        # Training package is post-flip upright → default False for Arena.
        flip_hw_180: bool = False,
        # Diagnosis: replace wire state (R2 OOD). See STATE_ABLATION_* constants.
        state_ablation: str = STATE_ABLATION_NONE,
        # Near-π axis-angle double-cover: flip into LIBERO stats hemisphere (default on).
        align_axis_angle: bool = True,
        # Map Arena env-relative eef into LIBERO training world-ish frame (default on).
        align_eef_frame: bool = True,
        # Added to Arena eef_pos before packing state. None → DEFAULT_EEF_POS_OFFSET when
        # align_eef_frame else zeros.
        eef_pos_offset: Sequence[float] | None = None,
        # libero_osc: OSC_POSE controller units → DiffIK env actions (training domain).
        # legacy: single ee_action_scale for pos+rot (old diagnosis path).
        action_unit: str = ACTION_UNIT_LIBERO_OSC,
    ) -> None:
        self.image_size = int(image_size)
        self.ee_action_scale = float(ee_action_scale)
        self.ee_pos_clip = float(ee_pos_clip)
        self.ee_rot_clip = float(ee_rot_clip)
        self.invert_gripper = bool(invert_gripper)
        self.binarize_gripper = bool(binarize_gripper)
        self.gripper_deadzone = float(gripper_deadzone)
        self.invert_ee_pos = bool(invert_ee_pos)
        self.invert_ee_rot = bool(invert_ee_rot)
        self.flip_hw_180 = bool(flip_hw_180)
        mode = str(state_ablation or STATE_ABLATION_NONE).strip().lower()
        if mode not in STATE_ABLATION_MODES:
            raise ValueError(
                f"state_ablation={state_ablation!r}; expected one of {STATE_ABLATION_MODES}"
            )
        self.state_ablation = mode
        self.align_axis_angle = bool(align_axis_angle)
        self.align_eef_frame = bool(align_eef_frame)
        if eef_pos_offset is not None:
            off = np.asarray(eef_pos_offset, dtype=np.float32).reshape(-1)
            if off.size != 3:
                raise ValueError(f"eef_pos_offset must have 3 elements, got {off.size}")
            self.eef_pos_offset = off.astype(np.float32, copy=True)
        elif self.align_eef_frame:
            self.eef_pos_offset = DEFAULT_EEF_POS_OFFSET.copy()
        else:
            self.eef_pos_offset = np.zeros(3, dtype=np.float32)
        unit = str(action_unit or ACTION_UNIT_LIBERO_OSC).strip().lower()
        if unit not in ACTION_UNIT_MODES:
            raise ValueError(f"action_unit={action_unit!r}; expected one of {ACTION_UNIT_MODES}")
        self.action_unit = unit

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
        if self.align_axis_angle:
            axis_angle = align_axis_angle_to_libero(axis_angle)

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

        eef_raw = eef_pos[:3].astype(np.float32, copy=True)
        # Wire eef in LIBERO training frame (env-relative Arena + constant offset).
        eef_wire = (eef_raw + self.eef_pos_offset).astype(np.float32)
        state = np.concatenate([eef_wire, axis_angle[:3], grip]).astype(np.float32)
        assert state.shape == (STATE_DIM,), f"state shape {state.shape}"

        agent = self._pick_camera(observation, env_id, self.agentview_keys)
        wrist = self._pick_camera(observation, env_id, self.wrist_keys)
        return SmolVlaExtracted(
            agentview_hwc=agent,
            wrist_hwc=wrist,
            state=state,
            eef_pos=eef_raw,
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
        wire_state = apply_state_ablation(extracted.state, self.state_ablation)
        return {
            "timestamp_ns": int(timestamp_ns),
            "instruction": instruction,
            "images": [
                image_wire(CAMERA1_KEY, img1),
                image_wire(CAMERA2_KEY, img2),
                image_wire(CAMERA3_KEY, img3),
            ],
            "state": wire_state.astype(np.float32).tolist(),
            "unnorm_key": unnorm_key,
            "vision_features": None,
            "lang_features": None,
        }

    def to_lerobot_async_raw_obs(
        self,
        extracted: SmolVlaExtracted,
        *,
        instruction: str,
    ) -> dict[str, Any]:
        """Pack Arena extract into LeRobot ``async_inference`` raw robot observation.

        Keys match :func:`smolvla_libero_lerobot_features` / ``build_dataset_frame``:
        scalar state names ``s0..s7``, camera HWCs named ``camera1/2/3``, plus ``task``.
        Unnorm / MEAN_STD happen on the LeRobot policy server preprocessor.
        """
        a = resize_hwc(extracted.agentview_hwc, self.image_size)
        w = resize_hwc(extracted.wrist_hwc, self.image_size)
        if self.flip_hw_180:
            a = flip_hw_180(a)
            w = flip_hw_180(w)
        wire_state = apply_state_ablation(extracted.state, self.state_ablation)
        raw: dict[str, Any] = {
            name: float(wire_state[i]) for i, name in enumerate(STATE_NAMES)
        }
        raw[CAMERA1_KEY] = np.ascontiguousarray(a, dtype=np.uint8)
        raw[CAMERA2_KEY] = np.ascontiguousarray(w, dtype=np.uint8)
        raw[CAMERA3_KEY] = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        raw["task"] = instruction
        return raw

    def diagnose_state(self, state: np.ndarray) -> dict[str, Any]:
        """Return wire state, z-scores vs LIBERO stats, and ablation result."""
        raw = np.asarray(state, dtype=np.float32).reshape(-1)
        z = state_z_scores(raw)
        wire = apply_state_ablation(raw, self.state_ablation)
        z_wire = state_z_scores(wire)
        return {
            "state_ablation": self.state_ablation,
            "align_eef_frame": self.align_eef_frame,
            "eef_pos_offset": self.eef_pos_offset.tolist(),
            "action_unit": self.action_unit,
            "raw": raw.tolist(),
            "z_raw": z.tolist(),
            "wire": wire.tolist(),
            "z_wire": z_wire.tolist(),
            "max_abs_z_raw": float(np.max(np.abs(z))) if z.size else 0.0,
            "max_abs_z_wire": float(np.max(np.abs(z_wire))) if z_wire.size else 0.0,
        }

    def action_row_to_env(self, action_7d: np.ndarray) -> np.ndarray:
        """Map one unnormed 7-D row to Arena ``franka_ik`` action space.

        * ``libero_osc`` (default): treat unnormed action as OSC controller units
          in ~[-1, 1], map to DiffIK env actions so physical delta ≈ OSC output_max.
        * ``legacy``: multiply all EE dims by ``ee_action_scale`` (old path; often
          saturates at ``ee_pos_clip``).
        """
        a = np.asarray(action_7d, dtype=np.float32).reshape(-1)
        if a.size < ACTION_DIM:
            raise ValueError(f"action dim {a.size} < {ACTION_DIM}")
        out = a[:ACTION_DIM].copy()
        if self.invert_ee_pos:
            out[0:3] *= -1.0
        if self.invert_ee_rot:
            out[3:6] *= -1.0
        if self.action_unit == ACTION_UNIT_LIBERO_OSC:
            pos_s = LIBERO_OSC_POS_TO_ENV
            rot_s = LIBERO_OSC_ROT_TO_ENV
        else:
            pos_s = self.ee_action_scale
            rot_s = self.ee_action_scale
        out[0:3] = np.clip(out[0:3] * pos_s, -self.ee_pos_clip, self.ee_pos_clip)
        out[3:6] = np.clip(out[3:6] * rot_s, -self.ee_rot_clip, self.ee_rot_clip)
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


def state_z_scores(
    state: np.ndarray,
    mean: np.ndarray = LIBERO_STATE_MEAN,
    std: np.ndarray = LIBERO_STATE_STD,
    eps: float = 1e-6,
) -> np.ndarray:
    """MEAN_STD z-scores vs package stats (same as dora-policy preprocessor)."""
    s = np.asarray(state, dtype=np.float32).reshape(-1)
    m = np.asarray(mean, dtype=np.float32).reshape(-1)
    d = np.asarray(std, dtype=np.float32).reshape(-1)
    n = min(s.size, m.size, d.size)
    z = np.zeros(n, dtype=np.float32)
    for i in range(n):
        z[i] = (s[i] - m[i]) / max(abs(float(d[i])), eps)
    return z


def smolvla_libero_lerobot_features(*, image_size: int = IMAGE_SIZE) -> dict[str, dict[str, Any]]:
    """LeRobot dataset-style features for ``RemotePolicyConfig.lerobot_features``."""
    h = w = int(image_size)
    feats: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": list(STATE_NAMES),
        },
    }
    for cam in (CAMERA1_KEY, CAMERA2_KEY, CAMERA3_KEY):
        feats[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": [h, w, 3],
            "names": ["height", "width", "channels"],
        }
    return feats


def apply_state_ablation(state: np.ndarray, mode: str) -> np.ndarray:
    """Apply diagnosis ablation to raw Arena state before wire."""
    raw = np.asarray(state, dtype=np.float32).reshape(-1)
    if raw.size != STATE_DIM:
        # Still produce a fixed-size vector for the engine.
        out = np.zeros(STATE_DIM, dtype=np.float32)
        n = min(raw.size, STATE_DIM)
        out[:n] = raw[:n]
        raw = out
    mode = str(mode or STATE_ABLATION_NONE).strip().lower()
    if mode == STATE_ABLATION_ZERO:
        return np.zeros(STATE_DIM, dtype=np.float32)
    if mode == STATE_ABLATION_MEAN:
        return LIBERO_STATE_MEAN.copy()
    if mode != STATE_ABLATION_NONE:
        raise ValueError(f"unknown state_ablation={mode!r}")
    return raw.astype(np.float32, copy=True)


def hwc_uint8_to_chw_f32(img: np.ndarray) -> np.ndarray:
    hwc = as_hwc_uint8(img).astype(np.float32) / 255.0
    return np.transpose(hwc, (2, 0, 1)).astype(np.float32)


def quat_xyzw_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Quaternion xyzw → axis-angle (3,), matching LeRobot ``LiberoProcessorStep._quat2axisangle``.

    Important: do **not** force ``w >= 0``. LeRobot keeps the raw ``w`` sign so that
    ``w < 0`` yields angle ``> π``. Forcing the other double-cover maps near-π poses
    into the opposite axis-angle hemisphere (e.g. ``−π`` vs training mean ``+π``),
    which explodes MEAN_STD z-scores (~18σ on Arena first frame).
    """
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    if q.size < 4:
        return np.zeros(3, dtype=np.float32)
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n < 1e-8:
        return np.zeros(3, dtype=np.float32)
    x, y, z, w = x / n, y / n, z / n, w / n
    w = float(np.clip(w, -1.0, 1.0))
    den = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if den <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * float(np.arccos(w))
    axis = np.array([x / den, y / den, z / den], dtype=np.float32)
    return (axis * angle).astype(np.float32)


def align_axis_angle_to_libero(
    aa: np.ndarray,
    mean: np.ndarray | None = None,
    *,
    pi_band: float = 0.75,
) -> np.ndarray:
    """Flip near-π axis-angle into the LIBERO stats hemisphere when safer.

    For θ ≈ π, ``R(n, π) ≈ R(−n, π)`` so ``−aa`` is nearly the same SO(3) pose but
    lands next to training ``observation.state.mean[3:6] ≈ (+π, …)`` instead of
    ``(−π, …)`` (which is ~18σ OOD under MEAN_STD).

    Only flips when ``|‖aa‖ − π| < pi_band`` and ``−aa`` is closer to ``mean``.
    """
    aa = np.asarray(aa, dtype=np.float32).reshape(-1)
    if aa.size < 3:
        out = np.zeros(3, dtype=np.float32)
        out[: aa.size] = aa
        return out
    aa = aa[:3].astype(np.float32, copy=True)
    m = np.asarray(mean if mean is not None else LIBERO_STATE_MEAN[3:6], dtype=np.float32).reshape(-1)[:3]
    angle = float(np.linalg.norm(aa))
    if abs(angle - float(np.pi)) > float(pi_band):
        return aa
    flipped = (-aa).astype(np.float32)
    if float(np.linalg.norm(flipped - m)) < float(np.linalg.norm(aa - m)):
        return flipped
    return aa
