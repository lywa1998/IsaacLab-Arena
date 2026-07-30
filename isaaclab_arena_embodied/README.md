# isaaclab_arena_embodied

Arena **world-side** bridge to [embodied-rs](https://github.com/ptrdlr14/embodied-rs) and (eval-only) LeRobot async server:

| Layer | Location |
|-------|----------|
| T1 (body layout) | `adapters/smolvla_libero.py` |
| Chunk pop | `EmbodiedDoraPolicy` / `EmbodiedLerobotAsyncPolicy` |
| **Production** T2 + predict + unnorm | **embodied-rs** `dora-policy` / PolicyEngine (stdio) |
| **Eval** T2 + predict + unnorm | LeRobot `async_inference.policy_server` (gRPC) |

Production transport remains **Dora / stdio** (DESIGN.md). LeRobot gRPC is for **protocol eval / diagnosis** only.

## Install

From the Arena repo root (package is included via `isaaclab_arena_embodied*` in `pyproject.toml`):

```bash
# already on PYTHONPATH when the Arena tree is installed editable
uv sync   # or your Arena docker workflow
```

## Unit tests (no Isaac Sim)

```bash
# From Arena root, with numpy/torch/pytest available:
PYTHONPATH=. pytest isaaclab_arena_embodied/tests -q
```

## Run with policy_runner

Terminal A or spawn from policy:

```bash
# Build embodied-rs first
cargo build --release --bin dora-policy   # from embodied-rs root

./target/release/dora-policy \
  --model-dir models/lerobot/smolvla_libero \
  --device cuda \
  --unnorm-key default \
  --stdio-loop
# wait for: dora-policy STDIO_LOOP_READY
```

With **spawn** (policy starts dora-policy):

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type isaaclab_arena_embodied.policy.embodied_dora_policy.EmbodiedDoraPolicy \
  --dora_policy_bin /path/to/embodied-rs/target/release/dora-policy \
  --model_dir /path/to/models/lerobot/smolvla_libero \
  --dora_device cuda \
  --policy_device cuda \
  --spawn_policy \
  --language_instruction "pick up the cube" \
  --num_steps 50 \
  cube_goal_pose
```

Prefer embodiment **`franka_ik`** (7-D relative EE) over absolute joint_pos for `smolvla_libero`.

## LIBERO-like scene (recommended for adapter smoke)

Environment name: **`libero_like_lift`** (`isaaclab_arena_environments/libero_like_lift_environment.py`).

| 项 | 选择 |
|----|------|
| 成功 | **仅抬高**（`target_z_range`，无 yaw-90） |
| 控制 | `franka_ik` relative EE |
| 相机 | agentview + wrist @ 256² |
| 指令 | 默认 `pick up the cube` |

```bash
# Dora
BACKEND=dora bash scripts/run_arena_libero_like_eval.sh   # from embodied-rs root on university

# LeRobot async (server already on :8080)
BACKEND=async bash scripts/run_arena_libero_like_eval.sh
```

Still **not** bit-exact MuJoCo LIBERO — for true suite SR use `scripts/run_lerobot_libero_eval.sh`.

## Eval: LeRobot async server + Arena client

Terminal A — start LeRobot policy server (loads model on first client `SendPolicyInstructions`):

```bash
# university: see embodied-rs scripts/run_lerobot_async_server.sh
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080 --fps=30
```

Terminal B — Arena closed-loop client:

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type isaaclab_arena_embodied.policy.lerobot_async_policy.EmbodiedLerobotAsyncPolicy \
  --pretrained_name_or_path /path/to/models/lerobot/smolvla_libero \
  --server_host 127.0.0.1 \
  --server_port 8080 \
  --server_policy_device cuda \
  --lerobot_policy_type smolvla \
  --policy_device cuda \
  --ee_action_scale 2.0 \
  --align_axis_angle \
  --language_instruction "pick up the cube" \
  --num_episodes 1 \
  --enable_cameras \
  --record_camera_video \
  cube_goal_pose
```

Or one-shot spawn from Arena: `--spawn_server --server_python /usr/bin/python3.12`.

Helper scripts (embodied-rs repo):

- `scripts/run_lerobot_async_server.sh`
- `scripts/run_arena_lerobot_async_eval.sh`

## Profile alignment

See embodied-rs `deploy/profiles/smolvla_libero.yaml`:

- state dim **8**: eef_pos + axis_angle + gripper_qpos(2)
- images: camera1/2/3, 256²; **flip_hw_180 default false** (Arena upright)
- action dim **7**, unnorm on server (dora-policy **or** LeRobot postprocessor)
- diagnosis: `--state_ablation none|zero|mean`, `--diag_log_chunks N`, `--invert_ee_pos`
