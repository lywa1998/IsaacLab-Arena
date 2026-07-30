# isaaclab_arena_embodied

Arena **world-side** bridge to [embodied-rs](https://github.com/ptrdlr14/embodied-rs):

| Layer | Location |
|-------|----------|
| T1 (body layout) | `adapters/smolvla_libero.py` |
| Chunk pop | `EmbodiedDoraPolicy.get_action` |
| T2 + predict + unnorm | **embodied-rs** `dora-policy` / PolicyEngine |

Transport: **stdio JSONL** (`dora-policy --stdio-loop`), not WebSocket/gRPC.

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
  --device cuda \
  --policy_device cuda \
  --spawn_policy \
  --language_instruction "pick up the cube" \
  --num_steps 50 \
  cube_goal_pose
```

Prefer embodiment **`franka_ik`** (7-D relative EE) over absolute joint_pos for `smolvla_libero`.

## Profile alignment

See embodied-rs `deploy/profiles/smolvla_libero.yaml`:

- state dim **8**: eef_pos + axis_angle + gripper_qpos(2)
- images: camera1/2/3, **180° flip** on 1/2, 256²
- action dim **7**, unnorm only in dora-policy
