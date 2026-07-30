# Copyright (c) 2026, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""embodied-rs bridge package (T1 adapter + PolicyBase clients).

Policies:

* ``embodied_dora`` — Dora / ``dora-policy`` stdio (**production**)
* ``embodied_lerobot_async`` — LeRobot ``async_inference`` gRPC (**eval only**)

Import to register::

    import isaaclab_arena_embodied.policy.embodied_dora_policy  # noqa: F401
    import isaaclab_arena_embodied.policy.lerobot_async_policy  # noqa: F401

Or pass the full class path to ``policy_runner --policy_type``.
"""
