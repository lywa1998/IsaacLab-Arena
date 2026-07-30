# Copyright (c) 2026, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Thin gRPC client for LeRobot ``async_inference.policy_server`` (AsyncInference).

Eval / diagnosis path — not the Dora production transport (see DESIGN.md).
"""

from __future__ import annotations

import pickle  # nosec B403 — LeRobot wire format
import time
from typing import Any

import numpy as np


class LerobotAsyncClient:
    """Synchronous request/response client over LeRobot AsyncInference RPCs.

    Protocol (per chunk):
      1. Ready (once at start)
      2. SendPolicyInstructions (once; loads model on server)
      3. loop: SendObservations → GetActions
    """

    def __init__(
        self,
        *,
        server_address: str = "127.0.0.1:8080",
        policy_type: str = "smolvla",
        pretrained_name_or_path: str,
        policy_device: str = "cuda",
        actions_per_chunk: int = 50,
        lerobot_features: dict[str, Any],
        rename_map: dict[str, str] | None = None,
        ready_timeout_s: float = 30.0,
        get_actions_timeout_s: float = 120.0,
    ) -> None:
        self.server_address = str(server_address)
        self.policy_type = str(policy_type)
        self.pretrained_name_or_path = str(pretrained_name_or_path)
        self.policy_device = str(policy_device)
        self.actions_per_chunk = int(actions_per_chunk)
        self.lerobot_features = lerobot_features
        self.rename_map = dict(rename_map or {})
        self.ready_timeout_s = float(ready_timeout_s)
        self.get_actions_timeout_s = float(get_actions_timeout_s)

        self._channel = None
        self._stub = None
        self._timestep = 0
        self._started = False

    def start(self) -> None:
        import grpc
        from lerobot.async_inference.helpers import RemotePolicyConfig
        from lerobot.transport import services_pb2, services_pb2_grpc
        from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

        self._services_pb2 = services_pb2
        self._send_bytes_in_chunks = send_bytes_in_chunks

        opts = grpc_channel_options()
        self._channel = grpc.insecure_channel(self.server_address, options=opts)
        self._stub = services_pb2_grpc.AsyncInferenceStub(self._channel)

        # Ready handshake
        t0 = time.perf_counter()
        while True:
            try:
                self._stub.Ready(services_pb2.Empty(), timeout=min(5.0, self.ready_timeout_s))
                break
            except grpc.RpcError as exc:
                if time.perf_counter() - t0 > self.ready_timeout_s:
                    raise TimeoutError(
                        f"LeRobot policy_server Ready() failed at {self.server_address}: {exc}"
                    ) from exc
                time.sleep(0.5)

        # Load policy on server
        policy_cfg = RemotePolicyConfig(
            policy_type=self.policy_type,
            pretrained_name_or_path=self.pretrained_name_or_path,
            lerobot_features=self.lerobot_features,  # type: ignore[arg-type]
            actions_per_chunk=self.actions_per_chunk,
            device=self.policy_device,
            rename_map=self.rename_map,
        )
        setup = services_pb2.PolicySetup(data=pickle.dumps(policy_cfg))
        print(
            f"[LerobotAsyncClient] SendPolicyInstructions type={self.policy_type} "
            f"path={self.pretrained_name_or_path} device={self.policy_device}"
        )
        self._stub.SendPolicyInstructions(setup, timeout=self.ready_timeout_s)
        self._started = True
        self._timestep = 0
        print(f"[LerobotAsyncClient] connected {self.server_address}")

    def predict_chunk(self, raw_observation: dict[str, Any], *, must_go: bool = True) -> np.ndarray:
        """Send one observation; return unnormalized action chunk ``[H, D]`` float32."""
        if not self._started or self._stub is None:
            raise RuntimeError("LerobotAsyncClient.start() not called")

        from lerobot.async_inference.helpers import TimedObservation

        timed = TimedObservation(
            timestamp=time.time(),
            observation=raw_observation,
            timestep=self._timestep,
            must_go=must_go,
        )
        obs_bytes = pickle.dumps(timed)
        obs_iter = self._send_bytes_in_chunks(
            obs_bytes,
            self._services_pb2.Observation,
            log_prefix="[Arena→LerobotAsync] Observation",
            silent=True,
        )
        _ = self._stub.SendObservations(obs_iter)

        # Poll GetActions until non-empty chunk (server returns Empty on timeout)
        deadline = time.perf_counter() + self.get_actions_timeout_s
        timed_actions = None
        while time.perf_counter() < deadline:
            actions_msg = self._stub.GetActions(
                self._services_pb2.Empty(),
                timeout=max(1.0, min(self.get_actions_timeout_s, 30.0)),
            )
            if actions_msg is None or len(actions_msg.data) == 0:
                continue
            timed_actions = pickle.loads(actions_msg.data)  # nosec B301
            break
        if timed_actions is None:
            raise TimeoutError(
                f"GetActions empty for {self.get_actions_timeout_s}s "
                f"(server may not have enqueued obs; check must_go / server logs)"
            )

        rows: list[np.ndarray] = []
        for ta in timed_actions:
            act = ta.get_action()
            if hasattr(act, "detach"):
                act = act.detach().cpu().numpy()
            row = np.asarray(act, dtype=np.float32).reshape(-1)
            rows.append(row)
        if not rows:
            raise RuntimeError("GetActions returned zero actions")
        chunk = np.stack(rows, axis=0).astype(np.float32, copy=False)
        self._timestep += 1
        return chunk

    def close(self) -> None:
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:  # noqa: BLE001
                pass
        self._channel = None
        self._stub = None
        self._started = False
