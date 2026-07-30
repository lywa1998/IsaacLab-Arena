# Copyright (c) 2026, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""JSONL client for embodied-rs ``dora-policy --stdio-loop``."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Sequence
from typing import Any, TextIO

import numpy as np


READY_MARKER = "dora-policy STDIO_LOOP_READY"


class DoraStdioClientError(RuntimeError):
    """stdio protocol or process failure."""


class DoraStdioClient:
    """Spawn (or wrap) dora-policy and exchange ObservationWire / ActionChunkWire lines."""

    def __init__(
        self,
        *,
        argv: Sequence[str],
        ready_timeout_s: float = 600.0,
        predict_timeout_s: float = 120.0,
        env: dict[str, str] | None = None,
    ) -> None:
        self._argv = list(argv)
        self._ready_timeout_s = float(ready_timeout_s)
        self._predict_timeout_s = float(predict_timeout_s)
        self._env = env
        self._proc: subprocess.Popen[str] | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._ready = threading.Event()

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def start(self) -> None:
        if self._proc is not None:
            return
        print(f"[DoraStdioClient] spawning: {' '.join(self._argv)}")
        child_env = os.environ.copy()
        if self._env:
            child_env.update(self._env)
        self._proc = subprocess.Popen(
            self._argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered where supported
            env=child_env,
        )
        assert self._proc.stdin is not None and self._proc.stdout is not None and self._proc.stderr is not None
        self._stderr_thread = threading.Thread(target=self._drain_stderr, args=(self._proc.stderr,), daemon=True)
        self._stderr_thread.start()
        if not self._ready.wait(timeout=self._ready_timeout_s):
            self.close()
            tail = "\n".join(self._stderr_lines[-40:])
            raise DoraStdioClientError(
                f"timeout waiting for {READY_MARKER!r} after {self._ready_timeout_s}s.\nstderr tail:\n{tail}"
            )
        print(f"[DoraStdioClient] ready pid={self._proc.pid}")

    def _drain_stderr(self, stream: TextIO) -> None:
        try:
            for line in stream:
                line = line.rstrip("\n")
                self._stderr_lines.append(line)
                if READY_MARKER in line:
                    self._ready.set()
                # Keep useful logs visible.
                if line:
                    print(f"[dora-policy] {line}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[DoraStdioClient] stderr reader stopped: {exc}")

    def meta(self) -> dict[str, Any]:
        return self._roundtrip({"cmd": "meta"})

    def predict(self, observation_wire: dict[str, Any]) -> dict[str, Any]:
        """Send ObservationWire dict; return ActionChunkWire dict."""
        out = self._roundtrip(observation_wire)
        if out.get("error"):
            raise DoraStdioClientError(f"predict error: {out['error']}")
        if out.get("normalized", True):
            raise DoraStdioClientError(
                f"expected normalized=false from dora-policy (server unnorm); got {out.get('normalized')!r}"
            )
        return out

    def _roundtrip(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise DoraStdioClientError("client not started")
        if self._proc.poll() is not None:
            raise DoraStdioClientError(f"dora-policy exited early code={self._proc.returncode}")

        line = json.dumps(payload, separators=(",", ":"))
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise DoraStdioClientError("broken pipe writing to dora-policy") from exc

        deadline = time.monotonic() + self._predict_timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise DoraStdioClientError(f"dora-policy exited during predict code={self._proc.returncode}")
            # Blocking readline with overall deadline via poll is awkward in pure text mode;
            # rely on process liveness + timeout loop with short sleeps if line empty.
            # Use a worker for timeout on readline.
            result: list[str | None] = [None]
            err: list[BaseException | None] = [None]

            def _read() -> None:
                try:
                    assert self._proc is not None and self._proc.stdout is not None
                    result[0] = self._proc.stdout.readline()
                except BaseException as e:  # noqa: BLE001
                    err[0] = e

            t = threading.Thread(target=_read, daemon=True)
            t.start()
            t.join(timeout=max(0.05, deadline - time.monotonic()))
            if t.is_alive():
                continue
            if err[0] is not None:
                raise DoraStdioClientError(f"read failed: {err[0]}") from err[0]
            raw = result[0]
            if raw is None or raw == "":
                if self._proc.poll() is not None:
                    raise DoraStdioClientError("EOF from dora-policy stdout")
                continue
            raw = raw.strip()
            if not raw:
                continue
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DoraStdioClientError(f"invalid JSON from dora-policy: {raw[:200]!r}") from exc

        raise DoraStdioClientError(f"predict timeout after {self._predict_timeout_s}s")

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None and self._proc.stdin is not None:
                try:
                    self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    self._proc.stdin.flush()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        finally:
            self._proc = None


def build_dora_policy_argv(
    *,
    dora_policy_bin: str,
    model_dir: str | None,
    device: str = "flex",
    unnorm_key: str | None = "default",
    inference_steps: int = 0,
    extra_args: Sequence[str] | None = None,
) -> list[str]:
    argv = [dora_policy_bin]
    if model_dir:
        argv.extend(["--model-dir", model_dir])
    argv.extend(["--device", device])
    if unnorm_key:
        argv.extend(["--unnorm-key", unnorm_key])
    if inference_steps > 0:
        argv.extend(["--inference-steps", str(inference_steps)])
    argv.append("--stdio-loop")
    if extra_args:
        argv.extend(list(extra_args))
    return argv


def action_chunk_wire_to_numpy(chunk: dict[str, Any]) -> np.ndarray:
    """Reshape ActionChunkWire ``values`` to ``[H, D]`` for batch=1."""
    values = np.asarray(chunk["values"], dtype=np.float32).reshape(-1)
    horizon = int(chunk["horizon"])
    action_dim = int(chunk["action_dim"])
    batch = int(chunk.get("batch", 1))
    expected = batch * horizon * action_dim
    if values.size != expected:
        raise DoraStdioClientError(f"values size {values.size} != batch*H*D={expected}")
    # batch outermost; take first env
    return values.reshape(batch, horizon, action_dim)[0]
