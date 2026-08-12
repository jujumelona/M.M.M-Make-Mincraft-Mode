from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any


class MineflayerBridgeError(RuntimeError):
    pass


class MineflayerBridge:
    """Persistent JSONL client for the first-party Mineflayer 1.20.1 bridge.

    Calls are serialized because the bridge protocol has one request/response stream.
    Dedicated reader threads continuously drain stdout/stderr so a noisy Node child
    cannot deadlock on a full OS pipe. Every request also has a host-side deadline;
    a timed-out or protocol-corrupt child is discarded before another call can run.
    """

    ACTIONS = frozenset(
        {
            "connect",
            "status",
            "walk_to",
            "interact_block",
            "use_item",
            "attack_entity",
            "inventory",
            "chat",
            "craft",
            "wait_for",
            "open_container",
            "click_slot",
            "disconnect",
        }
    )

    def __init__(self, bridge_path: str | Path | None = None) -> None:
        self.bridge_path = (
            Path(bridge_path).expanduser().resolve()
            if bridge_path is not None
            else _default_bridge_path()
        )
        self.process: subprocess.Popen[str] | None = None
        self._lock = threading.RLock()
        self._next_id = 1
        self._response_lines: queue.Queue[str | None] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=80)
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self.process and self.process.poll() is None:
                return
            self._dispose_process_locked()
            if not self.bridge_path.is_file():
                raise FileNotFoundError(self.bridge_path)
            process = subprocess.Popen(
                ["node", str(self.bridge_path)],
                cwd=str(self.bridge_path.parent),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                env={**os.environ, "NODE_NO_WARNINGS": "1"},
            )
            if process.stdin is None or process.stdout is None or process.stderr is None:
                process.kill()
                process.wait(timeout=5)
                raise MineflayerBridgeError("Mineflayer bridge pipes are unavailable.")

            response_lines: queue.Queue[str | None] = queue.Queue()
            stderr_tail: deque[str] = deque(maxlen=80)
            self.process = process
            self._response_lines = response_lines
            self._stderr_tail = stderr_tail
            self._stdout_thread = threading.Thread(
                target=self._drain_stdout,
                args=(process, response_lines),
                name="mmm-mineflayer-stdout",
                daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                args=(process, stderr_tail),
                name="mmm-mineflayer-stderr",
                daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread.start()

    @staticmethod
    def _drain_stdout(
        process: subprocess.Popen[str],
        output: queue.Queue[str | None],
    ) -> None:
        stream = process.stdout
        if stream is None:
            output.put(None)
            return
        try:
            for line in iter(stream.readline, ""):
                output.put(line)
        finally:
            output.put(None)

    @staticmethod
    def _drain_stderr(
        process: subprocess.Popen[str],
        tail: deque[str],
    ) -> None:
        stream = process.stderr
        if stream is None:
            return
        for line in iter(stream.readline, ""):
            tail.append(line.rstrip())

    def call(
        self,
        action: str,
        *,
        timeout_seconds: float | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        if action not in self.ACTIONS:
            raise MineflayerBridgeError(
                f"Mineflayer action is not allowlisted: {action}"
            )
        timeout = _call_timeout_seconds(timeout_seconds)
        with self._lock:
            self.start()
            process = self.process
            assert process is not None
            if process.stdin is None:
                self._abort_process_locked()
                raise MineflayerBridgeError("Mineflayer bridge stdin is unavailable.")

            request_id = self._next_id
            self._next_id += 1
            try:
                process.stdin.write(
                    json.dumps(
                        {"id": request_id, "action": action, "params": params},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                detail = self._stderr_text()
                self._abort_process_locked()
                raise MineflayerBridgeError(
                    "Mineflayer bridge request pipe failed"
                    + (f": {detail}" if detail else "")
                ) from exc

            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = self._stderr_text()
                    self._abort_process_locked()
                    raise MineflayerBridgeError(
                        f"Mineflayer action {action!r} timed out after {timeout:g}s"
                        + (f": {detail}" if detail else "")
                    )
                try:
                    line = self._response_lines.get(timeout=remaining)
                except queue.Empty:
                    detail = self._stderr_text()
                    self._abort_process_locked()
                    raise MineflayerBridgeError(
                        f"Mineflayer action {action!r} timed out after {timeout:g}s"
                        + (f": {detail}" if detail else "")
                    )
                if line is None:
                    detail = self._stderr_text()
                    self._abort_process_locked()
                    raise MineflayerBridgeError(
                        "Mineflayer bridge exited without a response"
                        + (f": {detail}" if detail else "")
                    )
                try:
                    response = json.loads(line)
                except json.JSONDecodeError as exc:
                    detail = line.strip()[-1000:]
                    self._abort_process_locked()
                    raise MineflayerBridgeError(
                        f"Mineflayer bridge emitted invalid JSON: {detail}"
                    ) from exc
                if not isinstance(response, dict):
                    self._abort_process_locked()
                    raise MineflayerBridgeError(
                        "Mineflayer bridge response must be a JSON object."
                    )
                if response.get("id") != request_id:
                    # There is only one in-flight request. A different id can only be
                    # stale/corrupt protocol data; keep waiting, but only until the
                    # same finite request deadline.
                    continue
                if not response.get("ok"):
                    raise MineflayerBridgeError(
                        str(response.get("error", "unknown error"))
                    )
                result = response.get("result", {})
                return result if isinstance(result, dict) else {"value": result}

    def close(self) -> None:
        with self._lock:
            process = self.process
            if process is None:
                return
            if process.poll() is None:
                try:
                    self.call("disconnect", timeout_seconds=10.0)
                except Exception:
                    pass
            self._terminate_process_locked()

    def _stderr_text(self) -> str:
        return "\n".join(self._stderr_tail)[-2000:]

    def _abort_process_locked(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                pass
        self._dispose_process_locked()

    def _terminate_process_locked(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                try:
                    process.wait(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    pass
        self._dispose_process_locked()

    def _dispose_process_locked(self) -> None:
        process = self.process
        if process is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        self.process = None
        self._stdout_thread = None
        self._stderr_thread = None


def _call_timeout_seconds(explicit: float | None) -> float:
    if explicit is None:
        raw = os.environ.get("MMM_MINEFLAYER_CALL_TIMEOUT_SECONDS", "180").strip()
        try:
            value = float(raw)
        except ValueError as exc:
            raise MineflayerBridgeError(
                "MMM_MINEFLAYER_CALL_TIMEOUT_SECONDS must be numeric."
            ) from exc
    else:
        value = float(explicit)
    if not math.isfinite(value) or value <= 0:
        raise MineflayerBridgeError(
            "Mineflayer call timeout must be a finite positive number."
        )
    return value


def _default_bridge_path() -> Path:
    repository = (
        Path(__file__).resolve().parents[1]
        / "integrations"
        / "mineflayer-1201"
        / "bridge.mjs"
    )
    if repository.is_file():
        return repository
    return (
        Path(__file__).resolve().parent
        / "integrations"
        / "mineflayer-1201"
        / "bridge.mjs"
    )
