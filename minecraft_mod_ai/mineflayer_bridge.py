from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any


class MineflayerBridgeError(RuntimeError):
    pass


class MineflayerBridge:
    """Persistent JSONL client for the first-party Mineflayer 1.20.1 bridge."""

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

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        if not self.bridge_path.is_file():
            raise FileNotFoundError(self.bridge_path)
        self.process = subprocess.Popen(
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

    def call(self, action: str, **params: Any) -> dict[str, Any]:
        if action not in self.ACTIONS:
            raise MineflayerBridgeError(
                f"Mineflayer action is not allowlisted: {action}"
            )
        with self._lock:
            self.start()
            assert self.process is not None
            if self.process.stdin is None or self.process.stdout is None:
                raise MineflayerBridgeError(
                    "Mineflayer bridge pipes are unavailable."
                )
            request_id = self._next_id
            self._next_id += 1
            self.process.stdin.write(
                json.dumps(
                    {"id": request_id, "action": action, "params": params},
                    ensure_ascii=False,
                )
                + "\n"
            )
            self.process.stdin.flush()
            while True:
                line = self.process.stdout.readline()
                if not line:
                    stderr = (
                        self.process.stderr.read()
                        if self.process.stderr is not None
                        else ""
                    )
                    raise MineflayerBridgeError(
                        f"Mineflayer bridge exited without a response: {stderr[-2000:]}"
                    )
                response = json.loads(line)
                if response.get("id") != request_id:
                    continue
                if not response.get("ok"):
                    raise MineflayerBridgeError(
                        str(response.get("error", "unknown error"))
                    )
                result = response.get("result", {})
                return result if isinstance(result, dict) else {"value": result}

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            try:
                self.call("disconnect")
            except Exception:
                pass
            self.process.terminate()
        self.process = None


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
