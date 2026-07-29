from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config_paths import config_path as resolve_config_path


class RuntimePolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    minecraft_version: str
    server_java_command: str
    server_memory_mb: int
    server_launcher_relative: str
    client_command_env: str
    allowed_server_commands: tuple[re.Pattern[str], ...]
    startup_ready_patterns: tuple[re.Pattern[str], ...]
    disposable_only: bool
    eula_must_be_explicitly_accepted: bool


class MinecraftRuntimeManager:
    """Bounded process manager for disposable Minecraft 1.20.1 instances."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        profile_name: str = "fabric_1201_disposable",
        config_path: str | Path | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        path = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else resolve_config_path("runtime_profiles.yaml")
        )
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != "mmm/runtime-profiles-v1":
            raise RuntimePolicyError("Unsupported runtime profile registry.")
        entry = raw.get("profiles", {}).get(profile_name)
        if not isinstance(entry, dict):
            raise RuntimePolicyError(f"Unknown runtime profile: {profile_name}")
        self.profile = RuntimeProfile(
            name=profile_name,
            minecraft_version=str(entry["minecraft_version"]),
            server_java_command=str(entry["server_java_command"]),
            server_memory_mb=int(entry["server_memory_mb"]),
            server_launcher_relative=str(entry["server_launcher_relative"]),
            client_command_env=str(entry["client_command_env"]),
            allowed_server_commands=tuple(
                re.compile(pattern) for pattern in entry["allowed_server_commands"]
            ),
            startup_ready_patterns=tuple(
                re.compile(pattern) for pattern in entry["startup_ready_patterns"]
            ),
            disposable_only=bool(entry["disposable_only"]),
            eula_must_be_explicitly_accepted=bool(
                entry["eula_must_be_explicitly_accepted"]
            ),
        )
        if self.profile.minecraft_version != "1.20.1" or not self.profile.disposable_only:
            raise RuntimePolicyError("Only disposable Minecraft 1.20.1 runtime is supported.")
        self.server_process: subprocess.Popen[str] | None = None
        self.client_process: subprocess.Popen[str] | None = None
        self.instance_root: Path | None = None
        self._server_log: list[str] = []
        self._client_log: list[str] = []
        self._lock = threading.RLock()

    def prepare_instance(
        self,
        instance_name: str,
        *,
        mod_jar: str | Path,
        server_launcher: str | Path,
        eula_accepted: bool,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[a-z][a-z0-9_-]{1,63}", instance_name):
            raise RuntimePolicyError("Invalid runtime instance name.")
        if self.profile.eula_must_be_explicitly_accepted and not eula_accepted:
            raise RuntimePolicyError("Minecraft EULA acceptance must be explicit.")
        root = self._new_child(Path("runtime-instances") / instance_name)
        mods = root / "mods"
        mods.mkdir(parents=True)
        jar = self._existing_file(mod_jar)
        launcher = self._existing_file(server_launcher)
        shutil.copy2(jar, mods / jar.name)
        runtime_launcher = root / "fabric-server-launch.jar"
        shutil.copy2(launcher, runtime_launcher)
        (root / "eula.txt").write_text("eula=true\n", encoding="utf-8")
        (root / "server.properties").write_text(
            "\n".join(
                [
                    "online-mode=false",
                    "enable-command-block=true",
                    "spawn-protection=0",
                    "view-distance=6",
                    "simulation-distance=5",
                    "max-players=4",
                    "motd=M.M.M disposable integration test",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        self.instance_root = root
        return {
            "schema_version": "mmm/runtime-instance-v1",
            "instance_root": str(root),
            "mod_jar": str(mods / jar.name),
            "server_launcher": str(runtime_launcher),
            "minecraft_version": self.profile.minecraft_version,
            "disposable": True,
        }

    def start_server(self, timeout_seconds: int = 180) -> dict[str, Any]:
        with self._lock:
            if self.instance_root is None:
                raise RuntimePolicyError("Prepare a runtime instance first.")
            if self.server_process and self.server_process.poll() is None:
                raise RuntimePolicyError("Minecraft server is already running.")
            command = [
                self.profile.server_java_command,
                f"-Xms1024M",
                f"-Xmx{self.profile.server_memory_mb}M",
                "-jar",
                "fabric-server-launch.jar",
                "nogui",
            ]
            self.server_process = subprocess.Popen(
                command,
                cwd=str(self.instance_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                start_new_session=(os.name != "nt"),
            )
            self._server_log.clear()
            thread = threading.Thread(
                target=self._read_stream,
                args=(self.server_process, self._server_log),
                daemon=True,
            )
            thread.start()
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.server_process.poll() is not None:
                raise RuntimePolicyError(
                    "Minecraft server exited before readiness.\n"
                    + "\n".join(self._server_log[-80:])
                )
            text = "\n".join(self._server_log[-80:])
            if any(pattern.search(text) for pattern in self.profile.startup_ready_patterns):
                return self.status()
            time.sleep(0.5)
        self.stop_server()
        raise TimeoutError("Minecraft server did not become ready before timeout.")

    def start_client(self) -> dict[str, Any]:
        if self.instance_root is None:
            raise RuntimePolicyError("Prepare a runtime instance first.")
        raw = os.environ.get(self.profile.client_command_env, "").strip()
        if not raw:
            raise RuntimePolicyError(
                f"{self.profile.client_command_env} must contain a JSON command array."
            )
        command = json.loads(raw)
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) or not item for item in command)
        ):
            raise RuntimePolicyError("Client command must be a non-empty JSON string array.")
        if self.client_process and self.client_process.poll() is None:
            raise RuntimePolicyError("Minecraft client is already running.")
        env = os.environ.copy()
        env["MMM_GAME_DIR"] = str(self.instance_root / "client")
        Path(env["MMM_GAME_DIR"]).mkdir(parents=True, exist_ok=True)
        self.client_process = subprocess.Popen(
            command,
            cwd=str(self.instance_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            env=env,
            start_new_session=(os.name != "nt"),
        )
        self._client_log.clear()
        threading.Thread(
            target=self._read_stream,
            args=(self.client_process, self._client_log),
            daemon=True,
        ).start()
        return self.status()

    def send_server_command(self, command: str) -> dict[str, Any]:
        command = command.strip()
        if not any(pattern.fullmatch(command) for pattern in self.profile.allowed_server_commands):
            raise RuntimePolicyError(f"Server command is not allowlisted: {command!r}")
        if not self.server_process or self.server_process.poll() is not None:
            raise RuntimePolicyError("Minecraft server is not running.")
        if self.server_process.stdin is None:
            raise RuntimePolicyError("Minecraft server stdin is unavailable.")
        self.server_process.stdin.write(command + "\n")
        self.server_process.stdin.flush()
        return {"status": "sent", "command": command}

    def register_screenshot(self, screenshot_path: str | Path) -> dict[str, Any]:
        path = self._existing_file(screenshot_path)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise RuntimePolicyError("Runtime screenshot has an unsupported format.")
        return {
            "schema_version": "mmm/runtime-screenshot-v1",
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }

    def tail_logs(self, lines: int = 120) -> dict[str, Any]:
        if not 1 <= lines <= 1000:
            raise RuntimePolicyError("Log line limit must be 1-1000.")
        return {
            "server": self._server_log[-lines:],
            "client": self._client_log[-lines:],
        }

    def stop_server(self) -> dict[str, Any]:
        with self._lock:
            process = self.server_process
            if process is None:
                return self.status()
            if process.poll() is None:
                try:
                    self.send_server_command("stop")
                    process.wait(timeout=30)
                except Exception:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
            self.server_process = None
            return self.status()

    def stop_client(self) -> dict[str, Any]:
        with self._lock:
            process = self.client_process
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
            self.client_process = None
            return self.status()

    def cleanup(self) -> dict[str, Any]:
        self.stop_client()
        self.stop_server()
        root = self.instance_root
        self.instance_root = None
        if root and root.is_dir():
            shutil.rmtree(root)
        return {"status": "cleaned", "removed": str(root) if root else None}

    def status(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/runtime-status-v1",
            "instance_root": str(self.instance_root) if self.instance_root else None,
            "server_running": bool(
                self.server_process and self.server_process.poll() is None
            ),
            "client_running": bool(
                self.client_process and self.client_process.poll() is None
            ),
            "server_log_lines": len(self._server_log),
            "client_log_lines": len(self._client_log),
        }

    @staticmethod
    def _read_stream(process: subprocess.Popen[str], target: list[str]) -> None:
        if process.stdout is None:
            return
        for line in iter(process.stdout.readline, ""):
            target.append(line.rstrip())
            if len(target) > 5000:
                del target[:1000]

    def _new_child(self, relative: Path) -> Path:
        target = (self.workspace_root / relative).resolve()
        self._assert_child(target)
        if target.exists():
            raise FileExistsError(target)
        target.mkdir(parents=True)
        return target

    def _existing_file(self, value: str | Path) -> Path:
        candidate = Path(value).expanduser()
        target = candidate.resolve() if candidate.is_absolute() else (self.workspace_root / candidate).resolve()
        self._assert_child(target)
        if not target.is_file() or target.is_symlink():
            raise FileNotFoundError(target)
        return target

    def _assert_child(self, target: Path) -> None:
        try:
            target.relative_to(self.workspace_root)
        except ValueError as exc:
            raise RuntimePolicyError("Runtime path escaped the configured workspace.") from exc
        if target == self.workspace_root:
            raise RuntimePolicyError("Runtime may not target the workspace root.")
