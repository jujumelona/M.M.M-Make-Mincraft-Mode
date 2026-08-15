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
    """Bounded process manager for one explicitly selected disposable Minecraft target."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        profile_name: str = "fabric_target_disposable",
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
        if not self.profile.minecraft_version.strip():
            raise RuntimePolicyError("Runtime profile must declare an explicit Minecraft target.")
        if not self.profile.disposable_only:
            raise RuntimePolicyError("MMM runtime profiles must be disposable-only.")
        self.server_process: subprocess.Popen[str] | None = None
        self.client_process: subprocess.Popen[str] | None = None
        self.instance_root: Path | None = None
        self._server_log: list[str] = []
        self._client_log: list[str] = []
        # Process/instance state and log buffers have different blocking behavior.
        # Never make the stdout reader wait behind a 30s process shutdown lock: a
        # chatty child could otherwise fill its pipe while stop_server waits.
        self._lock = threading.RLock()
        self._log_lock = threading.RLock()

    def prepare_instance(
        self,
        instance_name: str,
        *,
        mod_jar: str | Path,
        server_launcher: str | Path,
        eula_accepted: bool,
    ) -> dict[str, Any]:
        with self._lock:
            if self._process_running(self.server_process) or self._process_running(
                self.client_process
            ):
                raise RuntimePolicyError(
                    "Stop the active disposable runtime before preparing another instance."
                )
            if not re.fullmatch(r"[a-z][a-z0-9_-]{1,63}", instance_name):
                raise RuntimePolicyError("Invalid runtime instance name.")
            if self.profile.eula_must_be_explicitly_accepted and not eula_accepted:
                raise RuntimePolicyError("Minecraft EULA acceptance must be explicit.")

            # Validate all external inputs before creating the disposable directory.
            # A bad jar/launcher must not leave a reserved instance name behind.
            jar = self._existing_file(mod_jar)
            launcher = self._existing_file(server_launcher)
            root = self._new_child(Path("runtime-instances") / instance_name)
            try:
                mods = root / "mods"
                mods.mkdir(parents=True)
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
            except BaseException:
                shutil.rmtree(root, ignore_errors=True)
                raise
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
        if type(timeout_seconds) is not int or timeout_seconds < 1:
            raise RuntimePolicyError("Server startup timeout must be a positive integer.")
        with self._lock:
            if self.instance_root is None:
                raise RuntimePolicyError("Prepare a runtime instance first.")
            if self._process_running(self.server_process):
                raise RuntimePolicyError("Minecraft server is already running.")
            command = [
                self.profile.server_java_command,
                "-Xms1024M",
                f"-Xmx{self.profile.server_memory_mb}M",
                "-jar",
                "fabric-server-launch.jar",
                "nogui",
            ]
            process = subprocess.Popen(
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
            self.server_process = process
            with self._log_lock:
                self._server_log.clear()
            threading.Thread(
                target=self._read_stream,
                args=(process, self._server_log),
                name="mmm-runtime-server-log",
                daemon=True,
            ).start()

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                with self._lock:
                    if self.server_process is process:
                        self.server_process = None
                raise RuntimePolicyError(
                    "Minecraft server exited before readiness.\n"
                    + "\n".join(self._log_tail(self._server_log, 80))
                )
            text = "\n".join(self._log_tail(self._server_log, 80))
            if any(pattern.search(text) for pattern in self.profile.startup_ready_patterns):
                return self.status()
            time.sleep(0.5)

        with self._lock:
            if self.server_process is process:
                self._stop_process(process, graceful_server=True)
                self.server_process = None
        raise TimeoutError("Minecraft server did not become ready before timeout.")

    def start_client(self) -> dict[str, Any]:
        with self._lock:
            if self.instance_root is None:
                raise RuntimePolicyError("Prepare a runtime instance first.")
            raw = os.environ.get(self.profile.client_command_env, "").strip()
            if not raw:
                raise RuntimePolicyError(
                    f"{self.profile.client_command_env} must contain a JSON command array."
                )
            try:
                command = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimePolicyError("Client command must be valid JSON.") from exc
            if (
                not isinstance(command, list)
                or not command
                or any(not isinstance(item, str) or not item for item in command)
            ):
                raise RuntimePolicyError("Client command must be a non-empty JSON string array.")
            if self._process_running(self.client_process):
                raise RuntimePolicyError("Minecraft client is already running.")
            env = os.environ.copy()
            env["MMM_GAME_DIR"] = str(self.instance_root / "client")
            Path(env["MMM_GAME_DIR"]).mkdir(parents=True, exist_ok=True)
            process = subprocess.Popen(
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
            self.client_process = process
            with self._log_lock:
                self._client_log.clear()
            threading.Thread(
                target=self._read_stream,
                args=(process, self._client_log),
                name="mmm-runtime-client-log",
                daemon=True,
            ).start()
            return self.status()

    def send_server_command(self, command: str) -> dict[str, Any]:
        with self._lock:
            command = command.strip()
            if not any(
                pattern.fullmatch(command)
                for pattern in self.profile.allowed_server_commands
            ):
                raise RuntimePolicyError(
                    f"Server command is not allowlisted: {command!r}"
                )
            process = self.server_process
            if not self._process_running(process):
                raise RuntimePolicyError("Minecraft server is not running.")
            assert process is not None
            if process.stdin is None:
                raise RuntimePolicyError("Minecraft server stdin is unavailable.")
            try:
                process.stdin.write(command + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise RuntimePolicyError("Minecraft server stdin write failed.") from exc
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
        with self._log_lock:
            return {
                "server": list(self._server_log[-lines:]),
                "client": list(self._client_log[-lines:]),
            }

    def stop_server(self) -> dict[str, Any]:
        with self._lock:
            process = self.server_process
            if process is not None:
                self._stop_process(process, graceful_server=True)
                if self.server_process is process:
                    self.server_process = None
            return self.status()

    def stop_client(self) -> dict[str, Any]:
        with self._lock:
            process = self.client_process
            if process is not None:
                self._stop_process(process, graceful_server=False)
                if self.client_process is process:
                    self.client_process = None
            return self.status()

    def cleanup(self) -> dict[str, Any]:
        with self._lock:
            self.stop_client()
            self.stop_server()
            root = self.instance_root
            # Keep the path bound until deletion succeeds. If rmtree raises, callers
            # retain a valid cleanup target and can retry rather than leaking an
            # unreachable disposable instance.
            if root and root.is_dir():
                shutil.rmtree(root)
            if self.instance_root is root:
                self.instance_root = None
            return {"status": "cleaned", "removed": str(root) if root else None}

    def status(self) -> dict[str, Any]:
        with self._lock:
            instance_root = self.instance_root
            server_process = self.server_process
            client_process = self.client_process
        with self._log_lock:
            server_log_lines = len(self._server_log)
            client_log_lines = len(self._client_log)
        return {
            "schema_version": "mmm/runtime-status-v1",
            "instance_root": str(instance_root) if instance_root else None,
            "server_running": self._process_running(server_process),
            "client_running": self._process_running(client_process),
            "server_log_lines": server_log_lines,
            "client_log_lines": client_log_lines,
        }

    def _stop_process(
        self,
        process: subprocess.Popen[str],
        *,
        graceful_server: bool,
    ) -> None:
        if process.poll() is not None:
            # wait() also reaps an already-exited child on POSIX.
            try:
                process.wait(timeout=0)
            except (subprocess.TimeoutExpired, OSError):
                pass
            return
        if graceful_server and process.stdin is not None:
            try:
                process.stdin.write("stop\n")
                process.stdin.flush()
                process.wait(timeout=30)
                return
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                pass
        try:
            process.terminate()
        except OSError:
            pass
        try:
            process.wait(timeout=15 if not graceful_server else 10)
            return
        except (subprocess.TimeoutExpired, OSError):
            pass
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass

    @staticmethod
    def _process_running(process: subprocess.Popen[str] | None) -> bool:
        return bool(process is not None and process.poll() is None)

    def _read_stream(self, process: subprocess.Popen[str], target: list[str]) -> None:
        if process.stdout is None:
            return
        for line in iter(process.stdout.readline, ""):
            with self._log_lock:
                target.append(line.rstrip())
                if len(target) > 5000:
                    del target[:1000]

    def _log_tail(self, target: list[str], lines: int) -> list[str]:
        with self._log_lock:
            return list(target[-lines:])

    def _new_child(self, relative: Path) -> Path:
        target = (self.workspace_root / relative).resolve()
        self._assert_child(target)
        if target.exists():
            raise FileExistsError(target)
        target.mkdir(parents=True)
        return target

    def _existing_file(self, value: str | Path) -> Path:
        candidate = Path(value).expanduser()
        target = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.workspace_root / candidate).resolve()
        )
        self._assert_child(target)
        if not target.is_file() or target.is_symlink():
            raise FileNotFoundError(target)
        return target

    def _assert_child(self, target: Path) -> None:
        try:
            target.relative_to(self.workspace_root)
        except ValueError as exc:
            raise RuntimePolicyError(
                "Runtime path escaped the configured workspace."
            ) from exc
        if target == self.workspace_root:
            raise RuntimePolicyError("Runtime may not target the workspace root.")
