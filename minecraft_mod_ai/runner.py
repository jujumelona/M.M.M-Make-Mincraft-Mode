from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
import zipfile
from collections.abc import Iterable
from contextlib import contextmanager
from contextvars import copy_context
from dataclasses import asdict, dataclass
from pathlib import Path

from .platform_catalog import adapter_from_project


class BuildRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandResult:
    name: str
    command: tuple[str, ...]
    exit_code: int
    duration_seconds: float
    log_path: str
    timed_out: bool = False


@dataclass(frozen=True)
class BuildReport:
    status: str
    gradle_version: str
    commands: tuple[CommandResult, ...]
    jar_path: str | None
    gametest_report: str | None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "gradle_version": self.gradle_version,
            "commands": [asdict(command) for command in self.commands],
            "jar_path": self.jar_path,
            "gametest_report": self.gametest_report,
            "error": self.error,
        }


class GradleRunner:
    def __init__(
        self,
        cache_dir: Path,
        *,
        download_timeout_seconds: int = 300,
        command_timeout_seconds: int = 1200,
    ) -> None:
        self.cache_dir = cache_dir.resolve()
        self.download_timeout_seconds = download_timeout_seconds
        self.command_timeout_seconds = command_timeout_seconds

    def build(self, project_root: Path, *, run_gametest: bool = True) -> BuildReport:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with _exclusive_cache_lock(
            self.cache_dir,
            timeout_seconds=max(300, self.command_timeout_seconds * 3),
        ):
            return self._build_locked(project_root, run_gametest=run_gametest)

    def _build_locked(
        self,
        project_root: Path,
        *,
        run_gametest: bool,
    ) -> BuildReport:
        project_root = project_root.resolve()
        if not (project_root / "build.gradle").is_file():
            raise BuildRunnerError(f"Not a generated Gradle project: {project_root}")
        try:
            adapter = adapter_from_project(project_root)
        except ValueError as exc:
            raise BuildRunnerError(
                f"Project platform lock is missing, mixed, or unsupported: {exc}"
            ) from exc
        gradle_version = adapter.gradle
        gradle_sha256 = adapter.gradle_sha256
        logs = project_root / ".minecraft_ai" / "logs"
        logs.mkdir(parents=True, exist_ok=True)

        gradle = self._ensure_gradle(gradle_version, gradle_sha256)
        commands: list[CommandResult] = []
        environment = os.environ.copy()
        environment["GRADLE_USER_HOME"] = str(self.cache_dir / "gradle-user-home")
        environment["CI"] = "true"

        if not self._wrapper_is_current(project_root, gradle_version, gradle_sha256):
            wrapper_result = self._run(
                name="wrapper",
                executable=gradle,
                arguments=(
                    "--no-daemon",
                    "wrapper",
                    "--gradle-version",
                    gradle_version,
                    "--gradle-distribution-sha256-sum",
                    gradle_sha256,
                    "--stacktrace",
                ),
                cwd=project_root,
                env=environment,
                log_path=logs / "gradle-wrapper.log",
            )
            commands.append(wrapper_result)
            if wrapper_result.exit_code != 0:
                return BuildReport(
                    status="FAIL",
                    gradle_version=gradle_version,
                    commands=tuple(commands),
                    jar_path=None,
                    gametest_report=None,
                    error="Gradle wrapper generation failed.",
                )

        force_clean = os.environ.get("MMM_GRADLE_FORCE_CLEAN", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        build_arguments = (
            ("--no-daemon", "clean", "build", "--stacktrace")
            if force_clean
            else ("--no-daemon", "build", "--stacktrace")
        )
        build_result = self._run(
            name="clean_build" if force_clean else "build",
            executable=gradle,
            arguments=build_arguments,
            cwd=project_root,
            env=environment,
            log_path=logs / "gradle-build.log",
        )
        commands.append(build_result)
        if build_result.exit_code != 0:
            return BuildReport(
                status="FAIL",
                gradle_version=gradle_version,
                commands=tuple(commands),
                jar_path=None,
                gametest_report=None,
                error="Gradle build failed.",
            )

        if run_gametest:
            gametest_result = self._run(
                name="gametest",
                executable=gradle,
                arguments=("--no-daemon", "runGameTestServer", "--stacktrace"),
                cwd=project_root,
                env=environment,
                log_path=logs / "gradle-gametest.log",
            )
            commands.append(gametest_result)
            if gametest_result.exit_code != 0:
                return BuildReport(
                    status="FAIL",
                    gradle_version=gradle_version,
                    commands=tuple(commands),
                    jar_path=self._find_release_jar(project_root),
                    gametest_report=self._gametest_report(project_root),
                    error="Headless Fabric GameTest failed.",
                )

        jar_path = self._find_release_jar(project_root)
        if jar_path is None:
            return BuildReport(
                status="FAIL",
                gradle_version=gradle_version,
                commands=tuple(commands),
                jar_path=None,
                gametest_report=self._gametest_report(project_root),
                error="Gradle reported success but no remapped release JAR was found.",
            )
        return BuildReport(
            status="PASS",
            gradle_version=gradle_version,
            commands=tuple(commands),
            jar_path=jar_path,
            gametest_report=self._gametest_report(project_root),
            error=None,
        )

    @staticmethod
    def _wrapper_is_current(
        project_root: Path,
        gradle_version: str,
        gradle_sha256: str,
    ) -> bool:
        properties = project_root / "gradle" / "wrapper" / "gradle-wrapper.properties"
        launcher = project_root / ("gradlew.bat" if os.name == "nt" else "gradlew")
        if not properties.is_file() or not launcher.is_file():
            return False
        try:
            text = properties.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return (
            f"gradle-{gradle_version}-bin.zip" in text
            and gradle_sha256.lower() in text.lower()
        )

    def _ensure_gradle(self, gradle_version: str, gradle_sha256: str) -> Path:
        from .root_cause_trace import emit_root_cause

        emit_root_cause("gradle_distribution_start", stage="verify", operation="prepare_gradle", result="START",
                        details={"version": gradle_version, "sha256": gradle_sha256, "cache_dir": str(self.cache_dir)})
        distribution_dir = self.cache_dir / f"gradle-{gradle_version}"
        executable = distribution_dir / "bin" / (
            "gradle.bat" if os.name == "nt" else "gradle"
        )
        verification_marker = distribution_dir / ".minecraft-ai-gradle-sha256"

        if executable.is_file() and verification_marker.is_file():
            try:
                verified_sha256 = verification_marker.read_text(
                    encoding="ascii"
                ).strip().lower()
            except OSError:
                verified_sha256 = ""
            if verified_sha256 == gradle_sha256.lower():
                if os.name != "nt":
                    executable.chmod(0o755)
                return executable

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        archive = self.cache_dir / f"gradle-{gradle_version}-bin.zip"
        if archive.is_file() and _sha256(archive) != gradle_sha256:
            archive.unlink()
        if not archive.is_file():
            temporary = archive.with_suffix(".zip.part")
            url = (
                "https://services.gradle.org/distributions/"
                f"gradle-{gradle_version}-bin.zip"
            )
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "minecraft-mod-ai/0.8"},
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.download_timeout_seconds
                ) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            except Exception:
                if temporary.exists():
                    temporary.unlink()
                raise
            if _sha256(temporary) != gradle_sha256:
                temporary.unlink(missing_ok=True)
                raise BuildRunnerError("Gradle distribution SHA-256 verification failed.")
            temporary.replace(archive)

        extraction_root = self.cache_dir / f".extract-gradle-{gradle_version}"
        if extraction_root.exists():
            shutil.rmtree(extraction_root)
        extraction_root.mkdir(parents=True)
        try:
            with zipfile.ZipFile(archive) as zipped:
                _safe_extract(zipped, extraction_root)
            extracted = extraction_root / f"gradle-{gradle_version}"
            if not extracted.is_dir():
                raise BuildRunnerError("Gradle archive did not contain the expected directory.")
            if distribution_dir.exists():
                shutil.rmtree(distribution_dir)
            extracted.replace(distribution_dir)
        finally:
            if extraction_root.exists():
                shutil.rmtree(extraction_root)
        if not executable.is_file():
            raise BuildRunnerError("Gradle executable is missing after extraction.")
        verification_marker.write_text(
            gradle_sha256.lower() + "\n",
            encoding="ascii",
        )
        if os.name != "nt":
            executable.chmod(0o755)
        return executable

    def _run(
        self,
        *,
        name: str,
        executable: Path,
        arguments: tuple[str, ...],
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
    ) -> CommandResult:
        from .agent_tool_runtime import _redact_text, _sanitize_observation
        from .root_cause_trace import emit_root_cause

        command = (str(executable), *arguments)
        started = time.monotonic()
        timed_out = False
        creation_flags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if os.name == "nt"
            else 0
        )
        emit_root_cause(
            "gradle_command_start", stage="verify", operation=name, result="START",
            details={"command": _sanitize_observation(command), "cwd": str(cwd),
                     "log_path": str(log_path), "timeout_seconds": self.command_timeout_seconds,
                     "java_home": env.get("JAVA_HOME", ""), "gradle_user_home": env.get("GRADLE_USER_HOME", "")},
        )
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            creationflags=creation_flags,
            start_new_session=(os.name != "nt"),
        )
        reader_errors: list[BaseException] = []

        def read_output() -> None:
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    if process.stdout is None:
                        raise BuildRunnerError("Gradle output pipe is unavailable")
                    private_key = False
                    for raw in process.stdout:
                        if private_key:
                            if "-----END " in raw and "PRIVATE KEY-----" in raw:
                                private_key = False
                            continue
                        if "-----BEGIN " in raw and "PRIVATE KEY-----" in raw:
                            private_key = "-----END " not in raw
                            line = "[REDACTED_PRIVATE_KEY]\n"
                        else:
                            line = _redact_text(raw)
                        log.write(line)
                        log.flush()
                        emit_root_cause(
                            "gradle_command_output", stage="verify", operation=name, result="INFO",
                            details={"pid": process.pid, "log_path": str(log_path), "line": line.rstrip()},
                        )
            except Exception as exc:
                reader_errors.append(exc)
                emit_root_cause("gradle_output_failure", stage="verify", operation=name,
                                result="FAIL", reason=f"{type(exc).__name__}: {exc}", exc=exc)

        reader = threading.Thread(target=copy_context().run, args=(read_output,), daemon=True)
        reader.start()
        try:
            process.wait(timeout=self.command_timeout_seconds)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_tree(process)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            exit_code = 124
        except BaseException:
            _terminate_process_tree(process)
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            raise
        finally:
            reader.join(timeout=15)
            if not reader.is_alive() and process.stdout is not None:
                process.stdout.close()
        if reader.is_alive() or reader_errors:
            raise BuildRunnerError("Gradle command log could not be fully drained") from (
                reader_errors[0] if reader_errors else None
            )
        if timed_out:
            with log_path.open("a", encoding="utf-8") as log:
                log.write("\n[ M.M.M Make Mincraft Mode: command timed out; process tree terminated ]\n")
        duration = time.monotonic() - started
        emit_root_cause(
            "gradle_command_result", stage="verify", operation=name,
            result="TIMEOUT" if timed_out else "PASS" if exit_code == 0 else "FAIL",
            details={"pid": process.pid, "exit_code": exit_code, "duration_seconds": round(duration, 3),
                     "log_path": str(log_path)},
        )
        return CommandResult(
            name=name,
            command=command,
            exit_code=exit_code,
            duration_seconds=round(duration, 3),
            log_path=str(log_path),
            timed_out=timed_out,
        )

    @staticmethod
    def _find_release_jar(project_root: Path) -> str | None:
        from .final_artifact import FinalArtifactError, select_production_jar

        try:
            return str(select_production_jar(project_root))
        except FinalArtifactError as exc:
            if "found 0:" in str(exc):
                return None
            raise BuildRunnerError(str(exc)) from exc

    @staticmethod
    def _gametest_report(project_root: Path) -> str | None:
        report = project_root / "build" / "gametest-report.xml"
        return str(report.resolve()) if report.is_file() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (root / member.filename).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise BuildRunnerError(f"Unsafe path in Gradle archive: {member.filename}") from exc
    archive.extractall(root)


def _acquire_cache_lock_fd(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0x7FFFFFFF, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_cache_lock_fd(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0x7FFFFFFF, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def _exclusive_cache_lock(
    cache_dir: Path,
    *,
    timeout_seconds: int,
) -> Iterable[None]:
    from .root_cause_trace import emit_root_cause

    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise BuildRunnerError(
            "Gradle cache lock timeout must be a positive integer."
        )
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / ".minecraft-mod-ai-cache.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    deadline = time.monotonic() + timeout_seconds
    emit_root_cause("gradle_cache_lock_wait", stage="verify", operation="cache_lock", result="START",
                    details={"lock_path": str(lock_path), "timeout_seconds": timeout_seconds})
    try:
        while not acquired:
            try:
                _acquire_cache_lock_fd(fd)
                acquired = True
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise BuildRunnerError(
                        f"Timed out waiting for the Gradle cache lock: {lock_path}"
                    )
                time.sleep(0.2)

        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(
            fd,
            f"pid={os.getpid()}\nacquired={time.time()}\n".encode("ascii"),
        )
        os.fsync(fd)
        emit_root_cause("gradle_cache_lock_acquired", stage="verify", operation="cache_lock", result="PASS",
                        details={"lock_path": str(lock_path), "pid": os.getpid()})
        yield
    finally:
        if acquired:
            try:
                _release_cache_lock_fd(fd)
            except OSError:
                pass
        os.close(fd)


_exclusive_cache_lock._mmm_os_advisory_cache_lock = True  # type: ignore[attr-defined]


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ("taskkill", "/PID", str(process.pid), "/T", "/F"),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                shell=False,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except OSError:
            process.kill()
