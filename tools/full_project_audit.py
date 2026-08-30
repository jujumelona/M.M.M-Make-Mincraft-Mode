from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import queue
import re
import shlex
import smtplib
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tomllib
import yaml

if __package__:
    from .audit_stream_redactor import StreamingRedactor
else:
    from audit_stream_redactor import StreamingRedactor

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "audit"
REPORT_PATH = AUDIT_DIR / "FULL_PROJECT_AUDIT.json"
LOG_PATH = AUDIT_DIR / "FULL_PROJECT_AUDIT.log"

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"
STATUSES = {PASS, WARN, FAIL, SKIP}
_OUTPUT_TAIL_LIMIT = 1200
_OUTPUT_CHUNK_CHARS = 64 * 1024


@dataclass
class Check:
    name: str
    status: str
    detail: str
    duration_seconds: float = 0.0
    category: str = "core"

    @property
    def passed(self) -> bool:
        return self.status == PASS

    @property
    def blocking_failure(self) -> bool:
        return self.status == FAIL

    @property
    def non_blocking(self) -> bool:
        return self.status in {WARN, SKIP}

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["passed"] = self.passed
        value["blocking_failure"] = self.blocking_failure
        value["non_blocking"] = self.non_blocking
        return value


@dataclass(frozen=True)
class LoggedProcessResult:
    returncode: int | None
    output_tail: str
    timed_out: bool = False
    error: str | None = None


CHECKS: list[Check] = []


def _environment_secret_values() -> tuple[str, ...]:
    values: list[str] = []
    for name, secret in os.environ.items():
        upper = name.upper()
        if secret and any(
            part in upper
            for part in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "COOKIE")
        ):
            values.append(secret)
    return tuple(dict.fromkeys(values))


def redact(value: Any, *, secrets: Iterable[str] | None = None) -> str:
    secret_values = _environment_secret_values() if secrets is None else secrets
    redactor = StreamingRedactor(secret_values)
    return redactor.feed(str(value)) + redactor.finish()


def _append_log(value: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(redact(value))
        if value and not value.endswith("\n"):
            handle.write("\n")


def _drain_process_output(path: Path) -> str:
    """Redact process output in bounded chunks while retaining only a small tail."""

    redactor = StreamingRedactor(_environment_secret_values())
    tail = ""
    with path.open("r", encoding="utf-8", errors="replace") as source, LOG_PATH.open(
        "a", encoding="utf-8", errors="replace"
    ) as destination:
        while True:
            chunk = source.read(_OUTPUT_CHUNK_CHARS)
            if not chunk:
                break
            safe = redactor.feed(chunk)
            if safe:
                destination.write(safe)
                tail = (tail + safe)[-_OUTPUT_TAIL_LIMIT:]
        final = redactor.finish()
        if final:
            destination.write(final)
            tail = (tail + final)[-_OUTPUT_TAIL_LIMIT:]
    return " ".join(tail.split())


def _run_logged_process(
    args: list[str],
    *,
    timeout: int,
    cwd: Path,
    env: dict[str, str] | None = None,
    label: str | None = None,
) -> LoggedProcessResult:
    """Run a subprocess without retaining its full output in memory."""

    run_env = {**os.environ, "PYTHONUTF8": "1"}
    if env:
        run_env.update(env)
    _append_log(f"\n[{label or 'command'}] $ {shlex.join(args)}")

    temporary_path: Path | None = None
    returncode: int | None = None
    timed_out = False
    error: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=AUDIT_DIR,
            prefix=".process-output-",
            suffix=".log",
            delete=False,
        ) as raw_output:
            temporary_path = Path(raw_output.name)
            try:
                process = subprocess.run(
                    args,
                    cwd=cwd,
                    stdout=raw_output,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    env=run_env,
                    check=False,
                )
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
            except OSError as exc:
                error = f"{type(exc).__name__}: {exc}"
        tail = _drain_process_output(temporary_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    if error:
        _append_log(f"process_error={error}")
    if timed_out:
        _append_log(f"process_timeout={timeout}s")
    return LoggedProcessResult(
        returncode=returncode,
        output_tail=tail,
        timed_out=timed_out,
        error=error,
    )


def record(
    name: str,
    status: str,
    detail: str,
    duration: float = 0.0,
    *,
    category: str = "core",
) -> None:
    normalized = status.upper()
    if normalized not in STATUSES:
        raise ValueError(f"unsupported audit status: {status}")
    safe_detail = redact(detail)
    check = Check(name, normalized, safe_detail, round(duration, 3), category)
    CHECKS.append(check)
    _append_log(f"[{normalized}] [{category}] {name}: {safe_detail}")


def _record_internal_exception(name: str, exc: BaseException, duration: float = 0.0) -> None:
    """Keep programming failures compact in reports and detailed only in debug artifacts."""

    _append_log(
        "[INTERNAL TRACEBACK] "
        + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    )
    record(
        name,
        FAIL,
        f"{type(exc).__name__}: {exc}",
        duration,
        category="audit-internal",
    )


def isolated_probe(
    name: str,
    category: str,
    probe: Callable[[], tuple[str, str] | str | None],
) -> None:
    started = time.monotonic()
    try:
        result = probe()
        if result is None:
            status, detail = PASS, "ok"
        elif isinstance(result, tuple):
            status, detail = result
        else:
            status, detail = PASS, result
        record(name, status, detail, time.monotonic() - started, category=category)
    except AssertionError as exc:
        record(
            name,
            FAIL,
            f"{type(exc).__name__}: {exc}",
            time.monotonic() - started,
            category=category,
        )
    except Exception as exc:
        _record_internal_exception(name, exc, time.monotonic() - started)


def command(
    name: str,
    args: list[str],
    *,
    timeout: int = 1800,
    category: str = "core",
    failure_status: str = FAIL,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    started = time.monotonic()
    try:
        result = _run_logged_process(
            args,
            cwd=cwd or ROOT,
            timeout=timeout,
            env=env,
            label=name,
        )
    except Exception as exc:
        _record_internal_exception(name, exc, time.monotonic() - started)
        return

    if result.timed_out:
        status = failure_status
        detail = f"timeout={timeout}s; output_tail={result.output_tail}"
    elif result.error:
        status = failure_status
        detail = f"{result.error}; output_tail={result.output_tail}"
    else:
        status = PASS if result.returncode == 0 else failure_status
        detail = f"exit={result.returncode}; output_tail={result.output_tail}"
    record(name, status, detail, time.monotonic() - started, category=category)


def tracked_files() -> list[Path]:
    process = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [ROOT / item.decode("utf-8") for item in process.stdout.split(b"\0") if item]


def audit_syntax_and_data(files: list[Path]) -> None:
    failures: list[str] = []
    counts = {"python": 0, "json": 0, "yaml": 0, "notebook": 0}
    expected_parse_errors = (
        OSError,
        UnicodeError,
        SyntaxError,
        ValueError,
        yaml.YAMLError,
    )
    for path in files:
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        relative = path.relative_to(ROOT).as_posix()
        try:
            if suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=relative)
                counts["python"] += 1
            elif suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
                counts["json"] += 1
            elif suffix in {".yaml", ".yml"}:
                yaml.safe_load(path.read_text(encoding="utf-8"))
                counts["yaml"] += 1
            elif suffix == ".ipynb":
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("nbformat") != 4:
                    raise ValueError("unsupported notebook format")
                counts["notebook"] += 1
        except expected_parse_errors as exc:
            failures.append(f"{relative}: {type(exc).__name__}: {exc}")
    detail = ", ".join(f"{key}={value}" for key, value in counts.items())
    if failures:
        detail += "; failures=" + " | ".join(failures[:30])
    record("syntax_and_structured_data", FAIL if failures else PASS, detail, category="syntax")


def _mcp_server_literal_versions(source: str) -> set[str]:
    versions: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else ""
        )
        if name != "MCPServer":
            continue
        for keyword in node.keywords:
            value = keyword.value
            if (
                keyword.arg == "version"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                versions.add(value.value)
    return versions


def audit_versions() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = str(pyproject["project"]["version"])
    init_text = (ROOT / "minecraft_mod_ai/__init__.py").read_text(encoding="utf-8")
    init_match = re.search(r'^__version__\s*=\s*"([^"]+)"', init_text, re.MULTILINE)
    versions: dict[str, Any] = {
        "pyproject": expected,
        "__version__": init_match.group(1) if init_match else "missing",
    }
    mismatches: list[str] = []
    if versions["__version__"] != expected:
        mismatches.append("__version__")
    for relative in (
        "minecraft_mod_ai/mcp_server.py",
        "minecraft_mod_ai/mod_generation_mcp_server.py",
    ):
        path = ROOT / relative
        if not path.is_file():
            continue
        configured = _mcp_server_literal_versions(path.read_text(encoding="utf-8"))
        versions[relative] = sorted(configured) if configured else "no-literal-version"
        if configured and expected not in configured:
            mismatches.append(relative)
    record(
        "version_consistency",
        FAIL if mismatches else PASS,
        json.dumps({"expected": expected, "observed": versions, "mismatches": mismatches}, sort_keys=True),
        category="metadata",
    )


def module_readiness(name: str, module: str, *, optional: bool, category: str) -> None:
    def probe() -> tuple[str, str]:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ModuleNotFoundError) as exc:
            return (
                (SKIP if optional else FAIL),
                f"{'optional' if optional else 'required'} module not importable: {module} ({type(exc).__name__})",
            )
        if spec is not None:
            return PASS, f"module={module}"
        return (
            (SKIP if optional else FAIL),
            f"{'optional' if optional else 'required'} module not installed: {module}",
        )

    isolated_probe(name, category, probe)


def audit_real_subsystems() -> None:
    required = (
        ("mcp_server_module", "minecraft_mod_ai.mcp_server", "mcp"),
        ("mcp_tools_module", "minecraft_mod_ai.mcp_tools", "tools"),
        ("skill_catalog_module", "minecraft_mod_ai.skill_catalog", "skills"),
        ("retrieval_contract_module", "minecraft_mod_ai.adaptive_retrieval_contract", "retrieval"),
        ("routing_contract_module", "minecraft_mod_ai.agent_routing_intent_contract", "routing"),
        ("model_registry_module", "minecraft_mod_ai.model_registry", "model"),
        ("colab_run_modes_module", "minecraft_mod_ai.colab_run_modes", "pipeline"),
    )
    for name, module, category in required:
        module_readiness(name, module, optional=False, category=category)

    module_readiness("mcp_sdk", "mcp", optional=True, category="mcp")
    module_readiness("transformers_backend", "transformers", optional=True, category="model")
    module_readiness("faiss_backend", "faiss", optional=True, category="retrieval")
    module_readiness("llamaindex_backend", "llama_index", optional=True, category="retrieval")
    module_readiness("openai_backend", "openai", optional=True, category="model")


def audit_model_registry() -> None:
    def probe() -> tuple[str, str]:
        registry_path = ROOT / "minecraft_mod_ai/config/model_registry.yaml"
        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return FAIL, "model registry is not a mapping"
        profiles = raw.get("profiles")
        if not isinstance(profiles, dict) or not profiles:
            return FAIL, "model registry has no profiles"
        failures: list[str] = []
        role_count = 0
        for profile_name, profile in profiles.items():
            if not isinstance(profile, dict) or not isinstance(profile.get("roles"), dict):
                failures.append(f"{profile_name}: missing roles mapping")
                continue
            for role, config in profile["roles"].items():
                role_count += 1
                if not isinstance(config, dict):
                    failures.append(f"{profile_name}.{role}: config is not a mapping")
                    continue
                adapter = str(config.get("adapter", "")).strip()
                provider = str(config.get("provider", "local")).strip() or "local"
                if not adapter:
                    failures.append(f"{profile_name}.{role}: adapter missing")
                if provider == "openai_compatible":
                    for key in ("model_env", "base_url_env", "api_key_env"):
                        if not str(config.get(key, "")).strip():
                            failures.append(f"{profile_name}.{role}: {key} missing")
                elif not str(config.get("model_id", "")).strip():
                    failures.append(f"{profile_name}.{role}: model_id missing")
        if failures:
            return FAIL, " | ".join(failures[:30])
        selected = os.environ.get("MMM_MODEL_PROFILE", "").strip()
        detail = f"profiles={len(profiles)}; roles={role_count}; selected={selected or 'not-set'}"
        if selected:
            if selected not in profiles:
                return FAIL, detail + "; selected profile not found"
            from minecraft_mod_ai.model_registry import ModelRegistry

            registry = ModelRegistry(registry_path)
            loaded = registry.load_profile(selected)
            detail += f"; selected_roles={len(loaded.roles)}"
        return PASS, detail

    isolated_probe("model_registry_generic", "model", probe)


def audit_runtime_contracts() -> None:
    def tools_probe() -> str:
        from minecraft_mod_ai import mcp_server
        from minecraft_mod_ai.mcp_tools import MMMToolService

        stages = getattr(mcp_server, "_TOOL_STAGES", None)
        if not isinstance(stages, dict) or not stages:
            raise AssertionError("mcp_server._TOOL_STAGES is missing or empty")
        if not isinstance(MMMToolService, type):
            raise AssertionError("MMMToolService unavailable")
        return f"tool_stages={len(stages)}; service={MMMToolService.__name__}"

    def skills_probe() -> str:
        from minecraft_mod_ai.skill_catalog import validate_skill_catalog

        result = validate_skill_catalog()
        if not isinstance(result, dict) or result.get("passed") is not True:
            raise AssertionError(f"skill catalog validation failed: {result}")
        return "skill catalog validation passed"

    def retrieval_probe() -> str:
        module = importlib.import_module("minecraft_mod_ai.adaptive_retrieval_contract")
        public = [name for name in vars(module) if not name.startswith("_")]
        if not public:
            raise AssertionError("retrieval contract exports no public symbols")
        return f"adaptive retrieval contract exports={len(public)}"

    def routing_probe() -> str:
        module = importlib.import_module("minecraft_mod_ai.agent_routing_intent_contract")
        public = [name for name in vars(module) if not name.startswith("_")]
        if not public:
            raise AssertionError("routing contract exports no public symbols")
        return f"routing contract exports={len(public)}"

    isolated_probe("tool_registry_runtime", "tools", tools_probe)
    isolated_probe("skills_runtime", "skills", skills_probe)
    isolated_probe("retrieval_runtime", "retrieval", retrieval_probe)
    isolated_probe("routing_runtime", "routing", routing_probe)


def audit_environment_config() -> None:
    def probe() -> tuple[str, str]:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        if not pyproject.get("project", {}).get("dependencies"):
            return FAIL, "pyproject project.dependencies is empty"
        config_dir = ROOT / "minecraft_mod_ai/config"
        if not config_dir.is_dir():
            return FAIL, "minecraft_mod_ai/config is missing"
        names = sorted(
            name
            for name, value in os.environ.items()
            if value
            and (
                name.startswith("MMM_")
                or name.startswith("SMTP_")
                or name.startswith("MAIL_")
                or name.endswith("_API_KEY")
            )
        )
        return PASS, "config readable; configured env names=" + (",".join(names) if names else "none")

    isolated_probe("environment_and_config", "config", probe)


def audit_filesystem_permissions() -> None:
    def probe() -> str:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=AUDIT_DIR, prefix=".write-probe-", delete=False
        ) as handle:
            path = Path(handle.name)
            handle.write("ok")
        try:
            if path.read_text(encoding="utf-8") != "ok":
                raise RuntimeError("write/read mismatch")
        finally:
            path.unlink(missing_ok=True)
        return "audit directory write/read/delete passed"

    isolated_probe("filesystem_permissions", "filesystem", probe)


def audit_structured_output() -> None:
    def probe() -> str:
        from jsonschema import validate

        schema = {
            "type": "object",
            "required": ["ok", "items"],
            "properties": {
                "ok": {"type": "boolean"},
                "items": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        }
        payload = json.loads(json.dumps({"ok": True, "items": ["probe"]}))
        validate(instance=payload, schema=schema)
        return "JSON round-trip and jsonschema validation passed"

    isolated_probe("structured_output_schema", "schema", probe)


def audit_timeout_retry_primitives() -> None:
    def probe() -> str:
        try:
            subprocess.run(
                [sys.executable, "-c", "import time; time.sleep(2)"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=0.05,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "subprocess timeout enforcement passed"
        raise AssertionError("timeout probe unexpectedly completed")

    isolated_probe("timeout_enforcement", "resilience", probe)


def audit_concurrency_queue() -> None:
    def concurrency_probe() -> str:
        with ThreadPoolExecutor(max_workers=4) as executor:
            values = list(executor.map(lambda value: value * value, range(8)))
        if values != [value * value for value in range(8)]:
            raise AssertionError(f"unexpected executor result: {values}")
        return "ThreadPoolExecutor parallel task completion passed"

    def queue_probe() -> str:
        work: queue.Queue[int] = queue.Queue()
        for value in range(8):
            work.put(value)
        drained = [work.get_nowait() for _ in range(8)]
        if drained != list(range(8)) or not work.empty():
            raise AssertionError(f"unexpected queue result: {drained}")
        return "queue put/drain ordering passed"

    isolated_probe("concurrency_executor", "concurrency", concurrency_probe)
    isolated_probe("queue_runtime", "concurrency", queue_probe)


def audit_pipeline_wiring() -> None:
    def probe() -> tuple[str, str]:
        path = ROOT / "minecraft_mod_ai/colab_run_modes.py"
        source = path.read_text(encoding="utf-8")
        lowered = source.lower()
        missing = [mode for mode in ("full", "plan", "revise", "execute", "debug") if mode not in lowered]
        if missing:
            return FAIL, "run modes missing from colab_run_modes.py: " + ",".join(missing)
        if "full_project_audit.py" not in source:
            return FAIL, "Debug mode is not wired to full_project_audit.py"
        return PASS, "Full/Plan/Revise/Execute/Debug wiring present"

    isolated_probe("plan_revise_execute_debug_wiring", "pipeline", probe)


def audit_smtp() -> None:
    isolated_probe(
        "smtp_stdlib",
        "smtp",
        lambda: "smtplib available; Debug never sends a message"
        if smtplib.SMTP is not None
        else (FAIL, "smtplib.SMTP unavailable"),
    )

    def probe() -> tuple[str, str]:
        env = os.environ
        host = next(
            (
                env[name].strip()
                for name in ("SMTP_HOST", "MAIL_HOST", "MMM_SMTP_HOST")
                if env.get(name, "").strip()
            ),
            "",
        )
        if not host:
            return SKIP, "SMTP host not configured; no network probe"
        port_text = next(
            (
                env[name].strip()
                for name in ("SMTP_PORT", "MAIL_PORT", "MMM_SMTP_PORT")
                if env.get(name, "").strip()
            ),
            "587",
        )
        try:
            port = int(port_text)
        except ValueError:
            return FAIL, f"invalid SMTP port: {port_text!r}"
        socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        with socket.create_connection((host, port), timeout=8) as sock:
            peer = sock.getpeername()
        if port == 465:
            context = ssl.create_default_context()
            with socket.create_connection((host, port), timeout=8) as raw:
                with context.wrap_socket(raw, server_hostname=host) as tls:
                    tls.version()
            return PASS, f"DNS/TCP/TLS connectivity passed for {host}:{port}; peer={peer[0]}; no auth/send"
        return PASS, f"DNS/TCP connectivity passed for {host}:{port}; peer={peer[0]}; no message sent"

    isolated_probe("smtp_connectivity", "smtp", probe)


def audit_wheel_import() -> None:
    wheels = sorted((AUDIT_DIR / "dist").glob("*.whl"))
    if not wheels:
        record("wheel_clean_environment_import", FAIL, "no wheel produced", category="packaging")
        return

    wheel = wheels[-1]
    with tempfile.TemporaryDirectory(prefix="mmm-audit-target-") as temp:
        root = Path(temp)
        target = root / "site"
        run_cwd = root / "cwd"
        target.mkdir()
        run_cwd.mkdir()
        started = time.monotonic()
        install = _run_logged_process(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(target),
                str(wheel),
            ],
            cwd=run_cwd,
            timeout=900,
            label="wheel pip-target install",
        )
        smoke: LoggedProcessResult | None = None
        if install.returncode == 0 and not install.timed_out and not install.error:
            smoke_env = {**os.environ, "PYTHONPATH": str(target), "PYTHONUTF8": "1"}
            smoke_code = (
                "import pathlib, minecraft_mod_ai; "
                "p=pathlib.Path(minecraft_mod_ai.__file__).resolve(); "
                f"assert str(p).startswith(str(pathlib.Path({str(target)!r}).resolve())); "
                "from minecraft_mod_ai.model_registry import ModelRegistry; "
                "assert ModelRegistry().profile_names(); "
                "print(p)"
            )
            smoke = _run_logged_process(
                [sys.executable, "-c", smoke_code],
                cwd=run_cwd,
                timeout=120,
                env=smoke_env,
                label="wheel clean import",
            )
        passed = (
            install.returncode == 0
            and not install.timed_out
            and not install.error
            and smoke is not None
            and smoke.returncode == 0
            and not smoke.timed_out
            and not smoke.error
        )
        output_tail = smoke.output_tail if smoke is not None else install.output_tail
        record(
            "wheel_clean_environment_import",
            PASS if passed else FAIL,
            (
                f"wheel={wheel.name}; install_exit={install.returncode}; "
                f"smoke_exit={smoke.returncode if smoke is not None else 'not-run'}; "
                f"output_tail={output_tail}"
            ),
            time.monotonic() - started,
            category="packaging",
        )


def _initialize_artifacts() -> bool:
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("", encoding="utf-8")
        REPORT_PATH.unlink(missing_ok=True)
    except OSError as exc:
        print(f"[FAIL] [audit-runner] initialize_artifacts: {type(exc).__name__}: {exc}")
        return False
    return True


def main() -> int:
    CHECKS.clear()
    if not _initialize_artifacts():
        return 1

    try:
        files = tracked_files()
        record("tracked_file_inventory", PASS, f"tracked_files={len(files)}")
    except (OSError, subprocess.CalledProcessError) as exc:
        files = []
        record(
            "tracked_file_inventory",
            FAIL,
            f"{type(exc).__name__}: {exc}",
            category="git",
        )
    except Exception as exc:
        files = []
        _record_internal_exception("tracked_file_inventory", exc)

    families: tuple[tuple[str, Callable[[], None]], ...] = (
        ("syntax", lambda: audit_syntax_and_data(files)),
        ("versions", audit_versions),
        ("subsystems", audit_real_subsystems),
        ("model_registry", audit_model_registry),
        ("runtime_contracts", audit_runtime_contracts),
        ("environment", audit_environment_config),
        ("filesystem", audit_filesystem_permissions),
        ("structured_output", audit_structured_output),
        ("timeout_retry", audit_timeout_retry_primitives),
        ("concurrency_queue", audit_concurrency_queue),
        ("pipeline", audit_pipeline_wiring),
        ("smtp", audit_smtp),
    )
    for family_name, function in families:
        before = len(CHECKS)
        started = time.monotonic()
        try:
            function()
        except Exception as exc:
            _record_internal_exception(
                f"{family_name}_audit_internal",
                exc,
                time.monotonic() - started,
            )
        if len(CHECKS) == before:
            record(
                f"{family_name}_audit_internal",
                WARN,
                "audit family produced no checks",
                time.monotonic() - started,
                category="audit-runner",
            )

    command(
        "compileall",
        [sys.executable, "-m", "compileall", "-q", "minecraft_mod_ai", "tests", "tools"],
        category="python",
    )
    command(
        "pip_check",
        [sys.executable, "-m", "pip", "check"],
        timeout=300,
        category="dependencies",
        failure_status=WARN,
    )
    command(
        "pytest_full",
        [sys.executable, "-m", "pytest", "-q"],
        timeout=2400,
        category="tests",
    )
    command(
        "package_build",
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--outdir",
            str(AUDIT_DIR / "dist"),
        ],
        timeout=900,
        category="packaging",
    )
    try:
        audit_wheel_import()
    except Exception as exc:
        _record_internal_exception("wheel_clean_environment_import", exc)

    counts = {status: sum(check.status == status for check in CHECKS) for status in sorted(STATUSES)}
    failures = [check.name for check in CHECKS if check.status == FAIL]
    warnings = [check.name for check in CHECKS if check.status == WARN]
    skipped = [check.name for check in CHECKS if check.status == SKIP]
    overall_status = "failed" if failures else ("warning" if warnings else "passed")
    report: dict[str, Any] = {
        "schema_version": "mmm/full-project-audit-v3",
        "overall_status": overall_status,
        "python": sys.version,
        "tracked_file_count": len(files),
        "summary": {
            "total": len(CHECKS),
            "passed": counts[PASS],
            "warned": counts[WARN],
            "failed": counts[FAIL],
            "skipped": counts[SKIP],
        },
        "checks": [check.payload() for check in CHECKS],
        "failed_checks": failures,
        "warning_checks": warnings,
        "skipped_checks": skipped,
        "status_semantics": {
            "PASS": "passed=true; successful check",
            "WARN": "passed=false; non_blocking=true",
            "SKIP": "passed=false; non_blocking=true",
            "FAIL": "passed=false; blocking_failure=true",
        },
        "limitations": [
            "Debug does not launch Minecraft/Fabric clients or dedicated servers.",
            "Large local model weights are not downloaded merely to prove a hard-coded model identity.",
            "Configured providers are validated generically through registry/runtime readiness; explicit generation remains a runtime gate.",
            "SMTP connectivity never sends a message and secrets are redacted from report/log output.",
            "Optional uninstalled integrations are SKIP; host-environment pip conflicts are WARN rather than project FAIL.",
        ],
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
