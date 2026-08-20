from __future__ import annotations

import ast
import importlib.util
import json
import os
import queue
import re
import smtplib
import subprocess
import sys
import tempfile
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "audit"
REPORT_PATH = AUDIT_DIR / "FULL_PROJECT_AUDIT.json"
LOG_PATH = AUDIT_DIR / "FULL_PROJECT_AUDIT.log"

STATUSES = {"PASS", "WARN", "FAIL", "SKIP"}

OBSOLETE_FILES = (
    "mcp_gateway.py",
    "colab_app.py",
    "minecraft_mod_ai/builder_mcp_server.py",
    "minecraft_mod_ai/builder_contract_service.py",
    "minecraft_mod_ai/buildspec.py",
    "minecraft_mod_ai/config/buildspec_catalog.yaml",
    "minecraft_mod_ai/scalable_world_compiler.py",
    "minecraft_mod_ai/skill_scope_contract.py",
    "minecraft_mod_ai/world_compiler.py",
    "minecraft_mod_ai/world_runtime_generator.py",
    "docs/BUILDER_CONTRACT.md",
)

OBSOLETE_MODULES = {
    "minecraft_mod_ai.builder_mcp_server",
    "minecraft_mod_ai.builder_contract_service",
    "minecraft_mod_ai.buildspec",
    "minecraft_mod_ai.skill_scope_contract",
    "minecraft_mod_ai.scalable_world_compiler",
    "minecraft_mod_ai.world_compiler",
    "minecraft_mod_ai.world_runtime_generator",
}


@dataclass
class Check:
    name: str
    status: str
    detail: str
    duration_seconds: float = 0.0
    category: str = "core"

    @property
    def passed(self) -> bool:
        return self.status != "FAIL"

    def payload(self) -> dict[str, Any]:
        item = asdict(self)
        item["passed"] = self.passed
        return item


CHECKS: list[Check] = []
LOGS: list[str] = []


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
    check = Check(name, normalized, detail, round(duration, 3), category)
    CHECKS.append(check)
    LOGS.append(f"[{normalized}] [{category}] {name}: {detail}")


def command(
    name: str,
    args: list[str],
    *,
    timeout: int = 1800,
    category: str = "core",
) -> None:
    started = time.monotonic()
    try:
        process = subprocess.run(
            args,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        LOGS.append(f"\n$ {' '.join(args)}\n{output}\n")
        record(
            name,
            "FAIL",
            f"timeout={timeout}s",
            time.monotonic() - started,
            category=category,
        )
        return
    except Exception as exc:  # noqa: BLE001 - aggregate audit failures.
        record(
            name,
            "FAIL",
            f"{type(exc).__name__}: {exc}",
            time.monotonic() - started,
            category=category,
        )
        return

    output = process.stdout.strip()
    LOGS.append(f"\n$ {' '.join(args)}\n{output}\n")
    record(
        name,
        "PASS" if process.returncode == 0 else "FAIL",
        f"exit={process.returncode}; output_tail={output[-1200:]}",
        time.monotonic() - started,
        category=category,
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
            status, detail = "PASS", "ok"
        elif isinstance(result, tuple):
            status, detail = result
        else:
            status, detail = "PASS", result
        record(name, status, detail, time.monotonic() - started, category=category)
    except Exception as exc:  # noqa: BLE001 - every probe must be isolated.
        record(
            name,
            "FAIL",
            f"{type(exc).__name__}: {exc}",
            time.monotonic() - started,
            category=category,
        )


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
    for path in files:
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
        except Exception as exc:  # noqa: BLE001 - aggregate audit failures.
            failures.append(f"{relative}: {type(exc).__name__}: {exc}")

    detail = ", ".join(f"{key}={value}" for key, value in counts.items())
    if failures:
        detail += "; failures=" + " | ".join(failures[:30])
    record("syntax_and_structured_data", "FAIL" if failures else "PASS", detail)


def audit_relative_imports(files: list[Path]) -> None:
    package_root = ROOT / "minecraft_mod_ai"
    failures: list[str] = []
    checked = 0
    for path in files:
        if path.suffix != ".py" or package_root not in path.parents:
            continue
        module_parts = path.relative_to(ROOT).with_suffix("").parts
        current_package = module_parts[:-1]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level <= 0:
                continue
            checked += 1
            ascend = node.level - 1
            if ascend > len(current_package):
                failures.append(f"{path.relative_to(ROOT)}:{node.lineno}: escapes package")
                continue
            base = current_package[: len(current_package) - ascend]
            target = (*base, *((node.module or "").split(".") if node.module else ()))
            module_file = ROOT.joinpath(*target).with_suffix(".py")
            package_file = ROOT.joinpath(*target, "__init__.py")
            if not module_file.is_file() and not package_file.is_file():
                failures.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: missing relative module {'.'.join(target)}"
                )

    detail = f"checked={checked}"
    if failures:
        detail += "; failures=" + " | ".join(failures[:30])
    record("relative_import_resolution", "FAIL" if failures else "PASS", detail)


def audit_versions() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = pyproject["project"]["version"]
    init_text = (ROOT / "minecraft_mod_ai/__init__.py").read_text(encoding="utf-8")
    init_match = re.search(r'^__version__\s*=\s*"([^"]+)"', init_text, re.MULTILINE)
    versions = {
        "pyproject": expected,
        "__version__": init_match.group(1) if init_match else "missing",
    }
    for relative in (
        "minecraft_mod_ai/mcp_server.py",
        "minecraft_mod_ai/mod_generation_mcp_server.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        match = re.search(r'MCPServer\([^\n]+version="([^"]+)"', text)
        versions[relative] = match.group(1) if match else "missing"
    record(
        "version_consistency",
        "PASS" if all(value == expected for value in versions.values()) else "FAIL",
        json.dumps(versions, sort_keys=True),
    )


def audit_mod_only_surface() -> None:
    failures: list[str] = []
    forbidden_by_file = {
        "minecraft_mod_ai/mcp_server.py": ("generate_world_ir", "compile_world_ir"),
        "minecraft_mod_ai/mcp_tools.py": ("def generate_world_ir",),
        "minecraft_mod_ai/skill_catalog.py": ('"generate_world_ir"', '"compile_world_ir"'),
        "minecraft_mod_ai/packaged_skills.json": ("generate_world_ir", "compile_world_ir"),
    }
    for relative, forbidden in forbidden_by_file.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for value in forbidden:
            if value in text:
                failures.append(f"{relative}: contains {value}")

    for relative in OBSOLETE_FILES:
        if (ROOT / relative).exists():
            failures.append(f"obsolete file still exists: {relative}")

    for relative in (
        ".mcp.json",
        "plugins/mmm-minecraft-mod-ai/.mcp.json",
        "minecraft_mod_ai/config/external_mcp_registry.yaml",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        if "minecraft_mod_ai.mod_generation_mcp_server" not in text:
            failures.append(f"{relative}: generation server is not mod-only")

    record(
        "mod_only_public_surface",
        "FAIL" if failures else "PASS",
        "clean" if not failures else " | ".join(failures),
    )


def audit_obsolete_imports(files: list[Path]) -> None:
    failures: list[str] = []
    for path in files:
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name in OBSOLETE_MODULES or any(
                    name.endswith("." + item.rsplit(".", 1)[-1])
                    for item in OBSOLETE_MODULES
                ):
                    failures.append(f"{path.relative_to(ROOT)}:{node.lineno}: imports {name}")

    record(
        "obsolete_imports",
        "FAIL" if failures else "PASS",
        "clean" if not failures else " | ".join(failures[:30]),
    )


def audit_runtime_contracts() -> None:
    started = time.monotonic()
    try:
        import minecraft_mod_ai
        from minecraft_mod_ai import mcp_server
        from minecraft_mod_ai.mcp_tools import MMMToolService
        from minecraft_mod_ai.mod_development_methods import resolve_mod_development_methods
        from minecraft_mod_ai.skill_catalog import validate_skill_catalog

        expected_version = tomllib.loads(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]["version"]
        food = resolve_mod_development_methods("음식 아이템 모드를 만들어줘")
        biome = resolve_mod_development_methods("새 바이옴 구조물 모드를 만들어줘")
        exclusion = resolve_mod_development_methods("월드 ZIP은 만들지 말고 모드만 만들어줘")
        assertions = [
            minecraft_mod_ai.__version__ == expected_version,
            "generate_world_ir" not in mcp_server._TOOL_STAGES,
            "compile_world_ir" not in mcp_server._TOOL_STAGES,
            not hasattr(MMMToolService, "generate_world_ir"),
            "fabric_worldgen" not in food["method_ids"],
            "fabric_worldgen" in biome["method_ids"],
            exclusion["standalone_map_requested"] is False,
            validate_skill_catalog()["passed"] is True,
        ]
        if not all(assertions):
            raise AssertionError(f"runtime assertions={assertions}")
        record(
            "runtime_contract_smoke",
            "PASS",
            "canonical MCP, skill catalog and mod-scope routing passed",
            time.monotonic() - started,
            category="runtime",
        )
    except Exception as exc:  # noqa: BLE001 - aggregate audit failure.
        record(
            "runtime_contract_smoke",
            "FAIL",
            f"{type(exc).__name__}: {exc}",
            time.monotonic() - started,
            category="runtime",
        )


def module_readiness(name: str, module: str, *, optional: bool, category: str) -> None:
    def probe() -> tuple[str, str]:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ModuleNotFoundError) as exc:
            if optional:
                return "SKIP", f"optional module not installed: {module} ({type(exc).__name__})"
            return "FAIL", f"required module not importable: {module} ({type(exc).__name__})"
        if spec is not None:
            return "PASS", f"module={module}"
        if optional:
            return "SKIP", f"optional module not installed: {module}"
        return "FAIL", f"required module not importable: {module}"

    isolated_probe(name, category, probe)


def audit_environment_config() -> None:
    def probe() -> tuple[str, str]:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        if not pyproject.get("project", {}).get("dependencies"):
            return "FAIL", "pyproject project.dependencies is empty"
        config_dir = ROOT / "minecraft_mod_ai/config"
        if not config_dir.is_dir():
            return "FAIL", "minecraft_mod_ai/config is missing"
        interesting = sorted(
            name
            for name, value in os.environ.items()
            if value and (
                name.startswith("MMM_")
                or name.startswith("SMTP_")
                or name.endswith("_API_KEY")
            )
        )
        return "PASS", "config readable; configured env names=" + (
            ",".join(interesting) if interesting else "none"
        )

    isolated_probe("environment_and_config", "config", probe)


def audit_filesystem_permissions() -> None:
    def probe() -> str:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=AUDIT_DIR,
            prefix=".write-probe-",
            delete=False,
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


def audit_integration_readiness() -> None:
    required = (
        ("mcp_sdk", "mcp", "mcp"),
        ("mcp_server_module", "minecraft_mod_ai.mcp_server", "mcp"),
        ("mcp_tools_module", "minecraft_mod_ai.mcp_tools", "tools"),
        ("external_mcp_module", "minecraft_mod_ai.integrations.external_mcp", "mcp"),
        ("agent_tools_module", "minecraft_mod_ai.integrations.agent_tools", "tools"),
        ("skill_catalog_module", "minecraft_mod_ai.skill_catalog", "skills"),
        ("retrieval_module", "minecraft_mod_ai.architect.retrieval", "retrieval"),
        ("adaptive_retrieval_contract", "minecraft_mod_ai.adaptive_retrieval_contract", "retrieval"),
        ("routing_contract_module", "minecraft_mod_ai.agent_routing_intent_contract", "routing"),
        ("model_registry_module", "minecraft_mod_ai.config.model_registry", "model"),
    )
    for name, module, category in required:
        module_readiness(name, module, optional=False, category=category)

    for name, module, category in (
        ("transformers_backend", "transformers", "model"),
        ("faiss_backend", "faiss", "retrieval"),
        ("llamaindex_backend", "llama_index", "retrieval"),
        ("openai_backend", "openai", "model"),
    ):
        module_readiness(name, module, optional=True, category=category)

    isolated_probe(
        "smtp_stdlib",
        "smtp",
        lambda: "smtplib available; no connection or message send attempted"
        if smtplib.SMTP is not None
        else ("FAIL", "smtplib.SMTP unavailable"),
    )

    def smtp_config_probe() -> tuple[str, str]:
        names = sorted(
            name
            for name, value in os.environ.items()
            if value and ("SMTP" in name.upper() or name.upper().startswith("MAIL_"))
        )
        if not names:
            return "SKIP", "no SMTP/mail environment configured; network probe intentionally skipped"
        return "PASS", "SMTP/mail configuration names present (values redacted): " + ",".join(names)

    isolated_probe("smtp_runtime_config", "smtp", smtp_config_probe)

    isolated_probe(
        "external_service_connectivity",
        "external",
        lambda: (
            "SKIP",
            "network side effects disabled in Debug audit; service connectivity must be exercised by explicit runtime gates",
        ),
    )


def audit_wheel_import() -> None:
    wheels = sorted((AUDIT_DIR / "dist").glob("*.whl"))
    if not wheels:
        record("wheel_clean_environment_import", "FAIL", "no wheel produced", category="packaging")
        return

    with tempfile.TemporaryDirectory(prefix="mmm-audit-venv-") as temp:
        env_dir = Path(temp)
        started = time.monotonic()
        create = subprocess.run(
            [sys.executable, "-m", "venv", str(env_dir)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if create.returncode != 0:
            record(
                "wheel_clean_environment_import",
                "FAIL",
                create.stdout + create.stderr,
                time.monotonic() - started,
                category="packaging",
            )
            return

        python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        install = subprocess.run(
            [str(python), "-m", "pip", "install", str(wheels[-1])],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        smoke = None
        if install.returncode == 0:
            smoke = subprocess.run(
                [
                    str(python),
                    "-c",
                    "import minecraft_mod_ai; "
                    "from importlib.metadata import version; "
                    "from minecraft_mod_ai.mod_development_methods import resolve_mod_development_methods; "
                    "assert minecraft_mod_ai.__version__ == version('minecraft-mod-ai'); "
                    "assert resolve_mod_development_methods('food item mod')",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )

        output = install.stdout + install.stderr
        if smoke is not None:
            output += smoke.stdout + smoke.stderr
        passed = install.returncode == 0 and smoke is not None and smoke.returncode == 0
        LOGS.append(f"\n[wheel clean env]\n{output}\n")
        record(
            "wheel_clean_environment_import",
            "PASS" if passed else "FAIL",
            f"wheel={wheels[-1].name}; output_tail={output[-1200:]}",
            time.monotonic() - started,
            category="packaging",
        )


def main() -> int:
    CHECKS.clear()
    LOGS.clear()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        files = tracked_files()
        record("tracked_file_inventory", "PASS", f"tracked_files={len(files)}")
    except Exception as exc:  # noqa: BLE001 - keep audit progressing.
        files = []
        record("tracked_file_inventory", "FAIL", f"{type(exc).__name__}: {exc}")

    for name, function in (
        ("syntax_audit_internal", lambda: audit_syntax_and_data(files)),
        ("relative_import_audit_internal", lambda: audit_relative_imports(files)),
        ("version_audit_internal", audit_versions),
        ("public_surface_audit_internal", audit_mod_only_surface),
        ("obsolete_import_audit_internal", lambda: audit_obsolete_imports(files)),
        ("runtime_contract_audit_internal", audit_runtime_contracts),
        ("environment_config_audit_internal", audit_environment_config),
        ("filesystem_audit_internal", audit_filesystem_permissions),
        ("structured_output_audit_internal", audit_structured_output),
        ("timeout_audit_internal", audit_timeout_retry_primitives),
        ("concurrency_queue_audit_internal", audit_concurrency_queue),
        ("integration_readiness_audit_internal", audit_integration_readiness),
    ):
        before = len(CHECKS)
        started = time.monotonic()
        try:
            function()
        except Exception as exc:  # noqa: BLE001 - one audit family must not abort the run.
            record(
                name,
                "FAIL",
                f"{type(exc).__name__}: {exc}",
                time.monotonic() - started,
                category="audit-runner",
            )
        if len(CHECKS) == before:
            record(
                name,
                "WARN",
                "audit family produced no checks",
                time.monotonic() - started,
                category="audit-runner",
            )

    command(
        "compileall",
        [sys.executable, "-m", "compileall", "-q", "minecraft_mod_ai", "tests", "tools"],
        category="python",
    )
    command("pip_check", [sys.executable, "-m", "pip", "check"], timeout=300, category="dependencies")
    command("pytest_full", [sys.executable, "-m", "pytest", "-q"], timeout=2400, category="tests")
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
    except Exception as exc:  # noqa: BLE001 - keep report generation guaranteed.
        record(
            "wheel_clean_environment_import",
            "FAIL",
            f"{type(exc).__name__}: {exc}",
            category="packaging",
        )

    counts = {status: sum(check.status == status for check in CHECKS) for status in sorted(STATUSES)}
    failures = [check.name for check in CHECKS if check.status == "FAIL"]
    warnings = [check.name for check in CHECKS if check.status == "WARN"]
    skipped = [check.name for check in CHECKS if check.status == "SKIP"]
    overall_status = "failed" if failures else ("warning" if warnings else "passed")
    report: dict[str, Any] = {
        "schema_version": "mmm/full-project-audit-v2",
        "overall_status": overall_status,
        "python": sys.version,
        "tracked_file_count": len(files),
        "summary": {
            "total": len(CHECKS),
            "passed": counts["PASS"],
            "warned": counts["WARN"],
            "failed": counts["FAIL"],
            "skipped": counts["SKIP"],
        },
        "checks": [check.payload() for check in CHECKS],
        "failed_checks": failures,
        "warning_checks": warnings,
        "skipped_checks": skipped,
        "limitations": [
            "No Minecraft/Fabric client or dedicated server was launched by this Python repository audit.",
            "No GPU model, Blockbench, JDT LS, Modrinth publication or external service was contacted.",
            "SMTP and other external services are configuration/readiness checks only; no message or network side effect is performed.",
            "Optional uninstalled or unconfigured integrations are reported as SKIP rather than false failures.",
            "Runtime integration gates remain required when a request explicitly needs those external systems.",
        ],
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOG_PATH.write_text("\n".join(LOGS) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
