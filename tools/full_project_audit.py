from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "audit"
REPORT_PATH = AUDIT_DIR / "FULL_PROJECT_AUDIT.json"
LOG_PATH = AUDIT_DIR / "FULL_PROJECT_AUDIT.log"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    duration_seconds: float = 0.0


CHECKS: list[Check] = []
LOGS: list[str] = []


def record(name: str, passed: bool, detail: str, duration: float = 0.0) -> None:
    CHECKS.append(Check(name, passed, detail, round(duration, 3)))
    LOGS.append(f"[{ 'PASS' if passed else 'FAIL' }] {name}: {detail}")


def command(name: str, args: list[str], *, timeout: int = 1800) -> None:
    started = time.monotonic()
    process = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    output = process.stdout.strip()
    LOGS.append(f"\n$ {' '.join(args)}\n{output}\n")
    record(
        name,
        process.returncode == 0,
        f"exit={process.returncode}; output_tail={output[-1200:]}",
        time.monotonic() - started,
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
    python_count = json_count = yaml_count = notebook_count = 0
    for path in files:
        suffix = path.suffix.lower()
        relative = path.relative_to(ROOT).as_posix()
        try:
            if suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=relative)
                python_count += 1
            elif suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
                json_count += 1
            elif suffix in {".yaml", ".yml"}:
                yaml.safe_load(path.read_text(encoding="utf-8"))
                yaml_count += 1
            elif suffix == ".ipynb":
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("nbformat") not in {4}:
                    raise ValueError("unsupported notebook format")
                notebook_count += 1
        except Exception as exc:  # noqa: BLE001 - audit must aggregate all failures.
            failures.append(f"{relative}: {type(exc).__name__}: {exc}")
    detail = (
        f"python={python_count}, json={json_count}, yaml={yaml_count}, "
        f"notebooks={notebook_count}"
    )
    if failures:
        detail += "; failures=" + " | ".join(failures[:30])
    record("syntax_and_structured_data", not failures, detail)


def audit_relative_imports(files: list[Path]) -> None:
    package_root = ROOT / "minecraft_mod_ai"
    failures: list[str] = []
    checked = 0
    for path in files:
        if path.suffix != ".py" or package_root not in path.parents:
            continue
        relative_module = path.relative_to(ROOT).with_suffix("").parts
        if relative_module[-1] == "__init__":
            current_package = relative_module[:-1]
        else:
            current_package = relative_module[:-1]
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
            module_parts = tuple(node.module.split(".")) if node.module else ()
            target_parts = (*base, *module_parts)
            module_file = ROOT.joinpath(*target_parts).with_suffix(".py")
            package_file = ROOT.joinpath(*target_parts, "__init__.py")
            if not module_file.is_file() and not package_file.is_file():
                failures.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: "
                    f"missing relative module {'.'.join(target_parts)}"
                )
    detail = f"checked={checked}"
    if failures:
        detail += "; failures=" + " | ".join(failures[:30])
    record("relative_import_resolution", not failures, detail)


def audit_versions() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = pyproject["project"]["version"]
    init_text = (ROOT / "minecraft_mod_ai/__init__.py").read_text(encoding="utf-8")
    init_match = re.search(r'^__version__\s*=\s*"([^"]+)"', init_text, re.MULTILINE)
    server_versions: dict[str, str] = {}
    for relative in (
        "minecraft_mod_ai/mcp_server.py",
        "minecraft_mod_ai/mod_generation_mcp_server.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        match = re.search(r'MCPServer\([^\n]+version="([^"]+)"', text)
        server_versions[relative] = match.group(1) if match else "missing"
    actual = init_match.group(1) if init_match else "missing"
    mismatches = {
        "pyproject": expected,
        "__version__": actual,
        **server_versions,
    }
    passed = all(value == expected for value in mismatches.values())
    record("version_consistency", passed, json.dumps(mismatches, sort_keys=True))


def audit_mod_only_surface() -> None:
    forbidden_by_file = {
        "minecraft_mod_ai/mcp_server.py": (
            "generate_world_ir",
            "compile_world_ir",
        ),
        "minecraft_mod_ai/mcp_tools.py": ("def generate_world_ir",),
        "mcp_gateway.py": ('"generate_world_ir"', '"compile_world"'),
        "minecraft_mod_ai/skill_catalog.py": (
            '"generate_world_ir"',
            '"compile_world_ir"',
        ),
        "minecraft_mod_ai/packaged_skills.json": (
            "generate_world_ir",
            "compile_world_ir",
        ),
    }
    failures: list[str] = []
    for relative, forbidden in forbidden_by_file.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for value in forbidden:
            if value in text:
                failures.append(f"{relative}: contains {value}")

    removed_files = (
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
    for relative in removed_files:
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
        not failures,
        "clean" if not failures else " | ".join(failures),
    )


def audit_obsolete_imports(files: list[Path]) -> None:
    forbidden_modules = {
        "minecraft_mod_ai.builder_mcp_server",
        "minecraft_mod_ai.builder_contract_service",
        "minecraft_mod_ai.buildspec",
        "minecraft_mod_ai.skill_scope_contract",
        "minecraft_mod_ai.scalable_world_compiler",
        "minecraft_mod_ai.world_compiler",
        "minecraft_mod_ai.world_runtime_generator",
    }
    failures: list[str] = []
    for path in files:
        if path.suffix != ".py" or path.name == "reconcile_mod_scope.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name in forbidden_modules or any(
                    name.endswith("." + item.rsplit(".", 1)[-1])
                    for item in forbidden_modules
                ):
                    failures.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: imports {name}"
                    )
    record(
        "obsolete_imports",
        not failures,
        "clean" if not failures else " | ".join(failures[:30]),
    )


def audit_runtime_contracts() -> None:
    started = time.monotonic()
    try:
        import mcp_gateway
        import minecraft_mod_ai
        from minecraft_mod_ai import mcp_server
        from minecraft_mod_ai.mcp_tools import MMMToolService
        from minecraft_mod_ai.mod_development_methods import (
            resolve_mod_development_methods,
        )
        from minecraft_mod_ai.skill_catalog import validate_skill_catalog

        food = resolve_mod_development_methods("음식 아이템 모드를 만들어줘")
        biome = resolve_mod_development_methods("새 바이옴 구조물 모드를 만들어줘")
        exclusion = resolve_mod_development_methods(
            "월드 ZIP은 만들지 말고 모드만 만들어줘"
        )
        assertions = [
            minecraft_mod_ai.__version__ == "0.8.0",
            "generate_world_ir" not in mcp_server._TOOL_STAGES,
            "compile_world_ir" not in mcp_server._TOOL_STAGES,
            "generate_world_ir" not in mcp_gateway._CORE_TOOLS,
            "compile_world" not in mcp_gateway._PRODUCTION_TOOLS,
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
            True,
            "imports, skill catalog and mod-scope routing passed",
            time.monotonic() - started,
        )
    except Exception as exc:  # noqa: BLE001 - aggregate audit failure.
        record(
            "runtime_contract_smoke",
            False,
            f"{type(exc).__name__}: {exc}",
            time.monotonic() - started,
        )


def audit_wheel_import() -> None:
    wheels = sorted((AUDIT_DIR / "dist").glob("*.whl"))
    if not wheels:
        record("wheel_clean_environment_import", False, "no wheel produced")
        return
    with tempfile.TemporaryDirectory(prefix="mmm-audit-venv-") as temp:
        venv = Path(temp)
        command_args = [sys.executable, "-m", "venv", str(venv)]
        started = time.monotonic()
        create = subprocess.run(command_args, cwd=ROOT, capture_output=True, text=True)
        if create.returncode != 0:
            record(
                "wheel_clean_environment_import",
                False,
                create.stdout + create.stderr,
                time.monotonic() - started,
            )
            return
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        install = subprocess.run(
            [str(python), "-m", "pip", "install", str(wheels[-1])],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        smoke = subprocess.run(
            [
                str(python),
                "-c",
                "import minecraft_mod_ai; "
                "from minecraft_mod_ai.mod_development_methods import "
                "resolve_mod_development_methods; "
                "assert minecraft_mod_ai.__version__ == '0.8.0'; "
                "assert resolve_mod_development_methods('food item mod')",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        ) if install.returncode == 0 else None
        output = install.stdout + install.stderr
        if smoke is not None:
            output += smoke.stdout + smoke.stderr
        passed = install.returncode == 0 and smoke is not None and smoke.returncode == 0
        LOGS.append(f"\n[wheel clean env]\n{output}\n")
        record(
            "wheel_clean_environment_import",
            passed,
            f"wheel={wheels[-1].name}; output_tail={output[-1200:]}",
            time.monotonic() - started,
        )


def main() -> int:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    files = tracked_files()
    record("tracked_file_inventory", True, f"tracked_files={len(files)}")
    audit_syntax_and_data(files)
    audit_relative_imports(files)
    audit_versions()
    audit_mod_only_surface()
    audit_obsolete_imports(files)
    audit_runtime_contracts()

    command("compileall", [sys.executable, "-m", "compileall", "-q", "minecraft_mod_ai", "tests", "tools"])
    command("pip_check", [sys.executable, "-m", "pip", "check"], timeout=300)
    command("pytest_full", [sys.executable, "-m", "pytest", "-q"], timeout=2400)
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
    )
    audit_wheel_import()

    passed = all(check.passed for check in CHECKS)
    report: dict[str, Any] = {
        "schema_version": "mmm/full-project-audit-v1",
        "overall_status": "passed" if passed else "failed",
        "python": sys.version,
        "tracked_file_count": len(files),
        "checks": [asdict(check) for check in CHECKS],
        "failed_checks": [check.name for check in CHECKS if not check.passed],
        "limitations": [
            "No Minecraft/Fabric client or dedicated server was launched by this Python repository audit.",
            "No GPU model, Blockbench, JDT LS, Modrinth publication or external sidecar was executed.",
            "Those integrations remain release-time gates for requests that require them.",
        ],
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOG_PATH.write_text("\n".join(LOGS) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
