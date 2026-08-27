from __future__ import annotations

"""Real Gradle build verifier for isolated reuse-proof workspaces.

The verifier executes the target Gradle wrapper, records the observed toolchain,
and treats JUnit XML as the only authority for individual test identities.  Console
summaries may establish aggregate counts but can never fabricate a per-test PASS.
"""

import hashlib
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BuildToolchainReceipt:
    gradle_version: str = "8.8"
    distribution_sha256: str = ""
    java_version: str = "21"
    loader: str = "fabric"
    minecraft_version: str = "1.21.1"
    toolchain_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gradle_version": self.gradle_version,
            "distribution_sha256": self.distribution_sha256,
            "java_version": self.java_version,
            "loader": self.loader,
            "minecraft_version": self.minecraft_version,
            "toolchain_hash": self.toolchain_hash,
        }


@dataclass(frozen=True)
class BuildVerificationReceipt:
    build_tool: str
    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    compile_passed: bool
    tests_passed: bool
    unresolved_symbols: tuple[str, ...]
    missing_resources: tuple[str, ...]
    tests_executed: int = 0
    tests_passed_count: int = 0
    tests_failed_count: int = 0
    executed_test_ids: tuple[str, ...] = ()
    individual_test_results: Mapping[str, bool] = field(default_factory=dict)
    toolchain: BuildToolchainReceipt | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/build-verification-receipt-v1",
            "build_tool": self.build_tool,
            "command": list(self.command),
            "exit_code": self.exit_code,
            "stdout": self.stdout[:4096],
            "stderr": self.stderr[:4096],
            "compile_passed": self.compile_passed,
            "tests_passed": self.tests_passed,
            "unresolved_symbols": list(self.unresolved_symbols),
            "missing_resources": list(self.missing_resources),
            "tests_executed": self.tests_executed,
            "tests_passed_count": self.tests_passed_count,
            "tests_failed_count": self.tests_failed_count,
            "executed_test_ids": list(self.executed_test_ids),
            "individual_test_results": dict(self.individual_test_results),
            "toolchain": self.toolchain.to_dict() if self.toolchain else None,
        }


def _find_gradle_wrapper(workspace_root: Path) -> tuple[Path | None, str]:
    gradlew = workspace_root / ("gradlew.bat" if os.name == "nt" else "gradlew")
    wrapper_jar = workspace_root / "gradle" / "wrapper" / "gradle-wrapper.jar"
    props = workspace_root / "gradle" / "wrapper" / "gradle-wrapper.properties"

    if gradlew.exists() and wrapper_jar.exists() and props.exists():
        return gradlew, "gradle_wrapper"

    return None, "none"


def _canonical_test_id(class_name: str, case_name: str) -> str:
    """Return one stable FQCN.method test identity without permissive aliases."""

    cls = str(class_name or "").strip()
    method = str(case_name or "").strip()
    no_arg = re.fullmatch(r"([A-Za-z_$][A-Za-z0-9_$]*)\(\)", method)
    if no_arg:
        method = no_arg.group(1)
    if not cls or not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_.$]*", cls):
        return ""
    if not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", method):
        return ""
    return f"{cls}.{method}"


def _parse_test_results(
    workspace_root: Path,
    stdout: str,
) -> tuple[int, int, int, tuple[str, ...], dict[str, bool]]:
    """Extract aggregate counts plus exact individual results from JUnit XML only."""

    executed = 0
    failed = 0
    test_ids: list[str] = []
    individual_results: dict[str, bool] = {}

    test_results_dir = workspace_root / "build" / "test-results" / "test"
    if test_results_dir.is_dir():
        import xml.etree.ElementTree as ET

        for xml_file in sorted(test_results_dir.glob("*.xml")):
            try:
                root = ET.parse(xml_file).getroot()
                if root.tag != "testsuite":
                    continue
                suite_name = str(root.attrib.get("name") or "").strip()
                for case in root.iter("testcase"):
                    executed += 1
                    has_failure = (
                        case.find("failure") is not None
                        or case.find("error") is not None
                    )
                    is_skipped = case.find("skipped") is not None
                    if has_failure:
                        failed += 1
                    class_name = str(case.attrib.get("classname") or suite_name).strip()
                    case_name = str(case.attrib.get("name") or "").strip()
                    canonical_id = _canonical_test_id(class_name, case_name)
                    if not canonical_id:
                        continue
                    passed_individually = not has_failure and not is_skipped
                    existing = individual_results.get(canonical_id)
                    if existing is None:
                        individual_results[canonical_id] = passed_individually
                    else:
                        # Duplicate canonical IDs are fail-closed: all observations must pass.
                        individual_results[canonical_id] = bool(existing and passed_individually)
                    test_ids.append(canonical_id)
            except (ET.ParseError, OSError, ValueError):
                # Malformed/missing XML simply cannot provide individual proof.
                continue

    if executed == 0:
        # Aggregate console summaries are useful for build diagnostics only.  They do not
        # create individual test identities or PASS receipts.
        match = re.search(
            r"(\d+)\s+tests completed,\s+(\d+)\s+failed",
            stdout,
            re.IGNORECASE,
        )
        if match:
            executed = int(match.group(1))
            failed = int(match.group(2))

    passed = max(0, executed - failed)
    return executed, passed, failed, tuple(dict.fromkeys(test_ids)), individual_results


def _inspect_build_toolchain(workspace_root: Path) -> BuildToolchainReceipt:
    props = workspace_root / "gradle" / "wrapper" / "gradle-wrapper.properties"
    dist_sha = ""
    gradle_ver = "8.10.2"
    if props.exists():
        text = props.read_text(encoding="utf-8")
        url_match = re.search(r"distributionUrl=([^\r\n]+)", text)
        if url_match:
            dist_url = url_match.group(1).replace(r"\:", ":")
            v_match = re.search(r"gradle-([0-9.]+)-", dist_url)
            if v_match:
                gradle_ver = v_match.group(1)
        sha_match = re.search(r"distributionSha256Sum=([^\r\n]+)", text)
        if sha_match:
            dist_sha = sha_match.group(1).strip()

    loader = "fabric"
    mc_ver = "1.21.1"
    bg = workspace_root / "build.gradle"
    bg_kts = workspace_root / "build.gradle.kts"
    bg_text = ""
    if bg.exists():
        bg_text = bg.read_text(encoding="utf-8")
    elif bg_kts.exists():
        bg_text = bg_kts.read_text(encoding="utf-8")

    if "net.neoforged" in bg_text:
        loader = "neoforge"
    elif "net.minecraftforge" in bg_text:
        loader = "forge"

    mc_match = re.search(
        r"['\"]com\.mojang:minecraft:([^'\"]+)['\"]",
        bg_text,
    ) or re.search(
        r"minecraft_version\s*=\s*['\"]?([^'\"\s]+)",
        bg_text,
    )
    if mc_match:
        mc_ver = mc_match.group(1).strip()

    java_ver = "21"
    try:
        res_java = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        j_text = res_java.stderr + res_java.stdout
        j_match = re.search(r'version\s+"?([0-9._]+)"?', j_text)
        if j_match:
            v_str = j_match.group(1)
            parts = v_str.split(".")
            java_ver = parts[1] if parts[0] == "1" and len(parts) > 1 else parts[0]
    except (OSError, subprocess.SubprocessError):
        pass

    toolchain_id = f"{loader}:{gradle_ver}:{mc_ver}:{java_ver}:{dist_sha or 'local'}"
    toolchain_hash = "sha256:" + hashlib.sha256(toolchain_id.encode("utf-8")).hexdigest()

    return BuildToolchainReceipt(
        gradle_version=gradle_ver,
        distribution_sha256=dist_sha,
        java_version=java_ver,
        loader=loader,
        minecraft_version=mc_ver,
        toolchain_hash=toolchain_hash,
    )


def verify_scratch_workspace_build(
    workspace_root: str | Path,
    *,
    run_tests: bool = False,
    timeout_seconds: float = 60.0,
) -> BuildVerificationReceipt:
    """Execute two-stage (compileJava -> test) verification in the target scratch workspace."""
    ws = Path(workspace_root).resolve()
    gradlew, tool_name = _find_gradle_wrapper(ws)
    toolchain = _inspect_build_toolchain(ws)

    if gradlew:
        compile_cmd = [str(gradlew), "compileJava", "--no-daemon", "-q"]
        try:
            res_compile = subprocess.run(
                compile_cmd,
                cwd=str(ws),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            compile_stdout, compile_stderr = res_compile.stdout, res_compile.stderr
            compile_exit = res_compile.returncode
            compile_passed = compile_exit == 0

            unresolved = tuple(
                re.findall(
                    r"cannot find symbol\s+symbol:\s+class\s+([A-Za-z0-9_]+)",
                    compile_stderr + compile_stdout,
                )
            )

            tests_executed = 0
            tests_passed_count = 0
            tests_failed_count = 0
            executed_test_ids: tuple[str, ...] = ()
            individual_results: dict[str, bool] = {}
            tests_passed = False
            combined_stdout = compile_stdout
            combined_stderr = compile_stderr

            if compile_passed and run_tests:
                test_cmd = [str(gradlew), "test", "--no-daemon", "-q"]
                try:
                    res_test = subprocess.run(
                        test_cmd,
                        cwd=str(ws),
                        capture_output=True,
                        text=True,
                        timeout=timeout_seconds,
                    )
                    combined_stdout += "\n" + res_test.stdout
                    combined_stderr += "\n" + res_test.stderr
                    (
                        tests_executed,
                        tests_passed_count,
                        tests_failed_count,
                        executed_test_ids,
                        individual_results,
                    ) = _parse_test_results(ws, res_test.stdout)
                    tests_passed = (
                        tests_executed > 0
                        and tests_failed_count == 0
                        and res_test.returncode == 0
                    )
                except (OSError, subprocess.SubprocessError) as test_err:
                    combined_stderr += f"\nTest execution failed: {test_err}"
                    tests_passed = False

            return BuildVerificationReceipt(
                build_tool=tool_name,
                command=tuple(compile_cmd),
                exit_code=compile_exit,
                stdout=combined_stdout,
                stderr=combined_stderr,
                compile_passed=compile_passed,
                tests_passed=tests_passed,
                unresolved_symbols=unresolved,
                missing_resources=(),
                tests_executed=tests_executed,
                tests_passed_count=tests_passed_count,
                tests_failed_count=tests_failed_count,
                executed_test_ids=executed_test_ids,
                individual_test_results=individual_results,
                toolchain=toolchain,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return BuildVerificationReceipt(
                build_tool=tool_name,
                command=tuple(compile_cmd),
                exit_code=1,
                stdout="",
                stderr=str(exc),
                compile_passed=False,
                tests_passed=False,
                unresolved_symbols=(),
                missing_resources=(),
                tests_executed=0,
                tests_passed_count=0,
                tests_failed_count=0,
                executed_test_ids=(),
                toolchain=toolchain,
            )

    return BuildVerificationReceipt(
        build_tool="none",
        command=(),
        exit_code=1,
        stdout="",
        stderr="No Gradle build wrapper found in target workspace; compile proof cannot be attested.",
        compile_passed=False,
        tests_passed=False,
        unresolved_symbols=(),
        missing_resources=(),
        tests_executed=0,
        tests_passed_count=0,
        tests_failed_count=0,
        executed_test_ids=(),
        toolchain=toolchain,
    )
