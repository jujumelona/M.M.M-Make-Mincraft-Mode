from __future__ import annotations

"""Real Gradle build verifier for isolated reuse-proof workspaces.

The verifier executes only a checksum-attested Gradle wrapper, records the observed
Java/Gradle target identity, and treats JUnit XML as the only authority for individual
test identities. Console summaries may establish aggregate counts but can never
fabricate a per-test PASS.

Exact, successful scratch builds are content-addressed and single-flight within one
process. The cache key binds every proof input byte plus the attested toolchain. Failed
or timed-out executions are never persisted, so transient failures remain retryable.
"""

import hashlib
import os
import re
import subprocess
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from typing import Any


_BUILD_CACHE_LOCK = Lock()
_BUILD_CACHE: dict[tuple[str, str], "BuildVerificationReceipt"] = {}
_BUILD_INFLIGHT: dict[tuple[str, str], Future["BuildVerificationReceipt"]] = {}
_BUILD_CACHE_MAX_ENTRIES = 64
_BUILD_CACHE_MAX_INPUT_BYTES = 32 * 1024 * 1024
_BUILD_FINGERPRINT_EXCLUDED_DIRS = frozenset(
    {".git", ".gradle", "build", "__pycache__", ".idea", ".vscode", ".gemini"}
)


@dataclass(frozen=True)
class BuildToolchainReceipt:
    gradle_version: str = ""
    distribution_sha256: str = ""
    wrapper_sha256: str = ""
    java_version: str = ""
    loader: str = ""
    minecraft_version: str = ""
    wrapper_verified: bool = False
    distribution_verified: bool = False
    target_matrix_verified: bool = False
    toolchain_hash: str = ""

    @property
    def is_attested(self) -> bool:
        return bool(
            self.gradle_version
            and self.java_version
            and self.loader
            and self.minecraft_version
            and self.wrapper_verified
            and self.distribution_verified
            and self.target_matrix_verified
            and self.toolchain_hash
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gradle_version": self.gradle_version,
            "distribution_sha256": self.distribution_sha256,
            "wrapper_sha256": self.wrapper_sha256,
            "java_version": self.java_version,
            "loader": self.loader,
            "minecraft_version": self.minecraft_version,
            "wrapper_verified": self.wrapper_verified,
            "distribution_verified": self.distribution_verified,
            "target_matrix_verified": self.target_matrix_verified,
            "is_attested": self.is_attested,
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_gradle_wrapper(workspace_root: Path) -> tuple[Path | None, str]:
    gradlew = workspace_root / ("gradlew.bat" if os.name == "nt" else "gradlew")
    wrapper_jar = workspace_root / "gradle" / "wrapper" / "gradle-wrapper.jar"
    props = workspace_root / "gradle" / "wrapper" / "gradle-wrapper.properties"

    if (
        gradlew.is_file()
        and not gradlew.is_symlink()
        and wrapper_jar.is_file()
        and not wrapper_jar.is_symlink()
        and props.is_file()
        and not props.is_symlink()
    ):
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
                        individual_results[canonical_id] = bool(
                            existing and passed_individually
                        )
                    test_ids.append(canonical_id)
            except (ET.ParseError, OSError, ValueError):
                continue

    if executed == 0:
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


def _detect_loader_and_minecraft(build_text: str) -> tuple[str, str]:
    loader = ""
    if "net.neoforged" in build_text:
        loader = "neoforge"
    elif "net.minecraftforge" in build_text:
        loader = "forge"
    elif "fabric-loom" in build_text or "net.fabricmc" in build_text:
        loader = "fabric"

    patterns = (
        r"['\"]com\.mojang:minecraft:([^'\"]+)['\"]",
        r"minecraft_version\s*=\s*['\"]?([^'\"\s]+)",
        r"mappings\s+channel\s*:\s*['\"]official['\"]\s*,\s*version\s*:\s*['\"]([^'\"]+)['\"]",
        r"neoForge\s*\{[^}]*version\s*=\s*['\"]([0-9]+\.[0-9]+(?:\.[0-9]+)?)",
    )
    minecraft = ""
    for pattern in patterns:
        match = re.search(pattern, build_text, re.DOTALL)
        if match:
            minecraft = match.group(1).strip()
            break
    return loader, minecraft


def _java_major_version() -> str:
    java_home = os.environ.get("JAVA_HOME", "").strip()
    java_cmd = Path(java_home) / "bin" / ("java.exe" if os.name == "nt" else "java")
    command = str(java_cmd) if java_home and java_cmd.is_file() else "java"
    try:
        result = subprocess.run(
            [command, "-version"],
            capture_output=True,
            text=True,
            timeout=4.0,
        )
        text = result.stderr + result.stdout
        match = re.search(r'version\s+"?([0-9._]+)"?', text)
        if not match:
            return ""
        parts = match.group(1).split(".")
        return parts[1] if parts[0] == "1" and len(parts) > 1 else parts[0]
    except (OSError, subprocess.SubprocessError):
        return ""


def _inspect_build_toolchain(workspace_root: Path) -> BuildToolchainReceipt:
    from .platform_catalog import adapter_for_target
    from .verified_scaffold_registry import (
        GRADLE_DISTRIBUTION_SHA256S,
        GRADLE_WRAPPER_SHA256S,
        validate_scaffold_buildability,
    )

    props = workspace_root / "gradle" / "wrapper" / "gradle-wrapper.properties"
    wrapper_jar = workspace_root / "gradle" / "wrapper" / "gradle-wrapper.jar"

    gradle_version = ""
    distribution_sha256 = ""
    if props.is_file() and not props.is_symlink():
        try:
            text = props.read_text(encoding="utf-8")
        except OSError:
            text = ""
        url_match = re.search(r"distributionUrl=([^\r\n]+)", text)
        if url_match:
            distribution_url = url_match.group(1).replace(r"\:", ":")
            version_match = re.search(r"gradle-([0-9.]+)-", distribution_url)
            if version_match:
                gradle_version = version_match.group(1)
        sha_match = re.search(r"distributionSha256Sum=([^\r\n]+)", text)
        if sha_match:
            distribution_sha256 = sha_match.group(1).strip().casefold()

    wrapper_sha256 = ""
    if wrapper_jar.is_file() and not wrapper_jar.is_symlink():
        try:
            wrapper_sha256 = _sha256_file(wrapper_jar)
        except OSError:
            wrapper_sha256 = ""

    build_gradle = workspace_root / "build.gradle"
    build_gradle_kts = workspace_root / "build.gradle.kts"
    build_text = ""
    try:
        if build_gradle.is_file() and not build_gradle.is_symlink():
            build_text = build_gradle.read_text(encoding="utf-8")
        elif build_gradle_kts.is_file() and not build_gradle_kts.is_symlink():
            build_text = build_gradle_kts.read_text(encoding="utf-8")
    except OSError:
        build_text = ""

    loader, minecraft_version = _detect_loader_and_minecraft(build_text)
    java_version = _java_major_version()

    expected_wrapper = GRADLE_WRAPPER_SHA256S.get(gradle_version, "")
    expected_distribution = GRADLE_DISTRIBUTION_SHA256S.get(gradle_version, "")
    wrapper_verified = bool(
        expected_wrapper
        and wrapper_sha256
        and wrapper_sha256.casefold() == expected_wrapper.casefold()
    )
    distribution_verified = bool(
        expected_distribution
        and distribution_sha256
        and distribution_sha256.casefold() == expected_distribution.casefold()
    )

    provider_adapter = None
    if loader and minecraft_version:
        try:
            provider_adapter = adapter_for_target(minecraft_version, loader)
            validate_scaffold_buildability(provider_adapter)
        except (ValueError, RuntimeError):
            provider_adapter = None
    target_matrix_verified = bool(
        provider_adapter
        and str(provider_adapter.gradle) == gradle_version
        and str(provider_adapter.java_version) == java_version
        and str(provider_adapter.gradle_sha256).casefold() == distribution_sha256
    )

    identity = "\n".join(
        (
            loader,
            minecraft_version,
            gradle_version,
            java_version,
            distribution_sha256,
            wrapper_sha256,
            "1" if wrapper_verified else "0",
            "1" if distribution_verified else "0",
            "1" if target_matrix_verified else "0",
        )
    )
    toolchain_hash = (
        "sha256:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
        if all((loader, minecraft_version, gradle_version, java_version))
        else ""
    )

    return BuildToolchainReceipt(
        gradle_version=gradle_version,
        distribution_sha256=distribution_sha256,
        wrapper_sha256=wrapper_sha256,
        java_version=java_version,
        loader=loader,
        minecraft_version=minecraft_version,
        wrapper_verified=wrapper_verified,
        distribution_verified=distribution_verified,
        target_matrix_verified=target_matrix_verified,
        toolchain_hash=toolchain_hash,
    )


def _toolchain_failure_receipt(
    tool_name: str,
    toolchain: BuildToolchainReceipt,
) -> BuildVerificationReceipt:
    return BuildVerificationReceipt(
        build_tool=tool_name,
        command=(),
        exit_code=1,
        stdout="",
        stderr=(
            "Gradle toolchain attestation failed: wrapper/distribution/target matrix "
            "identity did not match the reviewed registry."
        ),
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


def _workspace_input_fingerprint(
    workspace_root: Path,
    toolchain: BuildToolchainReceipt,
) -> str:
    """Hash exact proof inputs while pruning build/cache outputs from traversal."""

    digest = hashlib.sha256()
    total_bytes = 0
    try:
        for root_text, dir_names, file_names in os.walk(workspace_root, followlinks=False):
            root = Path(root_text)
            kept_dirs: list[str] = []
            for name in sorted(dir_names):
                directory = root / name
                if name in _BUILD_FINGERPRINT_EXCLUDED_DIRS:
                    continue
                if directory.is_symlink():
                    return ""
                kept_dirs.append(name)
            dir_names[:] = kept_dirs

            for name in sorted(file_names):
                path = root / name
                if path.is_symlink() or not path.is_file():
                    return ""
                relative = path.relative_to(workspace_root).as_posix()
                stat = path.stat()
                total_bytes += int(stat.st_size)
                if total_bytes > _BUILD_CACHE_MAX_INPUT_BYTES:
                    return ""
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(stat.st_mode & 0o777).encode("ascii"))
                digest.update(b"\0")
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                digest.update(b"\0")
    except (OSError, RuntimeError, ValueError):
        return ""

    digest.update(toolchain.toolchain_hash.encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _cache_get(key: tuple[str, str], command: tuple[str, ...]) -> BuildVerificationReceipt | None:
    with _BUILD_CACHE_LOCK:
        receipt = _BUILD_CACHE.get(key)
        if receipt is None:
            return None
        _BUILD_CACHE.pop(key, None)
        _BUILD_CACHE[key] = receipt
    return replace(receipt, command=command)


def _cache_store(key: tuple[str, str], receipt: BuildVerificationReceipt) -> None:
    with _BUILD_CACHE_LOCK:
        _BUILD_CACHE.pop(key, None)
        _BUILD_CACHE[key] = receipt
        while len(_BUILD_CACHE) > _BUILD_CACHE_MAX_ENTRIES:
            _BUILD_CACHE.pop(next(iter(_BUILD_CACHE)))


def _claim_inflight(
    key: tuple[str, str],
) -> tuple[Future[BuildVerificationReceipt], bool]:
    with _BUILD_CACHE_LOCK:
        existing = _BUILD_INFLIGHT.get(key)
        if existing is not None:
            return existing, False
        future: Future[BuildVerificationReceipt] = Future()
        _BUILD_INFLIGHT[key] = future
        return future, True


def _finish_inflight(
    key: tuple[str, str],
    future: Future[BuildVerificationReceipt],
    *,
    receipt: BuildVerificationReceipt | None = None,
    error: BaseException | None = None,
) -> None:
    with _BUILD_CACHE_LOCK:
        if _BUILD_INFLIGHT.get(key) is future:
            _BUILD_INFLIGHT.pop(key, None)
    if error is not None:
        future.set_exception(error)
    elif receipt is not None:
        future.set_result(receipt)


def _run_compile_stage(
    ws: Path,
    gradlew: Path,
    tool_name: str,
    toolchain: BuildToolchainReceipt,
    fingerprint: str,
    timeout_seconds: float,
) -> BuildVerificationReceipt:
    compile_cmd = (str(gradlew), "compileJava", "--no-daemon", "-q")
    cache_key = (fingerprint, "compile") if fingerprint else None
    if cache_key is not None:
        cached = _cache_get(cache_key, compile_cmd)
        if cached is not None:
            return cached
        future, owner = _claim_inflight(cache_key)
        if not owner:
            return replace(future.result(), command=compile_cmd)
    else:
        future = None

    try:
        res_compile = subprocess.run(
            list(compile_cmd),
            cwd=str(ws),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        compile_stdout = res_compile.stdout
        compile_stderr = res_compile.stderr
        compile_exit = res_compile.returncode
        compile_passed = compile_exit == 0
        unresolved = tuple(
            re.findall(
                r"cannot find symbol\s+symbol:\s+class\s+([A-Za-z0-9_]+)",
                compile_stderr + compile_stdout,
            )
        )
        receipt = BuildVerificationReceipt(
            build_tool=tool_name,
            command=compile_cmd,
            exit_code=compile_exit,
            stdout=compile_stdout,
            stderr=compile_stderr,
            compile_passed=compile_passed,
            tests_passed=False,
            unresolved_symbols=unresolved,
            missing_resources=(),
            tests_executed=0,
            tests_passed_count=0,
            tests_failed_count=0,
            executed_test_ids=(),
            individual_test_results={},
            toolchain=toolchain,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        receipt = BuildVerificationReceipt(
            build_tool=tool_name,
            command=compile_cmd,
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
            individual_test_results={},
            toolchain=toolchain,
        )
    except BaseException as exc:
        if cache_key is not None and future is not None:
            _finish_inflight(cache_key, future, error=exc)
        raise

    if cache_key is not None and future is not None:
        if receipt.compile_passed:
            _cache_store(cache_key, receipt)
        _finish_inflight(cache_key, future, receipt=receipt)
    return receipt


def _run_test_stage(
    ws: Path,
    gradlew: Path,
    compile_receipt: BuildVerificationReceipt,
    fingerprint: str,
    timeout_seconds: float,
) -> BuildVerificationReceipt:
    compile_cmd = tuple(compile_receipt.command)
    cache_key = (fingerprint, "test") if fingerprint else None
    if cache_key is not None:
        cached = _cache_get(cache_key, compile_cmd)
        if cached is not None:
            return cached
        future, owner = _claim_inflight(cache_key)
        if not owner:
            return replace(future.result(), command=compile_cmd)
    else:
        future = None

    combined_stdout = compile_receipt.stdout
    combined_stderr = compile_receipt.stderr
    tests_executed = 0
    tests_passed_count = 0
    tests_failed_count = 0
    executed_test_ids: tuple[str, ...] = ()
    individual_results: dict[str, bool] = {}
    tests_passed = False

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
        receipt = replace(
            compile_receipt,
            stdout=combined_stdout,
            stderr=combined_stderr,
            tests_passed=tests_passed,
            tests_executed=tests_executed,
            tests_passed_count=tests_passed_count,
            tests_failed_count=tests_failed_count,
            executed_test_ids=executed_test_ids,
            individual_test_results=individual_results,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        receipt = replace(
            compile_receipt,
            stdout=combined_stdout,
            stderr=combined_stderr + f"\nTest execution failed: {exc}",
            tests_passed=False,
        )
    except BaseException as exc:
        if cache_key is not None and future is not None:
            _finish_inflight(cache_key, future, error=exc)
        raise

    if cache_key is not None and future is not None:
        if receipt.compile_passed and receipt.tests_passed:
            _cache_store(cache_key, receipt)
        _finish_inflight(cache_key, future, receipt=receipt)
    return receipt


def verify_scratch_workspace_build(
    workspace_root: str | Path,
    *,
    run_tests: bool = False,
    timeout_seconds: float = 60.0,
) -> BuildVerificationReceipt:
    """Execute compile/test proof once per exact input+toolchain identity."""

    ws = Path(workspace_root).resolve()
    gradlew, tool_name = _find_gradle_wrapper(ws)
    toolchain = _inspect_build_toolchain(ws)

    if not gradlew:
        return BuildVerificationReceipt(
            build_tool="none",
            command=(),
            exit_code=1,
            stdout="",
            stderr=(
                "No Gradle build wrapper found in target workspace; compile proof "
                "cannot be attested."
            ),
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

    if not toolchain.is_attested:
        return _toolchain_failure_receipt(tool_name, toolchain)

    fingerprint = _workspace_input_fingerprint(ws, toolchain)
    compile_receipt = _run_compile_stage(
        ws,
        gradlew,
        tool_name,
        toolchain,
        fingerprint,
        timeout_seconds,
    )
    if not compile_receipt.compile_passed or not run_tests:
        return compile_receipt

    return _run_test_stage(
        ws,
        gradlew,
        compile_receipt,
        fingerprint,
        timeout_seconds,
    )
