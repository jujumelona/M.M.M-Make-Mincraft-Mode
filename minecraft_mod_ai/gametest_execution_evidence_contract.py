from __future__ import annotations

"""GameTest contract validation and execution-attestation authority."""

import hashlib
import json
import os
import re
import threading
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any

SafeRegularFile = Callable[[Path, str | Path | None], Path | None]
PassingXml = Callable[[Path, str | Path | None], bool]


def _contract_fields(contract: dict[str, Any]) -> dict[str, str]:
    return {
        key: str(contract.get(key) or "").strip()
        for key in ("task", "report", "entrypoint", "source", "metadata", "mod_id")
    }


def _metadata_matches_contract(
    metadata: dict[str, Any],
    *,
    main_mod_id: str,
    contract: dict[str, str],
) -> bool:
    entrypoints = metadata.get("entrypoints")
    values = (
        entrypoints.get("fabric-gametest")
        if isinstance(entrypoints, dict)
        else None
    )
    depends = metadata.get("depends")
    return bool(
        metadata.get("id") == contract["mod_id"]
        and isinstance(values, list)
        and values == [contract["entrypoint"]]
        and isinstance(depends, dict)
        and depends.get(main_mod_id) == "*"
    )


def _source_matches_contract(
    source_text: str,
    *,
    main_mod_id: str,
    class_name: str,
) -> bool:
    required = (
        "import net.fabricmc.loader.api.FabricLoader;",
        "import net.minecraft.gametest.framework.GameTestHelper;",
        f"public final class {class_name}",
        f'FabricLoader.getInstance().isModLoaded("{main_mod_id}")',
        "context.succeed();",
    )
    if any(fragment not in source_text for fragment in required):
        return False
    if (
        "import net.fabricmc.fabric.api.gametest.v1.GameTest;" not in source_text
        and "import net.minecraft.gametest.framework.GameTest;" not in source_text
    ):
        return False
    return re.search(
        r"@GameTest(?:\s*\([^\n]*\))?\s+public\s+void\s+"
        r"generatedRegistriesAreLive\s*"
        r"\(\s*GameTestHelper\s+context\s*\)",
        source_text,
    ) is not None


def _build_matches_contract(build_text: str, *, test_mod_id: str) -> bool:
    required = (
        "configureTests",
        "createSourceSet = true",
        f'modId = "{test_mod_id}"',
        "enableGameTests = true",
        "enableClientGameTests = false",
        "// M.M.M host-owned GameTest source-set classpath bridge",
        "compileClasspath += sourceSets.main.output + sourceSets.main.compileClasspath",
        "runtimeClasspath += sourceSets.main.output + sourceSets.main.runtimeClasspath",
        "fabric-api.gametest.report-file",
    )
    return not any(fragment not in build_text for fragment in required)


def validate_host_gametest_contract(
    root: Path,
    payload: dict[str, Any],
    main_metadata: dict[str, Any],
    contract: dict[str, Any],
    *,
    safe_regular_file: SafeRegularFile,
) -> dict[str, str] | None:
    """Verify the provider-recorded dedicated GameTest receipt against live files."""

    if str(payload.get("loader") or "").strip().casefold() != "fabric":
        return None
    main_mod_id = str(main_metadata.get("id") or "").strip()
    values = _contract_fields(contract)
    if (
        not main_mod_id
        or values["task"] != "runGameTest"
        or values["report"] != "build/gametest-report.xml"
        or values["mod_id"] != f"{main_mod_id}_gametest"
        or values["metadata"] != "src/gametest/resources/fabric.mod.json"
        or "." not in values["entrypoint"]
    ):
        return None

    package_name, class_name = values["entrypoint"].rsplit(".", 1)
    expected_source = (
        "src/gametest/java/"
        + package_name.replace(".", "/")
        + f"/{class_name}.java"
    )
    if values["source"] != expected_source:
        return None

    source = safe_regular_file(root, root / values["source"])
    metadata_file = safe_regular_file(root, root / values["metadata"])
    build = safe_regular_file(root, root / "build.gradle")
    if source is None or metadata_file is None or build is None:
        return None
    try:
        source_text = source.read_text(encoding="utf-8", errors="strict")
        build_text = build.read_text(encoding="utf-8", errors="strict")
        test_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(test_metadata, dict):
        return None
    if not _metadata_matches_contract(
        test_metadata,
        main_mod_id=main_mod_id,
        contract=values,
    ):
        return None
    if not _source_matches_contract(
        source_text,
        main_mod_id=main_mod_id,
        class_name=class_name,
    ):
        return None
    if not _build_matches_contract(build_text, test_mod_id=values["mod_id"]):
        return None

    return {
        **values,
        "testcase": f"{class_name}.generatedRegistriesAreLive",
    }


def derived_host_gametest_contract(
    root: Path,
    payload: dict[str, Any],
    main_metadata: dict[str, Any],
    *,
    safe_regular_file: SafeRegularFile,
) -> dict[str, str] | None:
    """Reconstruct the current dedicated-source-set contract only as a fallback."""

    metadata_path = safe_regular_file(
        root, root / "src/gametest/resources/fabric.mod.json"
    )
    if metadata_path is None:
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    entrypoints = metadata.get("entrypoints")
    values = (
        entrypoints.get("fabric-gametest")
        if isinstance(entrypoints, dict)
        else None
    )
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], str)
        or "." not in values[0]
    ):
        return None
    entrypoint = values[0]
    package_name, class_name = entrypoint.rsplit(".", 1)
    candidate = {
        "task": "runGameTest",
        "report": "build/gametest-report.xml",
        "entrypoint": entrypoint,
        "source": (
            "src/gametest/java/"
            + package_name.replace(".", "/")
            + f"/{class_name}.java"
        ),
        "metadata": "src/gametest/resources/fabric.mod.json",
        "mod_id": str(metadata.get("id") or "").strip(),
    }
    return validate_host_gametest_contract(
        root,
        payload,
        main_metadata,
        candidate,
        safe_regular_file=safe_regular_file,
    )


def host_gametest_contract(
    root: Path,
    *,
    safe_regular_file: SafeRegularFile,
) -> dict[str, str] | None:
    """Resolve provider receipt first; reconstruct only when old locks lack it."""

    lock = safe_regular_file(root, root / ".minecraft_ai/platform-lock.json")
    main_path = safe_regular_file(root, root / "src/main/resources/fabric.mod.json")
    if lock is None or main_path is None:
        return None
    try:
        payload = json.loads(lock.read_text(encoding="utf-8"))
        main_metadata = json.loads(main_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(main_metadata, dict):
        return None
    bootstrap = payload.get("bootstrap")
    recorded = (
        bootstrap.get("gametest_contract")
        if isinstance(bootstrap, dict)
        else None
    )
    if isinstance(recorded, dict):
        return validate_host_gametest_contract(
            root,
            payload,
            main_metadata,
            recorded,
            safe_regular_file=safe_regular_file,
        )
    return derived_host_gametest_contract(
        root,
        payload,
        main_metadata,
        safe_regular_file=safe_regular_file,
    )


def gametest_pass_summary(log_path: str | Path) -> int | None:
    """Read Fabric's terminal required-test summary conservatively."""

    path = Path(log_path)
    if not path.is_file() or path.is_symlink():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    completed = re.findall(r"(?mi)=+\s*(\d+)\s+GAME TESTS COMPLETE\b", text)
    passed = re.findall(r"(?mi)\bAll\s+(\d+)\s+required tests passed\s*:\)", text)
    if not completed or not passed:
        return None
    try:
        completed_count = int(completed[-1])
        passed_count = int(passed[-1])
    except ValueError:
        return None
    return (
        completed_count
        if completed_count > 0 and completed_count == passed_count
        else None
    )


def report_contains_contract_testcase(
    root: Path,
    path_value: str | Path | None,
    contract: dict[str, str],
    *,
    safe_regular_file: SafeRegularFile,
) -> bool:
    """Require native XML to identify the exact mandatory host testcase."""

    path = safe_regular_file(root, path_value)
    expected = str(contract.get("testcase") or "").strip().casefold()
    if path is None or not expected:
        return False
    try:
        for _event, element in ET.iterparse(path, events=("end",)):
            tag = element.tag.rsplit("}", 1)[-1]
            if (
                tag == "testcase"
                and str(element.attrib.get("name") or "").strip().casefold()
                == expected
            ):
                return True
            element.clear()
    except (ET.ParseError, OSError):
        return False
    return False


def structured_gametest_report(
    root: Path,
    log_path: str | Path,
    native_report: str | Path,
    *,
    safe_regular_file: SafeRegularFile,
    passing_gametest_xml: PassingXml,
) -> Path | None:
    """Return exact native XML or build a host-bound attestation from Fabric output."""

    contract = host_gametest_contract(root, safe_regular_file=safe_regular_file)
    safe_native = safe_regular_file(root, native_report)
    if (
        contract is not None
        and safe_native is not None
        and passing_gametest_xml(root, safe_native)
        and report_contains_contract_testcase(
            root,
            safe_native,
            contract,
            safe_regular_file=safe_regular_file,
        )
    ):
        return safe_native

    passed_count = gametest_pass_summary(log_path)
    if contract is None or passed_count is None:
        return None
    log_file = safe_regular_file(root, log_path)
    if log_file is None:
        return None

    target = root / "build" / "mmm-gametest-attestation.xml"
    if target.is_symlink():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    suite = ET.Element(
        "testsuite",
        {
            "name": "mmm-host-required-gametest",
            "tests": "1",
            "failures": "0",
            "errors": "0",
            "skipped": "0",
        },
    )
    properties = ET.SubElement(suite, "properties")
    ET.SubElement(
        properties,
        "property",
        {
            "name": "mmm.runtime.required_tests_passed",
            "value": str(passed_count),
        },
    )
    ET.SubElement(
        properties,
        "property",
        {
            "name": "mmm.gradle_log_sha256",
            "value": "sha256:" + hashlib.sha256(log_file.read_bytes()).hexdigest(),
        },
    )
    ET.SubElement(
        suite,
        "testcase",
        {"name": contract["testcase"], "classname": contract["entrypoint"]},
    )
    try:
        ET.ElementTree(suite).write(
            temporary,
            encoding="utf-8",
            xml_declaration=True,
        )
        os.replace(temporary, target)
    except OSError:
        temporary.unlink(missing_ok=True)
        return None
    safe_target = safe_regular_file(root, target)
    if safe_target is None or not passing_gametest_xml(root, safe_target):
        return None
    return safe_target


__all__ = [
    "derived_host_gametest_contract",
    "gametest_pass_summary",
    "host_gametest_contract",
    "report_contains_contract_testcase",
    "structured_gametest_report",
    "validate_host_gametest_contract",
]
