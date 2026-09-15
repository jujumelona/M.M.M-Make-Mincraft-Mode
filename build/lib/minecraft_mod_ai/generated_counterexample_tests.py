from __future__ import annotations

"""Generate and execute candidate-neutral counterexample oracles from an A/B diff.

The same test specification is applied to both isolated candidate snapshots. Tests
are derived only from unchanged project consumers, focused resource contracts and
original verifier failures; a candidate never supplies its own expected answer.
"""

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_SCHEMA = "mmm/generated-counterexample-test-v1"
_ID = re.compile(r"\b([a-z0-9_.-]+):([a-z0-9_./-]+)\b")
_PACKAGE = re.compile(r"(?m)^\s*package\s+([A-Za-z_$][\w.$]*)\s*;")
# Java allows `package x; public final class Y {}` on one physical line. A line-start
# anchor therefore creates false missing-entrypoint failures. Match the declaration
# token boundary instead; package qualification remains independently parsed above.
_CLASS = re.compile(
    r"\b(?:public\s+|protected\s+|private\s+)?"
    r"(?:(?:final|abstract|sealed|non-sealed|static)\s+)*"
    r"(?:class|record|enum|interface)\s+([A-Za-z_$][\w$]*)\b"
)
_TEXT_SUFFIXES = {
    ".java",
    ".json",
    ".mcfunction",
    ".gradle",
    ".kts",
    ".properties",
    ".toml",
    ".yml",
    ".yaml",
}


def _paths(operations: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(item.get("path", "")).replace("\\", "/")
        for item in operations
        if str(item.get("path", "")).strip()
    }


def _fragments(operations: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for item in operations:
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
        replacements = item.get("replacements")
        if isinstance(replacements, Sequence) and not isinstance(
            replacements, (str, bytes, bytearray)
        ):
            for replacement in replacements:
                if not isinstance(replacement, Mapping):
                    continue
                for key in ("old", "new"):
                    value = replacement.get(key)
                    if isinstance(value, str):
                        parts.append(value)
    return "\n".join(parts)


def _project_namespaces(root: Path) -> set[str]:
    values: set[str] = set()
    fabric = root / "src/main/resources/fabric.mod.json"
    if fabric.is_file() and not fabric.is_symlink():
        try:
            payload = json.loads(fabric.read_text(encoding="utf-8"))
            mod_id = str(payload.get("id", "")).strip()
            if mod_id:
                values.add(mod_id)
        except Exception:
            pass
    for base in (
        root / "src/main/resources/assets",
        root / "src/main/resources/data",
    ):
        if base.is_dir():
            values.update(child.name for child in base.iterdir() if child.is_dir())
    return values


def _identifiers(text: str, namespaces: set[str]) -> set[str]:
    values = {
        f"{match.group(1)}:{match.group(2)}"
        for match in _ID.finditer(text)
        if not namespaces or match.group(1) in namespaces
    }
    constructor = re.compile(
        r"(?:new\s+Identifier|Identifier\.(?:of|tryParse))\s*\(\s*"
        r"\"([^\"]+)\"\s*,\s*\"([^\"]+)\"\s*\)"
    )
    for match in constructor.finditer(text):
        if not namespaces or match.group(1) in namespaces:
            values.add(f"{match.group(1)}:{match.group(2)}")
    return values


def _iter_project_text(root: Path):
    for base in (
        root / "src/main/java",
        root / "src/client/java",
        root / "src/main/resources",
    ):
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if (
                not path.is_file()
                or path.is_symlink()
                or path.suffix.casefold() not in _TEXT_SUFFIXES
            ):
                continue
            try:
                if path.stat().st_size > 1_000_000:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            yield path.relative_to(root).as_posix(), text


def _external_identifier_contracts(
    root: Path,
    focus: set[str],
    candidate_ids: set[str],
    namespaces: set[str],
) -> list[dict[str, Any]]:
    if not candidate_ids:
        return []
    refs: dict[str, list[str]] = {identifier: [] for identifier in candidate_ids}
    for relative, text in _iter_project_text(root):
        if relative in focus:
            continue
        found = _identifiers(text, namespaces) & candidate_ids
        for identifier in found:
            if len(refs[identifier]) < 8:
                refs[identifier].append(relative)
    return [
        {"identifier": identifier, "oracle_paths": sorted(paths)}
        for identifier, paths in sorted(refs.items())
        if paths
    ][:24]


def _resource_target(identifier: str, *, kind: str) -> str | None:
    match = _ID.fullmatch(identifier)
    if match is None:
        return None
    namespace, value = match.groups()
    if kind == "model":
        return f"src/main/resources/assets/{namespace}/models/{value}.json"
    if kind == "texture":
        return f"src/main/resources/assets/{namespace}/textures/{value}.png"
    return None


def _unchanged_resource_targets(
    root: Path,
    focus: set[str],
    touched: set[str],
    namespaces: set[str],
) -> list[dict[str, str]]:
    targets: dict[str, dict[str, str]] = {}
    for relative, text in _iter_project_text(root):
        if relative in focus or not relative.endswith(".json"):
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue

        def walk(value: Any, key: str = "") -> None:
            if isinstance(value, Mapping):
                for raw_key, child in value.items():
                    walk(child, str(raw_key))
                return
            if isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                for child in value:
                    walk(child, key)
                return
            if not isinstance(value, str) or ":" not in value:
                return
            match = _ID.fullmatch(value)
            if match is None or match.group(1) not in namespaces:
                return
            kind = (
                "model"
                if key in {"model", "parent"}
                else ("texture" if key in {"texture", "textures"} else "")
            )
            target = _resource_target(value, kind=kind) if kind else None
            if target and target in touched:
                targets[target] = {
                    "target": target,
                    "oracle_path": relative,
                    "reference": value,
                }

        walk(payload)
    return [targets[key] for key in sorted(targets)][:24]


def _diagnostic_seed_assertion(
    evidence_seed: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(evidence_seed, Mapping):
        return None
    diagnostics = evidence_seed.get("diagnostics")
    if not isinstance(diagnostics, Sequence) or isinstance(
        diagnostics, (str, bytes, bytearray)
    ):
        return None
    clean: list[dict[str, str]] = []
    for item in diagnostics[:16]:
        if not isinstance(item, Mapping):
            continue
        code = str(item.get("code", "")).strip()[:120]
        message = " ".join(str(item.get("message", "")).casefold().split())[:240]
        if code or message:
            clean.append({"code": code, "message": message})
    return (
        {"kind": "original_diagnostic_absence", "diagnostics": clean}
        if clean
        else None
    )


def build_generated_test_spec(
    root: Path,
    *,
    focus_paths: Sequence[str],
    left_operations: Sequence[Mapping[str, Any]],
    right_operations: Sequence[Mapping[str, Any]],
    evidence_seed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    focus = {
        str(path).replace("\\", "/")
        for path in focus_paths
        if str(path).strip()
    }
    touched = _paths(left_operations) | _paths(right_operations)
    namespaces = _project_namespaces(root)
    left_ids = _identifiers(_fragments(left_operations), namespaces)
    right_ids = _identifiers(_fragments(right_operations), namespaces)
    candidate_ids = left_ids ^ right_ids

    assertions: list[dict[str, Any]] = []
    if any(path.endswith(".json") for path in focus):
        assertions.append(
            {
                "kind": "json_parse_focus",
                "paths": sorted(
                    path for path in focus if path.endswith(".json")
                )[:24],
            }
        )
    assertions.append({"kind": "fabric_entrypoint_resolution"})
    assertions.append({"kind": "mixin_class_resolution"})
    if any("/assets/" in path and path.endswith(".json") for path in focus):
        assertions.append(
            {"kind": "model_resource_closure", "paths": sorted(focus)[:24]}
        )

    external_ids = _external_identifier_contracts(
        root, focus, candidate_ids, namespaces
    )
    if external_ids:
        assertions.append(
            {"kind": "external_identifier_contract", "contracts": external_ids}
        )
    targets = _unchanged_resource_targets(root, focus, touched, namespaces)
    if targets:
        assertions.append(
            {"kind": "unchanged_resource_reference", "contracts": targets}
        )
    diagnostic = _diagnostic_seed_assertion(evidence_seed)
    if diagnostic:
        assertions.append(diagnostic)

    return {
        "schema_version": _SCHEMA,
        "generator": "host-ab-diff-oracle-v1",
        "same_test_for_both_candidates": True,
        "focus_paths": sorted(focus)[:24],
        "project_namespaces": sorted(namespaces)[:16],
        "assertions": assertions[:16],
    }


def _class_sources(root: Path) -> set[str]:
    classes: set[str] = set()
    for base in (root / "src/main/java", root / "src/client/java"):
        if not base.is_dir():
            continue
        for path in base.rglob("*.java"):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            package = _PACKAGE.search(text)
            clazz = _CLASS.search(text)
            if clazz:
                classes.add(
                    f"{package.group(1)}.{clazz.group(1)}"
                    if package
                    else clazz.group(1)
                )
    return classes


def _entrypoint_failures(root: Path) -> list[str]:
    fabric = root / "src/main/resources/fabric.mod.json"
    if not fabric.is_file():
        return ["fabric.mod.json:missing"]
    try:
        payload = json.loads(fabric.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"fabric.mod.json:{type(exc).__name__}"]
    entrypoints = payload.get("entrypoints", {})
    if not isinstance(entrypoints, Mapping):
        return ["fabric.mod.json:entrypoints-not-object"]
    classes = _class_sources(root)
    failures: list[str] = []

    def values(raw: Any):
        if isinstance(raw, str):
            yield raw
        elif isinstance(raw, Mapping):
            value = raw.get("value")
            if isinstance(value, str):
                yield value
        elif isinstance(raw, Sequence) and not isinstance(
            raw, (str, bytes, bytearray)
        ):
            for item in raw:
                yield from values(item)

    for raw in entrypoints.values():
        for value in values(raw):
            class_name = value.split("::", 1)[0].strip()
            if class_name and class_name not in classes:
                failures.append(f"entrypoint:{class_name}")
    return failures[:24]


def _mixin_failures(root: Path) -> list[str]:
    fabric = root / "src/main/resources/fabric.mod.json"
    if not fabric.is_file():
        return []
    try:
        payload = json.loads(fabric.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw_mixins = payload.get("mixins", ())
    if isinstance(raw_mixins, (str, Mapping)):
        raw_mixins = [raw_mixins]
    if not isinstance(raw_mixins, Sequence):
        return []
    classes = _class_sources(root)
    failures: list[str] = []
    for item in raw_mixins[:24]:
        config_name = (
            item
            if isinstance(item, str)
            else (item.get("config") if isinstance(item, Mapping) else None)
        )
        if not isinstance(config_name, str) or not config_name.strip():
            continue
        config_path = root / "src/main/resources" / config_name
        if not config_path.is_file():
            failures.append(f"mixin-config:{config_name}")
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"mixin-config:{config_name}:{type(exc).__name__}")
            continue
        package = str(config.get("package", "")).strip()
        for key in ("mixins", "client", "server"):
            values = config.get(key, ())
            if not isinstance(values, Sequence) or isinstance(
                values, (str, bytes, bytearray)
            ):
                continue
            for value in values[:64]:
                name = str(value).strip()
                fqcn = f"{package}.{name}" if package and name else name
                if fqcn and fqcn not in classes:
                    failures.append(f"mixin-class:{fqcn}")
    return failures[:32]


def _model_failures(
    root: Path, focus_paths: Sequence[str], namespaces: set[str]
) -> list[str]:
    failures: list[str] = []

    def resolve(identifier: str, kind: str) -> Path | None:
        target = _resource_target(identifier, kind=kind)
        return root / target if target else None

    for relative in focus_paths[:24]:
        if not relative.endswith(".json") or "/assets/" not in relative:
            continue
        path = root / relative
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        def walk(value: Any, key: str = "") -> None:
            if isinstance(value, Mapping):
                for raw_key, child in value.items():
                    walk(child, str(raw_key))
                return
            if isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                for child in value:
                    walk(child, key)
                return
            if not isinstance(value, str) or ":" not in value:
                return
            match = _ID.fullmatch(value)
            if match is None or match.group(1) not in namespaces:
                return
            kind = (
                "model"
                if key in {"model", "parent"}
                else ("texture" if key in {"texture", "textures"} else "")
            )
            target = resolve(value, kind) if kind else None
            if target is not None and not target.is_file():
                failures.append(f"{kind}:{value}")

        walk(payload)
    return failures[:32]


def _focus_identifiers(
    root: Path, paths: Sequence[str], namespaces: set[str]
) -> set[str]:
    result: set[str] = set()
    for relative in paths[:24]:
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.suffix.casefold() not in _TEXT_SUFFIXES
        ):
            continue
        try:
            result.update(
                _identifiers(
                    path.read_text(encoding="utf-8", errors="replace"), namespaces
                )
            )
        except OSError:
            continue
    return result


def _diagnostic_failures(
    assertion: Mapping[str, Any],
    candidate_verifier: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(candidate_verifier, Mapping):
        return []
    observed = candidate_verifier.get("jdt_diagnostics")
    if not isinstance(observed, Sequence) or isinstance(
        observed, (str, bytes, bytearray)
    ):
        return []
    expected = assertion.get("diagnostics")
    if not isinstance(expected, Sequence) or isinstance(
        expected, (str, bytes, bytearray)
    ):
        return []
    failures: list[str] = []
    for original in expected:
        if not isinstance(original, Mapping):
            continue
        code = str(original.get("code", "")).strip()
        message = " ".join(
            str(original.get("message", "")).casefold().split()
        )
        for item in observed:
            if not isinstance(item, Mapping):
                continue
            current_code = str(item.get("code", "")).strip()
            current_message = " ".join(
                str(item.get("message", "")).casefold().split()
            )
            if (code and code == current_code) or (
                message and message == current_message
            ):
                failures.append(f"original-diagnostic:{code or message[:80]}")
                break
    return failures[:16]


def run_generated_test_spec(
    root: Path,
    spec: Mapping[str, Any],
    *,
    candidate_verifier: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if spec.get("schema_version") != _SCHEMA:
        return {
            "schema_version": "mmm/generated-counterexample-result-v1",
            "status": "ERROR",
            "assertions": [],
        }
    namespaces = {
        str(value) for value in spec.get("project_namespaces", ()) if str(value)
    }
    focus_paths = [
        str(value) for value in spec.get("focus_paths", ()) if str(value)
    ]
    results: list[dict[str, Any]] = []
    for assertion in spec.get("assertions", ()):
        if not isinstance(assertion, Mapping):
            continue
        kind = str(assertion.get("kind", ""))
        failures: list[str] = []
        if kind == "json_parse_focus":
            for relative in assertion.get("paths", ()):
                path = root / str(relative)
                if path.is_file():
                    try:
                        json.loads(path.read_text(encoding="utf-8"))
                    except Exception as exc:
                        failures.append(f"{relative}:{type(exc).__name__}")
        elif kind == "fabric_entrypoint_resolution":
            failures = _entrypoint_failures(root)
        elif kind == "mixin_class_resolution":
            failures = _mixin_failures(root)
        elif kind == "model_resource_closure":
            failures = _model_failures(
                root, assertion.get("paths", focus_paths), namespaces
            )
        elif kind == "external_identifier_contract":
            available = _focus_identifiers(root, focus_paths, namespaces)
            for contract in assertion.get("contracts", ()):
                if isinstance(contract, Mapping):
                    identifier = str(contract.get("identifier", ""))
                    if identifier and identifier not in available:
                        failures.append(f"identifier:{identifier}")
        elif kind == "unchanged_resource_reference":
            for contract in assertion.get("contracts", ()):
                if isinstance(contract, Mapping):
                    target = str(contract.get("target", ""))
                    if target and not (root / target).is_file():
                        failures.append(f"resource:{target}")
        elif kind == "original_diagnostic_absence":
            failures = _diagnostic_failures(assertion, candidate_verifier)
        results.append(
            {
                "kind": kind,
                "status": "PASS" if not failures else "FAIL",
                "failure_count": len(failures),
                "failures": failures[:16],
            }
        )
    overall = (
        "FAIL"
        if any(item["status"] == "FAIL" for item in results)
        else ("PASS" if results else "INCOMPLETE")
    )
    return {
        "schema_version": "mmm/generated-counterexample-result-v1",
        "status": overall,
        "same_test_for_both_candidates": True,
        "assertions": results,
    }


def _junit_mode(root: Path) -> str:
    text = ""
    for name in ("build.gradle", "gradle/libs.versions.toml"):
        path = root / name
        if path.is_file():
            text += "\n" + path.read_text(encoding="utf-8", errors="replace")
    lowered = text.casefold()
    if (
        "junit-jupiter" in lowered or "org.junit.jupiter" in lowered
    ) and "usejunitplatform" in lowered:
        return "junit5"
    if re.search(r"junit\s*[:'\"]", lowered) or "junit:junit" in lowered:
        return "junit4"
    return ""


def _junit_required_paths(spec: Mapping[str, Any]) -> list[str]:
    required: set[str] = set()
    for assertion in spec.get("assertions", ()):
        if not isinstance(assertion, Mapping):
            continue
        if assertion.get("kind") == "unchanged_resource_reference":
            for contract in assertion.get("contracts", ()):
                if isinstance(contract, Mapping) and str(contract.get("target", "")):
                    required.add(str(contract["target"]))
    return sorted(required)[:24]


def install_generated_junit(
    root: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Install the same generated JUnit oracle when the project already has JUnit."""

    mode = _junit_mode(root)
    required = _junit_required_paths(spec)
    if not mode or not required:
        return {
            "status": "NOT_APPLICABLE",
            "harness": mode or "none",
            "assertion_count": 0,
        }
    imports = (
        "import org.junit.jupiter.api.Test;\n"
        "import static org.junit.jupiter.api.Assertions.assertTrue;"
        if mode == "junit5"
        else "import org.junit.Test;\nimport static org.junit.Assert.assertTrue;"
    )
    lines = [
        "package mmm.generated;",
        "",
        imports,
        "import java.nio.file.Files;",
        "import java.nio.file.Path;",
        "",
        "public final class MmmGeneratedCounterexampleTest {",
        "    @Test",
        "    public void unchangedConsumersStillResolveTouchedResources() {",
    ]
    for relative in required:
        literal = json.dumps(relative)
        message = json.dumps("Missing referenced resource: " + relative)
        lines.append(
            f"        assertTrue(Files.isRegularFile(Path.of({literal})), {message});"
        )
    lines.extend(["    }", "}", ""])
    target = root / "src/test/java/mmm/generated/MmmGeneratedCounterexampleTest.java"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": "INSTALLED",
        "harness": mode,
        "assertion_count": len(required),
        "generated_path": "src/test/java/mmm/generated/MmmGeneratedCounterexampleTest.java",
    }


__all__ = [
    "build_generated_test_spec",
    "install_generated_junit",
    "run_generated_test_spec",
]
