from __future__ import annotations

"""Fail-closed authority for the final generated mod artifact.

The build, runtime, and downloadable bundle paths all bind to the receipt emitted
here. A filename heuristic is never sufficient authority for selecting a mod JAR.
"""

import hashlib
import json
import os
import re
import shutil
import tomllib
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_AUXILIARY_JAR = re.compile(
    r"-(?:sources?|dev|development|javadoc|docs?|api)\.jar$", re.IGNORECASE
)
_SHA256 = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
_MOD_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:/")
_METADATA_BY_LOADER = {
    "fabric": "fabric.mod.json",
    "forge": "META-INF/mods.toml",
    "neoforge": "META-INF/neoforge.mods.toml",
}


class FinalArtifactError(RuntimeError):
    """Raised when a final build cannot prove one exact production artifact."""


@dataclass(frozen=True)
class FinalModArtifactReceipt:
    status: str
    artifact: str
    artifact_path: str
    sha256: str
    size_bytes: int
    loader: str
    minecraft_version: str
    java: str
    gradle: str
    mod_id: str
    metadata_path: str
    integrity: str = "PASS"
    schema_version: str = "mmm/final-mod-artifact-receipt-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _lexical_absolute(value: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(value).expanduser())))


def _has_symlink_hop(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            if current.is_symlink():
                return True
        except OSError:
            return True
    return False


def _safe_existing_file(value: str | Path) -> Path | None:
    raw = _lexical_absolute(value)
    if _has_symlink_hop(raw):
        return None
    try:
        resolved = raw.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        return None
    return resolved if resolved.is_file() else None


def _safe_existing_directory(value: str | Path) -> Path | None:
    raw = _lexical_absolute(value)
    if _has_symlink_hop(raw):
        return None
    try:
        resolved = raw.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        return None
    return resolved if resolved.is_dir() else None


def _ensure_safe_parent(path: Path) -> None:
    current = path.parent
    while not current.exists():
        if current.is_symlink():
            raise FinalArtifactError(f"Output parent path is unsafe: {current}")
        parent = current.parent
        if parent == current:
            break
        current = parent
    if _has_symlink_hop(current) or not current.is_dir():
        raise FinalArtifactError(f"Output parent path is unsafe: {current}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if _has_symlink_hop(path.parent):
        raise FinalArtifactError(f"Output parent path is unsafe: {path.parent}")


def _safe_write_target(value: str | Path) -> Path:
    target = _lexical_absolute(value)
    if _has_symlink_hop(target):
        raise FinalArtifactError(f"Output target is unsafe: {target}")
    if target.exists():
        if not target.is_file():
            raise FinalArtifactError(f"Output target is not a regular file: {target}")
        return target
    _ensure_safe_parent(target)
    return target


def _safe_new_directory_target(value: str | Path) -> Path:
    target = _lexical_absolute(value)
    if target.exists() or target.is_symlink():
        raise FinalArtifactError(f"Download bundle path already exists: {target}")
    _ensure_safe_parent(target)
    return target


def sha256_file(path: str | Path) -> str:
    target = _safe_existing_file(path)
    if target is None:
        raise FinalArtifactError(f"Artifact is missing or unsafe: {_lexical_absolute(path)}")
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FinalArtifactError(f"Artifact could not be read safely: {target}") from exc
    return "sha256:" + digest.hexdigest()


def normalize_sha256(value: Any) -> str:
    match = _SHA256.fullmatch(str(value or "").strip())
    if match is None:
        raise FinalArtifactError("Expected a SHA-256 digest.")
    return "sha256:" + match.group(1).casefold()


def _require_artifact_sha(
    receipt: Mapping[str, Any],
    *,
    label: str,
    expected_sha256: str,
) -> None:
    try:
        observed = normalize_sha256(receipt.get("artifact_sha256"))
    except FinalArtifactError as exc:
        raise FinalArtifactError(
            f"{label} receipt is missing its artifact SHA-256."
        ) from exc
    if observed != expected_sha256:
        raise FinalArtifactError(f"{label} receipt does not bind to the final artifact.")


def select_production_jar(project_root: str | Path) -> Path:
    root = _project_root(project_root)
    libs = _safe_existing_directory(root / "build" / "libs")
    if libs is None:
        raise FinalArtifactError(
            f"Final Gradle output directory is missing or unsafe: {root / 'build' / 'libs'}"
        )
    jar_entries = sorted(libs.glob("*.jar"), key=lambda path: path.name.casefold())
    safe_entries: list[Path] = []
    unsafe: list[str] = []
    for path in jar_entries:
        safe = _safe_existing_file(path)
        if safe is None:
            unsafe.append(path.name)
        else:
            safe_entries.append(safe)
    if unsafe:
        raise FinalArtifactError(
            "Final Gradle output contains unsafe JAR entries: " + ", ".join(unsafe)
        )
    production = [path for path in safe_entries if not _AUXILIARY_JAR.search(path.name)]
    if len(production) != 1:
        names = ", ".join(path.name for path in production) or "none"
        raise FinalArtifactError(
            "Expected exactly one production JAR after excluding source/dev/javadoc "
            f"classifiers; found {len(production)}: {names}"
        )
    return production[0]


def verify_final_mod_artifact(
    project_root: str | Path,
    *,
    expected_mod_id: str = "",
    expected_loader: str = "",
    expected_minecraft_version: str = "",
    expected_java: str = "",
    expected_gradle: str = "",
    receipt_path: str | Path | None = None,
) -> FinalModArtifactReceipt:
    root = _project_root(project_root)
    project_identity = _project_identity(root)
    mod_id = _consistent_value("mod_id", expected_mod_id, project_identity["mod_id"])
    loader = _consistent_value(
        "loader", _normalize_loader(expected_loader), project_identity["loader"]
    )
    minecraft_version = _consistent_value(
        "minecraft_version",
        expected_minecraft_version,
        project_identity["minecraft_version"],
    )
    java = _consistent_value("java", expected_java, project_identity["java"])
    gradle = _consistent_value("gradle", expected_gradle, project_identity["gradle"])
    if not _MOD_ID.fullmatch(mod_id):
        raise FinalArtifactError(f"Final mod ID is missing or invalid: {mod_id!r}")
    if loader not in _METADATA_BY_LOADER:
        raise FinalArtifactError(f"Final target loader is missing or unsupported: {loader!r}")
    if not all((minecraft_version, java, gradle)):
        raise FinalArtifactError(
            "Final target receipt must bind Minecraft, Java, and Gradle versions."
        )

    jar = select_production_jar(root)
    metadata_path = _METADATA_BY_LOADER[loader]
    metadata, metadata_mod_ids, declared_minecraft = _read_jar_metadata(
        jar, loader=loader, metadata_path=metadata_path
    )
    del metadata
    if mod_id not in metadata_mod_ids:
        raise FinalArtifactError(
            f"Production JAR metadata does not declare expected mod ID {mod_id!r}."
        )
    if declared_minecraft and not _version_constraint_mentions(
        declared_minecraft, minecraft_version
    ):
        raise FinalArtifactError(
            "Production JAR Minecraft dependency disagrees with the target receipt: "
            f"target={minecraft_version!r}, metadata={declared_minecraft!r}."
        )

    receipt = FinalModArtifactReceipt(
        status="PASS",
        artifact=jar.relative_to(root).as_posix(),
        artifact_path=str(jar),
        sha256=sha256_file(jar),
        size_bytes=jar.stat().st_size,
        loader=loader,
        minecraft_version=minecraft_version,
        java=java,
        gradle=gradle,
        mod_id=mod_id,
        metadata_path=metadata_path,
    )
    if receipt_path is not None:
        _write_json_receipt(receipt_path, receipt.to_dict())
    return receipt


def verify_runtime_artifact_binding(
    runtime_receipt: Mapping[str, Any], expected_sha256: str
) -> dict[str, Any]:
    expected = normalize_sha256(expected_sha256)
    prepared = runtime_receipt.get("prepared")
    if not isinstance(prepared, Mapping):
        raise FinalArtifactError("Runtime receipt is missing its prepared instance receipt.")
    observed = {
        "runtime": runtime_receipt.get("artifact_sha256"),
        "source": prepared.get("source_mod_sha256"),
        "installed": prepared.get("installed_mod_sha256"),
    }
    for label, value in observed.items():
        try:
            actual = normalize_sha256(value)
        except FinalArtifactError as exc:
            raise FinalArtifactError(
                f"Runtime receipt is missing the {label} artifact SHA-256."
            ) from exc
        if actual != expected:
            raise FinalArtifactError(
                f"Runtime {label} artifact does not equal the final build artifact."
            )
    return {
        "schema_version": "mmm/runtime-artifact-binding-v1",
        "status": "PASS",
        "artifact_sha256": expected,
    }


def _debug_java_code_surface(source: str) -> tuple[str, str]:
    """Return commentless source plus executable-token surface for deterministic checks."""

    comments = re.compile(r"//[^\r\n]*|/\*.*?\*/", re.DOTALL)
    commentless = comments.sub(" ", source)
    literals = re.compile(
        r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
        re.DOTALL,
    )
    code = literals.sub(" ", commentless)
    return commentless, code


def _debug_host_symbol_used(code: str, symbol: Mapping[str, Any]) -> bool:
    owner = str(symbol.get("owner") or "").strip()
    name = str(symbol.get("name") or "").strip()
    kind = str(symbol.get("kind") or "").strip().casefold()
    if not owner or not name or kind not in {"method", "field"}:
        return False

    owner_leaf = owner.rsplit(".", 1)[-1]
    owner_simple = owner_leaf.split("$", 1)[0]
    owner_source = owner_leaf.replace("$", ".")
    compact = re.sub(r"\s+", "", code)
    static = symbol.get("static") is not False

    if kind == "method" and not static:
        # Instance methods such as Item.Properties.setId are invoked on an object.
        return bool(
            owner_source in compact
            and re.search(rf"\.\s*{re.escape(name)}\s*\(", code)
        )

    direct = f"{owner_simple}.{name}"
    if kind == "method":
        if f"{direct}(" in compact:
            return True
        return bool(
            f"importstatic{owner}.{name};" in compact
            and re.search(rf"(?<![\w.]){re.escape(name)}\s*\(", code)
        )

    if direct in compact:
        return True
    return bool(
        f"importstatic{owner}.{name};" in compact
        and re.search(rf"(?<![\w.]){re.escape(name)}\b", code)
    )

def verify_debug_fixture_source(
    project_root: str | Path,
    *,
    source_contract: Mapping[str, Any] | None,
    host_facts_json: str,
) -> dict[str, Any]:
    """Prove the Debug fixture's observable source semantics against host facts."""

    findings: list[str] = []
    contract = source_contract if isinstance(source_contract, Mapping) else {}
    if contract.get("schema_version") != "mmm/debug-source-contract-v1":
        findings.append("debug source contract is missing or has an unsupported schema")

    relative_path = str(contract.get("path") or "").strip()
    identifier = str(contract.get("identifier") or "").strip()
    binding_field = str(contract.get("binding_field") or "").strip()
    required_keys_raw = contract.get("required_host_symbol_keys")
    required_specs_raw = contract.get("required_host_symbol_specs")
    required_specs = (
        required_specs_raw if isinstance(required_specs_raw, Mapping) else {}
    )
    forbidden_raw = contract.get("forbidden_lifecycle_symbols")
    required_keys = (
        tuple(str(value).strip() for value in required_keys_raw if str(value).strip())
        if isinstance(required_keys_raw, Sequence)
        and not isinstance(required_keys_raw, (str, bytes, bytearray))
        else ()
    )
    forbidden = (
        tuple(str(value).strip() for value in forbidden_raw if str(value).strip())
        if isinstance(forbidden_raw, Sequence)
        and not isinstance(forbidden_raw, (str, bytes, bytearray))
        else ()
    )
    if not relative_path or Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        findings.append("debug source contract path is unsafe or empty")
    if not identifier:
        findings.append("debug source contract identifier is empty")
    if not binding_field:
        findings.append("debug source contract binding_field is empty")
    if not required_keys:
        findings.append("debug source contract has no required host symbols")

    try:
        host_facts = json.loads(host_facts_json)
    except (TypeError, json.JSONDecodeError):
        host_facts = {}
        findings.append("platform host facts are missing or malformed")
    api_symbols = host_facts.get("api_symbols") if isinstance(host_facts, Mapping) else None
    if not isinstance(api_symbols, Mapping):
        api_symbols = {}
        findings.append("platform host facts contain no api_symbols map")

    # Older immutable Debug plans may predate the explicit item_set_id contract.
    # Fail closed from HOST registration shape as well: keyed item registration
    # requires the same ResourceKey to be installed into Item.Properties.
    register_symbol = api_symbols.get("register_item")
    register_descriptor = (
        str(register_symbol.get("descriptor") or "")
        if isinstance(register_symbol, Mapping)
        else ""
    )
    keyed_registration = bool(
        "Lnet/minecraft/resources/ResourceKey;" in register_descriptor
        and "resource_key_create" in api_symbols
    )
    if keyed_registration and "item_set_id" not in required_keys:
        required_keys = (*required_keys, "item_set_id")
    if keyed_registration and "item_set_id" not in required_specs:
        required_specs = {
            **dict(required_specs),
            "item_set_id": {
                "owner": "net.minecraft.world.item.Item$Properties",
                "name": "setId",
                "descriptor": (
                    "(Lnet/minecraft/resources/ResourceKey;)"
                    "Lnet/minecraft/world/item/Item$Properties;"
                ),
                "kind": "method",
                "static": False,
                "side": "common",
                "namespace": "minecraft",
            },
        }

    root = Path(project_root).expanduser().resolve()
    target: Path | None = None
    source = ""
    source_sha256 = ""
    if relative_path and not findings[:1]:
        candidate = root / relative_path
        safe = _safe_existing_file(candidate)
        if safe is None:
            findings.append("debug source target is missing, unsafe, or not a regular file")
        else:
            try:
                safe.relative_to(root)
            except ValueError:
                findings.append("debug source target escaped the project root")
            else:
                target = safe
                try:
                    source = safe.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    findings.append("debug source target is not readable UTF-8")
                else:
                    source_sha256 = "sha256:" + hashlib.sha256(
                        source.encode("utf-8")
                    ).hexdigest()

    symbol_results: dict[str, bool] = {}
    identifier_present = False
    binding_assignment_proven = False
    lifecycle_clear = False
    if source:
        commentless, code = _debug_java_code_surface(source)
        identifier_present = f'"{identifier}"' in commentless
        if not identifier_present:
            findings.append(
                f"debug source does not contain the exact registry identifier {identifier!r}"
            )

        for key in required_keys:
            symbol = api_symbols.get(key)
            if not isinstance(symbol, Mapping):
                contract_symbol = required_specs.get(key)
                symbol = contract_symbol if isinstance(contract_symbol, Mapping) else None
            used = isinstance(symbol, Mapping) and _debug_host_symbol_used(code, symbol)
            symbol_results[key] = bool(used)
            if not isinstance(symbol, Mapping):
                findings.append(
                    f"required host symbol {key!r} is absent from platform facts and source contract"
                )
            elif not used:
                findings.append(f"debug source does not use required host symbol {key!r}")

        register_symbol = api_symbols.get("register_item")
        if isinstance(register_symbol, Mapping) and binding_field:
            register_owner = str(register_symbol.get("owner") or "").strip()
            register_name = str(register_symbol.get("name") or "").strip()
            register_simple = register_owner.rsplit(".", 1)[-1].split("$", 1)[0]
            compact = re.sub(r"\s+", "", code)
            assignment_pattern = (
                re.escape(binding_field)
                + r"=[^;]*"
                + re.escape(f"{register_simple}.{register_name}(")
            )
            binding_assignment_proven = (
                re.search(assignment_pattern, compact) is not None
            )
        if not binding_assignment_proven:
            findings.append(
                "debug binding field is not directly assigned from the host item registry call"
            )

        code_identifiers = set(re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*", code))
        lifecycle_hits = [
            token for token in forbidden if token and token in code_identifiers
        ]
        lifecycle_clear = not lifecycle_hits
        if lifecycle_hits:
            findings.append(
                "debug source introduces forbidden lifecycle surface: "
                + ", ".join(sorted(lifecycle_hits))
            )

    resource_surface = {
        "mod_id": "",
        "texture": False,
        "resource_document": False,
        "en_us": False,
        "ko_kr": False,
    }
    # This verifier owns source semantics only. Resource completeness is enforced by
    # project/JAR validation and must not make an otherwise valid source contract fail.
    metadata_path = root / "src/main/resources/fabric.mod.json"
    metadata_file = _safe_existing_file(metadata_path)
    if metadata_file is not None:
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            metadata = {}
        mod_id = str(metadata.get("id") or "").strip() if isinstance(metadata, Mapping) else ""
        resource_surface["mod_id"] = mod_id
        if mod_id:
            asset_root = root / "src/main/resources/assets" / mod_id
            texture = _safe_existing_file(
                asset_root / "textures" / "item" / f"{identifier}.png"
            )
            if texture is not None:
                try:
                    resource_surface["texture"] = texture.read_bytes().startswith(
                        b"\x89PNG\r\n\x1a\n"
                    )
                except OSError:
                    resource_surface["texture"] = False

            resource_surface["resource_document"] = any(
                "lang" not in path.parts and _safe_existing_file(path) is not None
                for path in asset_root.rglob(f"{identifier}.json")
            )
            translation_key = f"item.{mod_id}.{identifier}"
            for locale in ("en_us", "ko_kr"):
                lang_file = _safe_existing_file(asset_root / "lang" / f"{locale}.json")
                valid = False
                if lang_file is not None:
                    try:
                        lang_payload = json.loads(lang_file.read_text(encoding="utf-8"))
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                        lang_payload = {}
                    valid = bool(
                        isinstance(lang_payload, Mapping)
                        and isinstance(lang_payload.get(translation_key), str)
                        and str(lang_payload.get(translation_key) or "").strip()
                    )
                resource_surface[locale] = valid

    resource_surface_passed = all(
        bool(resource_surface[key])
        for key in ("texture", "resource_document", "en_us", "ko_kr")
    )
    passed = bool(
        target is not None
        and source
        and source_sha256
        and identifier_present
        and binding_assignment_proven
        and required_keys
        and all(symbol_results.get(key) is True for key in required_keys)
        and lifecycle_clear
        and not findings
    )
    diagnostics = [
        {
            "path": relative_path,
            "severity": 1,
            "source": "debug-source-contract",
            "code": "DEBUG_SOURCE_CONTRACT",
            "message": finding,
        }
        for finding in findings
    ]
    return {
        "schema_version": "mmm/debug-source-acceptance-v1",
        "status": "PASS" if passed else "BLOCKED",
        "source_path": relative_path,
        "source_sha256": source_sha256,
        "identifier": identifier,
        "binding_field": binding_field,
        "binding_assignment_proven": binding_assignment_proven,
        "required_host_symbol_keys": list(required_keys),
        "required_host_symbol_specs": {
            str(key): dict(value)
            for key, value in required_specs.items()
            if isinstance(value, Mapping)
        },
        "symbol_results": symbol_results,
        "forbidden_lifecycle_symbols": list(forbidden),
        "identifier_present": identifier_present,
        "lifecycle_clear": lifecycle_clear,
        "resource_surface": resource_surface,
        "resource_surface_passed": resource_surface_passed,
        "host_revision": (
            str(host_facts.get("host_revision") or "")
            if isinstance(host_facts, Mapping)
            else ""
        ),
        "findings": findings,
        "diagnostics": diagnostics,
    }


def build_debug_fixture_coverage_receipt(
    *,
    proposal_hash: str,
    acceptance_tests: Sequence[str],
    artifact_sha256: str,
    source_validation: Mapping[str, Any] | None,
    build_report: Mapping[str, Any] | None,
    jar_validation: Mapping[str, Any] | None,
    gametest_passed: bool,
    unresolved_gates: tuple[str, ...] | list[str],
    observable_acceptance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the host-owned Debug fixture to real deterministic verification evidence.

    This path is intentionally separate from production-contract coverage. It exists
    only for the planner-bypass Debug fixture, whose purpose is to exercise the normal
    implementation/build/JAR/GameTest/package pipeline without fabricating research or
    external-runtime quality evidence.
    """

    statements = [
        str(value).strip()
        for value in acceptance_tests
        if isinstance(value, str) and str(value).strip()
    ]
    unresolved = sorted(
        {str(item) for item in unresolved_gates if str(item).strip()}
    )

    source_passed = bool(
        isinstance(source_validation, Mapping)
        and source_validation.get("status") == "PASS"
        and isinstance(source_validation.get("checks_run"), int)
        and source_validation.get("checks_run", 0) > 0
        and isinstance(source_validation.get("findings"), list)
        and not any(
            isinstance(item, Mapping)
            and str(item.get("severity") or "").casefold() in {"error", "fatal"}
            for item in source_validation.get("findings", [])
        )
    )
    build_passed = bool(
        isinstance(build_report, Mapping)
        and build_report.get("status") == "PASS"
    )
    jar_passed = bool(
        isinstance(jar_validation, Mapping)
        and jar_validation.get("status") == "PASS"
        and isinstance(jar_validation.get("checks_run"), int)
        and jar_validation.get("checks_run", 0) > 0
        and isinstance(jar_validation.get("findings"), list)
        and not any(
            isinstance(item, Mapping)
            and str(item.get("severity") or "").casefold() in {"error", "fatal"}
            for item in jar_validation.get("findings", [])
        )
    )
    observable_passed = bool(
        isinstance(observable_acceptance, Mapping)
        and observable_acceptance.get("status") == "PASS"
        and isinstance(observable_acceptance.get("source_sha256"), str)
        and str(observable_acceptance.get("source_sha256") or "").startswith("sha256:")
        and not observable_acceptance.get("findings")
    )
    passed = bool(
        statements
        and source_passed
        and build_passed
        and jar_passed
        and gametest_passed is True
        and observable_passed
        and not unresolved
    )

    requirements = [
        {
            "requirement_ref": f"debug-acceptance:{index:08d}",
            "statement": statement,
            "coverage_group_ref": "debug-fixture:deterministic-verification",
            "status": "PASS" if passed else "BLOCKED",
        }
        for index, statement in enumerate(statements)
    ]
    core: dict[str, Any] = {
        "schema_version": "mmm/requirement-coverage-receipt-v1",
        "status": "PASS" if passed else "BLOCKED",
        "coverage_mode": "debug_fixture",
        "proposal_hash": str(proposal_hash),
        "production_contract_sha256": "",
        "artifact_sha256": normalize_sha256(artifact_sha256),
        "unresolved_gates": unresolved,
        "requirements": requirements,
        "verification": {
            "source_validation": source_passed,
            "build": build_passed,
            "jar_validation": jar_passed,
            "gametest": gametest_passed is True,
            "observable_acceptance": observable_passed,
        },
        "debug_source_acceptance": (
            dict(observable_acceptance)
            if isinstance(observable_acceptance, Mapping)
            else {"status": "BLOCKED", "findings": ["missing observable acceptance receipt"]}
        ),
    }
    core["coverage_sha256"] = _canonical_sha256(core)
    return core



_AUTHORED_INITIALIZE_RE = re.compile(
    r"\bpublic\s+static\s+void\s+initialize\s*\(\s*\)\s*\{"
)


def _authored_initialize_body(source: str, symbol: str) -> str | None:
    """Return executable initialize() body text with comments/literals removed."""

    _commentless, code = _debug_java_code_surface(source)
    if re.search(
        rf"\bpublic\s+final\s+class\s+{re.escape(symbol)}\b",
        code,
    ) is None:
        return None
    match = _AUTHORED_INITIALIZE_RE.search(code)
    if match is None:
        return None
    depth = 1
    start = match.end()
    for index in range(start, len(code)):
        token = code[index]
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth == 0:
                return code[start:index]
    return None


def _authored_feature_semantic_findings(
    project_root: str | Path,
    units: Sequence[Any],
) -> list[str]:
    """Reject host scaffold/no-op authored features before release certification."""

    findings: list[str] = []
    try:
        root = _project_root(project_root)
    except FinalArtifactError as exc:
        return [f"authored feature source root is unavailable: {exc}"]

    for ordinal, raw_unit in enumerate(units, start=1):
        if not isinstance(raw_unit, Mapping):
            continue
        module_id = str(raw_unit.get("module_id") or f"unit-{ordinal}").strip()
        relative = str(raw_unit.get("path") or "").replace("\\", "/").strip()
        symbol = str(raw_unit.get("symbol") or "").strip()
        if (
            not relative
            or not symbol
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not relative.casefold().endswith(".java")
        ):
            findings.append(
                f"authored module {module_id} has no safe exact Java source binding"
            )
            continue
        safe = _safe_existing_file(root / relative)
        if safe is None:
            findings.append(
                f"authored module {module_id} source is missing or unsafe: {relative}"
            )
            continue
        try:
            safe.relative_to(root)
            source = safe.read_text(encoding="utf-8")
        except (ValueError, OSError, UnicodeError):
            findings.append(
                f"authored module {module_id} source is unreadable or escaped the project root"
            )
            continue
        body = _authored_initialize_body(source, symbol)
        if body is None:
            findings.append(
                f"authored module {module_id} does not implement the required {symbol}.initialize() surface"
            )
            continue
        normalized = re.sub(r"\s+", "", body)
        if (
            not normalized
            or re.fullmatch(r"(?:;|return;)+", normalized) is not None
        ):
            findings.append(
                f"authored module {module_id} initialize() has no executable behavior"
            )
    return findings


def build_authored_design_coverage_receipt(
    *,
    proposal_hash: str,
    requested_prompt: str,
    authored_plan: Mapping[str, Any] | None,
    authored_manifest: Mapping[str, Any] | None,
    module_ids: Sequence[str],
    project_root: str | Path | None = None,
    artifact_sha256: str,
    source_validation: Mapping[str, Any] | None,
    build_report: Mapping[str, Any] | None,
    jar_validation: Mapping[str, Any] | None,
    gametest_passed: bool,
    unresolved_gates: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Bind a planner-bypassed saved design to exact host-lowered execution evidence.

    Saved authored production deliberately bypasses requirement extraction and the normal
    production contract. Release coverage therefore comes from the immutable authored
    text plus the exact-task manifest that partitions every UTF-8 byte of that text.
    """

    findings: list[str] = []
    plan = authored_plan if isinstance(authored_plan, Mapping) else {}
    manifest = authored_manifest if isinstance(authored_manifest, Mapping) else {}

    if plan.get("schema_version") != "mmm/authored-plan-v1":
        findings.append("authored plan is missing or has an unsupported schema")
    plan_prompt = str(plan.get("requested_prompt") or "").strip()
    expected_prompt = str(requested_prompt or "").strip()
    if not plan_prompt:
        findings.append("authored plan requested prompt is empty")
    elif plan_prompt != expected_prompt:
        findings.append("authored plan requested prompt does not match the approved proposal")

    raw_text = plan.get("text")
    text = raw_text if isinstance(raw_text, str) else ""
    if not text.strip():
        findings.append("authored plan text is empty")
    text_bytes = text.encode("utf-8")
    source_text_sha256 = "sha256:" + hashlib.sha256(text_bytes).hexdigest()

    if manifest.get("schema_version") != "mmm/authored-execution-manifest-v2":
        findings.append("authored execution manifest is missing or has an unsupported schema")
    authored_policy = str(manifest.get("policy") or "")
    if authored_policy not in {
        "host_exact_task_queue_no_coder_file_planning",
        "host_localize_freeze_exact_targets_before_coder",
    }:
        findings.append("authored execution manifest policy is not a host-owned task policy")
    if manifest.get("source_text_sha256") != source_text_sha256:
        findings.append("authored execution manifest does not bind the approved design text")
    if manifest.get("source_bytes") != len(text_bytes):
        findings.append("authored execution manifest source byte count does not match")

    supplied_manifest_sha256 = str(manifest.get("manifest_sha256") or "")
    manifest_payload = dict(manifest)
    manifest_payload.pop("manifest_sha256", None)
    expected_manifest_sha256 = _canonical_sha256(manifest_payload)
    if supplied_manifest_sha256 != expected_manifest_sha256:
        findings.append("authored execution manifest hash does not match its payload")

    raw_units = manifest.get("units")
    units = raw_units if isinstance(raw_units, list) else []
    unit_count = manifest.get("unit_count")
    if type(unit_count) is not int or unit_count <= 0:
        findings.append("authored execution manifest unit_count is invalid")
    elif unit_count != len(units):
        findings.append("authored execution manifest unit_count does not match units")
    if not units:
        findings.append("authored execution manifest contains no authored units")

    expected_module_ids = [str(value).strip() for value in module_ids]
    if not expected_module_ids or any(not value for value in expected_module_ids):
        findings.append("approved authored module IDs are missing or invalid")

    manifest_module_ids: list[str] = []
    requirement_rows: list[dict[str, str]] = []
    if plan_prompt:
        requirement_rows.append(
            {
                "requirement_ref": "authored-request:00000000",
                "statement": plan_prompt,
                "coverage_group_ref": "saved-authored-design:request",
            }
        )

    cursor = 0
    for ordinal, raw_unit in enumerate(units, start=1):
        if not isinstance(raw_unit, Mapping):
            findings.append(f"authored execution unit {ordinal} is not an object")
            continue
        start = raw_unit.get("start_byte")
        end = raw_unit.get("end_byte")
        if type(start) is not int or type(end) is not int:
            findings.append(f"authored execution unit {ordinal} has invalid byte bounds")
            continue
        if start != cursor or end <= start or end > len(text_bytes):
            findings.append(f"authored execution unit {ordinal} does not form a contiguous text partition")
            if 0 <= end <= len(text_bytes):
                cursor = end
            continue

        piece = text_bytes[start:end]
        try:
            statement = piece.decode("utf-8").strip()
        except UnicodeDecodeError:
            statement = ""
            findings.append(f"authored execution unit {ordinal} splits invalid UTF-8")
        unit_sha256 = "sha256:" + hashlib.sha256(piece).hexdigest()
        if str(raw_unit.get("text_sha256") or "") != unit_sha256:
            findings.append(f"authored execution unit {ordinal} text hash does not match")
        module_id = str(raw_unit.get("module_id") or "").strip()
        if not module_id:
            findings.append(f"authored execution unit {ordinal} has no module binding")
        manifest_module_ids.append(module_id)
        if not statement:
            findings.append(f"authored execution unit {ordinal} has no authored statement")
        else:
            requirement_rows.append(
                {
                    "requirement_ref": (
                        f"authored-unit:{ordinal:08d}:"
                        + unit_sha256.removeprefix("sha256:")
                    ),
                    "statement": statement,
                    "coverage_group_ref": (
                        f"authored-module:{module_id}" if module_id else "authored-module:missing"
                    ),
                }
            )
        cursor = end

    if cursor != len(text_bytes):
        findings.append("authored execution units do not cover every byte of the approved design")
    if manifest_module_ids != expected_module_ids:
        findings.append("authored execution manifest modules do not match the approved proposal modules")

    if (
        authored_policy == "host_exact_task_queue_no_coder_file_planning"
        and project_root is not None
    ):
        findings.extend(_authored_feature_semantic_findings(project_root, units))

    unresolved = sorted(
        {str(item) for item in unresolved_gates if str(item).strip()}
    )
    source_passed = bool(
        isinstance(source_validation, Mapping)
        and source_validation.get("status") == "PASS"
        and isinstance(source_validation.get("checks_run"), int)
        and source_validation.get("checks_run", 0) > 0
        and isinstance(source_validation.get("findings"), list)
        and not any(
            isinstance(item, Mapping)
            and str(item.get("severity") or "").casefold() in {"error", "fatal"}
            for item in source_validation.get("findings", [])
        )
    )
    build_passed = bool(
        isinstance(build_report, Mapping)
        and build_report.get("status") == "PASS"
    )
    jar_passed = bool(
        isinstance(jar_validation, Mapping)
        and jar_validation.get("status") == "PASS"
        and isinstance(jar_validation.get("checks_run"), int)
        and jar_validation.get("checks_run", 0) > 0
        and isinstance(jar_validation.get("findings"), list)
        and not any(
            isinstance(item, Mapping)
            and str(item.get("severity") or "").casefold() in {"error", "fatal"}
            for item in jar_validation.get("findings", [])
        )
    )
    binding_passed = not findings
    passed = bool(
        requirement_rows
        and binding_passed
        and source_passed
        and build_passed
        and jar_passed
        and gametest_passed is True
        and not unresolved
    )
    rows = [
        {**item, "status": "PASS" if passed else "BLOCKED"}
        for item in requirement_rows
    ]
    core: dict[str, Any] = {
        "schema_version": "mmm/requirement-coverage-receipt-v1",
        "status": "PASS" if passed else "BLOCKED",
        "coverage_mode": "saved_authored_design",
        "proposal_hash": str(proposal_hash),
        "production_contract_sha256": "",
        "artifact_sha256": normalize_sha256(artifact_sha256),
        "unresolved_gates": unresolved,
        "requirements": rows,
        "authored_design_binding": {
            "schema_version": "mmm/authored-design-binding-v1",
            "source_text_sha256": source_text_sha256,
            "manifest_sha256": supplied_manifest_sha256,
            "unit_count": len(units),
            "source_bytes": len(text_bytes),
        },
        "verification": {
            "authored_design_binding": binding_passed,
            "source_validation": source_passed,
            "build": build_passed,
            "jar_validation": jar_passed,
            "gametest": gametest_passed is True,
        },
        "findings": findings,
    }
    core["coverage_sha256"] = _canonical_sha256(core)
    return core

def build_requirement_coverage_receipt(
    *,
    contract: Mapping[str, Any] | None,
    proposal_hash: str,
    quality_report: Mapping[str, Any] | None,
    artifact_sha256: str,
    unresolved_gates: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    requirements = []
    if isinstance(contract, Mapping):
        raw_requirements = contract.get("requirement_catalog")
        if isinstance(raw_requirements, list):
            for item in raw_requirements:
                if not isinstance(item, Mapping):
                    continue
                requirement_ref = str(item.get("requirement_ref") or "").strip()
                statement = str(item.get("statement") or "").strip()
                if requirement_ref and statement:
                    requirements.append(
                        {
                            "requirement_ref": requirement_ref,
                            "statement": statement,
                            "coverage_group_ref": str(
                                item.get("coverage_group_ref") or ""
                            ),
                        }
                    )
    quality_passed = bool(
        isinstance(quality_report, Mapping)
        and quality_report.get("overall_status") == "PASS"
    )
    unresolved = sorted({str(item) for item in unresolved_gates if str(item).strip()})
    passed = bool(requirements and quality_passed and not unresolved)
    rows = [{**item, "status": "PASS" if passed else "BLOCKED"} for item in requirements]
    core: dict[str, Any] = {
        "schema_version": "mmm/requirement-coverage-receipt-v1",
        "status": "PASS" if passed else "BLOCKED",
        "proposal_hash": str(proposal_hash),
        "production_contract_sha256": (
            str(contract.get("contract_sha256") or "")
            if isinstance(contract, Mapping)
            else ""
        ),
        "artifact_sha256": normalize_sha256(artifact_sha256),
        "unresolved_gates": unresolved,
        "requirements": rows,
    }
    core["coverage_sha256"] = _canonical_sha256(core)
    return core


def empty_reuse_manifest(project_name: str) -> dict[str, Any]:
    return {
        "schema_version": "mmm/reuse-manifest-v1",
        "project_name": project_name,
        "total_reused_files": 0,
        "reused_file_count": 0,
        "donor_count": 0,
        "bundle_count": 0,
        "donors": [],
        "bundles": [],
        "files": [],
    }


def load_or_empty_reuse_manifest(
    project_root: str | Path, project_name: str
) -> dict[str, Any]:
    return _load_reuse_manifest(_project_root(project_root), project_name)


def write_build_artifact_bundle(
    output_zip: str | Path,
    *,
    artifact_receipt: Mapping[str, Any],
    build_receipt: Mapping[str, Any],
    unresolved_gates: Sequence[str],
    release_ready: bool,
    proposal_hash: str,
    receipts: Mapping[str, Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Package a successful build independently from release certification."""

    if artifact_receipt.get("status") != "PASS":
        raise FinalArtifactError("Build artifact bundle requires a passing artifact receipt.")
    artifact = _safe_existing_file(str(artifact_receipt.get("artifact_path") or ""))
    if artifact is None:
        raise FinalArtifactError("Build artifact path is missing or unsafe.")
    artifact_sha256 = normalize_sha256(artifact_receipt.get("sha256"))
    if sha256_file(artifact) != artifact_sha256:
        raise FinalArtifactError("Build artifact changed after verification.")
    if build_receipt.get("status") != "PASS":
        raise FinalArtifactError("Build artifact bundle requires a passing build receipt.")
    _require_artifact_sha(
        build_receipt,
        label="Build",
        expected_sha256=artifact_sha256,
    )

    target = _safe_write_target(output_zip)
    if target.exists():
        target.unlink()
    unresolved = sorted({str(value) for value in unresolved_gates if str(value).strip()})
    manifest = {
        "schema_version": "mmm/build-artifact-bundle-v1",
        "status": "BUILT",
        "release_ready": bool(release_ready),
        "release_certified": False,
        "artifact": artifact.name,
        "artifact_sha256": artifact_sha256,
        "proposal_hash": str(proposal_hash),
        "unresolved_gates": unresolved,
    }
    receipt_payloads: dict[str, Mapping[str, Any]] = {
        "artifact-receipt.json": dict(artifact_receipt),
        "build-receipt.json": dict(build_receipt),
    }
    for name, payload in sorted((receipts or {}).items()):
        if payload is None:
            continue
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise FinalArtifactError(f"Build bundle receipt name is unsafe: {name!r}")
        if not isinstance(payload, Mapping):
            raise FinalArtifactError(f"Build bundle receipt is invalid: {name}")
        receipt_payloads[name] = dict(payload)

    temp = target.with_name("." + target.name + ".tmp")
    if temp.exists():
        if temp.is_symlink() or not temp.is_file():
            raise FinalArtifactError("Build bundle temporary target is unsafe.")
        temp.unlink()
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(artifact, arcname="artifact/" + artifact.name)
            archive.writestr(
                "build-manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            for name, payload in sorted(receipt_payloads.items()):
                archive.writestr(
                    "receipts/" + name,
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                )
        temp.replace(target)
    except BaseException:
        temp.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise

    return {
        **manifest,
        "status": "PACKAGED",
        "build_status": "BUILT",
        "build_bundle_zip": str(target),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def write_downloadable_bundle(
    bundle_dir: str | Path,
    *,
    artifact_receipt: Mapping[str, Any],
    requirement_coverage: Mapping[str, Any],
    reuse_manifest: Mapping[str, Any],
    build_receipt: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    additional_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    proposal_hash: str = "",
) -> dict[str, Any]:
    expected_proposal_hash = str(proposal_hash or "").strip()
    coverage_proposal_hash = str(requirement_coverage.get("proposal_hash") or "").strip()
    if expected_proposal_hash:
        if coverage_proposal_hash != expected_proposal_hash:
            raise FinalArtifactError(
                "Requirement coverage is not bound to the requested pipeline proposal."
            )
        bound_proposal_hash = expected_proposal_hash
    else:
        bound_proposal_hash = coverage_proposal_hash

    if artifact_receipt.get("status") != "PASS":
        raise FinalArtifactError("Only a passing final artifact receipt may be bundled.")
    artifact = _safe_existing_file(str(artifact_receipt.get("artifact_path") or ""))
    if artifact is None:
        raise FinalArtifactError("Final artifact path is missing or unsafe.")
    expected_sha256 = normalize_sha256(artifact_receipt.get("sha256"))
    if sha256_file(artifact) != expected_sha256:
        raise FinalArtifactError("Final artifact changed after verification.")
    if build_receipt.get("status") != "PASS":
        raise FinalArtifactError("Download bundle requires a passing final build receipt.")
    if requirement_coverage.get("status") != "PASS":
        raise FinalArtifactError("Download bundle requires complete requirement coverage.")
    _require_artifact_sha(
        build_receipt,
        label="Build",
        expected_sha256=expected_sha256,
    )
    _require_artifact_sha(
        requirement_coverage,
        label="Requirement coverage",
        expected_sha256=expected_sha256,
    )
    runtime_status = str(runtime_receipt.get("status") or "")
    if runtime_status not in {"PASS", "NOT_REQUIRED"}:
        raise FinalArtifactError("Download bundle requires a terminal runtime receipt.")
    if runtime_status == "PASS":
        verify_runtime_artifact_binding(runtime_receipt, expected_sha256)
    else:
        _require_artifact_sha(
            runtime_receipt,
            label="Runtime",
            expected_sha256=expected_sha256,
        )

    target = _safe_new_directory_target(bundle_dir)
    target.mkdir()
    try:
        installed = target / artifact.name
        shutil.copy2(artifact, installed)
        if sha256_file(installed) != expected_sha256:
            raise FinalArtifactError("Bundled JAR changed while it was copied.")

        bundled_additional: dict[str, str] = {}
        for name, descriptor in sorted((additional_artifacts or {}).items()):
            if not isinstance(name, str) or not name or Path(name).name != name:
                raise FinalArtifactError("Additional artifact bundle name is unsafe.")
            if not isinstance(descriptor, Mapping):
                raise FinalArtifactError(
                    f"Additional artifact descriptor is invalid: {name}"
                )
            source = _safe_existing_file(str(descriptor.get("path") or ""))
            if source is None:
                raise FinalArtifactError(
                    f"Additional artifact path is missing or unsafe: {name}"
                )
            expected = normalize_sha256(descriptor.get("sha256"))
            if sha256_file(source) != expected:
                raise FinalArtifactError(
                    f"Additional artifact changed after verification: {name}"
                )
            destination = target / name
            if destination.exists():
                raise FinalArtifactError(
                    f"Additional artifact collides with bundle member: {name}"
                )
            shutil.copy2(source, destination)
            if sha256_file(destination) != expected:
                raise FinalArtifactError(
                    f"Additional artifact changed while copied: {name}"
                )
            bundled_additional[name] = expected

        receipts = {
            "artifact-receipt.json": dict(artifact_receipt),
            "requirement-coverage.json": dict(requirement_coverage),
            "reuse-manifest.json": dict(reuse_manifest),
            "build-receipt.json": dict(build_receipt),
            "runtime-receipt.json": dict(runtime_receipt),
        }
        for name, payload in receipts.items():
            _write_json_receipt(target / name, payload)
        members = []
        for path in sorted(target.iterdir(), key=lambda item: item.name):
            safe = _safe_existing_file(path)
            if safe is None:
                raise FinalArtifactError(f"Download bundle member is unsafe: {path}")
            members.append(
                {
                    "path": safe.name,
                    "sha256": sha256_file(safe),
                    "size_bytes": safe.stat().st_size,
                }
            )
        bundle_receipt: dict[str, Any] = {
            "schema_version": "mmm/downloadable-mod-bundle-v1",
            "status": "PASS",
            "artifact": artifact.name,
            "artifact_sha256": expected_sha256,
            "proposal_hash": bound_proposal_hash,
            "additional_artifacts": bundled_additional,
            "members": members,
        }
        bundle_receipt["manifest_sha256"] = _canonical_sha256(bundle_receipt)
        _write_json_receipt(target / "bundle-receipt.json", bundle_receipt)
        return {**bundle_receipt, "path": str(target)}
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise


def bundle_from_pipeline_result(
    result: Mapping[str, Any],
    bundle_dir: str | Path,
    *,
    require_runtime: bool = False,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise FinalArtifactError("Complete pipeline result must be a JSON object.")
    if result.get("status") != "VERIFIED" or result.get("release_ready") is not True:
        raise FinalArtifactError(
            "Download bundle requires a release-ready VERIFIED pipeline result."
        )
    result_proposal_hash = str(result.get("complete_proposal_hash") or "").strip()
    if not result_proposal_hash:
        raise FinalArtifactError("Complete pipeline result has no proposal hash.")
    project_root = _project_root(str(result.get("project_root") or ""))
    artifact = verify_final_mod_artifact(project_root).to_dict()
    result_jar = _safe_existing_file(str(result.get("jar_path") or ""))
    if result_jar is None or result_jar != Path(artifact["artifact_path"]):
        raise FinalArtifactError("Pipeline result JAR is not the sole verified production JAR.")
    build = result.get("build_report")
    if not isinstance(build, Mapping) or build.get("status") != "PASS":
        raise FinalArtifactError("Complete pipeline result has no passing final build.")
    reported_artifact = build.get("artifact_receipt")
    if not isinstance(reported_artifact, Mapping):
        raise FinalArtifactError("Final build did not persist an artifact receipt.")
    if normalize_sha256(reported_artifact.get("sha256")) != artifact["sha256"]:
        raise FinalArtifactError("Build receipt artifact SHA-256 no longer matches.")

    runtime = result.get("runtime_receipt")
    if require_runtime:
        if not isinstance(runtime, Mapping):
            raise FinalArtifactError("Production integration requires runtime evidence.")
        verify_runtime_artifact_binding(runtime, artifact["sha256"])
    runtime_payload = (
        dict(runtime)
        if isinstance(runtime, Mapping)
        else {
            "schema_version": "mmm/final-runtime-receipt-v1",
            "status": "NOT_REQUIRED",
            "artifact_sha256": artifact["sha256"],
        }
    )
    if isinstance(runtime, Mapping):
        runtime_payload.setdefault("status", "PASS")
        runtime_payload.setdefault("artifact_sha256", artifact["sha256"])

    coverage = _read_optional_json(project_root / ".minecraft_ai/requirement-coverage.json")
    if coverage is None:
        coverage = _coverage_from_project(project_root, result, artifact["sha256"])
    if str(coverage.get("proposal_hash") or "").strip() != result_proposal_hash:
        raise FinalArtifactError(
            "Requirement coverage is not bound to this pipeline result proposal."
        )
    reuse = _load_reuse_manifest(project_root, artifact["mod_id"])
    build_payload = _read_optional_json(project_root / ".minecraft_ai/build-receipt.json")
    if build_payload is None:
        build_payload = dict(build)
        build_payload["schema_version"] = "mmm/final-build-receipt-v1"
        build_payload["artifact_sha256"] = artifact["sha256"]
    asset_receipt = result.get("asset_receipt")
    resource_pack_bundle = (
        asset_receipt.get("resource_pack_bundle")
        if isinstance(asset_receipt, Mapping)
        and isinstance(asset_receipt.get("resource_pack_bundle"), Mapping)
        else None
    )
    additional_artifacts = (
        {"generated-resource-pack.zip": resource_pack_bundle}
        if resource_pack_bundle is not None
        else None
    )
    return write_downloadable_bundle(
        bundle_dir,
        artifact_receipt=artifact,
        requirement_coverage=coverage,
        reuse_manifest=reuse,
        build_receipt=build_payload,
        runtime_receipt=runtime_payload,
        additional_artifacts=additional_artifacts,
        proposal_hash=result_proposal_hash,
    )


def append_github_outputs(path: str | Path, bundle: Mapping[str, Any]) -> None:
    target = _safe_write_target(path)
    bundle_path = _safe_existing_directory(str(bundle.get("path") or ""))
    if bundle_path is None:
        raise FinalArtifactError("Download bundle path is missing or unsafe.")
    artifact_name = str(bundle.get("artifact") or "")
    if not artifact_name or Path(artifact_name).name != artifact_name:
        raise FinalArtifactError("Download bundle artifact name is unsafe.")
    artifact_path = _safe_existing_file(bundle_path / artifact_name)
    if artifact_path is None:
        raise FinalArtifactError("Download bundle artifact is missing or unsafe.")
    expected_sha256 = normalize_sha256(bundle.get("artifact_sha256"))
    if sha256_file(artifact_path) != expected_sha256:
        raise FinalArtifactError("Download bundle artifact SHA-256 does not match its receipt.")
    receipt_path = _safe_existing_file(bundle_path / "artifact-receipt.json")
    if receipt_path is None:
        raise FinalArtifactError("Download bundle artifact receipt is missing or unsafe.")
    values = {
        "artifact_path": str(artifact_path),
        "artifact_sha256": expected_sha256,
        "bundle_path": str(bundle_path),
        "receipt_path": str(receipt_path),
    }
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise FinalArtifactError("GitHub Actions output value contains a newline.")
            handle.write(f"{key}={value}\n")


def _project_root(value: str | Path) -> Path:
    raw = _lexical_absolute(value)
    if _has_symlink_hop(raw):
        raise FinalArtifactError("Final project root may not traverse symbolic links.")
    try:
        root = raw.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise FinalArtifactError(f"Final project root is missing: {raw}") from exc
    if not root.is_dir():
        raise FinalArtifactError(f"Final project root is missing: {root}")
    return root


def _project_identity(root: Path) -> dict[str, str]:
    loader, mod_id = _source_metadata_identity(root)
    minecraft = java = gradle = ""
    lock_path = root / ".minecraft_ai" / "platform-lock.json"
    lock = _optional_safe_file(lock_path)
    if lock is not None:
        try:
            raw = json.loads(lock.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FinalArtifactError("Generated platform lock is invalid.") from exc
        if not isinstance(raw, dict):
            raise FinalArtifactError("Generated platform lock must be an object.")
        lock_loader = _normalize_loader(raw.get("loader"))
        loader = _consistent_value("loader", loader, lock_loader)
        minecraft = str(raw.get("minecraft_version") or "").strip()
        java = str(raw.get("java_version") or "").strip()
        gradle = str(raw.get("gradle") or "").strip()
    return {
        "loader": loader,
        "mod_id": mod_id,
        "minecraft_version": minecraft,
        "java": java,
        "gradle": gradle,
    }


def _source_metadata_identity(root: Path) -> tuple[str, str]:
    resources = root / "src" / "main" / "resources"
    found: list[tuple[str, str]] = []
    for loader, relative in _METADATA_BY_LOADER.items():
        safe = _optional_safe_file(resources / Path(relative))
        if safe is None:
            continue
        raw = safe.read_bytes()
        _, ids, _ = _parse_metadata(raw, loader=loader, metadata_path=relative)
        if len(ids) != 1:
            raise FinalArtifactError(
                f"Source metadata must declare exactly one project mod ID: {relative}"
            )
        found.append((loader, ids[0]))
    if len(found) != 1:
        raise FinalArtifactError(
            "Final project must contain exactly one loader metadata authority."
        )
    return found[0]


def _read_jar_metadata(
    jar: Path, *, loader: str, metadata_path: str
) -> tuple[dict[str, Any], tuple[str, ...], Any]:
    safe_jar = _safe_existing_file(jar)
    if safe_jar is None or not zipfile.is_zipfile(safe_jar):
        raise FinalArtifactError("Production artifact is missing, unsafe, or not a ZIP/JAR archive.")
    try:
        with zipfile.ZipFile(safe_jar) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise FinalArtifactError("Production JAR contains duplicate entries.")
            for info in infos:
                normalized = info.filename.replace("\\", "/")
                if (
                    not normalized
                    or "\x00" in normalized
                    or normalized.startswith("/")
                    or normalized.startswith("//")
                    or _WINDOWS_DRIVE_PATH.match(normalized)
                    or ".." in Path(normalized).parts
                ):
                    raise FinalArtifactError(
                        f"Production JAR contains unsafe path: {info.filename}"
                    )
            corrupt = archive.testzip()
            if corrupt is not None:
                raise FinalArtifactError(
                    f"Production JAR entry failed its CRC check: {corrupt}"
                )
            known = [path for path in _METADATA_BY_LOADER.values() if path in names]
            if known != [metadata_path]:
                raise FinalArtifactError(
                    "Production JAR loader metadata is missing, mixed, or mismatched: "
                    + (", ".join(known) or "none")
                )
            info = archive.getinfo(metadata_path)
            if info.file_size > 2 * 1024 * 1024:
                raise FinalArtifactError("Production JAR metadata is unreasonably large.")
            raw = archive.read(metadata_path)
    except FinalArtifactError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError, ValueError) as exc:
        raise FinalArtifactError("Production JAR integrity verification failed.") from exc
    return _parse_metadata(raw, loader=loader, metadata_path=metadata_path)


def _parse_metadata(
    raw: bytes, *, loader: str, metadata_path: str
) -> tuple[dict[str, Any], tuple[str, ...], Any]:
    try:
        text = raw.decode("utf-8")
        if loader == "fabric":
            metadata = json.loads(text)
        else:
            metadata = tomllib.loads(text)
    except (UnicodeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise FinalArtifactError(f"Invalid loader metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise FinalArtifactError(f"Loader metadata is not an object: {metadata_path}")
    if loader == "fabric":
        mod_id = str(metadata.get("id") or "").strip()
        ids = (mod_id,) if mod_id else ()
        depends = metadata.get("depends")
        minecraft = depends.get("minecraft") if isinstance(depends, Mapping) else ""
        return metadata, ids, minecraft
    mods = metadata.get("mods")
    if isinstance(mods, Mapping):
        mod_rows = [mods]
    elif isinstance(mods, list):
        mod_rows = [row for row in mods if isinstance(row, Mapping)]
    else:
        mod_rows = []
    ids = tuple(
        dict.fromkeys(
            str(row.get("modId") or row.get("mod_id") or "").strip()
            for row in mod_rows
            if str(row.get("modId") or row.get("mod_id") or "").strip()
        )
    )
    minecraft: Any = ""
    dependencies = metadata.get("dependencies")
    if isinstance(dependencies, Mapping):
        for value in dependencies.values():
            rows = value if isinstance(value, list) else [value]
            for row in rows:
                if isinstance(row, Mapping) and str(row.get("modId") or "") == "minecraft":
                    minecraft = row.get("versionRange") or row.get("version_range") or ""
                    break
            if minecraft:
                break
    return metadata, ids, minecraft


def _version_constraint_mentions(value: Any, version: str) -> bool:
    if isinstance(value, str):
        return bool(re.search(rf"(?<![0-9]){re.escape(version)}(?![0-9])", value))
    if isinstance(value, (list, tuple)):
        return any(_version_constraint_mentions(item, version) for item in value)
    return False


def _normalize_loader(value: Any) -> str:
    loader = str(value or "").strip().casefold().replace("_", "").replace("-", "")
    return {"fabric": "fabric", "forge": "forge", "neoforge": "neoforge"}.get(
        loader, loader
    )


def _consistent_value(name: str, first: Any, second: Any) -> str:
    left = str(first or "").strip()
    right = str(second or "").strip()
    if left and right and left != right:
        raise FinalArtifactError(
            f"Final project {name} disagrees with the requested target: {left!r} != {right!r}."
        )
    return left or right


def _canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_json_receipt(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = _safe_write_target(path)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _optional_safe_file(path: Path) -> Path | None:
    raw = _lexical_absolute(path)
    if not raw.exists() and not raw.is_symlink():
        return None
    safe = _safe_existing_file(raw)
    if safe is None:
        raise FinalArtifactError(f"Receipt or metadata path is unsafe: {raw}")
    return safe


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    safe = _optional_safe_file(path)
    if safe is None:
        return None
    try:
        value = json.loads(safe.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalArtifactError(f"Invalid receipt JSON: {safe}") from exc
    if not isinstance(value, dict):
        raise FinalArtifactError(f"Receipt must be a JSON object: {safe}")
    return value


def _coverage_from_project(
    project_root: Path, result: Mapping[str, Any], artifact_sha256: str
) -> dict[str, Any]:
    try:
        from .proposal_store import load_sharded_complete_proposal

        proposal_path = _safe_existing_file(
            project_root / ".minecraft_ai/complete-proposal.json"
        )
        if proposal_path is None:
            raise FinalArtifactError("Complete proposal path is missing or unsafe.")
        proposal = load_sharded_complete_proposal(proposal_path)
    except Exception as exc:
        raise FinalArtifactError(
            "Final project has no readable requirement-bound complete proposal."
        ) from exc
    contract = proposal.game_design.get("_production_contract")
    if not isinstance(contract, Mapping):
        raise FinalArtifactError("Complete proposal has no production requirement contract.")
    quality = result.get("quality_report")
    return build_requirement_coverage_receipt(
        contract=contract,
        proposal_hash=str(result.get("complete_proposal_hash") or ""),
        quality_report=quality if isinstance(quality, Mapping) else None,
        artifact_sha256=artifact_sha256,
        unresolved_gates=list(result.get("unresolved_gates") or ()),
    )


def _load_reuse_manifest(project_root: Path, project_name: str) -> dict[str, Any]:
    candidates = (
        project_root / "reuse-manifest.json",
        project_root / ".minecraft_ai/reuse-manifest.json",
    )
    for path in candidates:
        value = _read_optional_json(path)
        if value is not None:
            if value.get("schema_version") != "mmm/reuse-manifest-v1":
                raise FinalArtifactError("Reuse manifest has an unsupported schema.")
            return value
    return empty_reuse_manifest(project_name)


__all__ = [
    "FinalArtifactError",
    "FinalModArtifactReceipt",
    "append_github_outputs",
    "build_authored_design_coverage_receipt",
    "build_requirement_coverage_receipt",
    "bundle_from_pipeline_result",
    "empty_reuse_manifest",
    "load_or_empty_reuse_manifest",
    "normalize_sha256",
    "select_production_jar",
    "sha256_file",
    "verify_final_mod_artifact",
    "verify_runtime_artifact_binding",
    "write_downloadable_bundle",
]
