"""Immutable, evidence-bearing inventory of an existing Minecraft project.

The scanner in this module is deliberately inspection-only.  It never imports
project classes, invokes Gradle, opens a JVM, evaluates build scripts, or follows
symbolic links.  Build files are treated as text and metadata as data.  Every
fact exposed to a planner is bound to a relative locator and a SHA-256 digest of
the bytes that were inspected.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "mmm/project-inventory-v1"
COMPONENT_CATALOG_SCHEMA = "mmm/component-catalog-v1"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".gradle",
        ".idea",
        ".minecraft_ai",
        ".vscode",
        "build",
        "node_modules",
        "out",
        "run",
        "target",
    }
)
_TEXT_SUFFIXES = frozenset(
    {
        ".accesswidener",
        ".gradle",
        ".java",
        ".json",
        ".kts",
        ".mcfunction",
        ".md",
        ".mcmeta",
        ".properties",
        ".scala",
        ".snbt",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
_TEXT_NAMES = frozenset(
    {
        "dockerfile",
        "gradlew",
        "jenkinsfile",
        "license",
        "notice",
        "readme",
    }
)
_SOURCE_LANGUAGES = frozenset({"java", "kotlin", "scala"})
_RESOURCE_ID = re.compile(r"(?<![A-Za-z0-9_.-])([a-z0-9_.-]+:[a-z0-9_./-]+)")
_TWO_PART_IDENTIFIER = re.compile(
    r"(?:Identifier|ResourceLocation)(?:\.of|\.tryParse)?\s*\(\s*"
    r"[\"']([a-z0-9_.-]+)[\"']\s*,\s*[\"']([a-z0-9_./-]+)[\"']\s*\)",
    re.IGNORECASE,
)
_PACKAGE = re.compile(r"(?m)^\s*package\s+([A-Za-z_][A-Za-z0-9_.]*)\s*;?")
_IMPORT = re.compile(r"(?m)^\s*import\s+(?:static\s+)?([A-Za-z_][A-Za-z0-9_.$*]*)")
_JAVA_TYPE = re.compile(
    r"(?:^|\s)(?:public\s+|protected\s+|private\s+|abstract\s+|final\s+|"
    r"static\s+|sealed\s+|non-sealed\s+|strictfp\s+)*"
    r"(?P<kind>@interface|class|interface|enum|record)\s+"
    r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\b"
)
_KOTLIN_TYPE = re.compile(
    r"(?:^|\s)(?:(?:public|protected|private|internal|open|abstract|sealed|data|"
    r"value|annotation|enum|fun)\s+)*(?P<kind>class|interface|object)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"
)
_JAVA_METHOD = re.compile(
    r"^\s*(?:(?:public|protected|private|static|final|abstract|synchronized|native|"
    r"default|strictfp)\s+)+(?:<[^>{};]+>\s+)?[A-Za-z_$][A-Za-z0-9_.$<>, ?\[\]@]*\s+"
    r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*\([^;{}]*\)"
)
_JAVA_FIELD = re.compile(
    r"^\s*(?:(?:public|protected|private|static|final|transient|volatile)\s+)+"
    r"[A-Za-z_$][A-Za-z0-9_.$<>, ?\[\]@]*\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*"
    r"(?:=|;)"
)
_KOTLIN_CALLABLE = re.compile(
    r"^\s*(?:(?:public|protected|private|internal|open|override|abstract|suspend|"
    r"inline|operator|infix|tailrec|external|const|lateinit)\s+)*"
    r"(?P<kind>fun|val|var)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"
)
_GRADLE_DEPENDENCY = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<configuration>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?:\(\s*)?[\"'](?P<coordinate>[^\"'\r\n]+)[\"']\s*\)?"
)
_GRADLE_PROJECT_DEPENDENCY = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<configuration>[A-Za-z_][A-Za-z0-9_]*)\s*\(?\s*"
    r"project\s*\(\s*[\"'](?P<project>:[^\"']+)[\"']\s*\)"
)
_GRADLE_PLUGIN = re.compile(
    r"(?:id\s*\(?\s*[\"'](?P<id>[^\"']+)[\"']\s*\)?|"
    r"alias\s*\(\s*libs\.plugins\.(?P<alias>[A-Za-z0-9_.-]+)\s*\))"
    r"(?:\s*version\s*[\"'](?P<version>[^\"']+)[\"'])?"
)
_QUOTED = re.compile(r"[\"']([^\"']+)[\"']")


class ProjectInventoryError(ValueError):
    """Raised when a project cannot produce trustworthy planning evidence."""


def _canonical_value(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonical_value(value.to_dict())
    if hasattr(value, "__dataclass_fields__") and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise ProjectInventoryError(f"Non-canonical inventory value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Encode one inventory payload without platform- or insertion-order drift."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidenceLocator:
    """An exact project-relative evidence location bound to inspected bytes."""

    locator_id: str
    path: str
    sha256: str
    size_bytes: int
    line_start: int = 0
    line_end: int = 0

    @property
    def locator(self) -> str:
        if self.line_start:
            suffix = f"#L{self.line_start}"
            if self.line_end and self.line_end != self.line_start:
                suffix += f"-L{self.line_end}"
            return self.path + suffix
        return self.path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["locator"] = self.locator
        return payload

    def validate(self) -> None:
        if not self.locator_id.startswith("evidence:"):
            raise ProjectInventoryError("Evidence locator ID is not host-owned.")
        if not self.path or "\\" in self.path or self.path.startswith("/"):
            raise ProjectInventoryError(f"Invalid evidence path: {self.path!r}")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ProjectInventoryError(f"Invalid evidence hash for {self.path!r}.")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ProjectInventoryError(f"Invalid evidence size for {self.path!r}.")
        if self.line_start < 0 or self.line_end < 0:
            raise ProjectInventoryError(f"Invalid evidence line range for {self.path!r}.")
        if self.line_start == 0 and self.line_end != 0:
            raise ProjectInventoryError(f"Incomplete evidence line range for {self.path!r}.")
        if self.line_start and self.line_end < self.line_start:
            raise ProjectInventoryError(f"Reversed evidence line range for {self.path!r}.")


@dataclass(frozen=True)
class SourceRoot:
    module_id: str
    source_set: str
    language: str
    path: str
    generated: bool
    test: bool


@dataclass(frozen=True)
class ProjectModule:
    module_id: str
    path: str
    build_files: tuple[str, ...]
    source_sets: tuple[str, ...]
    source_roots: tuple[SourceRoot, ...]
    generated_resource_roots: tuple[str, ...]
    test_roots: tuple[str, ...]
    dependency_ids: tuple[str, ...]
    depends_on_modules: tuple[str, ...]


@dataclass(frozen=True)
class DependencyRecord:
    dependency_id: str
    module_id: str
    configuration: str
    coordinate: str
    raw_coordinate: str
    group: str
    name: str
    version: str
    project_path: str
    resolved: bool
    evidence: EvidenceLocator


@dataclass(frozen=True)
class EntryPointRecord:
    loader: str
    group: str
    value: str
    adapter: str
    evidence: EvidenceLocator


@dataclass(frozen=True)
class ModMetadata:
    loader: str
    path: str
    mod_id: str
    mod_name: str
    mod_version: str
    environment: str
    license: str
    entrypoints: tuple[EntryPointRecord, ...]
    dependency_ids: tuple[str, ...]
    evidence: EvidenceLocator


@dataclass(frozen=True)
class ProjectTarget:
    minecraft_versions: tuple[str, ...]
    loaders: tuple[str, ...]
    loader_versions: tuple[str, ...]
    java_versions: tuple[str, ...]
    mappings: tuple[str, ...]
    gradle_versions: tuple[str, ...]
    evidence: tuple[EvidenceLocator, ...]


@dataclass(frozen=True)
class ComponentRecord:
    component_id: str
    kind: str
    name: str
    locator: str
    content_sha256: str
    module_id: str
    source_set: str
    side: str
    minecraft_versions: tuple[str, ...]
    loaders: tuple[str, ...]
    provides: tuple[str, ...]
    requires: tuple[str, ...]
    evidence: tuple[EvidenceLocator, ...]
    provenance: str = "same_project"
    license_refs: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.component_id.startswith("component:"):
            raise ProjectInventoryError("Component ID is not host-owned.")
        if not self.kind or not self.name or not self.locator:
            raise ProjectInventoryError(f"Incomplete component {self.component_id!r}.")
        if not _SHA256_RE.fullmatch(self.content_sha256):
            raise ProjectInventoryError(f"Invalid component hash for {self.component_id!r}.")
        if not self.evidence:
            raise ProjectInventoryError(f"Component {self.component_id!r} has no evidence.")
        if self.provenance != "same_project":
            raise ProjectInventoryError("Project inventory may only attest same-project provenance.")
        if self.provides != tuple(sorted(set(self.provides))):
            raise ProjectInventoryError(f"Unsorted component provides for {self.component_id!r}.")
        if self.requires != tuple(sorted(set(self.requires))):
            raise ProjectInventoryError(f"Unsorted component requires for {self.component_id!r}.")
        for locator in self.evidence:
            locator.validate()
        if self.content_sha256 not in {item.sha256 for item in self.evidence}:
            raise ProjectInventoryError(
                f"Component {self.component_id!r} is not bound to its content hash."
            )


@dataclass(frozen=True)
class ComponentCatalog:
    schema_version: str
    components: tuple[ComponentRecord, ...]
    catalog_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))

    def validate(self) -> None:
        if self.schema_version != COMPONENT_CATALOG_SCHEMA:
            raise ProjectInventoryError(
                f"Unsupported component catalog schema: {self.schema_version!r}"
            )
        ids = tuple(component.component_id for component in self.components)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ProjectInventoryError("Component catalog IDs must be sorted and unique.")
        for component in self.components:
            component.validate()
        expected = canonical_sha256(
            {
                "schema_version": self.schema_version,
                "components": self.components,
                "catalog_sha256": "",
            }
        )
        if self.catalog_sha256 != expected:
            raise ProjectInventoryError("Component catalog hash does not match its payload.")


@dataclass(frozen=True)
class ProjectInventory:
    schema_version: str
    source_kind: str
    root_name: str
    source_sha256: str
    imported_source_snapshot_sha256: str
    project_snapshot_sha256: str
    manifest: tuple[EvidenceLocator, ...]
    modules: tuple[ProjectModule, ...]
    target: ProjectTarget
    metadata: tuple[ModMetadata, ...]
    entrypoints: tuple[EntryPointRecord, ...]
    namespaces: tuple[str, ...]
    dependencies: tuple[DependencyRecord, ...]
    logical_resource_ids: tuple[str, ...]
    logical_resource_references: tuple[str, ...]
    component_catalog: ComponentCatalog
    warnings: tuple[str, ...]
    inventory_sha256: str

    @property
    def components(self) -> tuple[ComponentRecord, ...]:
        return self.component_catalog.components

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProjectInventory":
        """Reconstruct and fully verify an untrusted serialized inventory."""

        return _project_inventory_from_mapping(value)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ProjectInventoryError(f"Unsupported inventory schema: {self.schema_version!r}")
        if self.source_kind not in {"workspace", "archive"}:
            raise ProjectInventoryError(f"Unsupported inventory source: {self.source_kind!r}")
        for digest in (
            self.source_sha256,
            self.project_snapshot_sha256,
            self.inventory_sha256,
        ):
            if not _SHA256_RE.fullmatch(digest):
                raise ProjectInventoryError("Inventory contains an invalid SHA-256 receipt.")
        if self.imported_source_snapshot_sha256 and not _SHA256_RE.fullmatch(
            self.imported_source_snapshot_sha256
        ):
            raise ProjectInventoryError("Importer snapshot receipt is invalid.")
        manifest_paths = tuple(item.path for item in self.manifest)
        if manifest_paths != tuple(sorted(manifest_paths)) or len(manifest_paths) != len(
            set(manifest_paths)
        ):
            raise ProjectInventoryError("Manifest paths must be sorted and unique.")
        for locator in self.manifest:
            locator.validate()
            if locator.line_start or locator.line_end:
                raise ProjectInventoryError("Manifest evidence must bind complete files.")
        expected_snapshot = canonical_sha256(
            [
                {"path": item.path, "size_bytes": item.size_bytes, "sha256": item.sha256}
                for item in self.manifest
            ]
        )
        if self.project_snapshot_sha256 != expected_snapshot:
            raise ProjectInventoryError("Project snapshot hash does not match its manifest.")
        module_ids = tuple(item.module_id for item in self.modules)
        if module_ids != tuple(sorted(module_ids)) or len(module_ids) != len(set(module_ids)):
            raise ProjectInventoryError("Project modules must be sorted and unique.")
        for values, label in (
            (self.namespaces, "namespaces"),
            (self.logical_resource_ids, "logical resource IDs"),
            (self.logical_resource_references, "logical resource references"),
            (self.target.minecraft_versions, "Minecraft target versions"),
            (self.target.loaders, "loaders"),
            (self.target.loader_versions, "loader versions"),
            (self.target.mappings, "mappings"),
            (self.target.gradle_versions, "Gradle versions"),
        ):
            if values != tuple(sorted(set(values))):
                raise ProjectInventoryError(f"Inventory {label} must be sorted and unique.")
        self.component_catalog.validate()
        manifest_hashes = {item.path: item.sha256 for item in self.manifest}
        evidence_ids = {item.locator_id for item in self.manifest}
        bound_evidence: list[EvidenceLocator] = [*self.target.evidence]
        bound_evidence.extend(item.evidence for item in self.dependencies)
        bound_evidence.extend(item.evidence for item in self.entrypoints)
        bound_evidence.extend(item.evidence for item in self.metadata)
        for component in self.components:
            bound_evidence.extend(component.evidence)
            unknown_license_refs = set(component.license_refs) - evidence_ids
            if unknown_license_refs:
                raise ProjectInventoryError(
                    f"Component has unknown license evidence: {sorted(unknown_license_refs)}"
                )
            for evidence in component.evidence:
                if evidence.path.startswith("@archive/"):
                    if evidence.sha256 != self.source_sha256:
                        raise ProjectInventoryError("Archive evidence is not bound to archive bytes.")
                    continue
                if manifest_hashes.get(evidence.path) != evidence.sha256:
                    raise ProjectInventoryError(
                        f"Component evidence is absent or stale: {evidence.locator}"
                    )
        for evidence in bound_evidence:
            evidence.validate()
            if evidence.path.startswith("@archive/"):
                if evidence.sha256 != self.source_sha256:
                    raise ProjectInventoryError("Archive evidence is not bound to archive bytes.")
            elif manifest_hashes.get(evidence.path) != evidence.sha256:
                raise ProjectInventoryError(f"Inventory evidence is absent or stale: {evidence.locator}")
        dependency_ids = tuple(item.dependency_id for item in self.dependencies)
        if dependency_ids != tuple(sorted(dependency_ids)) or len(dependency_ids) != len(
            set(dependency_ids)
        ):
            raise ProjectInventoryError("Dependency IDs must be sorted and unique.")
        known_dependency_ids = set(dependency_ids)
        for module in self.modules:
            if not set(module.dependency_ids) <= known_dependency_ids:
                raise ProjectInventoryError(f"Module {module.module_id!r} has unknown dependencies.")
        for record in self.metadata:
            if not set(record.dependency_ids) <= known_dependency_ids:
                raise ProjectInventoryError(f"Metadata {record.path!r} has unknown dependencies.")
        expected = canonical_sha256(_inventory_hash_payload(self))
        if self.inventory_sha256 != expected:
            raise ProjectInventoryError("Inventory hash does not match its payload.")


@dataclass(frozen=True)
class _ScannedFile:
    absolute: Path
    path: str
    size_bytes: int
    sha256: str


def _inventory_hash_payload(inventory: ProjectInventory) -> dict[str, Any]:
    payload = inventory.to_dict()
    payload["inventory_sha256"] = ""
    return payload


def _evidence_id(path: str, sha256: str, line_start: int = 0, line_end: int = 0) -> str:
    digest = canonical_sha256(
        {
            "path": path,
            "sha256": sha256,
            "line_start": line_start,
            "line_end": line_end,
        }
    ).removeprefix("sha256:")
    return "evidence:" + digest


def _evidence(file: _ScannedFile, line_start: int = 0, line_end: int = 0) -> EvidenceLocator:
    return EvidenceLocator(
        locator_id=_evidence_id(file.path, file.sha256, line_start, line_end),
        path=file.path,
        sha256=file.sha256,
        size_bytes=file.size_bytes,
        line_start=line_start,
        line_end=line_end,
    )


def _archive_evidence(path: Path, digest: str) -> EvidenceLocator:
    relative = "@archive/" + path.name
    return EvidenceLocator(
        locator_id=_evidence_id(relative, digest),
        path=relative,
        sha256=digest,
        size_bytes=path.stat().st_size,
    )


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
            closed = os.fstat(stream.fileno())
    except OSError as exc:
        raise ProjectInventoryError(f"Could not inspect project file {path}: {exc}") from exc
    if (
        opened.st_size != closed.st_size
        or getattr(opened, "st_mtime_ns", 0) != getattr(closed, "st_mtime_ns", 0)
        or opened.st_size != size
    ):
        raise ProjectInventoryError(f"Project file changed during inventory: {path}")
    return size, "sha256:" + digest.hexdigest()


def _walk_files(root: Path) -> tuple[tuple[_ScannedFile, ...], tuple[str, ...]]:
    result: list[_ScannedFile] = []
    warnings: list[str] = []
    stack = [root]
    casefolded: dict[str, str] = {}
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise ProjectInventoryError(f"Could not enumerate project directory {directory}: {exc}") from exc
        child_directories: list[Path] = []
        for entry in entries:
            path = Path(entry.path)
            try:
                if entry.is_symlink():
                    relative = path.relative_to(root).as_posix()
                    warnings.append(f"symbolic_link_skipped:{relative}")
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name.casefold() in _IGNORED_DIRECTORIES:
                        continue
                    child_directories.append(path)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
            except OSError as exc:
                raise ProjectInventoryError(f"Could not stat project path {path}: {exc}") from exc
            relative = path.relative_to(root).as_posix()
            folded = relative.casefold()
            prior = casefolded.get(folded)
            if prior is not None and prior != relative:
                raise ProjectInventoryError(
                    f"Project contains case-colliding paths: {prior!r}, {relative!r}"
                )
            casefolded[folded] = relative
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError) as exc:
                raise ProjectInventoryError(f"Project path escapes its root: {relative}") from exc
            size, digest = _hash_file(path)
            result.append(_ScannedFile(path, relative, size, digest))
        stack.extend(reversed(child_directories))
    return tuple(sorted(result, key=lambda item: item.path)), tuple(sorted(set(warnings)))


def _read_bound_bytes(file: _ScannedFile) -> bytes:
    try:
        raw = file.absolute.read_bytes()
    except OSError as exc:
        raise ProjectInventoryError(f"Could not read project file {file.path}: {exc}") from exc
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    if digest != file.sha256 or len(raw) != file.size_bytes:
        raise ProjectInventoryError(f"Project file changed after manifest scan: {file.path}")
    return raw


def _read_bound_text(file: _ScannedFile) -> str:
    return _read_bound_bytes(file).decode("utf-8", errors="replace")


def _is_text(file: _ScannedFile) -> bool:
    name = Path(file.path).name.casefold()
    return Path(file.path).suffix.casefold() in _TEXT_SUFFIXES or name in _TEXT_NAMES


def _properties(files: Sequence[_ScannedFile]) -> dict[str, str]:
    values: dict[str, str] = {}
    for file in files:
        if Path(file.path).name.casefold() != "gradle.properties":
            continue
        for raw_line in _read_bound_text(file).splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", "!")):
                continue
            separator = "=" if "=" in line else ":" if ":" in line else ""
            if not separator:
                continue
            key, value = line.split(separator, 1)
            values[key.strip()] = value.strip()
    return values


def _resolve_properties(value: str, properties: Mapping[str, str]) -> tuple[str, bool]:
    result = value.strip()
    patterns = (
        re.compile(r"\$\{(?:project\.)?([A-Za-z_][A-Za-z0-9_.-]*)\}"),
        re.compile(r"(?<![A-Za-z0-9_])\$(?!\{)([A-Za-z_][A-Za-z0-9_.-]*)"),
        re.compile(
            r"(?:project\.)?(?:property|findProperty)\(\s*[\"']([A-Za-z_][A-Za-z0-9_.-]*)[\"']\s*\)"
        ),
        re.compile(
            r"providers\.gradleProperty\(\s*[\"']([A-Za-z_][A-Za-z0-9_.-]*)[\"']\s*\)(?:\.get\(\))?"
        ),
    )
    for _ in range(max(1, len(properties) + 1)):
        before = result
        for pattern in patterns:
            result = pattern.sub(lambda match: properties.get(match.group(1), match.group(0)), result)
        if result == before:
            break
    unresolved = bool(re.search(r"\$\{|(?<![A-Za-z0-9_])\$[A-Za-z_]|(?:find)?property\(", result))
    return result, not unresolved


def _settings_modules(files: Sequence[_ScannedFile]) -> tuple[dict[str, str], str]:
    settings = [
        file
        for file in files
        if Path(file.path).name.casefold() in {"settings.gradle", "settings.gradle.kts"}
    ]
    if not settings:
        return {":": "."}, ""
    root_settings = min(settings, key=lambda item: (item.path.count("/"), item.path))
    base = Path(root_settings.path).parent.as_posix()
    if base == ".":
        base = ""
    text = _read_bound_text(root_settings)
    modules: dict[str, str] = {":": base or "."}
    for match in re.finditer(r"(?m)^\s*include(?!Build)\s*(?:\((?P<paren>[^)]*)\)|(?P<plain>[^\r\n]+))", text):
        arguments = match.group("paren") if match.group("paren") is not None else match.group("plain")
        for raw in _QUOTED.findall(arguments or ""):
            module_id = ":" + raw.strip().strip(":")
            if module_id == ":":
                continue
            relative = raw.strip().strip(":").replace(":", "/")
            modules[module_id] = "/".join(part for part in (base, relative) if part) or "."
    project_dir = re.compile(
        r"project\s*\(\s*[\"'](?P<module>:[^\"']+)[\"']\s*\)\.projectDir\s*=\s*"
        r"(?:file\s*\()?\s*[\"'](?P<path>[^\"']+)[\"']"
    )
    for match in project_dir.finditer(text):
        raw_path = match.group("path").replace("\\", "/").strip("/")
        modules[match.group("module")] = "/".join(
            part for part in (base, raw_path) if part
        ) or "."
    root_name = ""
    name_match = re.search(r"rootProject\.name\s*=\s*[\"']([^\"']+)[\"']", text)
    if name_match:
        root_name = name_match.group(1).strip()
    return dict(sorted(modules.items())), root_name


def _module_for_path(path: str, module_paths: Mapping[str, str]) -> str:
    candidates: list[tuple[int, str]] = []
    for module_id, raw_prefix in module_paths.items():
        prefix = "" if raw_prefix == "." else raw_prefix.rstrip("/")
        if not prefix or path == prefix or path.startswith(prefix + "/"):
            candidates.append((len(prefix), module_id))
    return max(candidates, default=(0, ":"))[1]


def _relative_to_module(path: str, module_path: str) -> str:
    if module_path == ".":
        return path
    if path == module_path:
        return "."
    return path[len(module_path.rstrip("/")) + 1 :]


def _source_root_for_path(module_id: str, module_path: str, path: str) -> SourceRoot | None:
    relative = _relative_to_module(path, module_path)
    parts = relative.split("/")
    if len(parts) < 4 or parts[0] != "src":
        return None
    source_set, language = parts[1], parts[2].casefold()
    if language not in {*_SOURCE_LANGUAGES, "resources"}:
        return None
    root_relative = "/".join(parts[:3])
    root_path = "/".join(part for part in ("" if module_path == "." else module_path, root_relative) if part)
    lowered = (source_set + "/" + root_path).casefold()
    return SourceRoot(
        module_id=module_id,
        source_set=source_set,
        language=language,
        path=root_path,
        generated="generated" in lowered,
        test="test" in source_set.casefold() or "gametest" in lowered,
    )


def _generated_paths(text: str, module_path: str) -> set[str]:
    result: set[str] = set()
    for quoted in _QUOTED.findall(text):
        normalized = quoted.replace("\\", "/").strip("/")
        lowered = normalized.casefold()
        if "generated" not in lowered:
            continue
        if "resource" not in lowered and not normalized.startswith("src/"):
            continue
        if normalized.startswith(("http://", "https://")) or ":" in normalized[:3]:
            continue
        result.add(
            "/".join(part for part in ("" if module_path == "." else module_path, normalized) if part)
        )
    return result


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _dependency_id(
    *, module_id: str, configuration: str, coordinate: str, project_path: str = ""
) -> str:
    digest = canonical_sha256(
        {
            "module_id": module_id,
            "configuration": configuration,
            "coordinate": coordinate,
            "project_path": project_path,
        }
    ).removeprefix("sha256:")
    return "dependency:" + digest


def _coordinate_parts(coordinate: str) -> tuple[str, str, str]:
    parts = coordinate.split(":")
    if len(parts) >= 3:
        return parts[0], parts[1], ":".join(parts[2:])
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return "", coordinate, ""


def _is_dependency_configuration(configuration: str) -> bool:
    lowered = configuration.casefold()
    markers = (
        "api",
        "compile",
        "implementation",
        "include",
        "kapt",
        "minecraft",
        "mappings",
        "processor",
        "runtime",
        "forge",
        "loader",
    )
    return any(marker in lowered for marker in markers)


def _toml_catalog_dependencies(
    file: _ScannedFile,
    module_id: str,
) -> list[DependencyRecord]:
    text = _read_bound_text(file)
    section = ""
    versions: dict[str, str] = {}
    libraries: list[tuple[str, str, str, int]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        section_match = re.fullmatch(r"\[([^]]+)\]", line)
        if section_match:
            section = section_match.group(1).strip().casefold()
            continue
        if not line or line.startswith("#") or "=" not in line:
            continue
        alias, raw_value = (part.strip() for part in line.split("=", 1))
        if section == "versions":
            match = re.fullmatch(r"[\"']([^\"']+)[\"']", raw_value)
            if match:
                versions[alias] = match.group(1)
            continue
        if section != "libraries":
            continue
        module_match = re.search(r"\bmodule\s*=\s*[\"']([^\"']+)[\"']", raw_value)
        group_match = re.search(r"\bgroup\s*=\s*[\"']([^\"']+)[\"']", raw_value)
        name_match = re.search(r"\bname\s*=\s*[\"']([^\"']+)[\"']", raw_value)
        direct_match = re.fullmatch(r"[\"']([^\"']+)[\"']", raw_value)
        module = ""
        if module_match:
            module = module_match.group(1)
        elif group_match and name_match:
            module = f"{group_match.group(1)}:{name_match.group(1)}"
        elif direct_match:
            module = direct_match.group(1)
        if not module:
            continue
        version_match = re.search(r"\bversion\s*=\s*[\"']([^\"']+)[\"']", raw_value)
        ref_match = re.search(r"\bversion\.ref\s*=\s*[\"']([^\"']+)[\"']", raw_value)
        version = version_match.group(1) if version_match else versions.get(ref_match.group(1), "") if ref_match else ""
        coordinate = module + (":" + version if version else "")
        libraries.append((alias, coordinate, raw_value, line_number))
    result: list[DependencyRecord] = []
    for alias, coordinate, raw_value, line_number in libraries:
        group, name, version = _coordinate_parts(coordinate)
        evidence = _evidence(file, line_number, line_number)
        result.append(
            DependencyRecord(
                dependency_id=_dependency_id(
                    module_id=module_id,
                    configuration="version_catalog",
                    coordinate=coordinate,
                ),
                module_id=module_id,
                configuration="version_catalog",
                coordinate=coordinate,
                raw_coordinate=raw_value,
                group=group,
                name=name or alias,
                version=version,
                project_path="",
                resolved=bool(group and name and (version or coordinate.count(":") == 1)),
                evidence=evidence,
            )
        )
    return result


def _gradle_dependencies(
    files: Sequence[_ScannedFile],
    module_paths: Mapping[str, str],
    properties: Mapping[str, str],
) -> tuple[DependencyRecord, ...]:
    records: dict[str, DependencyRecord] = {}
    for file in files:
        name = Path(file.path).name.casefold()
        module_id = _module_for_path(file.path, module_paths)
        if name == "libs.versions.toml":
            for dependency in _toml_catalog_dependencies(file, module_id):
                records.setdefault(dependency.dependency_id, dependency)
            continue
        if name not in {"build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"}:
            continue
        text = _read_bound_text(file)
        for match in _GRADLE_PROJECT_DEPENDENCY.finditer(text):
            configuration = match.group("configuration")
            if not _is_dependency_configuration(configuration):
                continue
            project_path = match.group("project")
            coordinate = "project:" + project_path
            dependency_id = _dependency_id(
                module_id=module_id,
                configuration=configuration,
                coordinate=coordinate,
                project_path=project_path,
            )
            line = _line_number(text, match.start())
            records.setdefault(
                dependency_id,
                DependencyRecord(
                    dependency_id=dependency_id,
                    module_id=module_id,
                    configuration=configuration,
                    coordinate=coordinate,
                    raw_coordinate=project_path,
                    group="project",
                    name=project_path.strip(":").replace(":", "/"),
                    version="",
                    project_path=project_path,
                    resolved=project_path in module_paths,
                    evidence=_evidence(file, line, line),
                ),
            )
        for match in _GRADLE_DEPENDENCY.finditer(text):
            configuration = match.group("configuration")
            raw_coordinate = match.group("coordinate").strip()
            if not _is_dependency_configuration(configuration) or ":" not in raw_coordinate:
                continue
            coordinate, resolved = _resolve_properties(raw_coordinate, properties)
            group, dependency_name, version = _coordinate_parts(coordinate)
            dependency_id = _dependency_id(
                module_id=module_id,
                configuration=configuration,
                coordinate=coordinate,
            )
            line = _line_number(text, match.start())
            records.setdefault(
                dependency_id,
                DependencyRecord(
                    dependency_id=dependency_id,
                    module_id=module_id,
                    configuration=configuration,
                    coordinate=coordinate,
                    raw_coordinate=raw_coordinate,
                    group=group,
                    name=dependency_name,
                    version=version,
                    project_path="",
                    resolved=resolved,
                    evidence=_evidence(file, line, line),
                ),
            )
        for match in _GRADLE_PLUGIN.finditer(text):
            plugin_id = (match.group("id") or "").strip()
            alias = (match.group("alias") or "").strip()
            if not plugin_id and not alias:
                continue
            plugin_name = plugin_id or "libs.plugins." + alias
            raw_version = (match.group("version") or "").strip()
            version, resolved = _resolve_properties(raw_version, properties) if raw_version else ("", True)
            coordinate = "plugin:" + plugin_name + (":" + version if version else "")
            dependency_id = _dependency_id(
                module_id=module_id,
                configuration="plugin",
                coordinate=coordinate,
            )
            line = _line_number(text, match.start())
            records.setdefault(
                dependency_id,
                DependencyRecord(
                    dependency_id=dependency_id,
                    module_id=module_id,
                    configuration="plugin",
                    coordinate=coordinate,
                    raw_coordinate=match.group(0).strip(),
                    group="plugin",
                    name=plugin_name,
                    version=version,
                    project_path="",
                    resolved=resolved,
                    evidence=_evidence(file, line, line),
                ),
            )
    return tuple(sorted(records.values(), key=lambda item: item.dependency_id))


def _metadata_dependency(
    *,
    file: _ScannedFile,
    module_id: str,
    loader: str,
    dependency_name: str,
    version: str,
    configuration: str,
) -> DependencyRecord:
    coordinate = f"{loader}-mod:{dependency_name}:{version}" if version else f"{loader}-mod:{dependency_name}"
    dependency_id = _dependency_id(
        module_id=module_id,
        configuration=configuration,
        coordinate=coordinate,
    )
    return DependencyRecord(
        dependency_id=dependency_id,
        module_id=module_id,
        configuration=configuration,
        coordinate=coordinate,
        raw_coordinate=version,
        group=f"{loader}-mod",
        name=dependency_name,
        version=version,
        project_path="",
        resolved=True,
        evidence=_evidence(file),
    )


def _entrypoint_values(value: Any) -> Iterator[tuple[str, str]]:
    values = value if isinstance(value, list) else [value]
    for item in values:
        if isinstance(item, str) and item.strip():
            yield item.strip(), ""
        elif isinstance(item, Mapping):
            target = str(item.get("value", "")).strip()
            adapter = str(item.get("adapter", "")).strip()
            if target:
                yield target, adapter


def _fabric_or_quilt_metadata(
    file: _ScannedFile,
    module_id: str,
) -> tuple[ModMetadata | None, tuple[DependencyRecord, ...]]:
    try:
        payload = json.loads(_read_bound_text(file))
    except (TypeError, ValueError):
        return None, ()
    if not isinstance(payload, Mapping):
        return None, ()
    name = Path(file.path).name.casefold()
    loader = "quilt" if name == "quilt.mod.json" else "fabric"
    if loader == "quilt":
        quilt_loader = payload.get("quilt_loader")
        core = quilt_loader if isinstance(quilt_loader, Mapping) else {}
        mod_id = str(core.get("id", "")).strip()
        mod_version = str(core.get("version", "")).strip()
        metadata_value = core.get("metadata")
        nested_metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
        mod_name = str(nested_metadata.get("name", "")).strip()
        entrypoint_payload = core.get("entrypoints")
        dependency_payload = core.get("depends")
        license_value = nested_metadata.get("license", "")
        environment = str(core.get("intermediate_mappings", "")).strip()
    else:
        mod_id = str(payload.get("id", "")).strip()
        mod_version = str(payload.get("version", "")).strip()
        mod_name = str(payload.get("name", "")).strip()
        entrypoint_payload = payload.get("entrypoints")
        dependency_payload = payload.get("depends")
        license_value = payload.get("license", "")
        environment = str(payload.get("environment", "")).strip()
    license_text = (
        ",".join(str(item) for item in license_value)
        if isinstance(license_value, list)
        else str(license_value).strip()
    )
    evidence = _evidence(file)
    entrypoints: list[EntryPointRecord] = []
    if isinstance(entrypoint_payload, Mapping):
        for group, raw_values in sorted(entrypoint_payload.items(), key=lambda item: str(item[0])):
            for value, adapter in _entrypoint_values(raw_values):
                entrypoints.append(
                    EntryPointRecord(
                        loader=loader,
                        group=str(group),
                        value=value,
                        adapter=adapter,
                        evidence=evidence,
                    )
                )
    dependencies: list[DependencyRecord] = []
    if isinstance(dependency_payload, Mapping):
        for dependency_name, raw_version in sorted(dependency_payload.items(), key=lambda item: str(item[0])):
            if isinstance(raw_version, list):
                version = " || ".join(str(item) for item in raw_version)
            elif isinstance(raw_version, Mapping):
                version = canonical_json(raw_version)
            else:
                version = str(raw_version)
            dependencies.append(
                _metadata_dependency(
                    file=file,
                    module_id=module_id,
                    loader=loader,
                    dependency_name=str(dependency_name),
                    version=version,
                    configuration="metadata:depends",
                )
            )
    return (
        ModMetadata(
            loader=loader,
            path=file.path,
            mod_id=mod_id,
            mod_name=mod_name,
            mod_version=mod_version,
            environment=environment,
            license=license_text,
            entrypoints=tuple(sorted(entrypoints, key=lambda item: (item.group, item.value, item.adapter))),
            dependency_ids=tuple(sorted(item.dependency_id for item in dependencies)),
            evidence=evidence,
        ),
        tuple(dependencies),
    )


def _toml_value(block: str, key: str) -> str:
    match = re.search(
        rf"(?m)^\s*{re.escape(key)}\s*=\s*[\"']([^\"']*)[\"']",
        block,
    )
    return match.group(1).strip() if match else ""


def _forge_metadata(
    file: _ScannedFile,
    module_id: str,
) -> tuple[tuple[ModMetadata, ...], tuple[DependencyRecord, ...]]:
    text = _read_bound_text(file)
    loader = "neoforge" if Path(file.path).name.casefold() == "neoforge.mods.toml" else "forge"
    header_license = _toml_value(text, "license")
    mod_markers = list(re.finditer(r"(?m)^\s*\[\[mods\]\]\s*$", text))
    metadata: list[ModMetadata] = []
    dependencies: list[DependencyRecord] = []
    for index, marker in enumerate(mod_markers):
        end = mod_markers[index + 1].start() if index + 1 < len(mod_markers) else len(text)
        block = text[marker.end() : end]
        mod_id = _toml_value(block, "modId")
        if not mod_id:
            continue
        local_dependencies: list[DependencyRecord] = []
        dependency_pattern = re.compile(
            rf"(?ms)^\s*\[\[dependencies\.{re.escape(mod_id)}\]\]\s*(.*?)(?=^\s*\[\[|\Z)"
        )
        for dep_match in dependency_pattern.finditer(text):
            dep_block = dep_match.group(1)
            dependency_name = _toml_value(dep_block, "modId")
            if not dependency_name:
                continue
            dependency = _metadata_dependency(
                file=file,
                module_id=module_id,
                loader=loader,
                dependency_name=dependency_name,
                version=_toml_value(dep_block, "versionRange"),
                configuration="metadata:mandatory" if _toml_value(dep_block, "mandatory").casefold() == "true" else "metadata:depends",
            )
            local_dependencies.append(dependency)
            dependencies.append(dependency)
        metadata.append(
            ModMetadata(
                loader=loader,
                path=file.path,
                mod_id=mod_id,
                mod_name=_toml_value(block, "displayName"),
                mod_version=_toml_value(block, "version"),
                environment="",
                license=header_license,
                entrypoints=(),
                dependency_ids=tuple(sorted(item.dependency_id for item in local_dependencies)),
                evidence=_evidence(file),
            )
        )
    return tuple(metadata), tuple(dependencies)


def _metadata_records(
    files: Sequence[_ScannedFile],
    module_paths: Mapping[str, str],
) -> tuple[tuple[ModMetadata, ...], tuple[EntryPointRecord, ...], tuple[DependencyRecord, ...]]:
    metadata: list[ModMetadata] = []
    dependencies: dict[str, DependencyRecord] = {}
    for file in files:
        name = Path(file.path).name.casefold()
        module_id = _module_for_path(file.path, module_paths)
        if name in {"fabric.mod.json", "quilt.mod.json"}:
            record, found_dependencies = _fabric_or_quilt_metadata(file, module_id)
            if record is not None:
                metadata.append(record)
                for dependency in found_dependencies:
                    dependencies.setdefault(dependency.dependency_id, dependency)
        elif name in {"mods.toml", "neoforge.mods.toml"} and "/meta-inf/" in ("/" + file.path.casefold()):
            found_metadata, found_dependencies = _forge_metadata(file, module_id)
            metadata.extend(found_metadata)
            for dependency in found_dependencies:
                dependencies.setdefault(dependency.dependency_id, dependency)
    metadata_tuple = tuple(sorted(metadata, key=lambda item: (item.path, item.mod_id, item.loader)))
    entrypoints = tuple(
        sorted(
            (entry for record in metadata_tuple for entry in record.entrypoints),
            key=lambda item: (item.loader, item.group, item.value, item.adapter),
        )
    )
    return metadata_tuple, entrypoints, tuple(sorted(dependencies.values(), key=lambda item: item.dependency_id))


def _complete_module_paths(
    initial: Mapping[str, str],
    files: Sequence[_ScannedFile],
) -> dict[str, str]:
    result = dict(initial)
    occupied = {path: module_id for module_id, path in result.items()}
    for file in files:
        if Path(file.path).name.casefold() not in {"build.gradle", "build.gradle.kts"}:
            continue
        directory = Path(file.path).parent.as_posix()
        if directory == ".":
            directory = "."
        if directory in occupied:
            continue
        module_id = ":" + directory.replace("/", ":").strip(":")
        if module_id == ":":
            continue
        suffix = 2
        base = module_id
        while module_id in result and result[module_id] != directory:
            module_id = f"{base}_{suffix}"
            suffix += 1
        result[module_id] = directory
        occupied[directory] = module_id
    return dict(sorted(result.items()))


def _build_modules(
    files: Sequence[_ScannedFile],
    module_paths: Mapping[str, str],
    dependencies: Sequence[DependencyRecord],
) -> tuple[ProjectModule, ...]:
    dependency_by_module: dict[str, list[DependencyRecord]] = {}
    for dependency in dependencies:
        dependency_by_module.setdefault(dependency.module_id, []).append(dependency)
    result: list[ProjectModule] = []
    for module_id, module_path in sorted(module_paths.items()):
        owned = [file for file in files if _module_for_path(file.path, module_paths) == module_id]
        build_files = tuple(
            sorted(
                file.path
                for file in owned
                if Path(file.path).name.casefold()
                in {"build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts", "gradle.properties", "libs.versions.toml"}
            )
        )
        source_roots = {
            root
            for file in owned
            if (root := _source_root_for_path(module_id, module_path, file.path)) is not None
        }
        generated = {root.path for root in source_roots if root.generated and root.language == "resources"}
        for file in owned:
            if Path(file.path).name.casefold() not in {"build.gradle", "build.gradle.kts"}:
                continue
            generated.update(_generated_paths(_read_bound_text(file), module_path))
        module_dependencies = dependency_by_module.get(module_id, [])
        result.append(
            ProjectModule(
                module_id=module_id,
                path=module_path,
                build_files=build_files,
                source_sets=tuple(sorted({root.source_set for root in source_roots})),
                source_roots=tuple(sorted(source_roots, key=lambda item: (item.path, item.language))),
                generated_resource_roots=tuple(sorted(generated)),
                test_roots=tuple(sorted({root.path for root in source_roots if root.test})),
                dependency_ids=tuple(sorted(item.dependency_id for item in module_dependencies)),
                depends_on_modules=tuple(
                    sorted({item.project_path for item in module_dependencies if item.project_path})
                ),
            )
        )
    return tuple(result)


def _target_from_evidence(
    files: Sequence[_ScannedFile],
    properties: Mapping[str, str],
    dependencies: Sequence[DependencyRecord],
    metadata: Sequence[ModMetadata],
) -> ProjectTarget:
    minecraft_versions: set[str] = set()
    loaders: set[str] = {item.loader for item in metadata if item.loader}
    loader_versions: set[str] = set()
    java_versions: set[str] = set()
    mappings: set[str] = set()
    gradle_versions: set[str] = set()
    evidence: dict[str, EvidenceLocator] = {}

    normalized_properties = {
        re.sub(r"[^a-z0-9]", "", key.casefold()): value.strip()
        for key, value in properties.items()
        if value.strip()
    }
    for key, value in normalized_properties.items():
        if key in {"minecraftversion", "mcversion"}:
            minecraft_versions.add(value)
        elif key in {
            "loaderversion",
            "fabricloaderversion",
            "quiltloaderversion",
            "neoforgeversion",
            "forgeversion",
            "neoversion",
        }:
            loader_versions.add(value)
        elif key in {"javaversion", "javatarget", "javarelease"}:
            java_versions.add(value)
        elif "mappings" in key or key == "yarnversion":
            mappings.add(value)

    for dependency in dependencies:
        coordinate_lower = dependency.coordinate.casefold()
        group_name = f"{dependency.group}:{dependency.name}".casefold()
        if group_name == "com.mojang:minecraft" or dependency.configuration.casefold() == "minecraft":
            if dependency.version:
                minecraft_versions.add(dependency.version)
        if "fabric-loader" in coordinate_lower:
            loaders.add("fabric")
            if dependency.version:
                loader_versions.add(dependency.version)
        if "quilt-loader" in coordinate_lower:
            loaders.add("quilt")
            if dependency.version:
                loader_versions.add(dependency.version)
        if dependency.name.casefold() == "neoforge" or "net.neoforged:neoforge" in coordinate_lower:
            loaders.add("neoforge")
            if dependency.version:
                loader_versions.add(dependency.version)
        elif dependency.name.casefold() == "forge" or "net.minecraftforge:forge" in coordinate_lower:
            loaders.add("forge")
            if dependency.version:
                loader_versions.add(dependency.version)
        if dependency.configuration.casefold() == "mappings" or "yarn" in coordinate_lower:
            mappings.add(dependency.coordinate)
        evidence[dependency.evidence.locator_id] = dependency.evidence

    for record in metadata:
        evidence[record.evidence.locator_id] = record.evidence
        for dependency_id in record.dependency_ids:
            dependency = next(
                (item for item in dependencies if item.dependency_id == dependency_id),
                None,
            )
            if dependency is not None and dependency.name.casefold() == "minecraft" and dependency.version:
                minecraft_versions.add(dependency.version)

    java_patterns = (
        re.compile(r"options\.release(?:\.set)?\s*\(?\s*(?:JavaLanguageVersion\.of\s*\()?\s*(\d+)", re.IGNORECASE),
        re.compile(r"JavaVersion\.VERSION_(\d+)", re.IGNORECASE),
        re.compile(r"(?:java|jvm)Toolchain\s*\(?\s*(\d+)", re.IGNORECASE),
        re.compile(r"JavaLanguageVersion\.of\s*\(\s*(\d+)\s*\)", re.IGNORECASE),
        re.compile(r"jvmTarget\s*=\s*[\"']?(\d+)", re.IGNORECASE),
    )
    for file in files:
        name = Path(file.path).name.casefold()
        if name == "gradle.properties":
            evidence[_evidence(file).locator_id] = _evidence(file)
        if name in {"build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"}:
            text = _read_bound_text(file)
            lowered = text.casefold()
            if "fabric-loom" in lowered or "fabricmc" in lowered:
                loaders.add("fabric")
            if "org.quiltmc" in lowered or "quilt-loom" in lowered:
                loaders.add("quilt")
            if "net.neoforged" in lowered or "neoforge" in lowered:
                loaders.add("neoforge")
            if "net.minecraftforge" in lowered or "forgegradle" in lowered:
                loaders.add("forge")
            if "officialmojangmappings" in lowered:
                mappings.add("mojang")
            for pattern in java_patterns:
                java_versions.update(match.group(1) for match in pattern.finditer(text))
            evidence[_evidence(file).locator_id] = _evidence(file)
        if file.path.casefold().endswith("gradle/wrapper/gradle-wrapper.properties"):
            text = _read_bound_text(file)
            match = re.search(r"gradle-([0-9][0-9A-Za-z_.-]*)-(?:bin|all)\.zip", text)
            if match:
                gradle_versions.add(match.group(1))
            evidence[_evidence(file).locator_id] = _evidence(file)

    return ProjectTarget(
        minecraft_versions=tuple(sorted(minecraft_versions)),
        loaders=tuple(sorted(loaders)),
        loader_versions=tuple(sorted(loader_versions)),
        java_versions=tuple(sorted(java_versions, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))),
        mappings=tuple(sorted(mappings)),
        gradle_versions=tuple(sorted(gradle_versions)),
        evidence=tuple(sorted(evidence.values(), key=lambda item: item.locator_id)),
    )


def _resource_identity(path: str) -> tuple[tuple[str, ...], str, str]:
    parts = path.split("/")
    marker_index = next((index for index, part in enumerate(parts) if part in {"assets", "data"}), -1)
    if marker_index < 0 or len(parts) <= marker_index + 2:
        return (), "", ""
    scope = parts[marker_index]
    namespace = parts[marker_index + 1].casefold()
    relative_parts = parts[marker_index + 2 :]
    filename = relative_parts[-1]
    suffix = Path(filename).suffix
    if suffix:
        relative_parts[-1] = filename[: -len(suffix)]
    relative = "/".join(relative_parts).casefold()
    category, _, remainder = relative.partition("/")
    kind_map = {
        "advancements": "advancement",
        "blockstates": "blockstate",
        "functions": "function",
        "lang": "lang",
        "loot_tables": "loot_table",
        "models": "model",
        "recipes": "recipe",
        "tags": "tag",
        "textures": "texture",
        "worldgen": "worldgen",
    }
    kind = kind_map.get(category, "resource")
    semantic_path = remainder if remainder else category
    provides = {
        f"resource:{scope}:{namespace}:{relative}",
        f"{kind}:{namespace}:{semantic_path}",
    }
    return tuple(sorted(provides)), namespace, kind


def _resource_references(text: str) -> tuple[str, ...]:
    references = {match.group(1).casefold() for match in _RESOURCE_ID.finditer(text)}
    references.update(
        f"{match.group(1).casefold()}:{match.group(2).casefold()}"
        for match in _TWO_PART_IDENTIFIER.finditer(text)
    )
    for match in re.finditer(r"[\"']parent[\"']\s*:\s*[\"']([^\"':]+/[^\"']+)[\"']", text):
        references.add("minecraft:" + match.group(1).casefold())
    return tuple(sorted(references))


def _snake_case_identifier(value: str) -> str:
    """Return one deterministic semantic token; this is not fuzzy matching."""

    separated = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated)
    separated = re.sub(r"[^A-Za-z0-9]+", "_", separated).strip("_").casefold()
    return re.sub(r"_+", "_", separated)


def _capability_aliases(name: str) -> tuple[str, ...]:
    """Derive exact host-owned aliases from a symbol or logical resource name."""

    candidates: list[str] = []
    if "#" in name:
        owner, member = name.rsplit("#", 1)
        owner_name = owner.rsplit(".", 1)[-1]
        candidates.extend((member, owner_name + "_" + member))
    else:
        tail = name.rsplit(":", 1)[-1].rsplit("/", 1)[-1].rsplit(".", 1)[-1]
        candidates.append(tail)
    aliases = {
        "capability:" + normalized
        for candidate in candidates
        if (normalized := _snake_case_identifier(candidate))
    }
    return tuple(sorted(aliases))


def _test_capability_aliases(name: str) -> tuple[str, ...]:
    """Bind conventional test owners to the exact subject name they verify."""

    owner = name.split("#", 1)[0]
    prefix, separator, simple = owner.rpartition(".")
    subject = re.sub(r"(?:GameTests?|Tests?)$", "", simple)
    if not subject or subject == simple:
        return ()
    qualified = f"{prefix}{separator}{subject}" if prefix else subject
    return _capability_aliases(qualified)


def _component_id(*, kind: str, name: str, locator: str, module_id: str) -> str:
    digest = canonical_sha256(
        {"kind": kind, "name": name, "locator": locator, "module_id": module_id}
    ).removeprefix("sha256:")
    return f"component:{kind}:{digest}"


def _component(
    *,
    kind: str,
    name: str,
    evidence: EvidenceLocator,
    module_id: str,
    source_set: str,
    side: str,
    target: ProjectTarget,
    provides: Iterable[str],
    requires: Iterable[str],
    license_refs: tuple[str, ...],
) -> ComponentRecord:
    locator = evidence.locator
    return ComponentRecord(
        component_id=_component_id(
            kind=kind,
            name=name,
            locator=locator,
            module_id=module_id,
        ),
        kind=kind,
        name=name,
        locator=locator,
        content_sha256=evidence.sha256,
        module_id=module_id,
        source_set=source_set,
        side=side,
        minecraft_versions=target.minecraft_versions,
        loaders=target.loaders,
        provides=tuple(sorted({str(item).strip() for item in provides if str(item).strip()})),
        requires=tuple(sorted({str(item).strip() for item in requires if str(item).strip()})),
        evidence=(evidence,),
        license_refs=license_refs,
    )


def _source_location(file: _ScannedFile, module_paths: Mapping[str, str]) -> tuple[str, str, str]:
    module_id = _module_for_path(file.path, module_paths)
    root = _source_root_for_path(module_id, module_paths[module_id], file.path)
    if root is None:
        return module_id, "", "common"
    lowered = root.source_set.casefold()
    side = "client" if "client" in lowered else "server" if "server" in lowered else "common"
    return module_id, root.source_set, side


def _source_components(
    files: Sequence[_ScannedFile],
    module_paths: Mapping[str, str],
    target: ProjectTarget,
    license_refs: tuple[str, ...],
) -> tuple[list[ComponentRecord], set[str], set[str]]:
    components: list[ComponentRecord] = []
    namespaces: set[str] = set()
    references: set[str] = set()
    for file in files:
        suffix = Path(file.path).suffix.casefold()
        if suffix not in {".java", ".kt"}:
            continue
        text = _read_bound_text(file)
        package_match = _PACKAGE.search(text)
        package = package_match.group(1) if package_match else ""
        if package:
            namespaces.add(package)
        imports = {"symbol:" + item.rstrip(".*") for item in _IMPORT.findall(text)}
        resource_refs = set(_resource_references(text))
        references.update(resource_refs)
        requires = imports | {"resource_ref:" + item for item in resource_refs}
        module_id, source_set, side = _source_location(file, module_paths)
        is_test = "test" in source_set.casefold() or Path(file.path).stem.endswith("Test")
        owner = ""
        found = False
        lines = text.splitlines()
        for line_number, line in enumerate(lines, start=1):
            type_match = _JAVA_TYPE.search(line) if suffix == ".java" else _KOTLIN_TYPE.search(line)
            if type_match:
                simple_name = type_match.group("name")
                owner = f"{package}.{simple_name}" if package else simple_name
                evidence = _evidence(file, line_number, line_number)
                symbol = "symbol:" + owner
                provides = {symbol, "namespace:" + package} if package else {symbol}
                provides.update(_capability_aliases(owner))
                if is_test:
                    provides.add("test:" + owner)
                    provides.update(_test_capability_aliases(owner))
                components.append(
                    _component(
                        kind="test" if is_test else "symbol",
                        name=owner,
                        evidence=evidence,
                        module_id=module_id,
                        source_set=source_set,
                        side=side,
                        target=target,
                        provides=provides,
                        requires=requires,
                        license_refs=license_refs,
                    )
                )
                found = True
                continue
            callable_match = _JAVA_METHOD.match(line) if suffix == ".java" else _KOTLIN_CALLABLE.match(line)
            if callable_match and callable_match.group("name") not in {"if", "for", "while", "switch", "catch"}:
                callable_name = callable_match.group("name")
                qualified = (owner or package or Path(file.path).stem) + "#" + callable_name
                evidence = _evidence(file, line_number, line_number)
                provides = {"symbol:" + qualified, *_capability_aliases(qualified)}
                if is_test or "@Test" in line or "@GameTest" in line:
                    provides.add("test:" + qualified)
                    provides.update(_test_capability_aliases(qualified))
                components.append(
                    _component(
                        kind="test" if is_test else "symbol",
                        name=qualified,
                        evidence=evidence,
                        module_id=module_id,
                        source_set=source_set,
                        side=side,
                        target=target,
                        provides=provides,
                        requires=requires,
                        license_refs=license_refs,
                    )
                )
                found = True
                continue
            if suffix == ".java":
                field_match = _JAVA_FIELD.match(line)
                if field_match:
                    field_name = field_match.group("name")
                    qualified = (owner or package or Path(file.path).stem) + "#" + field_name
                    components.append(
                        _component(
                            kind="symbol",
                            name=qualified,
                            evidence=_evidence(file, line_number, line_number),
                            module_id=module_id,
                            source_set=source_set,
                            side=side,
                            target=target,
                            provides={"symbol:" + qualified, *_capability_aliases(qualified)},
                            requires=requires,
                            license_refs=license_refs,
                        )
                    )
                    found = True
        if not found:
            name = f"{package}.{Path(file.path).stem}" if package else Path(file.path).stem
            components.append(
                _component(
                    kind="test" if is_test else "symbol",
                    name=name,
                    evidence=_evidence(file),
                    module_id=module_id,
                    source_set=source_set,
                    side=side,
                    target=target,
                    provides={
                        ("test-file:" if is_test else "source-file:") + file.path,
                        *_capability_aliases(name),
                        *(_test_capability_aliases(name) if is_test else ()),
                    },
                    requires=requires,
                    license_refs=license_refs,
                )
            )
    return components, namespaces, references


def _resource_components(
    files: Sequence[_ScannedFile],
    module_paths: Mapping[str, str],
    target: ProjectTarget,
    license_refs: tuple[str, ...],
) -> tuple[list[ComponentRecord], set[str], set[str], set[str]]:
    components: list[ComponentRecord] = []
    namespaces: set[str] = set()
    logical_ids: set[str] = set()
    references: set[str] = set()
    for file in files:
        provides, namespace, kind = _resource_identity(file.path)
        if not provides:
            continue
        namespaces.add(namespace)
        logical_ids.update(provides)
        semantic_provides = {*provides}
        for logical_id in provides:
            semantic_provides.update(_capability_aliases(logical_id))
        found_references: tuple[str, ...] = ()
        if _is_text(file):
            found_references = _resource_references(_read_bound_text(file))
            references.update(found_references)
        module_id, source_set, side = _source_location(file, module_paths)
        components.append(
            _component(
                kind="resource",
                name=provides[0],
                evidence=_evidence(file),
                module_id=module_id,
                source_set=source_set,
                side=side,
                target=target,
                provides=semantic_provides,
                requires={"resource_ref:" + item for item in found_references},
                license_refs=license_refs,
            )
        )
    return components, namespaces, logical_ids, references


def _config_components(
    files: Sequence[_ScannedFile],
    module_paths: Mapping[str, str],
    modules: Sequence[ProjectModule],
    dependencies: Sequence[DependencyRecord],
    metadata: Sequence[ModMetadata],
    target: ProjectTarget,
    license_refs: tuple[str, ...],
) -> list[ComponentRecord]:
    components: list[ComponentRecord] = []
    dependencies_by_module: dict[str, list[DependencyRecord]] = {}
    for dependency in dependencies:
        dependencies_by_module.setdefault(dependency.module_id, []).append(dependency)
        components.append(
            _component(
                kind="dependency",
                name=dependency.coordinate,
                evidence=dependency.evidence,
                module_id=dependency.module_id,
                source_set="",
                side="build",
                target=target,
                provides={"dependency:" + dependency.coordinate},
                requires={"module:" + dependency.project_path} if dependency.project_path else (),
                license_refs=license_refs,
            )
        )
    modules_by_id = {module.module_id: module for module in modules}
    metadata_by_path = {record.path: record for record in metadata}
    for file in files:
        name = Path(file.path).name.casefold()
        module_id = _module_for_path(file.path, module_paths)
        module = modules_by_id.get(module_id)
        if name in {
            "build.gradle",
            "build.gradle.kts",
            "settings.gradle",
            "settings.gradle.kts",
            "gradle.properties",
            "libs.versions.toml",
        } or file.path.casefold().endswith("gradle/wrapper/gradle-wrapper.properties"):
            provides = {"build_config:" + file.path, "module:" + module_id}
            if module is not None:
                provides.update("source_set:" + module_id + ":" + item for item in module.source_sets)
                provides.update("generated_resources:" + item for item in module.generated_resource_roots)
            components.append(
                _component(
                    kind="build_config",
                    name=file.path,
                    evidence=_evidence(file),
                    module_id=module_id,
                    source_set="",
                    side="build",
                    target=target,
                    provides=provides,
                    requires={"dependency:" + item.coordinate for item in dependencies_by_module.get(module_id, [])},
                    license_refs=license_refs,
                )
            )
        record = metadata_by_path.get(file.path)
        if record is not None:
            provides = {
                f"metadata:{record.loader}:{record.mod_id}",
                "namespace:" + record.mod_id,
            }
            provides.update(
                f"entrypoint:{entry.loader}:{entry.group}:{entry.value}"
                for entry in record.entrypoints
            )
            requires = {"symbol:" + entry.value for entry in record.entrypoints}
            requires.update(
                "dependency:" + dependency.coordinate
                for dependency in dependencies
                if dependency.dependency_id in record.dependency_ids
            )
            components.append(
                _component(
                    kind="build_config",
                    name=file.path,
                    evidence=record.evidence,
                    module_id=module_id,
                    source_set="resources",
                    side=record.environment or "common",
                    target=target,
                    provides=provides,
                    requires=requires,
                    license_refs=license_refs,
                )
            )
        lowered_path = file.path.casefold()
        release_name = name in {
            "changelog.md",
            "jenkinsfile",
            "license",
            "license.md",
            "license.txt",
            "notice",
            "notice.txt",
        }
        release_path = (
            lowered_path.startswith(".github/workflows/")
            or name in {"azure-pipelines.yml", ".gitlab-ci.yml"}
            or "publish" in name
            or "/release" in lowered_path
        )
        if release_name or release_path:
            components.append(
                _component(
                    kind="release_config",
                    name=file.path,
                    evidence=_evidence(file),
                    module_id=module_id,
                    source_set="",
                    side="build",
                    target=target,
                    provides={"release_config:" + file.path},
                    requires={"module:" + module_id},
                    license_refs=license_refs,
                )
            )
    return components


def _deduplicate_components(components: Iterable[ComponentRecord]) -> tuple[ComponentRecord, ...]:
    result: dict[str, ComponentRecord] = {}
    for component in components:
        prior = result.get(component.component_id)
        if prior is not None and prior != component:
            raise ProjectInventoryError(f"Component ID collision: {component.component_id}")
        result[component.component_id] = component
    return tuple(sorted(result.values(), key=lambda item: item.component_id))


def _build_component_catalog(components: Iterable[ComponentRecord]) -> ComponentCatalog:
    ordered = _deduplicate_components(components)
    catalog = ComponentCatalog(
        schema_version=COMPONENT_CATALOG_SCHEMA,
        components=ordered,
        catalog_sha256="",
    )
    catalog = replace(
        catalog,
        catalog_sha256=canonical_sha256(
            {
                "schema_version": catalog.schema_version,
                "components": catalog.components,
                "catalog_sha256": "",
            }
        ),
    )
    catalog.validate()
    return catalog


def _inspect_project_inventory(
    root: Path,
    *,
    source_kind: str,
    source_sha256: str,
    imported_source_snapshot_sha256: str,
    extra_warnings: Iterable[str] = (),
) -> ProjectInventory:
    files, scan_warnings = _walk_files(root)
    manifest = tuple(_evidence(file) for file in files)
    project_snapshot_sha256 = canonical_sha256(
        [
            {"path": item.path, "size_bytes": item.size_bytes, "sha256": item.sha256}
            for item in manifest
        ]
    )
    properties = _properties(files)
    initial_module_paths, settings_root_name = _settings_modules(files)
    module_paths = _complete_module_paths(initial_module_paths, files)
    gradle_dependencies = _gradle_dependencies(files, module_paths, properties)
    metadata, entrypoints, metadata_dependencies = _metadata_records(files, module_paths)
    dependency_map = {item.dependency_id: item for item in (*gradle_dependencies, *metadata_dependencies)}
    dependencies = tuple(sorted(dependency_map.values(), key=lambda item: item.dependency_id))
    modules = _build_modules(files, module_paths, dependencies)
    target = _target_from_evidence(files, properties, dependencies, metadata)
    license_refs = tuple(
        sorted(
            _evidence(file).locator_id
            for file in files
            if Path(file.path).name.casefold()
            in {"license", "license.md", "license.txt", "copying", "copying.md", "notice", "notice.txt"}
        )
    )
    source_components, source_namespaces, source_references = _source_components(
        files, module_paths, target, license_refs
    )
    resource_components, resource_namespaces, logical_ids, resource_references = _resource_components(
        files, module_paths, target, license_refs
    )
    config_components = _config_components(
        files,
        module_paths,
        modules,
        dependencies,
        metadata,
        target,
        license_refs,
    )
    component_catalog = _build_component_catalog(
        (*config_components, *source_components, *resource_components)
    )
    namespaces = set(source_namespaces) | set(resource_namespaces)
    namespaces.update(record.mod_id for record in metadata if record.mod_id)
    actual_root_name = settings_root_name or root.name
    effective_source_sha256 = source_sha256 or project_snapshot_sha256
    inventory = ProjectInventory(
        schema_version=SCHEMA_VERSION,
        source_kind=source_kind,
        root_name=actual_root_name,
        source_sha256=effective_source_sha256,
        imported_source_snapshot_sha256=imported_source_snapshot_sha256,
        project_snapshot_sha256=project_snapshot_sha256,
        manifest=manifest,
        modules=modules,
        target=target,
        metadata=metadata,
        entrypoints=entrypoints,
        namespaces=tuple(sorted(namespaces)),
        dependencies=dependencies,
        logical_resource_ids=tuple(sorted(logical_ids)),
        logical_resource_references=tuple(sorted(source_references | resource_references)),
        component_catalog=component_catalog,
        warnings=tuple(sorted(set((*scan_warnings, *extra_warnings)))),
        inventory_sha256="",
    )
    inventory = replace(inventory, inventory_sha256=canonical_sha256(_inventory_hash_payload(inventory)))
    inventory.validate()
    return inventory


def inspect_project_inventory(project_root: str | Path) -> ProjectInventory:
    """Inspect a workspace tree as immutable planning evidence without executing it."""

    root = Path(project_root).expanduser()
    try:
        if root.is_symlink() or not root.is_dir():
            raise ProjectInventoryError(f"Project root must be a regular directory: {root}")
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ProjectInventoryError(f"Project root cannot be resolved: {root}") from exc
    return _inspect_project_inventory(
        resolved,
        source_kind="workspace",
        source_sha256="",
        imported_source_snapshot_sha256="",
    )


def inspect_existing_archive_inventory(archive_path: str | Path) -> ProjectInventory:
    """Safely validate, extract, and inventory an existing project archive.

    Extraction is delegated to the fail-closed existing-project importer and is
    confined to a temporary directory.  Neither this function nor the importer
    executes any extracted file or build script.
    """

    from .importer import inspect_existing_project_archive

    archive = Path(archive_path).expanduser()
    with tempfile.TemporaryDirectory(prefix="mmm-project-inventory-") as temporary:
        report = inspect_existing_project_archive(archive, extract_root=temporary)
        if not report.extracted_to:
            raise ProjectInventoryError("Safe importer did not produce an inspectable extraction.")
        inventory = _inspect_project_inventory(
            Path(report.extracted_to).resolve(strict=True),
            source_kind="archive",
            source_sha256=report.archive_sha256,
            imported_source_snapshot_sha256=report.source_snapshot_hash,
            extra_warnings=report.warnings,
        )
        if inventory.target.minecraft_versions or inventory.target.loaders:
            return inventory
        # A distribution-only archive can hide metadata inside a JAR.  The safe
        # importer has already parsed that ZIP member as data.  Bind these summary
        # facts to the exact outer archive bytes rather than inventing file evidence.
        if not report.minecraft_versions and not report.loader:
            return inventory
        archive_locator = _archive_evidence(archive.resolve(strict=True), report.archive_sha256)
        target = replace(
            inventory.target,
            minecraft_versions=tuple(sorted(set(report.minecraft_versions))),
            loaders=(report.loader,) if report.loader else (),
            evidence=tuple(sorted((*inventory.target.evidence, archive_locator), key=lambda item: item.locator_id)),
        )
        provides = {
            "archive_metadata:" + (report.mod_id or archive.name),
            *( {"namespace:" + report.mod_id} if report.mod_id else set() ),
        }
        archive_component = _component(
            kind="build_config",
            name=report.fabric_metadata_paths[0] if report.fabric_metadata_paths else archive.name,
            evidence=archive_locator,
            module_id=":",
            source_set="distribution",
            side="common",
            target=target,
            provides=provides,
            requires=(),
            license_refs=(),
        )
        catalog = _build_component_catalog((*inventory.components, archive_component))
        augmented = replace(inventory, target=target, component_catalog=catalog, inventory_sha256="")
        augmented = replace(augmented, inventory_sha256=canonical_sha256(_inventory_hash_payload(augmented)))
        augmented.validate()
        return augmented


def _strict_mapping(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectInventoryError(f"{label} must be an object.")
    actual = {str(key) for key in value}
    missing = fields - actual
    unknown = actual - fields
    if missing:
        raise ProjectInventoryError(f"{label} is missing fields: {sorted(missing)}")
    if unknown:
        raise ProjectInventoryError(f"{label} has unknown fields: {sorted(unknown)}")
    return value


def _strict_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ProjectInventoryError(f"{label} must be an array.")
    return value


def _strict_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ProjectInventoryError(f"{label} must be a string.")
    return value


def _strict_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ProjectInventoryError(f"{label} must be an integer.")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ProjectInventoryError(f"{label} must be a boolean.")
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    return tuple(_strict_string(item, f"{label}[]") for item in _strict_sequence(value, label))


def _evidence_from_mapping(value: Any, label: str) -> EvidenceLocator:
    payload = _strict_mapping(
        value,
        {"locator_id", "path", "sha256", "size_bytes", "line_start", "line_end"},
        label,
    )
    result = EvidenceLocator(
        locator_id=_strict_string(payload["locator_id"], label + ".locator_id"),
        path=_strict_string(payload["path"], label + ".path"),
        sha256=_strict_string(payload["sha256"], label + ".sha256"),
        size_bytes=_strict_int(payload["size_bytes"], label + ".size_bytes"),
        line_start=_strict_int(payload["line_start"], label + ".line_start"),
        line_end=_strict_int(payload["line_end"], label + ".line_end"),
    )
    result.validate()
    expected_id = _evidence_id(
        result.path,
        result.sha256,
        result.line_start,
        result.line_end,
    )
    if result.locator_id != expected_id:
        raise ProjectInventoryError(f"{label} locator ID does not match its evidence payload.")
    return result


def _source_root_from_mapping(value: Any, label: str) -> SourceRoot:
    payload = _strict_mapping(
        value,
        {"module_id", "source_set", "language", "path", "generated", "test"},
        label,
    )
    return SourceRoot(
        module_id=_strict_string(payload["module_id"], label + ".module_id"),
        source_set=_strict_string(payload["source_set"], label + ".source_set"),
        language=_strict_string(payload["language"], label + ".language"),
        path=_strict_string(payload["path"], label + ".path"),
        generated=_strict_bool(payload["generated"], label + ".generated"),
        test=_strict_bool(payload["test"], label + ".test"),
    )


def _module_from_mapping(value: Any, label: str) -> ProjectModule:
    payload = _strict_mapping(
        value,
        {
            "module_id",
            "path",
            "build_files",
            "source_sets",
            "source_roots",
            "generated_resource_roots",
            "test_roots",
            "dependency_ids",
            "depends_on_modules",
        },
        label,
    )
    roots = tuple(
        _source_root_from_mapping(item, f"{label}.source_roots[{index}]")
        for index, item in enumerate(_strict_sequence(payload["source_roots"], label + ".source_roots"))
    )
    return ProjectModule(
        module_id=_strict_string(payload["module_id"], label + ".module_id"),
        path=_strict_string(payload["path"], label + ".path"),
        build_files=_string_tuple(payload["build_files"], label + ".build_files"),
        source_sets=_string_tuple(payload["source_sets"], label + ".source_sets"),
        source_roots=roots,
        generated_resource_roots=_string_tuple(
            payload["generated_resource_roots"], label + ".generated_resource_roots"
        ),
        test_roots=_string_tuple(payload["test_roots"], label + ".test_roots"),
        dependency_ids=_string_tuple(payload["dependency_ids"], label + ".dependency_ids"),
        depends_on_modules=_string_tuple(
            payload["depends_on_modules"], label + ".depends_on_modules"
        ),
    )


def _dependency_from_mapping(value: Any, label: str) -> DependencyRecord:
    payload = _strict_mapping(
        value,
        {
            "dependency_id",
            "module_id",
            "configuration",
            "coordinate",
            "raw_coordinate",
            "group",
            "name",
            "version",
            "project_path",
            "resolved",
            "evidence",
        },
        label,
    )
    result = DependencyRecord(
        dependency_id=_strict_string(payload["dependency_id"], label + ".dependency_id"),
        module_id=_strict_string(payload["module_id"], label + ".module_id"),
        configuration=_strict_string(payload["configuration"], label + ".configuration"),
        coordinate=_strict_string(payload["coordinate"], label + ".coordinate"),
        raw_coordinate=_strict_string(payload["raw_coordinate"], label + ".raw_coordinate"),
        group=_strict_string(payload["group"], label + ".group"),
        name=_strict_string(payload["name"], label + ".name"),
        version=_strict_string(payload["version"], label + ".version"),
        project_path=_strict_string(payload["project_path"], label + ".project_path"),
        resolved=_strict_bool(payload["resolved"], label + ".resolved"),
        evidence=_evidence_from_mapping(payload["evidence"], label + ".evidence"),
    )
    expected_id = _dependency_id(
        module_id=result.module_id,
        configuration=result.configuration,
        coordinate=result.coordinate,
        project_path=result.project_path,
    )
    if result.dependency_id != expected_id:
        raise ProjectInventoryError(f"{label} dependency ID does not match its payload.")
    return result


def _entrypoint_from_mapping(value: Any, label: str) -> EntryPointRecord:
    payload = _strict_mapping(value, {"loader", "group", "value", "adapter", "evidence"}, label)
    return EntryPointRecord(
        loader=_strict_string(payload["loader"], label + ".loader"),
        group=_strict_string(payload["group"], label + ".group"),
        value=_strict_string(payload["value"], label + ".value"),
        adapter=_strict_string(payload["adapter"], label + ".adapter"),
        evidence=_evidence_from_mapping(payload["evidence"], label + ".evidence"),
    )


def _metadata_from_mapping(value: Any, label: str) -> ModMetadata:
    payload = _strict_mapping(
        value,
        {
            "loader",
            "path",
            "mod_id",
            "mod_name",
            "mod_version",
            "environment",
            "license",
            "entrypoints",
            "dependency_ids",
            "evidence",
        },
        label,
    )
    entrypoints = tuple(
        _entrypoint_from_mapping(item, f"{label}.entrypoints[{index}]")
        for index, item in enumerate(_strict_sequence(payload["entrypoints"], label + ".entrypoints"))
    )
    return ModMetadata(
        loader=_strict_string(payload["loader"], label + ".loader"),
        path=_strict_string(payload["path"], label + ".path"),
        mod_id=_strict_string(payload["mod_id"], label + ".mod_id"),
        mod_name=_strict_string(payload["mod_name"], label + ".mod_name"),
        mod_version=_strict_string(payload["mod_version"], label + ".mod_version"),
        environment=_strict_string(payload["environment"], label + ".environment"),
        license=_strict_string(payload["license"], label + ".license"),
        entrypoints=entrypoints,
        dependency_ids=_string_tuple(payload["dependency_ids"], label + ".dependency_ids"),
        evidence=_evidence_from_mapping(payload["evidence"], label + ".evidence"),
    )


def _target_from_mapping(value: Any, label: str) -> ProjectTarget:
    payload = _strict_mapping(
        value,
        {
            "minecraft_versions",
            "loaders",
            "loader_versions",
            "java_versions",
            "mappings",
            "gradle_versions",
            "evidence",
        },
        label,
    )
    return ProjectTarget(
        minecraft_versions=_string_tuple(payload["minecraft_versions"], label + ".minecraft_versions"),
        loaders=_string_tuple(payload["loaders"], label + ".loaders"),
        loader_versions=_string_tuple(payload["loader_versions"], label + ".loader_versions"),
        java_versions=_string_tuple(payload["java_versions"], label + ".java_versions"),
        mappings=_string_tuple(payload["mappings"], label + ".mappings"),
        gradle_versions=_string_tuple(payload["gradle_versions"], label + ".gradle_versions"),
        evidence=tuple(
            _evidence_from_mapping(item, f"{label}.evidence[{index}]")
            for index, item in enumerate(_strict_sequence(payload["evidence"], label + ".evidence"))
        ),
    )


def _component_from_mapping(value: Any, label: str) -> ComponentRecord:
    payload = _strict_mapping(
        value,
        {
            "component_id",
            "kind",
            "name",
            "locator",
            "content_sha256",
            "module_id",
            "source_set",
            "side",
            "minecraft_versions",
            "loaders",
            "provides",
            "requires",
            "evidence",
            "provenance",
            "license_refs",
        },
        label,
    )
    result = ComponentRecord(
        component_id=_strict_string(payload["component_id"], label + ".component_id"),
        kind=_strict_string(payload["kind"], label + ".kind"),
        name=_strict_string(payload["name"], label + ".name"),
        locator=_strict_string(payload["locator"], label + ".locator"),
        content_sha256=_strict_string(payload["content_sha256"], label + ".content_sha256"),
        module_id=_strict_string(payload["module_id"], label + ".module_id"),
        source_set=_strict_string(payload["source_set"], label + ".source_set"),
        side=_strict_string(payload["side"], label + ".side"),
        minecraft_versions=_string_tuple(
            payload["minecraft_versions"], label + ".minecraft_versions"
        ),
        loaders=_string_tuple(payload["loaders"], label + ".loaders"),
        provides=_string_tuple(payload["provides"], label + ".provides"),
        requires=_string_tuple(payload["requires"], label + ".requires"),
        evidence=tuple(
            _evidence_from_mapping(item, f"{label}.evidence[{index}]")
            for index, item in enumerate(_strict_sequence(payload["evidence"], label + ".evidence"))
        ),
        provenance=_strict_string(payload["provenance"], label + ".provenance"),
        license_refs=_string_tuple(payload["license_refs"], label + ".license_refs"),
    )
    expected_id = _component_id(
        kind=result.kind,
        name=result.name,
        locator=result.locator,
        module_id=result.module_id,
    )
    if result.component_id != expected_id:
        raise ProjectInventoryError(f"{label} component ID does not match its payload.")
    result.validate()
    return result


def _catalog_from_mapping(value: Any, label: str) -> ComponentCatalog:
    payload = _strict_mapping(value, {"schema_version", "components", "catalog_sha256"}, label)
    catalog = ComponentCatalog(
        schema_version=_strict_string(payload["schema_version"], label + ".schema_version"),
        components=tuple(
            _component_from_mapping(item, f"{label}.components[{index}]")
            for index, item in enumerate(_strict_sequence(payload["components"], label + ".components"))
        ),
        catalog_sha256=_strict_string(payload["catalog_sha256"], label + ".catalog_sha256"),
    )
    catalog.validate()
    return catalog


def _project_inventory_from_mapping(value: Mapping[str, Any]) -> ProjectInventory:
    fields = {
        "schema_version",
        "source_kind",
        "root_name",
        "source_sha256",
        "imported_source_snapshot_sha256",
        "project_snapshot_sha256",
        "manifest",
        "modules",
        "target",
        "metadata",
        "entrypoints",
        "namespaces",
        "dependencies",
        "logical_resource_ids",
        "logical_resource_references",
        "component_catalog",
        "warnings",
        "inventory_sha256",
    }
    payload = _strict_mapping(value, fields, "project inventory")
    inventory = ProjectInventory(
        schema_version=_strict_string(payload["schema_version"], "schema_version"),
        source_kind=_strict_string(payload["source_kind"], "source_kind"),
        root_name=_strict_string(payload["root_name"], "root_name"),
        source_sha256=_strict_string(payload["source_sha256"], "source_sha256"),
        imported_source_snapshot_sha256=_strict_string(
            payload["imported_source_snapshot_sha256"], "imported_source_snapshot_sha256"
        ),
        project_snapshot_sha256=_strict_string(
            payload["project_snapshot_sha256"], "project_snapshot_sha256"
        ),
        manifest=tuple(
            _evidence_from_mapping(item, f"manifest[{index}]")
            for index, item in enumerate(_strict_sequence(payload["manifest"], "manifest"))
        ),
        modules=tuple(
            _module_from_mapping(item, f"modules[{index}]")
            for index, item in enumerate(_strict_sequence(payload["modules"], "modules"))
        ),
        target=_target_from_mapping(payload["target"], "target"),
        metadata=tuple(
            _metadata_from_mapping(item, f"metadata[{index}]")
            for index, item in enumerate(_strict_sequence(payload["metadata"], "metadata"))
        ),
        entrypoints=tuple(
            _entrypoint_from_mapping(item, f"entrypoints[{index}]")
            for index, item in enumerate(_strict_sequence(payload["entrypoints"], "entrypoints"))
        ),
        namespaces=_string_tuple(payload["namespaces"], "namespaces"),
        dependencies=tuple(
            _dependency_from_mapping(item, f"dependencies[{index}]")
            for index, item in enumerate(_strict_sequence(payload["dependencies"], "dependencies"))
        ),
        logical_resource_ids=_string_tuple(
            payload["logical_resource_ids"], "logical_resource_ids"
        ),
        logical_resource_references=_string_tuple(
            payload["logical_resource_references"], "logical_resource_references"
        ),
        component_catalog=_catalog_from_mapping(payload["component_catalog"], "component_catalog"),
        warnings=_string_tuple(payload["warnings"], "warnings"),
        inventory_sha256=_strict_string(payload["inventory_sha256"], "inventory_sha256"),
    )
    inventory.validate()
    return inventory


def validate_project_inventory_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on a serialized inventory and return its canonical mapping."""

    return ProjectInventory.from_dict(value).to_dict()


__all__ = [
    "COMPONENT_CATALOG_SCHEMA",
    "SCHEMA_VERSION",
    "ComponentCatalog",
    "ComponentRecord",
    "DependencyRecord",
    "EntryPointRecord",
    "EvidenceLocator",
    "ModMetadata",
    "ProjectInventory",
    "ProjectInventoryError",
    "ProjectModule",
    "ProjectTarget",
    "SourceRoot",
    "canonical_json",
    "canonical_sha256",
    "inspect_existing_archive_inventory",
    "inspect_project_inventory",
    "validate_project_inventory_payload",
]
