from __future__ import annotations
"""Local source compatibility metadata and immutable manifest validation.

External repository search, snapshot inspection and downloads were removed.
"""
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from .artifact_dependency_graph import (ArtifactDependencyGraph, ArtifactEdge, ArtifactKind, ArtifactNode, UnresolvedArtifactEdge)
from .ecosystem_discovery import ProviderRequestFailure
from .platform_catalog import PlatformAdapter
from .reuse_license import is_reusable_source_license

_TYPE_DECL = re.compile(
    r"\b(?:(?:data|sealed|value|annotation|enum)\s+class|class|interface|object|record|enum)"
    r"\s+([A-Za-z_][A-Za-z0-9_]*)"
)
_METHOD_DECL = re.compile(
    r"\b(?:public|protected|private|static|final|synchronized|abstract|default|native|\s)+"
    r"[A-Za-z_$][A-Za-z0-9_$<>?,.\[\]\s]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_KOTLIN_FUN_DECL = re.compile(
    r"\bfun\s+(?:<[^>]+>\s*)?(?:[A-Za-z_][A-Za-z0-9_?.<>]*\.)?"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_MINECRAFT_PROPERTIES = frozenset({
    "minecraft_version", "minecraftVersion", "minecraft.version", "mc_version", "mcVersion",
})
_MINECRAFT_VERSION = re.compile(r"(?:\d+(?:\.\d+)+(?:-(?:pre|rc)\d+)?|\d{2}w\d{2}[a-z])")
_METADATA_NAMES = frozenset({
    "gradle.properties", "build.gradle", "build.gradle.kts", "settings.gradle",
    "settings.gradle.kts", "libs.versions.toml", "fabric.mod.json", "mods.toml",
    "neoforge.mods.toml",
})
_GRADLE_DEP = re.compile(
    r"(?m)^\s*(?:modImplementation|implementation|api|compileOnly|runtimeOnly)"
    r"\s*[\( ]\s*['\"]([^'\"]+)"
)


@dataclass(frozen=True)
class CompatibilityEvidence:
    minecraft_version: str
    loader: str
    loader_version: str = ""
    java_version: str = ""
    fabric_api_dependency: str = ""
    mixin_count: int = 0
    has_access_widener: bool = False
    status: str = "unverified"  # "metadata_exact" | "metadata_adapt" | "unverified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "minecraft_version": self.minecraft_version,
            "loader": self.loader,
            "loader_version": self.loader_version,
            "java_version": self.java_version,
            "fabric_api_dependency": self.fabric_api_dependency,
            "mixin_count": self.mixin_count,
            "has_access_widener": self.has_access_widener,
            "status": self.status,
        }


@dataclass(frozen=True)
class DonorFile:
    path: str
    blob_sha: str
    sha256: str
    size_bytes: int
    symbols: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DonorFile:
        size = value.get("size_bytes")
        if size is None:
            size = value.get("size", 0)
        return cls(
            path=str(value["path"]),
            blob_sha=str(value["blob_sha"]),
            sha256=str(value["sha256"]),
            size_bytes=int(size),
            symbols=tuple(value.get("symbols", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "blob_sha": self.blob_sha,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "symbols": list(self.symbols),
        }


def _artifact_kind_from_dict(value: Mapping[str, Any]) -> ArtifactKind:
    raw = str(value.get("kind") or value.get("artifact_kind") or "").strip()
    if raw:
        try:
            return ArtifactKind(raw.upper())
        except ValueError:
            pass
    path = str(value.get("id") or value.get("path") or "")
    return ArtifactDependencyGraph.kind_for_path(path)


def _artifact_node_from_dict(value: Mapping[str, Any]) -> ArtifactNode:
    path = str(value.get("id") or value.get("path") or "")
    return ArtifactNode(
        id=path,
        kind=_artifact_kind_from_dict(value),
        namespace=str(value.get("namespace") or "common"),
        logical_id=str(value.get("logical_id") or ""),
        environment=str(value.get("environment") or "common"),
        source_set=str(value.get("source_set") or "main"),
        rel_path=str(value.get("rel_path") or path),
        symbols_defined=tuple(
            value.get("symbols_defined") or value.get("declared_symbols") or ()
        ),
        symbols_referenced=tuple(
            value.get("symbols_referenced") or value.get("referenced_symbols") or ()
        ),
    )


def _artifact_edge_from_dict(value: Mapping[str, Any]) -> ArtifactEdge:
    return ArtifactEdge(
        source_id=str(value.get("source_id") or value.get("source_path") or ""),
        target_id=str(value.get("target_id") or value.get("target_path") or ""),
        dependency_type=str(
            value.get("dependency_type") or value.get("relation") or "reference"
        ),
        is_unresolved=bool(value.get("is_unresolved", False)),
        is_mandatory=bool(value.get("is_mandatory", True)),
    )


def _unresolved_edge_from_dict(value: Mapping[str, Any]) -> UnresolvedArtifactEdge:
    return UnresolvedArtifactEdge(
        source_id=str(value.get("source_id") or value.get("source_path") or ""),
        requested_target=str(
            value.get("requested_target")
            or value.get("target_id")
            or value.get("target_path")
            or ""
        ),
        relation=str(value.get("relation") or value.get("dependency_type") or "reference"),
        reason=str(value.get("reason") or "REPOSITORY_GRAPH_UNRESOLVED"),
    )


@dataclass(frozen=True)
class DonorSlice:
    capability: str
    repository: str
    commit_sha: str
    license_id: str
    source_url: str
    target_compatibility: str
    files: tuple[DonorFile, ...]
    seed_files: tuple[str, ...]
    source_symbols: tuple[str, ...]
    required_dependencies: tuple[str, ...]
    donor_tests: tuple[str, ...]
    confidence: float
    adaptation_cost: float = 0.0
    closure_complete: bool = True
    truncation_reason: str = ""
    artifact_nodes: tuple[ArtifactNode, ...] = ()
    artifact_edges: tuple[ArtifactEdge, ...] = ()
    unresolved_edges: tuple[UnresolvedArtifactEdge, ...] = ()
    compatibility_evidence: CompatibilityEvidence | None = None

    @property
    def metadata_match(self) -> bool:
        return self.target_compatibility in {"exact", "metadata_exact"} and self.closure_complete

    @property
    def exact_target(self) -> bool:
        return self.metadata_match

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DonorSlice:
        files = tuple(DonorFile.from_dict(f) for f in value.get("files", ()))
        artifact_nodes = tuple(
            _artifact_node_from_dict(n)
            for n in value.get("artifact_nodes", ())
            if isinstance(n, Mapping)
        )
        artifact_edges = tuple(
            _artifact_edge_from_dict(e)
            for e in value.get("artifact_edges", ())
            if isinstance(e, Mapping)
        )
        unresolved_edges = tuple(
            _unresolved_edge_from_dict(e)
            for e in value.get("unresolved_edges", ())
            if isinstance(e, Mapping)
        )
        compat_raw = value.get("compatibility_evidence")
        compat = None
        if compat_raw:
            compat = CompatibilityEvidence(
                minecraft_version=str(compat_raw.get("minecraft_version", "")),
                loader=str(compat_raw.get("loader", "")),
                loader_version=str(compat_raw.get("loader_version", "")),
                java_version=str(compat_raw.get("java_version", "")),
                fabric_api_dependency=str(compat_raw.get("fabric_api_dependency", "")),
                mixin_count=int(compat_raw.get("mixin_count", 0)),
                has_access_widener=bool(compat_raw.get("has_access_widener", False)),
                status=str(compat_raw.get("status", "unverified")),
            )
        return cls(
            capability=str(value.get("capability", "")),
            repository=str(value.get("repository", "")),
            commit_sha=str(value.get("commit_sha", "")),
            license_id=str(value.get("license_id", "")),
            source_url=str(value.get("source_url", "")),
            target_compatibility=str(value.get("target_compatibility", "exact")),
            files=files,
            seed_files=tuple(value.get("seed_files", ())),
            source_symbols=tuple(value.get("source_symbols", ())),
            required_dependencies=tuple(value.get("required_dependencies", ())),
            donor_tests=tuple(value.get("donor_tests", ())),
            confidence=float(value.get("confidence", 0.0)),
            adaptation_cost=float(value.get("adaptation_cost", 0.0)),
            closure_complete=bool(value.get("closure_complete", True)),
            truncation_reason=str(value.get("truncation_reason", "")),
            artifact_nodes=artifact_nodes,
            artifact_edges=artifact_edges,
            unresolved_edges=unresolved_edges,
            compatibility_evidence=compat,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/source-transplant-slice-v1",
            "capability": self.capability,
            "repository": self.repository,
            "commit_sha": self.commit_sha,
            "license_id": self.license_id,
            "source_url": self.source_url,
            "target_compatibility": self.target_compatibility,
            "files": [item.to_dict() for item in self.files],
            "seed_files": list(self.seed_files),
            "source_symbols": list(self.source_symbols),
            "required_dependencies": list(self.required_dependencies),
            "donor_tests": list(self.donor_tests),
            "confidence": self.confidence,
            "adaptation_cost": self.adaptation_cost,
            "closure_complete": self.closure_complete,
            "truncation_reason": self.truncation_reason,
            "artifact_nodes": [n.to_dict() for n in self.artifact_nodes],
            "artifact_edges": [e.to_dict() for e in self.artifact_edges],
            "unresolved_edges": [e.to_dict() for e in self.unresolved_edges],
            "compatibility_evidence": self.compatibility_evidence.to_dict() if self.compatibility_evidence else None,
        }


class SourceTransplantError(RuntimeError):
    def __init__(self, message: str = "", *, request_failure: ProviderRequestFailure | None = None) -> None:
        super().__init__(message)
        self.request_failure = request_failure


def validate_donor_slice_manifest(donor_slice: DonorSlice) -> None:
    """Validate immutable donor identity and every manifest path/hash before I/O."""

    repository = str(donor_slice.repository or "").strip()
    if repository.count("/") != 1 or any(
        not part or part in {".", ".."} for part in repository.split("/")
    ):
        raise SourceTransplantError("Donor repository identity is invalid.")
    if not re.fullmatch(r"[0-9a-f]{40,64}", str(donor_slice.commit_sha or "")):
        raise SourceTransplantError("Donor commit is not an immutable full SHA.")
    if not is_reusable_source_license(donor_slice.license_id):
        raise SourceTransplantError("Donor source license is not admitted for reuse.")
    if not donor_slice.files:
        raise SourceTransplantError("Donor source-slice manifest is empty.")

    seen_paths: set[str] = set()
    for donor_file in donor_slice.files:
        path = str(donor_file.path or "")
        normalized = path.replace("\\", "/").strip()
        parts = normalized.split("/")
        if (
            not normalized
            or normalized != path
            or normalized.startswith("/")
            or re.match(r"^[A-Za-z]:", normalized)
            or any(part in {"", ".", ".."} for part in parts)
            or normalized in seen_paths
        ):
            raise SourceTransplantError("Donor manifest contains an unsafe or duplicate path.")
        if not re.fullmatch(r"[0-9a-f]{40,64}", str(donor_file.blob_sha or "")):
            raise SourceTransplantError("Donor manifest contains a non-immutable blob SHA.")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(donor_file.sha256 or "").casefold()):
            raise SourceTransplantError("Donor manifest contains an invalid SHA-256 binding.")
        if donor_file.size_bytes < 0:
            raise SourceTransplantError("Donor manifest contains a negative file size.")
        seen_paths.add(normalized)


def donor_closure_sha256(donor_slice: DonorSlice) -> str:
    """Bind proof identity to every immutable donor file attribute."""

    payload = [
        [item.path, item.blob_sha, item.sha256, item.size_bytes]
        for item in sorted(donor_slice.files, key=lambda entry: entry.path)
    ]
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validated_reuse_donor(decision: Mapping[str, Any]) -> DonorSlice:
    """Validate one source-reuse decision and its executable proof as one unit."""

    if str(decision.get("mode") or "").strip().casefold() not in {
        "source_transplant",
        "adapt",
    }:
        raise SourceTransplantError("Reuse decision is not a source donor action.")
    raw_donor = decision.get("donor")
    if not isinstance(raw_donor, Mapping):
        raise SourceTransplantError("Reuse decision has no donor manifest.")
    if raw_donor.get("schema_version") != "mmm/source-transplant-slice-v1":
        raise SourceTransplantError("Reuse donor schema is invalid.")
    try:
        donor = DonorSlice.from_dict(raw_donor)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise SourceTransplantError("Reuse donor manifest is malformed.") from exc
    validate_donor_slice_manifest(donor)
    if not donor.closure_complete or donor.target_compatibility not in {"exact", "adapt"}:
        raise SourceTransplantError("Reuse donor closure or target compatibility is unverified.")

    capability = str(decision.get("capability") or "").strip().casefold()
    donor_capability = str(donor.capability or "").strip().casefold()
    if capability.removeprefix("capability:") != donor_capability.removeprefix("capability:"):
        raise SourceTransplantError("Reuse decision capability does not match its donor.")

    proof = decision.get("proof_receipt")
    if not isinstance(proof, Mapping):
        raise SourceTransplantError("Reuse decision has no executable proof receipt.")
    if proof.get("schema_version") != "mmm/reuse-proof-receipt-v1":
        raise SourceTransplantError("Reuse proof schema is invalid.")
    from .proof_level import ProofLevel

    if not ProofLevel.from_value(proof.get("proof_level")).is_verified():
        raise SourceTransplantError("Reuse proof level is not verified.")
    if proof.get("authoritative_compile") is not True or proof.get("compile_passed") is not True:
        raise SourceTransplantError("Reuse proof has no authoritative compile pass.")
    # A source transplant claims working functionality, not merely importable
    # Java. Enforce that distinction at the shared donor-validation boundary
    # as well as the production candidate selector.
    if str(decision.get("mode") or "").strip().casefold() == "source_transplant":
        if (proof.get("tests_passed") is not True
                or not proof.get("matched_capability_tests")
                or ProofLevel.from_value(proof.get("proof_level")) not in {
                    ProofLevel.BEHAVIOR_VERIFIED, ProofLevel.HOST_VERIFIED
                }):
            raise SourceTransplantError("SOURCE_REUSE_BEHAVIOR_PROOF_REQUIRED")
    candidate_id = f"{donor.repository}@{donor.commit_sha}"
    if str(proof.get("candidate_id") or "") != candidate_id:
        raise SourceTransplantError("Reuse proof candidate identity does not match its donor.")
    if str(proof.get("commit_sha") or "") != donor.commit_sha:
        raise SourceTransplantError("Reuse proof commit does not match its donor.")
    proof_capability = str(proof.get("capability") or "").strip().casefold()
    if proof_capability.removeprefix("capability:") != donor_capability.removeprefix("capability:"):
        raise SourceTransplantError("Reuse proof capability does not match its donor.")
    if str(proof.get("closure_hash") or "") != donor_closure_sha256(donor):
        raise SourceTransplantError("Reuse proof closure hash does not match its donor manifest.")
    verified_capabilities = {
        str(item).strip().casefold().removeprefix("capability:")
        for item in proof.get("verified_capabilities", ())
        if str(item).strip()
    }
    if donor_capability.removeprefix("capability:") not in verified_capabilities:
        raise SourceTransplantError("Reuse proof did not verify the selected capability.")
    # A compile receipt lists final paths, not which pinned donor file each
    # output came from. Require the exact per-file attestation created by the
    # production host after re-running deterministic source adaptation.
    from .curseforge_source_attestation import check_attested_files
    try:
        check_attested_files(donor, proof)
    except ValueError as exc:
        raise SourceTransplantError(str(exc)) from exc
    return donor


def _metadata_scope(path: str) -> str | None:
    """Return the Gradle/source module owning a conventional metadata path."""
    parts = PurePosixPath(path).parts
    if not parts or parts[-1] not in _METADATA_NAMES:
        return None
    if "src" in parts and parts[-1] in {"fabric.mod.json", "mods.toml", "neoforge.mods.toml"}:
        return "/".join(parts[:parts.index("src")])
    if len(parts) >= 2 and parts[-2:] == ("gradle", "libs.versions.toml"):
        return "/".join(parts[:-2])
    return "/".join(parts[:-1])


def _is_license_path(path: str) -> bool:
    return PurePosixPath(path).name.casefold() in {
        "license", "license.txt", "license.md", "copying", "copying.txt", "copying.md",
    }


def _source_module(path: str, metadata: Mapping[str, str | None]) -> str:
    parts = PurePosixPath(path).parts
    if "src" in parts:
        return "/".join(parts[:parts.index("src")])
    scopes = {scope for item in metadata if (scope := _metadata_scope(item)) is not None}
    return max((scope for scope in scopes if not scope or path.startswith(scope + "/")),
               key=len, default="")


def _without_metadata_comments(text: str) -> str:
    # Preserve quoted strings (including URLs and Gradle interpolation) while
    # removing comments before extracting any version authority.
    return re.sub(r'''(?:"""[\s\S]*?"""|\x27\x27\x27[\s\S]*?\x27\x27\x27)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(/\*[\s\S]*?\*/|//[^\n]*|\#[^\n]*)''',
                  lambda match: match.group(1) or "\n" * match.group(0).count("\n"), text)


def _metadata_assignments(path: str, text: str) -> dict[str, str]:
    if path.endswith("libs.versions.toml"):
        section = ""
        lines = []
        for line in text.splitlines():
            if line.strip().startswith("["):
                section = line.strip()
            elif section == "[versions]":
                lines.append(line)
        text = "\n".join(lines)
    separator = r"[=:]" if path.endswith("gradle.properties") else "="
    pattern = (r"(?m)^\s*(?:(?:val|var|def)\s+)?(?:(?:ext|project|rootProject)\.)?"
               r"([A-Za-z_][\w.-]*)(?:[ \t]*:[ \t]*[\w?.<>]+)?[ \t]*" + separator + r"[ \t]*([^\n]+)")
    values = {match[1]: match[2].strip().rstrip(";") for match in re.finditer(pattern, text)}
    for match in re.finditer(r'''\b(?:extra|ext)\[(["'])([A-Za-z_][\w.-]*)\1\]\s*=\s*([^\n]+)''', text):
        values[match[2]] = match[3].strip().rstrip(";")
    if path.endswith("libs.versions.toml"):
        values = {"libs.versions." + key: value for key, value in values.items()}
    return values


def _resolve_metadata_value(value: str, properties: Mapping[str, str], seen=frozenset()) -> str:
    value = value.strip().rstrip(";").strip()
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        value = value[1:-1]
    key = re.sub(r"^(?:project\.|ext\.)+", "", value)
    key = key.replace("rootProject.ext.", "rootProject.", 1)
    accessor = re.fullmatch(
        r'''(?:(rootProject|project)\.)?(?:findProperty|property)\(["']([\w.-]+)["']\)(?:\s+as\s+String)?''', value)
    provider = re.fullmatch(r'''providers\.gradleProperty\(["']([\w.-]+)["']\)\.get\(\)''', value)
    if accessor:
        key = ("rootProject." if accessor[1] == "rootProject" else "") + accessor[2]
    elif provider:
        key = provider[1]
    if key.startswith("libs.versions.") and key.endswith(".get()"):
        key = key[:-6]
    if key in properties and key not in seen:
        return _resolve_metadata_value(properties[key], properties, seen | {key})

    def interpolate(match):
        reference = match[1] or match[2]
        resolved = _resolve_metadata_value(reference, properties, seen)
        # Missing variables must stay unresolved; replacing them with empty
        # text could turn an unknown version expression into an exact release.
        return resolved if resolved != reference else match[0]

    return re.sub(r"\$\{([^{}]+)\}|\$([A-Za-z_][\w.]*)", interpolate, value)


def _single_minecraft_version(values: Sequence[str]) -> str:
    versions = set(values)
    if len(versions) != 1:
        return ""
    value = next(iter(versions))
    return value if _MINECRAFT_VERSION.fullmatch(value) else ""


def _gradle_owned_metadata(text: str, *, inherited: bool) -> tuple[str, str] | None:
    """Separate local dependencies from explicitly shared project configuration.

    Project selectors and conditional Minecraft mutations need Gradle evaluation;
    they cannot supply global version authority to an unrelated source module.
    """
    quoted = re.compile(r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*' '''.rstrip())
    masked = quoted.sub(lambda match: ' ' * len(match[0]), text)
    properties, coordinates = [], []
    stack = [(True, not inherited, False, 0)]
    previous = header_start = 0
    authority = re.compile(r'(?:com\.mojang:minecraft:|net\.minecraftforge:forge:)')
    for token in re.finditer(r'[{}]', masked):
        start = token.start()
        include_properties, include_coordinates, unknown, opened = stack[-1]
        segment = text[previous:start]
        blank = ''.join('\n' if char == '\n' else ' ' for char in segment)
        properties.append(segment if include_properties else blank)
        coordinates.append(segment if include_coordinates else blank)
        if token[0] == '{':
            header = masked[header_start:start]
            shared = bool(re.search(r'\b(?:allprojects|subprojects)\s*$', header))
            subprojects = bool(re.search(r'\bsubprojects\s*$', header))
            # Only these declarative project contexts grant authority. A task,
            # function, conditional or project selector is not a global scope.
            scoped = not bool(re.search(r'(?<![\w.])(?:ext|dependencies|allprojects|subprojects)\s*$', header))
            stack.append((include_properties and not scoped and not (subprojects and not inherited),
                          (include_coordinates or (shared and inherited)) and not scoped
                          and not (subprojects and not inherited), scoped, token.end()))
        elif len(stack) > 1:
            if unknown:
                body = text[opened:start]
                if authority.search(body):
                    return None
                mutations = set(_metadata_assignments('build.gradle', body))
                mutations.update(match[1] for match in re.finditer(
                    r'''(?:setProperty|extraProperties\.set|extra\.set)\(\s*["']([\w.-]+)["']''', body))
                # Keep unknown writes in the property graph. Silently dropping
                # an alias write would revive its earlier literal version.
                properties.append('\n' + '\n'.join(
                    key + ' = <unresolved-scoped-mutation>' for key in sorted(mutations)) + '\n')
            stack.pop()
        else:
            return None
        # Braces are not version literals. Keeping line boundaries is enough
        # for the assignment/dependency extractor, including inline blocks.
        properties.append('\n')
        coordinates.append('\n')
        previous = token.end()
        header_start = previous
    if len(stack) != 1:
        return None
    tail = text[previous:]
    properties.append(tail)
    coordinates.append(tail if not inherited else ''.join('\n' if char == '\n' else ' ' for char in tail))
    return ''.join(properties), ''.join(coordinates)


def _module_metadata_context(metadata: Mapping[str, str | None], module: str) -> tuple[str, str]:
    """Resolve declarative module/ancestor properties, never sibling modules.

    This handles conventional properties, simple Gradle interpolation and version
    catalogs; executable Gradle expressions remain unknown rather than guessed.
    """
    groups: dict[str, list[tuple[str, str | None]]] = {}
    for path, text in metadata.items():
        scope = _metadata_scope(path)
        if scope is None or (scope and module != scope and not module.startswith(scope + "/")):
            continue
        if PurePosixPath(path).name in {"fabric.mod.json", "mods.toml", "neoforge.mods.toml"} and scope != module:
            continue  # Resource manifests are not inherited by child projects.
        groups.setdefault(scope, []).append((path, text))
    properties: dict[str, str] = {}
    version = ""
    has_build_version = False
    shared_coordinates: list[str] = []
    chunks = []
    for scope in sorted(groups, key=lambda value: (len(value), value)):
        files = groups[scope]
        if any(text is None for _path, text in files):
            return "", ""  # Missing nearer metadata cannot authorize inherited source.
        assigned = set()
        cleaned = []
        root_properties = {}
        for path, text in sorted(files, key=lambda item: (not item[0].endswith("gradle.properties"), item[0])):
            chunks.append(f"\n# MMM_METADATA_PATH {path}\n{text}")
            clean = _without_metadata_comments(text)
            if scope and re.search(r"\brootProject\.(?:ext\.)?[\w.]+\s*=", clean):
                return "", ""  # Cross-project mutations require executing Gradle.
            property_text = dependency_text = clean
            if path.endswith(('.gradle', '.gradle.kts')):
                owned = _gradle_owned_metadata(clean, inherited=scope != module)
                if owned is None:
                    return '', ''
                property_text, dependency_text = owned
            cleaned.append((path, clean, dependency_text))
            if not scope:
                root_property_text = clean
                if path.endswith(('.gradle', '.gradle.kts')):
                    root_owned = _gradle_owned_metadata(clean, inherited=False)
                    if root_owned is None:
                        return '', ''
                    root_property_text = root_owned[0]
                for key, value in _metadata_assignments(path, root_property_text).items():
                    root_properties[key] = _resolve_metadata_value(value, root_properties)
            for key, value in _metadata_assignments(path, property_text).items():
                properties[key] = _resolve_metadata_value(value, properties)
                assigned.add(key)
        if not scope:
            for key, value in root_properties.items():
                resolved = _resolve_metadata_value(value, root_properties)
                properties["rootProject." + key] = (
                    resolved if "$" not in resolved else "<unresolved-root-property>" + resolved)
        declarations = [properties[key] for key in assigned if key in _MINECRAFT_PROPERTIES or
                        (key.startswith("libs.versions.") and key[14:] in _MINECRAFT_PROPERTIES | {"minecraft", "mc"})]
        own_coordinates = []
        for path, _clean, dependency_text in cleaned:
            if not path.endswith((".gradle", ".gradle.kts")):
                continue
            for match in re.finditer(
                r'''\b(?:minecraft|modImplementation|implementation|api|compileOnly)\s*(?:\(\s*)?(["'])((?:com\.mojang:minecraft|net\.minecraftforge:forge):[^"']+)\1''', dependency_text):
                own_coordinates.append(match[2])
        if scope != module:
            shared_coordinates.extend(own_coordinates)
        coordinates = []
        for raw_coordinate in dict.fromkeys([*shared_coordinates, *(own_coordinates if scope == module else [])]):
            coordinate = _resolve_metadata_value(raw_coordinate, properties)
            group, artifact, declared = coordinate.split(":", 2)
            if (group, artifact) == ("net.minecraftforge", "forge"):
                match = re.fullmatch(r'(\d+(?:\.\d+)+(?:-(?:pre|rc)\d+)?)-[0-9][A-Za-z0-9.+-]*', declared)
                declared = match[1] if match else ''
            coordinates.append(declared)
        if coordinates or declarations:
            # An actual dependency declaration overrides inherited default
            # properties. Conflicting/unknown values never select the target.
            version = _single_minecraft_version(coordinates or [
                _resolve_metadata_value(value, properties) for value in declarations])
            has_build_version = True
        for _path, clean, _dependency_text in cleaned:
            try:
                manifest = json.loads(clean)
            except (ValueError, TypeError):
                continue
            depends = manifest.get("depends") if isinstance(manifest, Mapping) else None
            declared = depends.get("minecraft") if isinstance(depends, Mapping) else None
            values = declared if isinstance(declared, list) else [declared]
            exact = _single_minecraft_version([
                _resolve_metadata_value(value, properties) if isinstance(value, str) else ""
                for value in values])
            if exact and has_build_version and version != exact:
                version = ""
            elif exact and not has_build_version:
                version = exact
    return version, "\n".join(chunks)


def _compatibility_evidence(text: str, minecraft_version: str, *, adapter: PlatformAdapter) -> CompatibilityEvidence:
    folded = _without_metadata_comments(text).casefold()
    loader = adapter.loader.casefold()
    loader_aliases = {
        "fabric": ("fabric", "fabricloader", "fabric.mod.json"),
        "forge": ("forge", "net.minecraftforge", "mods.toml", "javafml"),
        "neoforge": ("neoforge", "net.neoforged", "neoforge.mods.toml"),
    }
    loader_evidenced = any(
        alias in folded for alias in loader_aliases.get(loader, (loader,))
    )
    mixin_count = len(re.findall(r"\bmixins?\b", folded))
    has_access_widener = ".accesswidener" in folded or "accesswidener" in folded

    if (
        minecraft_version == adapter.minecraft_version
    ) and loader_evidenced:
        status = "metadata_exact"
    elif loader_evidenced or minecraft_version:
        status = "metadata_adapt"
    else:
        status = "unverified"

    return CompatibilityEvidence(
        minecraft_version=minecraft_version,
        loader=adapter.loader if loader_evidenced else "",
        mixin_count=mixin_count,
        has_access_widener=has_access_widener,
        status=status,
    )


def _target_compatibility_evidence(text: str, *, adapter: PlatformAdapter) -> CompatibilityEvidence:
    matches = list(re.finditer(r"(?m)^# MMM_METADATA_PATH (.+)\n", text))
    metadata = {match[1]: text[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(text)]
                for index, match in enumerate(matches)} if matches else {"build.gradle": text}
    version, scoped_text = _module_metadata_context(metadata, "")
    return _compatibility_evidence(scoped_text, version, adapter=adapter)


def _target_compatibility(text: str, *, adapter: PlatformAdapter) -> str:
    evidence = _target_compatibility_evidence(text, adapter=adapter)
    return (
        "exact"
        if evidence.status == "metadata_exact"
        else ("adapt" if evidence.status == "metadata_adapt" else "unverified")
    )


def _declared_dependencies(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    values: list[str] = []
    for value in _GRADLE_DEP.findall(text):
        cleaned = " ".join(value.split())[:256]
        if cleaned and cleaned not in values:
            values.append(cleaned)
    return tuple(values)


def materialize_source_slices(project_root: str | Path, reuse_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Install only compiled and SHA-bound donors into a staging directory."""
    from .curseforge_reuse_production import stage_verified_donors
    return stage_verified_donors(project_root, reuse_plan)

def materialize_pinned_donor(donor_slice: DonorSlice, discovery_client: Any = None) -> dict[str, bytes]:
    """Re-fetch exact CurseForge-linked source commit and attest every byte."""
    from .curseforge_reuse_production import read_pinned_manifest
    return read_pinned_manifest(donor_slice)

__all__ = ["ArtifactEdge", "ArtifactNode", "DonorFile", "DonorSlice", "SourceTransplantError", "donor_closure_sha256", "materialize_pinned_donor", "materialize_source_slices", "validate_donor_slice_manifest", "validated_reuse_donor"]
