from __future__ import annotations

"""Pinned, capability-slice source transplantation for permissive OSS donors.

A repository hit is never a reusable implementation by itself. Reuse is admitted
only after an immutable commit, SPDX license, exact target metadata (or an explicit
adaptation classification), a bounded Java source slice, and hashes for every source
blob have been recorded. Execution refetches the pinned blobs and verifies the same
hashes before exposing them to the coder.
"""

import base64
import hashlib
import json
import os
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlparse

import httpx

from .platform_catalog import PlatformAdapter

_PERMISSIVE = frozenset({
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Zlib",
    "Unlicense", "CC0-1.0",
})
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,127}")
_TYPE_DECL = re.compile(r"\b(?:class|interface|record|enum)\s+([A-Za-z_][A-Za-z0-9_]*)")
_METHOD_DECL = re.compile(
    r"\b(?:public|protected|private|static|final|synchronized|abstract|default|native|\s)+"
    r"[A-Za-z_$][A-Za-z0-9_$<>?,.\[\]\s]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_MINECRAFT_PROP = re.compile(r"(?m)^\s*minecraft_version\s*=\s*([^\s#]+)")
_GRADLE_DEP = re.compile(r"(?m)^\s*(?:modImplementation|implementation|api|compileOnly|runtimeOnly)\s*[\( ]\s*['\"]([^'\"]+)")
_MAX_TREE_FILES = 20_000
_MAX_SEEDS = 8
_MAX_CLOSURE_FILES = 64
_MAX_SLICE_BYTES = 1024 * 1024
_SNAPSHOT_LOCK = Lock()
_SNAPSHOT_CACHE: dict[str, Mapping[str, Any] | None] = {}
_SNAPSHOT_INFLIGHT: dict[str, Event] = {}
_BLOB_LOCK = Lock()
_BLOB_CACHE: dict[tuple[str, str], bytes] = {}
_BLOB_INFLIGHT: dict[tuple[str, str], Event] = {}


@dataclass(frozen=True)
class ArtifactNode:
    path: str
    artifact_kind: str  # "java_source" | "json_model" | "texture_png" | "loot_table" | "recipe" | "tag_json" | "mixin_config" | "access_widener" | "mod_metadata" | "gradle_build"
    blob_sha: str
    sha256: str
    size_bytes: int
    declared_symbols: tuple[str, ...] = ()
    referenced_symbols: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "artifact_kind": self.artifact_kind,
            "blob_sha": self.blob_sha,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "declared_symbols": list(self.declared_symbols),
            "referenced_symbols": list(self.referenced_symbols),
        }


@dataclass(frozen=True)
class ArtifactEdge:
    source_path: str
    target_path: str
    relation: str  # "import" | "type_ref" | "model_parent" | "texture_ref" | "loot_ref" | "mixin_target" | "registry_ref" | "access_widener_ref"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "target_path": self.target_path,
            "relation": self.relation,
        }


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "blob_sha": self.blob_sha,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "symbols": list(self.symbols),
        }


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
    unresolved_edges: tuple[ArtifactEdge, ...] = ()
    compatibility_evidence: CompatibilityEvidence | None = None

    @property
    def exact_target(self) -> bool:
        return self.target_compatibility in {"exact", "metadata_exact"} and self.closure_complete

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
    pass


def repository_from_candidate(candidate: Mapping[str, Any]) -> str:
    """Return owner/repo from a GitHub discovery candidate without guessing."""

    candidate_id = str(candidate.get("candidate_id") or "")
    if candidate_id.startswith("github:"):
        value = candidate_id.split(":", 1)[1].strip()
        if value.count("/") == 1:
            return value
    source = str(candidate.get("source_url") or "")
    parsed = urlparse(source)
    if parsed.hostname in {"github.com", "www.github.com"}:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1].removesuffix('.git')}"
    return ""


def inspect_repository_slice(
    *,
    repository: str,
    capability: str,
    adapter: PlatformAdapter,
    discovery_client: Any,
) -> DonorSlice | None:
    """Inspect one repo and return a pinned minimal Java closure when evidence suffices."""

    snapshot = _repository_snapshot(repository, discovery_client)
    if not isinstance(snapshot, Mapping):
        return None
    license_id = str(snapshot.get("license_id") or "")
    commit_sha = str(snapshot.get("commit_sha") or "")
    blobs = snapshot.get("blobs")
    if (
        license_id not in _PERMISSIVE
        or not re.fullmatch(r"[0-9a-f]{40,64}", commit_sha)
        or not isinstance(blobs, Mapping)
    ):
        return None

    token = str(getattr(discovery_client, "github_token", "") or "").strip()
    client = _github_client(token)
    try:
        metadata_text = _build_metadata_text(client, repository=repository, blobs=blobs)
        compatibility = _target_compatibility(metadata_text, adapter=adapter)
        # A donor with no target evidence is not an adaptation plan; it is an
        # unverified repository hit. Do not admit it into reuse scoring.
        if compatibility not in {"exact", "adapt"}:
            return None
        required_dependencies = _declared_dependencies(metadata_text)
        donor_tests = tuple(
            sorted(
                path for path in blobs
                if path.endswith(".java") and "/test/" in f"/{path.casefold()}/"
            )[:24]
        )
        java_paths = tuple(
            path for path in blobs
            if path.endswith(".java") and "/src/" in f"/{path}"
        )
        if not java_paths:
            java_paths = tuple(path for path in blobs if path.endswith(".java"))
        if not java_paths:
            return None

        capability_tokens = _semantic_tokens(capability)
        ranked = sorted(
            java_paths,
            key=lambda path: (_path_score(path, capability_tokens), -len(path), path),
            reverse=True,
        )
        seed_paths = tuple(path for path in ranked[:_MAX_SEEDS] if _path_score(path, capability_tokens) > 0)
        if not seed_paths:
            return None

        contents: dict[str, bytes] = {}
        declarations: dict[str, str] = {}
        # Building this map is only O(file-count) string work. Truncating it made
        # dependency closure depend on tree ordering and could silently miss a class
        # beyond the first 512 paths.
        for path in java_paths:
            stem = Path(path).stem
            if stem and stem not in declarations:
                declarations[stem] = path

        pending = deque(seed_paths)
        selected: list[str] = []
        selected_set: set[str] = set()
        total_bytes = 0
        artifact_nodes: list[ArtifactNode] = []
        artifact_edges: list[ArtifactEdge] = []
        unresolved_edges: list[ArtifactEdge] = []
        truncation_reason = ""
        closure_complete = True

        # Index resource files in repo using typed canonical identifiers
        resource_index: dict[str, str] = {}
        for path in blobs:
            clean_path = path.casefold()
            resource_index[clean_path] = path
            m = re.search(r"src/main/resources/(assets|data)/([^/]+)/([^/]+)/(.+)$", path, re.IGNORECASE)
            if m:
                _area, ns, category, subpath = m.group(1).casefold(), m.group(2).casefold(), m.group(3).casefold(), m.group(4).casefold()
                subpath_stem = Path(subpath).stem.casefold()
                resource_index[f"{ns}:{category}/{subpath}"] = path
                resource_index[f"{ns}:{subpath_stem}"] = path
                resource_index[f"{ns}:{subpath}"] = path
                if category in {"models", "model"}:
                    resource_index[f"model:{ns}:{subpath_stem}"] = path
                    resource_index[f"model:{ns}:{subpath}"] = path
                    resource_index[f"{ns}:item/{subpath_stem}"] = path
                    resource_index[f"{ns}:block/{subpath_stem}"] = path
                elif category in {"textures", "texture"}:
                    resource_index[f"texture:{ns}:{subpath_stem}"] = path
                    resource_index[f"texture:{ns}:{subpath}"] = path
                    resource_index[f"{ns}:textures/{subpath_stem}"] = path
            else:
                stem = Path(path).stem.casefold()
                if stem not in resource_index:
                    resource_index[stem] = path

        while pending:
            if len(selected) >= _MAX_CLOSURE_FILES:
                closure_complete = False
                truncation_reason = f"Exceeded MAX_CLOSURE_FILES ({_MAX_CLOSURE_FILES})"
                while pending:
                    rem = pending.popleft()
                    unresolved_edges.append(ArtifactEdge(source_path="closure_root", target_path=rem, relation="type_ref"))
                break

            path = pending.popleft()
            if path in selected_set:
                continue
            raw = _fetch_blob_bytes(client, repository, blobs[path])
            if not raw or total_bytes + len(raw) > _MAX_SLICE_BYTES:
                if total_bytes + len(raw) > _MAX_SLICE_BYTES:
                    closure_complete = False
                    truncation_reason = f"Exceeded MAX_SLICE_BYTES ({_MAX_SLICE_BYTES})"
                    unresolved_edges.append(ArtifactEdge(source_path="closure_root", target_path=path, relation="size_overflow"))
                continue

            selected.append(path)
            selected_set.add(path)
            contents[path] = raw
            total_bytes += len(raw)

            text = raw.decode("utf-8", errors="replace")
            kind = "java_source" if path.endswith(".java") else ("json_model" if path.endswith(".json") else "other")
            local_declared = tuple(sorted(set(_TYPE_DECL.findall(text))))
            local_referenced = tuple(sorted(set(_TOKEN.findall(text))))

            artifact_nodes.append(
                ArtifactNode(
                    path=path,
                    artifact_kind=kind,
                    blob_sha=blobs[path],
                    sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
                    size_bytes=len(raw),
                    declared_symbols=local_declared,
                    referenced_symbols=local_referenced[:64],
                )
            )

            # Java symbol closure
            if path.endswith(".java"):
                for symbol in sorted(set(_TOKEN.findall(text))):
                    dep_path = declarations.get(symbol)
                    if dep_path:
                        artifact_edges.append(ArtifactEdge(source_path=path, target_path=dep_path, relation="type_ref"))
                        if dep_path not in selected_set:
                            pending.append(dep_path)

                # Resource Identifier closure from Java code
                for id_match in re.findall(r'"([a-z0-9_.-]+:[a-z0-9_/.-]+)"', text):
                    parts = id_match.split(":", 1)
                    if len(parts) == 2:
                        ns, subpath = parts[0].casefold(), parts[1].casefold()
                        if ns not in {"minecraft", "fabric", "forge", "neoforge", "c"}:
                            res_name = subpath.split("/")[-1]
                            matched_res = (
                                resource_index.get(id_match.casefold())
                                or resource_index.get(f"{ns}:{subpath}")
                                or resource_index.get(f"{ns}:{res_name}")
                                or resource_index.get(res_name)
                            )
                            if matched_res:
                                if matched_res not in selected_set:
                                    artifact_edges.append(ArtifactEdge(source_path=path, target_path=matched_res, relation="registry_ref"))
                                    pending.append(matched_res)
                            else:
                                unresolved_edges.append(ArtifactEdge(source_path=path, target_path=id_match, relation="missing_internal_resource"))
                                closure_complete = False
                                if not truncation_reason:
                                    truncation_reason = f"Missing internal resource: {id_match}"

            # JSON model parent/texture closure
            elif path.endswith(".json"):
                # 1. Model Parent
                for parent_match in re.findall(r'"parent"\s*:\s*"([^"]+)"', text):
                    parts = parent_match.split(":", 1)
                    ns = parts[0].casefold() if len(parts) == 2 else ""
                    if ns not in {"minecraft", "builtin"}:
                        sub = parts[1] if len(parts) == 2 else parent_match
                        parent_stem = sub.split("/")[-1].casefold()
                        matched_parent = (
                            resource_index.get(f"model:{ns}:{parent_stem}")
                            or resource_index.get(f"{ns}:{sub.casefold()}")
                            or resource_index.get(parent_stem)
                        )
                        if matched_parent:
                            if matched_parent not in selected_set:
                                artifact_edges.append(ArtifactEdge(source_path=path, target_path=matched_parent, relation="model_parent"))
                                pending.append(matched_parent)
                        elif ns:
                            unresolved_edges.append(ArtifactEdge(source_path=path, target_path=parent_match, relation="missing_model_parent"))
                            closure_complete = False
                            if not truncation_reason:
                                truncation_reason = f"Missing parent model: {parent_match}"

                # 2. Textures in Model JSON
                for tex_match in re.findall(r'"(?:layer[0-9]|texture|particle|all|top|bottom|side)"\s*:\s*"([^"]+)"', text):
                    parts = tex_match.split(":", 1)
                    ns = parts[0].casefold() if len(parts) == 2 else ""
                    if ns not in {"minecraft"}:
                        sub = parts[1] if len(parts) == 2 else tex_match
                        tex_stem = sub.split("/")[-1].casefold()
                        matched_tex = (
                            resource_index.get(f"texture:{ns}:{tex_stem}")
                            or resource_index.get(f"{ns}:{sub.casefold()}")
                            or resource_index.get(tex_stem)
                        )
                        if matched_tex and matched_tex not in selected_set:
                            artifact_edges.append(ArtifactEdge(source_path=path, target_path=matched_tex, relation="texture_ref"))
                            pending.append(matched_tex)

                # 3. Mixin JSON target classes with package resolution
                if "mixin" in path.casefold():
                    pkg_match = re.search(r'"package"\s*:\s*"([^"]+)"', text)
                    pkg_prefix = (pkg_match.group(1).strip() + ".") if pkg_match else ""
                    for mixin_class in re.findall(r'"([A-Za-z0-9_]+)"', text):
                        full_class = f"{pkg_prefix}{mixin_class}"
                        matched_class = declarations.get(full_class) or declarations.get(mixin_class)
                        if matched_class and matched_class not in selected_set:
                            artifact_edges.append(ArtifactEdge(source_path=path, target_path=matched_class, relation="mixin_target"))
                            pending.append(matched_class)

        if not selected:
            return None

        # Build DonorFile list
        files: list[DonorFile] = []
        symbols: set[str] = set()
        for path in selected:
            raw = contents[path]
            text = raw.decode("utf-8", errors="replace")
            local_symbols = tuple(sorted(set(_TYPE_DECL.findall(text)) | set(_METHOD_DECL.findall(text))))
            symbols.update(local_symbols)
            files.append(
                DonorFile(
                    path=path,
                    blob_sha=blobs[path],
                    sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
                    size_bytes=len(raw),
                    symbols=local_symbols,
                )
            )

        overlap = len({token.casefold() for token in symbols} & capability_tokens)
        confidence = min(
            0.99,
            0.55
            + (0.25 if compatibility in {"exact", "metadata_exact"} else 0.05)
            + min(0.15, 0.03 * overlap)
            + min(0.04, 0.01 * len(files))
            - (0.20 if not closure_complete else 0.0),
        )
        adaptation_cost = round(
            10.0 * len(required_dependencies)
            + (0.0 if compatibility in {"exact", "metadata_exact"} else 25.0)
            + (5.0 if "mixin" in metadata_text.casefold() else 0.0)
            + 0.002 * (total_bytes / 1024.0)
            + (15.0 if not closure_complete else 0.0),
            2,
        )
        ev = _target_compatibility_evidence(metadata_text, adapter=adapter)

        return DonorSlice(
            capability=capability,
            repository=repository,
            commit_sha=commit_sha,
            license_id=license_id,
            source_url=str(snapshot.get("source_url") or f"https://github.com/{repository}"),
            target_compatibility=compatibility if closure_complete else "unverified",
            files=tuple(files),
            seed_files=seed_paths,
            source_symbols=tuple(sorted(symbols)),
            required_dependencies=required_dependencies,
            donor_tests=donor_tests,
            confidence=round(max(0.1, confidence), 4),
            adaptation_cost=adaptation_cost,
            closure_complete=closure_complete,
            truncation_reason=truncation_reason,
            artifact_nodes=tuple(artifact_nodes),
            artifact_edges=tuple(artifact_edges),
            unresolved_edges=tuple(unresolved_edges),
            compatibility_evidence=ev,
        )
    except Exception:
        return None
    finally:
        client.close()


def materialize_source_slices(
    project_root: str | Path,
    reuse_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Refetch pinned donor blobs, verify hashes, and expose them as immutable evidence."""

    root = Path(project_root).expanduser().resolve()
    target_root = root / ".minecraft_ai" / "reuse" / "donors"
    target_root.mkdir(parents=True, exist_ok=True)
    decisions = reuse_plan.get("capabilities") if isinstance(reuse_plan, Mapping) else None
    if not isinstance(decisions, Sequence) or isinstance(decisions, (str, bytes)):
        return {"schema_version": "mmm/reuse-materialization-v1", "donors": [], "count": 0}

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    client = _github_client(token)
    receipts: list[dict[str, Any]] = []
    try:
        for decision in decisions:
            if not isinstance(decision, Mapping) or decision.get("mode") not in {"source_transplant", "adapt"}:
                continue
            donor = decision.get("donor")
            if not isinstance(donor, Mapping):
                continue
            repository = str(donor.get("repository") or "")
            commit_sha = str(donor.get("commit_sha") or "")
            files = donor.get("files")
            if repository.count("/") != 1 or not re.fullmatch(r"[0-9a-f]{40,64}", commit_sha):
                raise SourceTransplantError("Reuse plan contains an unpinned donor.")
            if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
                raise SourceTransplantError("Reuse plan donor has no source-slice manifest.")
            donor_key = _donor_materialization_key(decision, donor, files)
            donor_root = target_root / donor_key
            donor_root.mkdir(parents=True, exist_ok=True)
            written: list[dict[str, Any]] = []
            for item in files:
                if not isinstance(item, Mapping):
                    raise SourceTransplantError("Malformed donor file manifest.")
                path = str(item.get("path") or "").replace("\\", "/")
                blob_sha = str(item.get("blob_sha") or "")
                expected = str(item.get("sha256") or "")
                if not path or path.startswith("/") or ".." in path.split("/"):
                    raise SourceTransplantError("Unsafe donor source path.")
                raw = _fetch_blob_bytes(client, repository, blob_sha)
                actual = "sha256:" + hashlib.sha256(raw).hexdigest()
                if actual != expected:
                    raise SourceTransplantError(
                        f"Pinned donor hash mismatch for {repository}@{commit_sha}:{path}."
                    )
                destination = (donor_root / path).resolve()
                destination.relative_to(donor_root.resolve())
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
                written.append({"path": str(destination), "sha256": actual, "size_bytes": len(raw)})
            manifest = {
                "repository": repository,
                "commit_sha": commit_sha,
                "license_id": donor.get("license_id"),
                "capability": decision.get("capability"),
                "required_dependencies": list(donor.get("required_dependencies") or ()),
                "donor_tests": list(donor.get("donor_tests") or ()),
                "files": written,
            }
            (donor_root / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            receipts.append(manifest)
    finally:
        client.close()
    return {
        "schema_version": "mmm/reuse-materialization-v1",
        "donors": receipts,
        "count": len(receipts),
    }


def _donor_materialization_key(
    decision: Mapping[str, Any],
    donor: Mapping[str, Any],
    files: Sequence[Any],
) -> str:
    identity = {
        "repository": str(donor.get("repository") or ""),
        "commit_sha": str(donor.get("commit_sha") or ""),
        "capability": str(decision.get("capability") or ""),
        "files": [
            [
                str(item.get("path") or ""),
                str(item.get("blob_sha") or ""),
                str(item.get("sha256") or ""),
            ]
            for item in files
            if isinstance(item, Mapping)
        ],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _repository_snapshot(repository: str, discovery_client: Any) -> Mapping[str, Any] | None:
    """Cache immutable repository metadata/tree across target-version evaluation."""

    owner = False
    with _SNAPSHOT_LOCK:
        if repository in _SNAPSHOT_CACHE:
            return _SNAPSHOT_CACHE[repository]
        event = _SNAPSHOT_INFLIGHT.get(repository)
        if event is None:
            event = Event()
            _SNAPSHOT_INFLIGHT[repository] = event
            owner = True
    if not owner:
        event.wait(timeout=30.0)
        with _SNAPSHOT_LOCK:
            return _SNAPSHOT_CACHE.get(repository)

    snapshot: Mapping[str, Any] | None = None
    try:
        evidence = discovery_client.inspect_github_repository(repository)
        if not isinstance(evidence, Mapping):
            return None
        commit_sha = str(evidence.get("commit_sha") or "")
        license_id = str(evidence.get("license_id") or "")
        if license_id not in _PERMISSIVE or not re.fullmatch(r"[0-9a-f]{40,64}", commit_sha):
            return None
        token = str(getattr(discovery_client, "github_token", "") or "").strip()
        client = _github_client(token)
        try:
            tree = _github_json(
                client,
                f"https://api.github.com/repos/{repository}/git/trees/{commit_sha}",
                params={"recursive": "1"},
            )
        finally:
            client.close()
        entries = tree.get("tree") if isinstance(tree, Mapping) else None
        if not isinstance(entries, list) or len(entries) > _MAX_TREE_FILES:
            return None
        blobs = {
            str(item.get("path")): str(item.get("sha"))
            for item in entries
            if isinstance(item, Mapping)
            and item.get("type") == "blob"
            and isinstance(item.get("path"), str)
            and isinstance(item.get("sha"), str)
        }
        snapshot = {
            "commit_sha": commit_sha,
            "license_id": license_id,
            "source_url": str(evidence.get("source_url") or f"https://github.com/{repository}"),
            "blobs": blobs,
        }
        return snapshot
    except Exception:
        return None
    finally:
        with _SNAPSHOT_LOCK:
            _SNAPSHOT_CACHE[repository] = snapshot
            pending = _SNAPSHOT_INFLIGHT.pop(repository, None)
            if pending is not None:
                pending.set()


def _github_client(token: str) -> httpx.Client:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "mmm-source-transplant"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(timeout=15.0, headers=headers, follow_redirects=False)


def _github_json(client: httpx.Client, url: str, *, params: Mapping[str, str] | None = None) -> Any:
    response = client.get(url, params=params)
    response.raise_for_status()
    if len(response.content) > 4 * 1024 * 1024:
        raise SourceTransplantError("GitHub response exceeded source-transplant byte policy.")
    return response.json()


def _fetch_blob_bytes(client: httpx.Client, repository: str, blob_sha: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40,64}", blob_sha):
        raise SourceTransplantError("Donor blob is not immutable.")
    key = (repository, blob_sha)
    owner = False
    with _BLOB_LOCK:
        cached = _BLOB_CACHE.get(key)
        if cached is not None:
            return cached
        event = _BLOB_INFLIGHT.get(key)
        if event is None:
            event = Event()
            _BLOB_INFLIGHT[key] = event
            owner = True
    if not owner:
        if not event.wait(timeout=30.0):
            raise SourceTransplantError("Timed out waiting for a concurrent donor blob download.")
        with _BLOB_LOCK:
            cached = _BLOB_CACHE.get(key)
        if cached is None:
            raise SourceTransplantError("Concurrent donor blob download failed.")
        return cached

    try:
        value = _github_json(
            client,
            f"https://api.github.com/repos/{repository}/git/blobs/{quote(blob_sha, safe='')}",
        )
        if not isinstance(value, Mapping) or value.get("encoding") != "base64":
            raise SourceTransplantError("GitHub donor blob is not base64 encoded.")
        raw = base64.b64decode(str(value.get("content") or "").replace("\n", ""), validate=True)
        if len(raw) > _MAX_SLICE_BYTES:
            raise SourceTransplantError("Single donor blob exceeds source-transplant byte policy.")
        with _BLOB_LOCK:
            while len(_BLOB_CACHE) >= 512:
                _BLOB_CACHE.pop(next(iter(_BLOB_CACHE)))
            _BLOB_CACHE[key] = raw
        return raw
    finally:
        with _BLOB_LOCK:
            pending = _BLOB_INFLIGHT.pop(key, None)
            if pending is not None:
                pending.set()


def _build_metadata_text(
    client: httpx.Client,
    *,
    repository: str,
    blobs: Mapping[str, str],
) -> str:
    metadata_paths = (
        "gradle.properties",
        "fabric.mod.json",
        "src/main/resources/fabric.mod.json",
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "settings.gradle.kts",
    )
    chunks: list[str] = []
    for path in metadata_paths:
        blob = blobs.get(path)
        if not blob:
            continue
        try:
            chunks.append(_fetch_blob_bytes(client, repository, blob).decode("utf-8", errors="replace"))
        except Exception:
            continue
    return "\n".join(chunks)


def _target_compatibility_evidence(text: str, *, adapter: PlatformAdapter) -> CompatibilityEvidence:
    if not text:
        return CompatibilityEvidence(minecraft_version="", loader="", status="unverified")
    folded = text.casefold()
    loader = adapter.loader.casefold()
    loader_evidenced = loader in folded or (loader == "fabric" and "fabricloader" in folded)
    prop = _MINECRAFT_PROP.search(text)
    mc_ver = prop.group(1).strip() if prop else ""
    if not mc_ver:
        for v in (adapter.minecraft_version, "1.21", "1.20", "1.19"):
            if v in text:
                mc_ver = v
                break
    mixin_count = len(re.findall(r"\bmixins?\b", folded))
    has_aw = ".accesswidener" in text or "accessWidener" in text

    if (mc_ver == adapter.minecraft_version or adapter.minecraft_version in text) and loader_evidenced:
        status = "metadata_exact"
    elif loader_evidenced:
        status = "metadata_adapt"
    else:
        status = "unverified"

    return CompatibilityEvidence(
        minecraft_version=mc_ver,
        loader=adapter.loader if loader_evidenced else "",
        mixin_count=mixin_count,
        has_access_widener=has_aw,
        status=status,
    )


def _target_compatibility(text: str, *, adapter: PlatformAdapter) -> str:
    ev = _target_compatibility_evidence(text, adapter=adapter)
    return "exact" if ev.status == "metadata_exact" else ("adapt" if ev.status == "metadata_adapt" else "unverified")


def _declared_dependencies(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    values: list[str] = []
    for value in _GRADLE_DEP.findall(text):
        cleaned = " ".join(value.split())[:256]
        if cleaned and cleaned not in values:
            values.append(cleaned)
        if len(values) >= 32:
            break
    return tuple(values)


def _semantic_tokens(value: str) -> set[str]:
    stop = {"minecraft", "fabric", "forge", "neoforge", "mod", "mods", "java", "system"}
    return {
        token.casefold()
        for token in _TOKEN.findall(value.replace(".", " ").replace("-", " "))
        if len(token) > 2 and token.casefold() not in stop
    }


def _path_score(path: str, tokens: set[str]) -> int:
    haystack = set(_TOKEN.findall(path.replace("/", " ").replace(".", " ").casefold()))
    return 4 * len(haystack & tokens) + sum(token in path.casefold() for token in tokens)


__all__ = [
    "DonorFile",
    "DonorSlice",
    "SourceTransplantError",
    "inspect_repository_slice",
    "materialize_source_slices",
    "repository_from_candidate",
]
