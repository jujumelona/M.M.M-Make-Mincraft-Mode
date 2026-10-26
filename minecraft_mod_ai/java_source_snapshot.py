"""Hash-bound project source snapshots shared by compilation and attribution."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath


def _source_digest(source):
    return hashlib.sha256(source.strip().replace("\r\n", "\n").encode("utf-8")).hexdigest()


def direct_source_dependencies(context):
    return {'include_qualified_sources': True} if context.get('host_direct_feature_source') else {}


def _source_records(sources):
    return tuple({"path": name, "sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                  "authority": "materialized_prerequisite_source"}
                 for name, code in sources.items())


class JavaSourceCapture:
    """Per-generation handoff of sources actually accepted by the full compiler."""

    def __init__(self):
        self._compiled = {}

    def record(self, source, *, target_path, sources):
        self._compiled[target_path, _source_digest(source)] = _source_records(sources)

    def source_context(self, source, *, project_root, target_path, source_context=()):
        verified_java_companions(project_root=project_root, source_context=source_context)
        key = target_path, _source_digest(source)
        if self._compiled and key not in self._compiled:
            raise ValueError("JAVA_COMPILED_SOURCE_CONTEXT_MISMATCH")
        merged = {row["path"]: row for row in source_context}
        for row in self._compiled.get(key, ()):
            previous = merged.get(row["path"])
            if previous and previous["sha256"] != row["sha256"]:
                raise ValueError("JAVA_COMPANION_SOURCE_CHANGED")
            merged[row["path"]] = row
        return materialized_java_source_context(source, project_root=project_root,
            target_path=target_path, source_context=tuple(merged.values()))


def record_compiled_sources(context, *, source, target_path, sources):
    capture = context.get("_java_source_capture")
    if capture is not None:
        capture.record(source, target_path=target_path, sources=sources)


def verified_java_companions(
    *, project_root: str | Path, source_context: Sequence[Mapping[str, str]],
) -> dict[str, str]:
    """Read exact admitted bytes, preserving package-qualified file identities."""
    root = Path(project_root).resolve()
    result = {}
    for entry in source_context:
        if entry.get("authority") != "materialized_prerequisite_source":
            raise ValueError("JAVA_COMPANION_UNVERIFIED_AUTHORITY")
        relative = entry.get("path")
        if (not isinstance(relative, str)
                or not relative.startswith(("src/main/java/", "src/client/java/"))
                or not relative.endswith(".java") or "\\" in relative
                or ".." in PurePosixPath(relative).parts
                or PurePosixPath(relative).as_posix() != relative):
            raise ValueError("JAVA_COMPANION_PATH_INVALID")
        current = root
        for section in PurePosixPath(relative).parts:
            current /= section
            if current.is_symlink():
                raise ValueError("JAVA_COMPANION_SYMLINK_INVALID")
        real = current.resolve()
        if not real.is_relative_to(root) or not real.is_file():
            raise ValueError("JAVA_COMPANION_NOT_MATERIALIZED")
        raw = real.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
            raise ValueError("JAVA_COMPANION_SOURCE_CHANGED")
        if relative in result:
            raise ValueError("JAVA_COMPANION_DUPLICATE_SOURCE_IDENTITY: " + relative)
        result[relative] = raw.decode("utf-8")
    return result


def materialized_java_sources(source, *, project_root, target_path, source_context=(),
                              include_qualified_sources=False):
    """Read the compiler closure once, preserving every admitted hash constraint."""
    from .java_source_closure import (
        MAX_CLOSURE_BYTES, MAX_SOURCE_BYTES, _discover_closure,
        _materialized_file, _relative_java_path,
    )

    root = Path(project_root).resolve()
    admitted = [entry for entry in source_context if entry.get("authority") is not None]
    verified_java_companions(project_root=root, source_context=admitted)
    expected = {entry["path"]: entry["sha256"] for entry in admitted}
    related = _discover_closure(source, root=root, target=_relative_java_path(target_path),
                                source_context=source_context,
                                include_qualified_sources=include_qualified_sources)
    result, total = {}, 0
    for relative, _ in related:
        path = _materialized_file(root, relative)
        if path is None:
            raise ValueError("JAVA_COMPANION_NOT_MATERIALIZED")
        raw = path.read_bytes()
        total += len(raw)
        if len(raw) > MAX_SOURCE_BYTES or total > MAX_CLOSURE_BYTES:
            raise ValueError("CONSTRUCTION_COMPILE_SOURCE_CLOSURE_TOO_LARGE")
        digest = hashlib.sha256(raw).hexdigest()
        name = relative.as_posix()
        if name in expected and digest != expected[name]:
            raise ValueError("JAVA_COMPANION_SOURCE_CHANGED")
        result[name] = raw.decode("utf-8")
    return result


def materialized_java_source_context(source, *, project_root, target_path, source_context=()):
    """Snapshot post-generation sources without refreshing away initial receipts.

    New producer identities come from the compiler's explicit source capture,
    never from interpreting an arbitrary dotted expression as a package name.
    """
    verified_java_companions(project_root=project_root, source_context=source_context)
    sources = materialized_java_sources(source, project_root=project_root,
        target_path=target_path, source_context=source_context)
    return _source_records(sources)
