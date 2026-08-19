from __future__ import annotations

"""Narrow hot-path hardening for coder research -> repair evidence reuse.

The semantic feature stays owned by :mod:`research_coder_repair_reuse`. This module
only removes repeated whole-repository scans, caps query-path fanout at the canonical
budget, tail-reads build logs, and keeps receipt writes project-scoped. It does not
introduce a second retriever, scorer, repair engine, or runtime composition owner.
"""

import json
import os
import threading
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Iterator, Mapping

_MARKER = "_mmm_research_coder_repair_performance_v1"
_INDEX_ATTR = "_mmm_dependency_neighborhood_index_v1"
_DEFAULT_QUERY_PATH_BUDGET = 5
_LOG_READ_BYTES = 32_768
_LOG_TEXT_CHARS = 16_000
_COMMAND_TEXT_CHARS = 4_096
_BUILD_NAMES = {
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
    "gradle.properties",
    "fabric.mod.json",
    "libs.versions.toml",
}


class _ProjectLockPool:
    """Serialize one project's receipt file without blocking unrelated projects."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._entries: dict[str, tuple[threading.RLock, int]] = {}

    @contextmanager
    def hold(self, key: str) -> Iterator[None]:
        with self._guard:
            lock, refs = self._entries.get(key, (threading.RLock(), 0))
            self._entries[key] = (lock, refs + 1)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
            with self._guard:
                current = self._entries.get(key)
                if current is not None and current[0] is lock:
                    remaining = current[1] - 1
                    if remaining <= 0:
                        self._entries.pop(key, None)
                    else:
                        self._entries[key] = (lock, remaining)


_RECEIPT_LOCKS = _ProjectLockPool()


def _query_path_budget() -> int:
    raw = os.environ.get("MMM_CODE_RESEARCH_QUERY_PATHS", "").strip()
    try:
        value = int(raw) if raw else _DEFAULT_QUERY_PATH_BUDGET
    except ValueError:
        value = _DEFAULT_QUERY_PATH_BUDGET
    # The canonical implementation already tops out at five paths. Dependency-directed
    # retrieval may replace a broad vocabulary lane, but must not increase total fanout.
    return max(2, min(_DEFAULT_QUERY_PATH_BUDGET, value))


def _dependency_index(research_module: Any, context: Any) -> dict[str, Any]:
    cached = getattr(context, _INDEX_ATTR, None)
    if isinstance(cached, dict):
        return cached

    by_fqcn: dict[str, str] = {}
    by_type: dict[str, list[str]] = {}
    by_package: dict[str, list[str]] = {}
    token_to_paths: dict[str, set[str]] = {}
    reverse_exact: dict[str, set[str]] = {}
    reverse_wildcard: dict[str, set[str]] = {}
    contracts: list[str] = []

    for path, unit in context.units.items():
        by_package.setdefault(unit.package, []).append(path)
        tokens = (
            research_module._tokens(path)
            | research_module._tokens(unit.package)
            | research_module._tokens(unit.types)
        )
        for token in tokens:
            token_to_paths.setdefault(token, set()).add(path)
        for type_name in unit.types:
            fqcn = f"{unit.package}.{type_name}" if unit.package else type_name
            by_fqcn[fqcn] = path
            by_type.setdefault(type_name, []).append(path)
        lowered = path.casefold()
        if any(term in lowered for term in ("contract", "interface", "api")):
            contracts.append(path)

    for path, unit in context.units.items():
        for imported in unit.imports:
            if imported.endswith(".*"):
                reverse_wildcard.setdefault(imported[:-2], set()).add(path)
            else:
                reverse_exact.setdefault(imported, set()).add(path)

    build_paths = tuple(
        sorted(
            indexed.path
            for indexed in context.index.files
            if Path(indexed.path).name in _BUILD_NAMES
        )
    )
    index = {
        "by_fqcn": by_fqcn,
        "by_type": {key: tuple(sorted(value)) for key, value in by_type.items()},
        "by_package": {key: tuple(sorted(value)) for key, value in by_package.items()},
        "token_to_paths": {key: tuple(sorted(value)) for key, value in token_to_paths.items()},
        "reverse_exact": {key: tuple(sorted(value)) for key, value in reverse_exact.items()},
        "reverse_wildcard": {
            key: tuple(sorted(value)) for key, value in reverse_wildcard.items()
        },
        "contracts": tuple(sorted(contracts)),
        "build_paths": build_paths,
    }
    setattr(context, _INDEX_ATTR, index)
    return index


def _dependency_neighborhood_query(
    research_module: Any,
    context: Any,
    query: str,
    plan_step: Any | None,
) -> str:
    query_tokens = set(research_module._tokens(query))
    if plan_step is not None:
        query_tokens |= research_module._tokens(getattr(plan_step, "required_symbols", ()))
        query_tokens |= research_module._tokens(getattr(plan_step, "capability", ""))
    if not query_tokens:
        return ""

    index = _dependency_index(research_module, context)
    match_counts: dict[str, int] = {}
    for token in query_tokens:
        for path in index["token_to_paths"].get(token, ()):
            match_counts[path] = match_counts.get(path, 0) + 1
    seeds = [
        path
        for path, _count in sorted(
            match_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:4]
    ]
    if not seeds:
        return ""

    direct: set[str] = set()
    reverse: set[str] = set()
    seed_fqcns: set[str] = set()
    seed_packages: set[str] = set()
    for seed in seeds:
        unit = context.units.get(seed)
        if unit is None:
            continue
        seed_packages.add(unit.package)
        for type_name in unit.types:
            seed_fqcns.add(f"{unit.package}.{type_name}" if unit.package else type_name)
        for imported in unit.imports:
            if imported.endswith(".*"):
                direct.update(index["by_package"].get(imported[:-2], ()))
                continue
            target = index["by_fqcn"].get(imported)
            if target:
                direct.add(target)
                continue
            direct.update(index["by_type"].get(imported.rsplit(".", 1)[-1], ()))

    for fqcn in seed_fqcns:
        reverse.update(index["reverse_exact"].get(fqcn, ()))
    for package in seed_packages:
        reverse.update(index["reverse_wildcard"].get(package, ()))
    direct.difference_update(seeds)
    reverse.difference_update(seeds)

    contract_tokens = query_tokens | {"contract", "interface", "api"}
    contracts: set[str] = set()
    for path in index["contracts"]:
        if path in seeds or path in direct or path in reverse:
            continue
        unit = context.units[path]
        tokens = research_module._tokens(path) | research_module._tokens(unit.types)
        if research_module._overlap(contract_tokens, tokens) > 0.0:
            contracts.add(path)

    wants_build = bool(
        query_tokens
        & {
            "dependency",
            "dependencies",
            "import",
            "api",
            "registry",
            "build",
            "gradle",
            "fabric",
        }
    )
    ordered = list(
        dict.fromkeys(
            [
                *seeds,
                *sorted(direct),
                *sorted(reverse),
                *sorted(contracts),
                *(index["build_paths"] if wants_build else ()),
            ]
        )
    )[:14]
    if len(ordered) <= len(seeds):
        return ""
    return research_module._join_query(
        "repository dependency neighborhood direct reverse shared contracts",
        *ordered,
    )


def _bounded_log_tail(command: Mapping[str, Any]) -> str:
    pieces = [
        command[key][-_COMMAND_TEXT_CHARS:]
        for key in ("error", "stderr", "stdout", "output")
        if isinstance(command.get(key), str)
    ]
    log_path = command.get("log_path")
    if isinstance(log_path, str):
        path = Path(log_path)
        if path.is_file() and not path.is_symlink():
            try:
                with path.open("rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - _LOG_READ_BYTES), os.SEEK_SET)
                    tail = handle.read(_LOG_READ_BYTES)
                pieces.append(tail.decode("utf-8", errors="replace"))
            except OSError:
                pass
    return "\n".join(pieces)[-_LOG_TEXT_CHARS:]


def _persist_research_receipt(
    reuse_module: Any,
    root: Path,
    receipt: Mapping[str, Any],
    *,
    module_id: str,
) -> None:
    bundle = str(receipt.get("bundle_sha256", ""))
    if not bundle:
        return
    normalized_root = root.expanduser().resolve()
    path = reuse_module._receipt_file(normalized_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_key = str(path.resolve())
    with _RECEIPT_LOCKS.hold(lock_key):
        receipts = reuse_module._load_receipts(normalized_root)
        entry = {
            "module_id": module_id,
            "bundle_sha256": bundle,
            "receipt": dict(receipt),
        }
        key = (module_id, bundle)
        existing = next(
            (
                item
                for item in receipts
                if (
                    str(item.get("module_id", "")),
                    str(item.get("bundle_sha256", "")),
                )
                == key
            ),
            None,
        )
        if existing == entry:
            return
        receipts = [
            item
            for item in receipts
            if (
                str(item.get("module_id", "")),
                str(item.get("bundle_sha256", "")),
            )
            != key
        ]
        receipts.append(entry)
        payload = {
            "schema_version": "mmm/research-code-context-receipt-store-v1",
            "receipts": receipts[-32:],
        }
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _install_helper_replacements(reuse_module: Any) -> None:
    current_dependency = reuse_module._dependency_neighborhood_query
    if not getattr(current_dependency, _MARKER, False):

        @wraps(current_dependency)
        def dependency_query(
            module: Any,
            context: Any,
            query: str,
            plan_step: Any | None,
        ) -> str:
            return _dependency_neighborhood_query(module, context, query, plan_step)

        setattr(dependency_query, _MARKER, True)
        reuse_module._dependency_neighborhood_query = dependency_query

    current_log_tail = reuse_module._read_log_tail
    if not getattr(current_log_tail, _MARKER, False):

        @wraps(current_log_tail)
        def read_log_tail(command: Mapping[str, Any]) -> str:
            return _bounded_log_tail(command)

        setattr(read_log_tail, _MARKER, True)
        reuse_module._read_log_tail = read_log_tail

    current_persist = reuse_module._persist_research_receipt
    if not getattr(current_persist, _MARKER, False):

        @wraps(current_persist)
        def persist(
            root: Path,
            receipt: Mapping[str, Any],
            *,
            module_id: str,
        ) -> None:
            _persist_research_receipt(
                reuse_module,
                root,
                receipt,
                module_id=module_id,
            )

        setattr(persist, _MARKER, True)
        reuse_module._persist_research_receipt = persist


def _install_query_path_budget(research_module: Any) -> None:
    cls = research_module.ResearchCodeContext
    current = cls._query_paths
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def query_paths(self: Any, query: str, plan_step: Any | None):
        paths = list(dict.fromkeys(current(self, query, plan_step)))
        budget = _query_path_budget()
        if len(paths) <= budget:
            return tuple(paths)
        # Keep exact and dependency/plan lanes first. The broad vocabulary lane is the
        # first lane to drop when dependency-directed retrieval already filled the cap.
        focused = [path for path in paths if "known repository vocabulary" not in path]
        broad = [path for path in paths if "known repository vocabulary" in path]
        return tuple([*focused, *broad][:budget])

    setattr(query_paths, _MARKER, True)
    query_paths._mmm_query_path_budget = _DEFAULT_QUERY_PATH_BUDGET  # type: ignore[attr-defined]
    cls._query_paths = query_paths


def harden() -> None:
    """Apply performance-only hardening after the semantic reuse contract is wired."""

    from . import research_code_context
    from . import research_coder_repair_reuse

    _install_helper_replacements(research_coder_repair_reuse)
    _install_query_path_budget(research_code_context)


__all__ = [
    "_bounded_log_tail",
    "_dependency_neighborhood_query",
    "_query_path_budget",
    "harden",
]
