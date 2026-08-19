from __future__ import annotations

"""Cross-phase hardening for bounded coder RAG and repair evidence reuse.

The canonical retrieval, scoring, repair and generation implementations stay in their
owning modules. This module adds bounded state, dependency-directed query hints, and
compact host receipts so repair can reuse already-paid-for coder evidence without a
second retriever or an unbounded research loop.
"""

import copy
import hashlib
import json
import os
import re
import threading
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

_MARKER = "_mmm_research_coder_repair_reuse_v1"
_RECEIPT_PATH = Path(".minecraft_ai/research-code-context-receipts.json")
_DEPENDENCY_INDEX_ATTR = "_mmm_dependency_neighborhood_index_v1"
_QUERY_PATH_MAX = 5
_LOG_READ_BYTES = 32_768
_LOG_TEXT_CHARS = 16_000
_COMMAND_TEXT_CHARS = 4_096
_EXCEPTION = re.compile(r"\b[A-Za-z_$][A-Za-z0-9_.$]*(?:Exception|Error)\b")
_SYMBOL = re.compile(
    r"(?:cannot\s+find\s+symbol|symbol\s*:)\s*(?:class|method|variable)?\s*([A-Za-z_$][A-Za-z0-9_$]*)",
    re.IGNORECASE,
)
_SOURCE_LINE = re.compile(r"(?P<path>[^\s:]+\.java):(?P<line>\d+)(?::\d+)?")
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
    """Serialize one project's receipt store without blocking unrelated projects."""

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


def _round_budget() -> int:
    raw = os.environ.get("MMM_CODE_RESEARCH_EVOLUTION_ROUNDS", "2").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 2
    # RepoCoder-style draft->retrieve evolution is deliberately small. Operators may
    # lower the budget, but cannot silently turn the coder into an unbounded agent.
    return max(1, min(2, value))


def _bounded_evolution_state_budget() -> int:
    return _round_budget()


def _query_path_budget() -> int:
    raw = os.environ.get("MMM_CODE_RESEARCH_QUERY_PATHS", "").strip()
    try:
        value = int(raw) if raw else _QUERY_PATH_MAX
    except ValueError:
        value = _QUERY_PATH_MAX
    return max(2, min(_QUERY_PATH_MAX, value))


def _dependency_index(research_module: Any, context: Any) -> dict[str, Any]:
    cached = getattr(context, _DEPENDENCY_INDEX_ATTR, None)
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
        "build_paths": tuple(
            sorted(
                indexed.path
                for indexed in context.index.files
                if Path(indexed.path).name in _BUILD_NAMES
            )
        ),
    }
    setattr(context, _DEPENDENCY_INDEX_ATTR, index)
    return index


def _dependency_neighborhood_query(
    research_module: Any,
    context: Any,
    query: str,
    plan_step: Any | None,
) -> str:
    """Return one compact direct/reverse dependency query, not a second retriever."""

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
            match_counts.items(), key=lambda item: (-item[1], item[0])
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


def _reusable_evidence(context: Any, *, limit: int = 8) -> list[dict[str, Any]]:
    ranked = sorted(
        context.evidence.values(),
        key=lambda item: (-float(item.bestfit_score), item.path, item.start_line),
    )
    result: list[dict[str, Any]] = []
    for item in ranked[:limit]:
        result.append(
            {
                "evidence_id": item.evidence_id,
                "source_type": item.source_type,
                "path": item.path,
                "sha256": item.sha256,
                "start_line": item.start_line,
                "end_line": item.end_line,
                "symbols": list(item.symbols[:12]),
                "snippet": item.text[:640],
            }
        )
    return result


def _install_research_context_hardening(research_module: Any) -> None:
    cls = research_module.ResearchCodeContext

    current_init = cls.__init__
    if not getattr(current_init, _MARKER, False):

        @wraps(current_init)
        def init(self: Any, *args: Any, **kwargs: Any) -> None:
            current_init(self, *args, **kwargs)
            self._mmm_generation_retrieval_rounds = 0
            self._mmm_generation_retrieval_round_budget = _round_budget()

        setattr(init, _MARKER, True)
        init.__wrapped__ = current_init  # type: ignore[attr-defined]
        cls.__init__ = init

    current_paths = cls._query_paths
    if not getattr(current_paths, _MARKER, False):

        @wraps(current_paths)
        def query_paths(self: Any, query: str, plan_step: Any | None):
            paths = list(dict.fromkeys(current_paths(self, query, plan_step)))
            dependency_query = _dependency_neighborhood_query(
                research_module, self, query, plan_step
            )
            if dependency_query and dependency_query not in paths:
                # Exact query remains first. Dependency evidence displaces the broad
                # vocabulary fallback rather than increasing retrieval fanout.
                paths.insert(1 if paths else 0, dependency_query)
            budget = _query_path_budget()
            if len(paths) <= budget:
                return tuple(paths)
            focused = [path for path in paths if "known repository vocabulary" not in path]
            broad = [path for path in paths if "known repository vocabulary" in path]
            return tuple([*focused, *broad][:budget])

        setattr(query_paths, _MARKER, True)
        query_paths._mmm_query_path_budget = _QUERY_PATH_MAX  # type: ignore[attr-defined]
        query_paths.__wrapped__ = current_paths  # type: ignore[attr-defined]
        cls._query_paths = query_paths

    current_evolve = cls.evolve_from_generation
    if not getattr(current_evolve, _MARKER, False):

        @wraps(current_evolve)
        def evolve_from_generation(self: Any, text: str):
            rounds = int(getattr(self, "_mmm_generation_retrieval_rounds", 0))
            budget = int(
                getattr(self, "_mmm_generation_retrieval_round_budget", _round_budget())
            )
            if rounds >= budget:
                violations = self.monitor.validate_model_output(text)
                return (self.bundle(), violations) if violations else (None, ())
            before_queries = len(self.query_history)
            result = current_evolve(self, text)
            if len(self.query_history) > before_queries:
                self._mmm_generation_retrieval_rounds = rounds + 1
            return result

        setattr(evolve_from_generation, _MARKER, True)
        evolve_from_generation.__wrapped__ = current_evolve  # type: ignore[attr-defined]
        cls.evolve_from_generation = evolve_from_generation

    current_receipt = cls.receipt
    if not getattr(current_receipt, _MARKER, False):

        @wraps(current_receipt)
        def receipt(self: Any) -> dict[str, Any]:
            payload = dict(current_receipt(self))
            reusable = _reusable_evidence(self)
            payload.update(
                {
                    "target": {
                        "minecraft_version": self.minecraft_version,
                        "loader": self.loader,
                        "mappings": self.mappings,
                    },
                    "retrieval_round_budget": int(
                        getattr(self, "_mmm_generation_retrieval_round_budget", _round_budget())
                    ),
                    "retrieval_round_count": int(
                        getattr(self, "_mmm_generation_retrieval_rounds", 0)
                    ),
                    "query_history_tail": list(self.query_history[-8:]),
                    "reusable_evidence": reusable,
                    "reusable_evidence_sha256": research_module._sha(reusable),
                    "dependency_directed_retrieval": True,
                }
            )
            return payload

        setattr(receipt, _MARKER, True)
        receipt.__wrapped__ = current_receipt  # type: ignore[attr-defined]
        cls.receipt = receipt


def _receipt_file(root: Path) -> Path:
    return root / _RECEIPT_PATH


def _load_receipts(root: Path) -> list[dict[str, Any]]:
    path = _receipt_file(root)
    if not path.is_file() or path.is_symlink():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, Mapping):
        return []
    raw = payload.get("receipts", [])
    return [dict(item) for item in raw if isinstance(item, Mapping)] if isinstance(raw, list) else []


def _persist_research_receipt(
    root: Path,
    receipt: Mapping[str, Any],
    *,
    module_id: str,
) -> None:
    bundle = str(receipt.get("bundle_sha256", ""))
    if not bundle:
        return
    normalized_root = root.expanduser().resolve()
    path = _receipt_file(normalized_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _RECEIPT_LOCKS.hold(str(path.resolve())):
        receipts = _load_receipts(normalized_root)
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


def _install_generation_receipt_persistence(custom_module_generator_module: Any) -> None:
    cls = custom_module_generator_module.CustomModuleGenerator
    current = cls.generate
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def generate(self: Any, project_root: str | Path, *args: Any, **kwargs: Any):
        result = current(self, project_root, *args, **kwargs)
        if not isinstance(result, Mapping):
            return result
        receipt = result.get("research_code_context")
        if not isinstance(receipt, Mapping):
            return result
        module = kwargs.get("module")
        module_id = str(getattr(module, "module_id", "") or getattr(module, "kind", ""))
        _persist_research_receipt(
            Path(project_root).expanduser().resolve(),
            receipt,
            module_id=module_id,
        )
        return result

    setattr(generate, _MARKER, True)
    generate.__wrapped__ = current  # type: ignore[attr-defined]
    cls.generate = generate


def _flatten_diagnostics(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    diagnostics = evidence.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        return []
    raw = diagnostics.get("diagnostics", [])
    if isinstance(raw, Mapping):
        return [
            dict(item)
            for group in raw.values()
            if isinstance(group, list)
            for item in group
            if isinstance(item, Mapping)
        ]
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    return []


def _read_log_tail(command: Mapping[str, Any]) -> str:
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
                    pieces.append(handle.read(_LOG_READ_BYTES).decode("utf-8", errors="replace"))
            except OSError:
                pass
    return "\n".join(pieces)[-_LOG_TEXT_CHARS:]


def _diagnostic_signature_payload(evidence: Mapping[str, Any]) -> dict[str, Any]:
    files: set[str] = set()
    lines: set[str] = set()
    symbols: set[str] = set()
    exceptions: set[str] = set()
    messages: list[str] = []
    for item in _flatten_diagnostics(evidence):
        path = item.get("path") or item.get("uri")
        if isinstance(path, str) and path:
            files.add(path)
        start = item.get("range", {})
        if isinstance(start, Mapping):
            start = start.get("start", {})
            if isinstance(start, Mapping) and isinstance(start.get("line"), int):
                lines.add(f"{path}:{int(start['line']) + 1}")
        message = str(item.get("message", ""))
        if message:
            messages.append(message[:1000])
            symbols.update(_SYMBOL.findall(message))
            exceptions.update(_EXCEPTION.findall(message))
            for match in _SOURCE_LINE.finditer(message):
                files.add(match.group("path"))
                lines.add(f"{match.group('path')}:{match.group('line')}")

    build = evidence.get("build", {})
    tasks: set[str] = set()
    if isinstance(build, Mapping):
        build_error = build.get("error")
        if isinstance(build_error, str):
            messages.append(build_error[:2000])
            symbols.update(_SYMBOL.findall(build_error))
            exceptions.update(_EXCEPTION.findall(build_error))
        commands = build.get("commands", [])
        if isinstance(commands, list):
            for command in commands[-4:]:
                if not isinstance(command, Mapping):
                    continue
                task = command.get("task") or command.get("command") or command.get("name")
                if isinstance(task, str) and task:
                    tasks.add(task[:240])
                tail = _read_log_tail(command)
                if tail:
                    symbols.update(_SYMBOL.findall(tail))
                    exceptions.update(_EXCEPTION.findall(tail))
                    for match in _SOURCE_LINE.finditer(tail):
                        files.add(match.group("path"))
                        lines.add(f"{match.group('path')}:{match.group('line')}")
                    messages.append(" ".join(tail.split())[-2000:])

    return {
        "files": sorted(files)[:16],
        "lines": sorted(lines)[:16],
        "symbols": sorted(symbols)[:16],
        "tasks": sorted(tasks)[:8],
        "exceptions": sorted(exceptions)[:8],
        "messages": messages[-6:],
        "build_status": build.get("status") if isinstance(build, Mapping) else None,
    }


def _signature_key(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _prior_evidence_for_diagnostic(
    receipts: Sequence[Mapping[str, Any]],
    signature: Mapping[str, Any],
    *,
    limit: int = 8,
) -> tuple[list[dict[str, Any]], list[str]]:
    needles = {
        token.casefold()
        for key in ("files", "symbols", "tasks", "exceptions")
        for value in signature.get(key, [])
        if isinstance(value, str)
        for token in re.findall(r"[A-Za-z_$][A-Za-z0-9_.$/-]+", value)
    }
    selected: list[dict[str, Any]] = []
    parent_bundles: list[str] = []
    seen: set[str] = set()
    fallback: list[dict[str, Any]] = []
    for stored in reversed(receipts):
        bundle = str(stored.get("bundle_sha256", ""))
        if bundle:
            parent_bundles.append(bundle)
        receipt = stored.get("receipt", {})
        if not isinstance(receipt, Mapping):
            continue
        reusable = receipt.get("reusable_evidence", [])
        if not isinstance(reusable, list):
            continue
        for raw in reusable:
            if not isinstance(raw, Mapping):
                continue
            item = dict(raw)
            evidence_id = str(item.get("evidence_id", ""))
            if not evidence_id or evidence_id in seen:
                continue
            haystack = " ".join(
                [
                    str(item.get("path", "")),
                    " ".join(
                        str(value)
                        for value in item.get("symbols", [])
                        if isinstance(value, str)
                    ),
                    str(item.get("snippet", "")),
                ]
            ).casefold()
            fallback.append(item)
            if needles and any(needle in haystack for needle in needles):
                seen.add(evidence_id)
                selected.append(item)
                if len(selected) >= limit:
                    break
        if len(selected) >= limit:
            break
    if not selected:
        for item in fallback:
            evidence_id = str(item.get("evidence_id", ""))
            if evidence_id and evidence_id not in seen:
                seen.add(evidence_id)
                selected.append(item)
                if len(selected) >= min(4, limit):
                    break
    return selected, list(dict.fromkeys(parent_bundles))[:16]


def _install_repair_context_reuse(repair_module: Any) -> None:
    cls = repair_module.RepairEngine

    current_signature = cls._signature
    if not getattr(current_signature, _MARKER, False):

        def signature(evidence: dict[str, Any]) -> str:
            return _signature_key(_diagnostic_signature_payload(evidence))

        setattr(signature, _MARKER, True)
        signature.__wrapped__ = current_signature  # type: ignore[attr-defined]
        cls._signature = staticmethod(signature)

    current_context = cls._context
    if getattr(current_context, _MARKER, False):
        return

    @wraps(current_context)
    def context(self: Any, root: Path, evidence: dict[str, Any]) -> dict[str, Any]:
        normalized_root = root.expanduser().resolve()
        diagnostic = _diagnostic_signature_payload(evidence)
        signature = _signature_key(diagnostic)
        cache_key = f"{normalized_root}:{signature}"
        cache = getattr(self, "_mmm_diagnostic_context_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._mmm_diagnostic_context_cache = cache
        cached = cache.get(cache_key)
        if isinstance(cached, Mapping):
            return copy.deepcopy(dict(cached))

        base = dict(current_context(self, normalized_root, evidence))
        stored = _load_receipts(normalized_root)
        reused, parent_bundles = _prior_evidence_for_diagnostic(stored, diagnostic)
        receipt_payload = {
            "schema_version": "mmm/narrow-repair-evidence-receipt-v1",
            "diagnostic": diagnostic,
            "parent_research_bundles": parent_bundles,
            "reused_evidence_ids": [item.get("evidence_id") for item in reused],
            "manifest": base.get("manifest"),
        }
        diagnostic_receipt = _signature_key(receipt_payload)
        previous = getattr(self, "_mmm_last_diagnostic_receipt", "")
        base.update(
            {
                "diagnostic_signature": diagnostic,
                "prior_research_evidence": reused,
                "repair_evidence_receipt": {
                    **receipt_payload,
                    "receipt_sha256": "sha256:"
                    + hashlib.sha256(diagnostic_receipt.encode("utf-8")).hexdigest(),
                    "previous_diagnostic_receipt_sha256": previous,
                    "novel_diagnostic": True,
                },
                "retrieval_policy": {
                    "reuse_plan_coder_evidence_first": True,
                    "same_diagnostic_memoized": True,
                    "diagnostic_paths_and_symbols_first": True,
                    "full_project_reresearch": False,
                },
            }
        )
        self._mmm_last_diagnostic_receipt = base["repair_evidence_receipt"]["receipt_sha256"]
        cache[cache_key] = copy.deepcopy(base)
        while len(cache) > 16:
            cache.pop(next(iter(cache)))
        return base

    setattr(context, _MARKER, True)
    context.__wrapped__ = current_context  # type: ignore[attr-defined]
    context._mmm_narrow_diagnostic_repair_rag = True  # type: ignore[attr-defined]
    cls._context = context


def harden() -> None:
    """Apply only late, idempotent hardeners; package bootstrap owns composition."""

    from . import custom_generation_search_contract
    from . import custom_module_generator
    from . import repair_engine
    from . import research_code_context

    _install_research_context_hardening(research_code_context)
    _install_generation_receipt_persistence(custom_module_generator)
    _install_repair_context_reuse(repair_engine)
    custom_generation_search_contract._evolution_state_budget = _bounded_evolution_state_budget


__all__ = [
    "_bounded_evolution_state_budget",
    "_dependency_neighborhood_query",
    "_diagnostic_signature_payload",
    "_persist_research_receipt",
    "_prior_evidence_for_diagnostic",
    "_query_path_budget",
    "_round_budget",
    "harden",
]
