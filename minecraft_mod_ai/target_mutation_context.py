from __future__ import annotations

"""Target mutation localization and evidence-derived context.

This module owns target-path normalization for observations, task-anchor authority
projection, context merging, and extraction of mutation context from tool payloads.
The execution loop consumes this state but does not reimplement it.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from .owned_target_contract import (
    normalize_target_status,
    target_is_creatable,
    target_is_existing,
    target_is_writable,
)
from .value_shapes import as_sequence as _sequence


class LocalizationStage(str, Enum):
    NEED_FILE = "NEED_FILE"
    NEED_SYMBOL = "NEED_SYMBOL"
    NEED_BODY = "NEED_BODY"
    READY = "READY"


_EXISTING_TARGET_EVIDENCE_SOURCES = frozenset({
    "host_exact_source",
    "mutation_receipt",
    "search_code_rag",
    "workspace_existing_target",
    "verifier_workspace_source",
})
_CODE_MARKERS = frozenset({
    "class ", "interface ", "enum ", "record ", "public ", "private ", "protected ",
    "package ", "import ", "void ", "return ", "final ", "static ", "new ",
    "extends ", "implements ", "override", "{", "}", ";", "(", ")",
})


def _canonical_mutation_path(value: Any) -> str:
    clean = str(value or "").strip().replace("\\", "/")
    return re.sub(r"^(?:\./)+", "", clean)


def _is_workspace_file_path(path: str) -> bool:
    clean = _canonical_mutation_path(path)
    if not clean or clean.startswith("/") or ":" in clean:
        return False
    if ".." in Path(clean).parts:
        return False
    return "/" in clean or Path(clean).suffix.casefold() in {
        ".java", ".json", ".toml", ".gradle", ".properties", ".txt", ".md",
        ".kt", ".groovy",
    }


def _is_code_bearing_text(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if len(stripped) < 15:
        return False
    lower = stripped.casefold()
    return any(marker in lower for marker in _CODE_MARKERS)


def _evidence_task_from_module(module: Any) -> Mapping[str, Any] | None:
    if not isinstance(module, Mapping):
        return None
    direct = module.get("evidence_task")
    if isinstance(direct, Mapping):
        return direct
    config = module.get("config")
    if isinstance(config, Mapping):
        nested = config.get("evidence_task")
        if isinstance(nested, Mapping):
            return nested
    return None


def _anchor_path_and_symbol(anchor: Mapping[str, Any]) -> tuple[str, str]:
    locator = str(anchor.get("locator") or "").replace("\\", "/").strip()
    raw_path, sep, symbol = locator.partition("#")
    path = _canonical_mutation_path(raw_path)
    if not _is_workspace_file_path(path):
        return "", ""
    return path, symbol.strip() if sep else ""


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _anchor_candidate(
    anchor: Mapping[str, Any],
    *,
    reuse: str,
) -> tuple[str, str, bool, bool] | None:
    path, symbol = _anchor_path_and_symbol(anchor)
    if not path:
        return None
    status = normalize_target_status(anchor.get("status"))
    if status and not target_is_writable(status):
        return None
    if target_is_existing(status):
        fresh = False
    elif reuse:
        fresh = reuse == "fresh"
    else:
        fresh = target_is_creatable(status)
    symbolic = str(anchor.get("kind") or "").strip().casefold() == "symbol"
    return path, symbol, fresh, symbolic


def _collect_task_anchors(
    anchors: Sequence[Any],
    *,
    reuse: str,
    candidates: list[tuple[str, str, bool]],
    writable: list[str],
    creatable: list[str],
) -> None:
    for raw_anchor in anchors:
        if not isinstance(raw_anchor, Mapping):
            continue
        candidate = _anchor_candidate(raw_anchor, reuse=reuse)
        if candidate is None:
            continue
        path, symbol, fresh, symbolic = candidate
        _append_unique(writable, path)
        reserved = target_is_creatable(raw_anchor.get("status"))
        if fresh or reserved:
            _append_unique(creatable, path)
        if symbolic:
            candidates.append((path, symbol, fresh))


def _collect_authority_candidates(
    task: Mapping[str, Any] | None,
    *,
    direct_reuse: str,
    writable: list[str],
    creatable: list[str],
) -> list[tuple[str, str, bool]]:
    candidates: list[tuple[str, str, bool]] = []
    if not isinstance(task, Mapping):
        return candidates
    task_reuse = str(task.get("reuse_action") or "").strip().casefold()
    for raw_binding in _sequence(task.get("production_bindings")):
        if not isinstance(raw_binding, Mapping):
            continue
        reuse = str(
            raw_binding.get("reuse_action") or task_reuse or direct_reuse
        ).strip().casefold()
        _collect_task_anchors(
            _sequence(raw_binding.get("owned_anchors")),
            reuse=reuse,
            candidates=candidates,
            writable=writable,
            creatable=creatable,
        )
    task_candidates = candidates if not candidates else []
    _collect_task_anchors(
        _sequence(task.get("owned_anchors")),
        reuse=task_reuse or direct_reuse,
        candidates=task_candidates,
        writable=writable,
        creatable=creatable,
    )
    return candidates


def _choose_authority_candidate(
    direct_primary: str,
    direct_reuse: str,
    candidates: Sequence[tuple[str, str, bool]],
) -> tuple[str, str, bool] | None:
    if direct_primary:
        matching = next((item for item in candidates if item[0] == direct_primary), None)
        return matching or (direct_primary, "", direct_reuse == "fresh")
    unique = tuple(dict.fromkeys(candidates))
    return unique[0] if len(unique) == 1 else None


def _task_anchor_status(task: Mapping[str, Any] | None, path: str) -> str:
    if not isinstance(task, Mapping):
        return ""
    groups: list[Any] = []
    for binding in _sequence(task.get("production_bindings")):
        if isinstance(binding, Mapping):
            groups.extend(_sequence(binding.get("owned_anchors")))
    groups.extend(_sequence(task.get("owned_anchors")))
    for anchor in groups:
        if not isinstance(anchor, Mapping):
            continue
        anchor_path, _symbol = _anchor_path_and_symbol(anchor)
        if anchor_path == path:
            status = normalize_target_status(anchor.get("status"))
            if status:
                return status
    return ""


@dataclass(frozen=True)
class TargetMutationContext:
    target_path: str | None = None
    target_symbol: str | None = None
    source_body: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    is_new_file: bool = False
    evidence_source: str | None = None
    base_revision_sha: str | None = None
    writable_paths: tuple[str, ...] = ()
    creatable_paths: tuple[str, ...] = ()
    target_pinned: bool = False

    @property
    def localization_stage(self) -> LocalizationStage:
        if self.is_new_file:
            return (
                LocalizationStage.READY
                if _canonical_mutation_path(self.target_path)
                else LocalizationStage.NEED_FILE
            )
        if not _canonical_mutation_path(self.target_path):
            return LocalizationStage.NEED_FILE
        if self.source_body and _is_code_bearing_text(self.source_body):
            return LocalizationStage.READY
        if not self.target_symbol and self.start_line is None:
            return LocalizationStage.NEED_SYMBOL
        return LocalizationStage.NEED_BODY

    @property
    def is_mutation_ready(self) -> bool:
        return self.localization_stage == LocalizationStage.READY

    def merge(self, other: "TargetMutationContext") -> "TargetMutationContext":
        return _merge_target_context(self, other)


def _task_authority_context(payload: Mapping[str, Any]) -> TargetMutationContext | None:
    task = _evidence_task_from_module(payload.get("module"))
    direct_primary = _canonical_mutation_path(payload.get("primary_path"))
    direct_reuse = str(payload.get("reuse_action") or "").strip().casefold()
    writable = [
        path
        for item in _sequence(payload.get("writable_paths"))
        if _is_workspace_file_path(path := _canonical_mutation_path(item))
    ]
    creatable: list[str] = []
    candidates = _collect_authority_candidates(
        task,
        direct_reuse=direct_reuse,
        writable=writable,
        creatable=creatable,
    )
    if direct_primary:
        _append_unique(writable, direct_primary)
    chosen = _choose_authority_candidate(direct_primary, direct_reuse, candidates)
    if chosen is None:
        return None
    path, symbol, fresh = chosen
    _append_unique(writable, path)
    if fresh:
        _append_unique(creatable, path)
    status = _task_anchor_status(task, path)
    if isinstance(task, Mapping) and target_is_existing(status):
        evidence_source = "evidence_existing_owned_anchor"
    elif fresh and isinstance(task, Mapping):
        evidence_source = "evidence_fresh_owned_anchor"
    elif isinstance(task, Mapping) and target_is_creatable(status):
        evidence_source = "evidence_host_reserved_owned_anchor"
    else:
        evidence_source = "host_task_authority"
    return TargetMutationContext(
        target_path=path,
        target_symbol=symbol or None,
        is_new_file=fresh,
        evidence_source=evidence_source,
        writable_paths=tuple(writable),
        creatable_paths=tuple(creatable),
        target_pinned=True,
    )


def _paths_conflict(left: str, right: str) -> bool:
    return bool(left and right and left != right)


def _conflict_resolution(
    current: TargetMutationContext,
    other: TargetMutationContext,
    left: str,
    right: str,
) -> TargetMutationContext | None:
    if not _paths_conflict(left, right):
        return None
    if current.target_pinned:
        return current
    return other


def _existing_target_context(
    current: TargetMutationContext,
    other: TargetMutationContext,
) -> TargetMutationContext | None:
    if (
        not other.is_new_file
        and str(other.evidence_source or "").strip() in _EXISTING_TARGET_EVIDENCE_SOURCES
    ):
        return other
    if (
        not current.is_new_file
        and str(current.evidence_source or "").strip() in _EXISTING_TARGET_EVIDENCE_SOURCES
    ):
        return current
    return None


def _preferred_context_value(
    existing: TargetMutationContext | None,
    existing_value: Any,
    other_value: Any,
    current_value: Any,
) -> Any:
    if existing is not None and existing_value is not None:
        return existing_value
    if other_value is not None:
        return other_value
    return current_value


def _without_target_path(paths: Sequence[str], target: str) -> tuple[str, ...]:
    return tuple(item for item in paths if _canonical_mutation_path(item) != target)


def _merged_is_new_file(
    existing: TargetMutationContext | None,
    current: TargetMutationContext,
    other: TargetMutationContext,
) -> bool:
    if existing is not None:
        return False
    return any((current.is_new_file, other.is_new_file))


def _merged_evidence_source(
    existing: TargetMutationContext | None,
    current: TargetMutationContext,
    other: TargetMutationContext,
) -> str | None:
    if existing is not None:
        return existing.evidence_source
    return other.evidence_source or current.evidence_source


def _merge_target_context(
    current: TargetMutationContext,
    other: TargetMutationContext,
) -> TargetMutationContext:
    left = _canonical_mutation_path(current.target_path)
    right = _canonical_mutation_path(other.target_path)
    conflict = _conflict_resolution(current, other, left, right)
    if conflict is not None:
        return conflict
    writable = tuple(dict.fromkeys((*current.writable_paths, *other.writable_paths)))
    creatable = tuple(dict.fromkeys((*current.creatable_paths, *other.creatable_paths)))
    existing = _existing_target_context(current, other)
    target_path = other.target_path or current.target_path
    target = _canonical_mutation_path(target_path)
    if existing is not None and target:
        creatable = _without_target_path(creatable, target)
    return TargetMutationContext(
        target_path=target_path,
        target_symbol=other.target_symbol or current.target_symbol,
        source_body=_preferred_context_value(
            existing,
            existing.source_body if existing is not None else None,
            other.source_body,
            current.source_body,
        ),
        start_line=_preferred_context_value(
            existing,
            existing.start_line if existing is not None else None,
            other.start_line,
            current.start_line,
        ),
        end_line=_preferred_context_value(
            existing,
            existing.end_line if existing is not None else None,
            other.end_line,
            current.end_line,
        ),
        is_new_file=_merged_is_new_file(existing, current, other),
        evidence_source=_merged_evidence_source(existing, current, other),
        base_revision_sha=_preferred_context_value(
            existing,
            existing.base_revision_sha if existing is not None else None,
            other.base_revision_sha,
            current.base_revision_sha,
        ),
        writable_paths=writable,
        creatable_paths=creatable,
        target_pinned=any((current.target_pinned, other.target_pinned)),
    )


def _first_mapping_value(
    mappings: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> Any:
    for mapping in mappings:
        for key in keys:
            value = mapping.get(key)
            if value not in (None, ""):
                return value
    return None


def _search_hit_text(hit: Mapping[str, Any]) -> str | None:
    raw = _first_mapping_value(
        (hit,),
        ("text", "snippet", "code", "content", "source"),
    )
    if isinstance(raw, (list, tuple)):
        return "\n".join(str(item) for item in raw)
    return raw if isinstance(raw, str) else None


def _search_hit_context(hit: Mapping[str, Any]) -> TargetMutationContext | None:
    raw_meta = hit.get("metadata")
    meta = raw_meta if isinstance(raw_meta, Mapping) else {}
    path = _canonical_mutation_path(
        _first_mapping_value(
            (hit, meta),
            ("source_path", "path", "file", "uri"),
        )
    )
    if not _is_workspace_file_path(path):
        return None
    text = _search_hit_text(hit)
    symbol = str(
        _first_mapping_value((hit, meta), ("symbol", "function", "name")) or ""
    ).strip()
    start_line = hit.get("start_line")
    end_line = hit.get("end_line")
    return TargetMutationContext(
        target_path=path,
        target_symbol=symbol or None,
        source_body=text if _is_code_bearing_text(text) else None,
        start_line=start_line if isinstance(start_line, int) else None,
        end_line=end_line if isinstance(end_line, int) else None,
        evidence_source="search_code_rag",
    )


def _extract_search_context(payload: Mapping[str, Any]) -> TargetMutationContext | None:
    hits = payload.get("hits") or payload.get("results") or payload.get("sources")
    for hit in _sequence(hits):
        if not isinstance(hit, Mapping):
            continue
        context = _search_hit_context(hit)
        if context is not None:
            return context
    return None


def _extract_mutation_context_from_payload(payload: Any) -> TargetMutationContext | None:
    if not isinstance(payload, Mapping):
        for item in _sequence(payload):
            context = _extract_mutation_context_from_payload(item)
            if context is not None:
                return context
        return None

    authority = _task_authority_context(payload)
    if authority is not None:
        initial = payload.get("initial_exact_source_context")
        if isinstance(initial, Mapping):
            exact_candidates: list[TargetMutationContext] = []
            direct_exact = _extract_mutation_context_from_payload(initial)
            if direct_exact is not None:
                exact_candidates.append(direct_exact)
            for record in _sequence(initial.get("records")):
                exact = _extract_mutation_context_from_payload(record)
                if exact is not None:
                    exact_candidates.append(exact)
            expected_path = _canonical_mutation_path(authority.target_path)
            exact = next(
                (
                    candidate
                    for candidate in exact_candidates
                    if _canonical_mutation_path(candidate.target_path) == expected_path
                    and candidate.source_body
                ),
                None,
            )
            if exact is not None:
                return replace(
                    authority,
                    source_body=exact.source_body,
                    start_line=exact.start_line,
                    end_line=exact.end_line,
                    is_new_file=False,
                    evidence_source="host_exact_source",
                    creatable_paths=_without_target_path(
                        authority.creatable_paths,
                        expected_path,
                    ),
                )
        return authority

    for key in (
        "structured_content",
        "result",
        "data",
        "body",
        "_mmm_observation",
        "raw_result",
        "structured",
        "observation",
    ):
        wrapped = payload.get(key)
        if isinstance(wrapped, (Mapping, list, tuple)) and wrapped is not payload:
            context = _extract_mutation_context_from_payload(wrapped)
            if context is not None:
                return context

    context = _extract_search_context(payload)
    if context is not None:
        return context

    symbols = payload.get("symbols")
    for symbol in _sequence(symbols):
        if not isinstance(symbol, Mapping):
            continue
        location = symbol.get("location")
        if not isinstance(location, Mapping):
            continue
        uri = str(location.get("uri") or "")
        path = _canonical_mutation_path(uri.replace("file:///", "").replace("file://", ""))
        if not path:
            continue
        name = str(symbol.get("name") or "").strip()
        container_name = str(
            symbol.get("containerName")
            or symbol.get("container_name")
            or symbol.get("container")
            or ""
        ).strip()
        target_symbol = f"{container_name}#{name}" if container_name and name else name or None
        symbol_range = location.get("range")
        start_line = None
        end_line = None
        if isinstance(symbol_range, Mapping):
            start = symbol_range.get("start")
            end = symbol_range.get("end")
            if isinstance(start, Mapping) and isinstance(start.get("line"), int):
                start_line = int(start["line"])
            if isinstance(end, Mapping) and isinstance(end.get("line"), int):
                end_line = int(end["line"])
        return TargetMutationContext(
            target_path=path,
            target_symbol=target_symbol,
            start_line=start_line,
            end_line=end_line,
            evidence_source="java_workspace_symbols",
        )

    files = payload.get("files")
    if isinstance(files, Mapping):
        for raw_path, raw_content in files.items():
            path = _canonical_mutation_path(raw_path)
            if _is_workspace_file_path(path):
                return TargetMutationContext(
                    target_path=path,
                    source_body=(
                        str(raw_content)
                        if _is_code_bearing_text(str(raw_content))
                        else None
                    ),
                    evidence_source="files_map",
                )

    target = _canonical_mutation_path(payload.get("target_file") or payload.get("path"))
    if target and _is_workspace_file_path(target):
        source = payload.get("source") or payload.get("content") or payload.get("code")
        return TargetMutationContext(
            target_path=target,
            source_body=source if _is_code_bearing_text(source) else None,
            evidence_source="target_path_field",
        )
    return None


def _mutation_context_dict(ctx: TargetMutationContext | None) -> dict[str, Any] | None:
    if ctx is None:
        return None
    return {
        "target_path": ctx.target_path,
        "target_symbol": ctx.target_symbol,
        "source_body_len": len(ctx.source_body) if ctx.source_body else 0,
        "start_line": ctx.start_line,
        "end_line": ctx.end_line,
        "is_new_file": ctx.is_new_file,
        "localization_stage": ctx.localization_stage.value,
        "evidence_source": ctx.evidence_source,
        "writable_paths": list(ctx.writable_paths),
        "creatable_paths": list(ctx.creatable_paths),
        "target_pinned": ctx.target_pinned,
    }
