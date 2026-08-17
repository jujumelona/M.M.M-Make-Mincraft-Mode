from __future__ import annotations

import re
from functools import wraps
from typing import Any, Mapping, Sequence

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.$:/-]{1,127}")
_ANCHOR_WORDS = frozenset(
    {"api", "contract", "dependency", "implements", "interface", "register", "required", "schema"}
)
_FAILURE_MARKERS = (
    "validation failure",
    "execution & validation failure",
    "failed with reason",
    "compile error",
    "compilation error",
    "diagnostic",
)
_RETRIEVAL_REPAIR_MARKERS = (
    "compile error",
    "compilation error",
    "javac",
    "jdt",
    "cannot find symbol",
    "cannot resolve symbol",
    "unresolved symbol",
    "package does not exist",
    "package ",
    "no suitable method",
    "cannot be applied to given types",
    "incompatible types",
    "has private access",
    "missing method",
    "missing field",
    "unknown method",
    "unknown class",
    "api mismatch",
    "mapping mismatch",
    "yarn mapping",
    "dependency",
    "gradle",
    "maven",
    "version catalog",
    "classpath",
    "mixin target",
    "registry id",
    "registry entry",
    "resource id",
)


def _message_tail(messages: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(
        str(message.get("content", ""))
        for message in messages[-4:]
        if isinstance(message.get("content"), str)
    ).casefold()


def _is_structural_patch_repair(messages: Sequence[Mapping[str, Any]]) -> bool:
    tail = _message_tail(messages)
    return any(
        marker in tail
        for marker in (
            "repairing only the json/patch/precondition shape",
            "correct that exact structural failure",
            "구조 검증 피드백 기반",
        )
    )


def _is_repair_failure(messages: Sequence[Mapping[str, Any]]) -> bool:
    tail = _message_tail(messages)
    return any(marker in tail for marker in _FAILURE_MARKERS)


def _needs_retrieval_repair(messages: Sequence[Mapping[str, Any]]) -> bool:
    tail = _message_tail(messages)
    return any(marker in tail for marker in _RETRIEVAL_REPAIR_MARKERS)


def _compact_anchor(record: Mapping[str, Any], query_tokens: set[str]) -> dict[str, Any]:
    text = str(record.get("text", ""))
    tokens = list(dict.fromkeys(_TOKEN.findall(text)))
    selected: list[str] = []
    for token in tokens:
        lowered = token.casefold()
        if (
            lowered in query_tokens
            or lowered in _ANCHOR_WORDS
            or "_" in token
            or any(character.isupper() for character in token[1:])
        ):
            selected.append(token)
        if len(selected) >= 16:
            break
    if not selected:
        selected = tokens[:8]
    return {
        "observation_id": str(record.get("observation_id", "")),
        "path": str(record.get("path", "")),
        "sha256": str(record.get("sha256", "")),
        "content_start_bytes": int(record.get("content_start_bytes", 0) or 0),
        "content_end_bytes": int(record.get("content_end_bytes", 0) or 0),
        "source_page_index": int(record.get("source_page_index", 0) or 0),
        "kind": "exact_source_anchor_ref",
        "text": "anchor-ref symbols: " + " ".join(selected),
    }


def _install_anchor_compaction(custom_module_generator_module: Any) -> None:
    current = custom_module_generator_module._observation_context_pages
    if getattr(current, "_mmm_first_page_anchor_payload", False):
        return

    @wraps(current)
    def compacted(
        ledger: dict[str, Any],
        *,
        query: str,
        byte_budget: int,
    ):
        pages = [dict(page) for page in current(ledger, query=query, byte_budget=byte_budget)]
        if not pages:
            return tuple(pages)
        query_tokens = {token.casefold() for token in _TOKEN.findall(query)}
        anchors = pages[0].get("global_anchors", [])
        anchors = anchors if isinstance(anchors, list) else []
        refs = [
            _compact_anchor(record, query_tokens)
            for record in anchors
            if isinstance(record, Mapping)
        ]
        for index, page in enumerate(pages):
            policy = page.get("policy")
            policy = dict(policy) if isinstance(policy, Mapping) else {}
            policy["global_anchor_source_payload"] = "first_page_only"
            page["policy"] = policy
            page["global_anchor_payload"] = "exact_source" if index == 0 else "compact_refs"
            if index > 0:
                page["global_anchors"] = refs
            page["global_anchor_ref_bytes"] = (
                custom_module_generator_module._json_size(refs) if refs else 0
            )
        return tuple(pages)

    compacted._mmm_first_page_anchor_payload = True  # type: ignore[attr-defined]
    compacted.__wrapped__ = current  # type: ignore[attr-defined]
    custom_module_generator_module._observation_context_pages = compacted


def _install_structural_repair_bypass(custom_generation_search_module: Any) -> None:
    cls = custom_generation_search_module._ResearchEvidenceRouter
    current = cls.generate_text
    if getattr(current, "_mmm_structural_no_rag", False):
        return

    @wraps(current)
    def generate_text(
        self: Any,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> str:
        repair_failure = role == "coder" and _is_repair_failure(messages)
        use_current_evidence_only = repair_failure and (
            _is_structural_patch_repair(messages)
            or not _needs_retrieval_repair(messages)
        )
        if use_current_evidence_only:
            sanitized = custom_generation_search_module._sanitized_messages(
                messages,
                minecraft_version=self._minecraft_version,
                loader=self._loader,
                mappings=self._mappings,
            )
            return self._router.generate_text(role, sanitized, **kwargs)
        return current(self, role, messages, **kwargs)

    generate_text._mmm_structural_no_rag = True  # type: ignore[attr-defined]
    generate_text._mmm_selective_repair_rag = True  # type: ignore[attr-defined]
    generate_text.__wrapped__ = current  # type: ignore[attr-defined]
    cls.generate_text = generate_text


def install() -> None:
    from . import custom_generation_search_contract, custom_module_generator

    _install_anchor_compaction(custom_module_generator)
    _install_structural_repair_bypass(custom_generation_search_contract)


__all__ = ["install"]
