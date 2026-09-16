from __future__ import annotations

"""Close the fresh-Java OBSERVE -> ACT grounding transition.

A host-reserved fresh Java target is already localized: its path and symbol are immutable
host authority, while only implementation/API evidence is missing. The generic progress
loop deliberately does not count novel evidence as implementation progress unless it
changes file/symbol/body localization. That rule is correct for existing-file repair but
left fresh targets trapped in OBSERVE after successful code RAG.

This installer keeps the generic rule intact and adds the narrower fresh-Java invariant:
request code-bearing RAG first, then move directly to ACT once a usable code-RAG receipt
has been recorded. Infrastructure-only symbol lookup is therefore not required to create
a brand-new, already host-localized Java file.
"""

from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

_MARKER = "_mmm_fresh_java_grounding_transition"
_CODE_RAG_SCHEMA = "mmm/code-rag-result-v1"
_CODE_RAG_TOOL = "search_code_rag"


def _contains_code_rag_hits(value: Any) -> bool:
    if isinstance(value, Mapping):
        if str(value.get("schema_version") or "").strip() == _CODE_RAG_SCHEMA:
            hits = value.get("hits")
            if (
                isinstance(hits, Sequence)
                and not isinstance(hits, (str, bytes, bytearray))
                and bool(hits)
            ):
                receipt = value.get("receipt")
                if not isinstance(receipt, Mapping):
                    return True
                try:
                    return int(receipt.get("result_count", len(hits)) or 0) > 0
                except (TypeError, ValueError, OverflowError):
                    return False
        return any(_contains_code_rag_hits(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_code_rag_hits(child) for child in value)
    return False


def _is_fresh_java_ready(context: Any) -> bool:
    if context is None:
        return False
    path = str(getattr(context, "target_path", "") or "").replace("\\", "/").strip()
    return bool(
        getattr(context, "is_new_file", False)
        and getattr(context, "is_mutation_ready", False)
        and path.endswith(".java")
    )


def install(progress_loop_module: Any) -> None:
    state_cls = progress_loop_module.HostRunState

    current_filter = progress_loop_module._filter_tools_for_phase
    if not getattr(current_filter, _MARKER, False):

        @wraps(current_filter)
        def filter_tools_for_fresh_java(
            exposed_tools: Any,
            phase: Any,
            role: str,
            *,
            mutation_context: Any = None,
            attempted_sources: Any = frozenset(),
            localization_active: bool | None = None,
            semantic_retrieval_choice: bool = False,
        ) -> tuple[Mapping[str, Any], ...]:
            selected = tuple(
                current_filter(
                    exposed_tools,
                    phase,
                    role,
                    mutation_context=mutation_context,
                    attempted_sources=attempted_sources,
                    localization_active=localization_active,
                    semantic_retrieval_choice=semantic_retrieval_choice,
                )
            )
            if not (
                phase == progress_loop_module.LoopPhase.OBSERVE
                and semantic_retrieval_choice
                and _is_fresh_java_ready(mutation_context)
            ):
                return selected
            code_rag = tuple(
                schema
                for schema in selected
                if progress_loop_module._tool_name(schema) == _CODE_RAG_TOOL
            )
            return code_rag or selected

        setattr(filter_tools_for_fresh_java, _MARKER, True)
        filter_tools_for_fresh_java.__wrapped__ = current_filter
        progress_loop_module._filter_tools_for_phase = filter_tools_for_fresh_java

    current_record = state_cls.record_evidence
    if not getattr(current_record, _MARKER, False):

        @wraps(current_record)
        def record_evidence_and_advance(
            self: Any,
            value: Any,
            *,
            usable: bool,
        ) -> bool:
            recorded = current_record(self, value, usable=usable)
            if not recorded or not usable or not _contains_code_rag_hits(value):
                return recorded

            transitioned = False
            with self._lock:
                if (
                    self.phase == progress_loop_module.LoopPhase.OBSERVE
                    and _is_fresh_java_ready(self.mutation_context)
                    and not self.workspace_changed
                ):
                    self.phase = progress_loop_module.LoopPhase.ACT
                    # A successful baseline code-RAG observation is real causal progress
                    # for a fresh target even though localization itself was already READY.
                    self.seen_no_progress_digests.clear()
                    self.semantic_fixed_point = False
                    self.no_progress_streak = 0
                    self.last_failure_reason = None
                    self.last_failure_digest = None
                    transitioned = True

            if transitioned:
                progress_loop_module.emit_root_cause(
                    "fresh_java_grounding_transition",
                    stage="generation",
                    operation="search_code_rag",
                    gate="progress_adjudication",
                    result="PASS",
                    reason="usable code RAG satisfied fresh-target grounding; advancing OBSERVE to ACT",
                    details={
                        "target_path": getattr(self.mutation_context, "target_path", None),
                        "phase": self.phase.value,
                    },
                )
            return recorded

        setattr(record_evidence_and_advance, _MARKER, True)
        record_evidence_and_advance.__wrapped__ = current_record
        state_cls.record_evidence = record_evidence_and_advance


def assert_installed(progress_loop_module: Any) -> None:
    checks = (
        getattr(progress_loop_module._filter_tools_for_phase, _MARKER, False),
        getattr(progress_loop_module.HostRunState.record_evidence, _MARKER, False),
    )
    if not all(checks):
        raise RuntimeError("Fresh Java grounding transition is not installed.")


__all__ = ["assert_installed", "install"]
