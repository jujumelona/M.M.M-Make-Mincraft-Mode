from __future__ import annotations

"""Late inference-time extensions for maximizing one frozen small agent.

This contract intentionally reuses the production owners that already exist in MMM:

* verifier-qualified v3 trajectories remain the only durable experience source;
* ProjectRAGIndex remains the only code-RAG implementation;
* causal_tool_frontier remains the tool-set/action-surface owner;
* deterministic validators remain authoritative over model self-judgement.

The extensions here only fill research-backed gaps around adaptive compute, procedural
skill evolution, intent-routed experience retrieval, instruction projection, and
executor-specific skill rendering. No model weights are trained or modified.
"""

import json
import os
import re
from collections import Counter
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence

from .qwen_model_profiles import qwen_family

_INSTALLED = False
_CAPABILITY_PREFIX = "MMM reviewed Skill/tool/Minecraft-MCP routing context:\n"
_TEMPORARY_SKILL_PREFIX = "MMM TEMPORARY VERIFIED SKILL:\n"
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.:$<>/-]{1,127}|[가-힣]{2,}")
_MODEL_FAMILY: ContextVar[str] = ContextVar(
    "mmm_small_model_executor_family",
    default="generic",
)
_COMPLEX_MEMORY_MARKERS = (
    "multi-file",
    "multi file",
    "cross-file",
    "cross file",
    "stack trace",
    "migration",
    "integration",
    "dependency chain",
    "call graph",
    "regression",
    "multiple errors",
    "여러 파일",
    "의존성",
    "마이그레이션",
    "통합",
)


def _tokens(value: str) -> set[str]:
    return {item.casefold() for item in _TOKEN.findall(value)}


def _memory_route(query: str, task_class: str, requested_limit: int) -> tuple[str, int]:
    """Choose retrieval depth with cheap host features, never another model call."""

    limit = max(2, min(12, int(requested_limit)))
    terms = _tokens(query)
    lowered = query.casefold()
    high_risk = task_class in {"repair", "build", "runtime", "release", "quality"}
    complex_query = (
        len(terms) >= 72
        or sum(marker in lowered for marker in _COMPLEX_MEMORY_MARKERS) >= 2
        or query.count("\n") >= 12
    )
    if complex_query:
        return "deep", max(limit, 8)
    if high_risk or len(terms) >= 28:
        return "targeted", limit
    return "exact", min(limit, 3)


def _explicit_legacy_workflow_path() -> Path | None:
    """Keep the old planner workflow file as explicit opt-in compatibility only.

    Durable success/failure experience is already owned by trajectory_memory and the
    remote mmm-data v3 store. Falling back to MMM_WORKSPACE here would create a second
    competing persistent memory.
    """

    explicit = os.environ.get("MMM_AGENT_WORKFLOW_MEMORY_PATH", "").strip()
    return Path(explicit).expanduser() if explicit else None


def _schema_tool_names(tool_schemas: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    result: set[str] = set()
    for schema in tool_schemas:
        function = schema.get("function")
        if not isinstance(function, Mapping):
            continue
        name = str(function.get("name", "")).strip()
        if name:
            result.add(name)
    return frozenset(result)


def _project_capability_payload(
    payload: Mapping[str, Any],
    *,
    exposed_tools: frozenset[str],
) -> dict[str, Any]:
    """Project canonical Skill instructions onto the already-authorized tool frontier."""

    projected = dict(payload)
    raw_skills = payload.get("eligible_skills")
    skills = list(raw_skills) if isinstance(raw_skills, list) else []
    if not exposed_tools or not skills:
        return projected

    relevant: list[dict[str, Any]] = []
    for raw in skills:
        if not isinstance(raw, Mapping):
            continue
        model_tools = {
            str(item).strip()
            for item in raw.get("model_tools", ())
            if str(item).strip()
        }
        if model_tools & exposed_tools:
            relevant.append(dict(raw))

    # The execution router/Skill catalog remains the authorization owner. This
    # projection only reduces prompt instructions, so an empty relevant set is valid
    # for meta-tools such as external MCP discovery.
    projected["eligible_skills"] = relevant
    projected["instruction_projection"] = {
        "schema_version": "mmm/instruction-projection-v1",
        "source": "authorized_tool_frontier",
        "candidate_skill_count": len(skills),
        "selected_skill_count": len(relevant),
        "exposed_tool_count": len(exposed_tools),
        "policy": "prompt_projection_only_authorization_unchanged",
    }
    return projected


def _install_instruction_projection(agent_capability_context: Any) -> None:
    current = agent_capability_context.build_agent_capability_context
    if getattr(current, "_mmm_instruction_projection_v1", False):
        return

    @wraps(current)
    def projected(
        stage: str,
        tool_schemas: Sequence[Mapping[str, Any]],
        *,
        model_role: str = "",
    ) -> str:
        rendered = current(stage, tool_schemas, model_role=model_role)
        if not rendered.startswith(_CAPABILITY_PREFIX):
            return rendered
        try:
            payload = json.loads(rendered[len(_CAPABILITY_PREFIX) :])
        except (TypeError, json.JSONDecodeError):
            return rendered
        if not isinstance(payload, Mapping):
            return rendered
        compact = _project_capability_payload(
            payload,
            exposed_tools=_schema_tool_names(tool_schemas),
        )
        return _CAPABILITY_PREFIX + json.dumps(
            compact,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    projected._mmm_instruction_projection_v1 = True  # type: ignore[attr-defined]
    projected.__wrapped__ = current  # type: ignore[attr-defined]
    agent_capability_context.build_agent_capability_context = projected


def _install_intent_routed_memory(trajectory_memory: Any) -> None:
    current = trajectory_memory.relevant_trajectories
    if getattr(current, "_mmm_intent_routed_memory_v1", False):
        return

    @wraps(current)
    def routed(
        base: str | Path,
        query: str,
        *,
        task_class: str,
        router: Any | None = None,
        limit: int = 6,
        current_context: Mapping[str, Any] | None = None,
    ):
        _tier, routed_limit = _memory_route(query, task_class, limit)
        return current(
            base,
            query,
            task_class=task_class,
            router=router,
            limit=routed_limit,
            current_context=current_context,
        )

    routed._mmm_intent_routed_memory_v1 = True  # type: ignore[attr-defined]
    routed.__wrapped__ = current  # type: ignore[attr-defined]
    trajectory_memory.relevant_trajectories = routed


def _verified_failure(record: Mapping[str, Any]) -> bool:
    from .trajectory_record_integrity import derive_levels, validate_trajectory_record

    derived = derive_levels(record)
    return bool(
        derived
        and validate_trajectory_record(record)
        and derived.get("verified_failure") is True
    )


def _skill_relations(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive a compact procedural relation graph from verified action sequences."""

    from .procedure_trace import sequence_actions
    from .trajectory_record_integrity import record_strong_skill_eligible

    proven: Counter[tuple[str, str]] = Counter()
    avoid: Counter[tuple[str, str]] = Counter()
    for record in records[:16]:
        strong = record_strong_skill_eligible(record)
        failed = _verified_failure(record)
        if not strong and not failed:
            continue
        procedure = record.get("procedure")
        actions = sequence_actions(procedure if isinstance(procedure, Mapping) else None)
        target = proven if strong else avoid
        for left, right in zip(actions[:16], actions[1:17]):
            if left and right and left != right:
                target[(left, right)] += 1

    def rows(counter: Counter[tuple[str, str]], limit: int) -> list[dict[str, Any]]:
        return [
            {"from": left, "to": right, "support": count}
            for (left, right), count in counter.most_common(limit)
        ]

    return {
        "schema_version": "mmm/procedural-relation-graph-v1",
        "proven_transitions": rows(proven, 8),
        "avoid_transitions": rows(avoid, 6),
    }


def _evolve_skill(
    base_skill: Mapping[str, Any],
    query: str,
    records: Sequence[Mapping[str, Any]],
    *,
    task_class: str,
) -> dict[str, Any]:
    """Add a fingerprint-cached evolving playbook without creating another store."""

    skill = dict(base_skill)
    relations = _skill_relations(records)
    tier, _limit = _memory_route(query, task_class, 6)
    skill["memory_route"] = {
        "schema_version": "mmm/intent-routed-memory-v1",
        "tier": tier,
        "task_class": task_class,
    }
    skill["skill_relations"] = relations
    skill["evolving_playbook"] = {
        "schema_version": "mmm/evolving-playbook-v1",
        "activation_terms": sorted(_tokens(query))[:24],
        "proven_transitions": relations["proven_transitions"][:6],
        "avoid_transitions": relations["avoid_transitions"][:4],
        "verification_contract": (
            "Replay only transitions supported by verifier-qualified trajectories; "
            "re-check current preconditions and let current compiler/test/runtime "
            "evidence override remembered procedure."
        ),
        "persistence": "derived_from_v3_trajectory_corpus_not_separate_memory",
    }
    return skill


def _install_skill_evolution(trajectory_memory: Any, temporary_skill_contract: Any) -> None:
    current = trajectory_memory.synthesize_temporary_skill
    if getattr(current, "_mmm_evolving_skill_v1", False):
        temporary_skill_contract.synthesize_temporary_skill = current
        return

    @wraps(current)
    def evolved(
        query: str,
        records: Sequence[Mapping[str, Any]],
        *,
        task_class: str,
    ):
        skill = current(query, records, task_class=task_class)
        if not isinstance(skill, Mapping):
            return skill
        return _evolve_skill(
            skill,
            query,
            records,
            task_class=task_class,
        )

    evolved._mmm_evolving_skill_v1 = True  # type: ignore[attr-defined]
    evolved.__wrapped__ = current  # type: ignore[attr-defined]
    trajectory_memory.synthesize_temporary_skill = evolved
    # temporary_skill_contract imported the function directly, so update that bound
    # reference as well rather than creating a second synthesis path.
    temporary_skill_contract.synthesize_temporary_skill = evolved


def _model_family(config: Any) -> str:
    values: list[str] = []
    if isinstance(config, Mapping):
        values.extend(
            str(config.get(key, ""))
            for key in ("model_id", "model", "repo_id", "name", "path")
        )
    else:
        values.extend(
            str(getattr(config, key, ""))
            for key in ("model_id", "model", "repo_id", "name", "path")
        )
    return qwen_family(" ".join(values)) or "generic"


def _compact_executor_skill(skill: Mapping[str, Any], family: str) -> dict[str, Any]:
    """Render the same canonical Skill with an executor-appropriate prompt surface."""

    if family not in {"qwen3.5", "qwen3.6"}:
        return dict(skill)

    keys = (
        "schema_version",
        "ephemeral",
        "task_class",
        "current_query_terms",
        "procedural_hierarchy",
        "proven_patterns",
        "avoid_patterns",
        "verifier_hints",
        "memory_route",
        "skill_relations",
        "evolving_playbook",
        "rule",
    )
    result = {key: skill[key] for key in keys if key in skill}
    terms = result.get("current_query_terms")
    if isinstance(terms, list):
        result["current_query_terms"] = terms[:18 if family == "qwen3.5" else 28]

    hierarchy = result.get("procedural_hierarchy")
    if family == "qwen3.5" and isinstance(hierarchy, Mapping):
        # Prefer concrete function/subtask motifs for the smaller executor; workflow
        # provenance remains host-side in the canonical cached skill.
        result["procedural_hierarchy"] = {
            key: value
            for key, value in hierarchy.items()
            if key in {"function", "subtask"}
        }

    result["executor_rendering"] = {
        "schema_version": "mmm/executor-skill-rendering-v1",
        "family": family,
        "canonical_skill_unchanged": True,
    }
    return result


def _install_executor_skill_rendering(
    model_router: Any,
    temporary_skill_contract: Any,
) -> None:
    current_prepare = model_router.ModelRouter._prepare_generation_request
    if not getattr(current_prepare, "_mmm_executor_skill_context_v1", False):

        @wraps(current_prepare)
        def prepare(
            self: Any,
            role: str,
            messages: Sequence[Mapping[str, Any]],
            **kwargs: Any,
        ):
            token = _MODEL_FAMILY.set(_model_family(kwargs.get("config")))
            try:
                return current_prepare(self, role, messages, **kwargs)
            finally:
                _MODEL_FAMILY.reset(token)

        prepare._mmm_executor_skill_context_v1 = True  # type: ignore[attr-defined]
        prepare.__wrapped__ = current_prepare  # type: ignore[attr-defined]
        model_router.ModelRouter._prepare_generation_request = prepare

    current_inject = temporary_skill_contract._inject
    if getattr(current_inject, "_mmm_executor_skill_rendering_v1", False):
        return

    @wraps(current_inject)
    def inject(
        messages: Sequence[Mapping[str, Any]],
        skill: Mapping[str, Any],
    ):
        return current_inject(
            messages,
            _compact_executor_skill(skill, _MODEL_FAMILY.get()),
        )

    inject._mmm_executor_skill_rendering_v1 = True  # type: ignore[attr-defined]
    inject.__wrapped__ = current_inject  # type: ignore[attr-defined]
    temporary_skill_contract._inject = inject


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import (
        agent_capability_context,
        agentic_research_game_design,
        central_intelligence_amplifier,
        complete_planner,
        model_router,
        small_model_adaptive_compute,
        small_model_agent_policy,
        temporary_skill_contract,
        trajectory_memory,
    )

    # The v3 trajectory store is the durable memory owner. Keep the old planner
    # workflow JSONL only when a caller explicitly opts into that legacy path.
    small_model_agent_policy._memory_path = _explicit_legacy_workflow_path

    small_model_adaptive_compute.harden(
        agentic_research_game_design,
        central_intelligence_amplifier,
    )
    small_model_agent_policy.enhance_planner(complete_planner)
    _install_instruction_projection(agent_capability_context)
    _install_intent_routed_memory(trajectory_memory)
    _install_skill_evolution(trajectory_memory, temporary_skill_contract)
    _install_executor_skill_rendering(model_router, temporary_skill_contract)
    _INSTALLED = True


__all__ = [
    "install",
]
