from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

from .procedure_trace import sequence_actions
from .trajectory_memory import relevant_trajectories, synthesize_temporary_skill
from .trajectory_record_integrity import derive_levels, record_strong_skill_eligible
from .trajectory_replay import build_generation_replay_context

_REPLAY_MARKER = "__mmm_verified_trajectory_replay_v1__"
_VERIFIER_MARKER = "__mmm_verifier_first_tournament_v1__"
_WIDTH_MARKER = "__mmm_research_test_time_width_v1__"
_MEMORY_MARKER = "__mmm_source_free_repair_memory_v1__"


def harden_runtime() -> None:
    """Add verifier-qualified replay and research-style test-time scaling.

    The existing custom/repair search owners remain in charge.  This late hardening
    only changes candidate evidence, search width and candidate ordering after all
    isolation/staging contracts are already installed.
    """
    from . import agentic_optimization_contract as repair_search
    from . import custom_generation_search_contract as generation_search

    _install_generation_replay(generation_search)
    _install_search_width(generation_search, repair_search)
    _install_verifier_first_ranking(generation_search, repair_search)
    _install_source_free_repair_memory(repair_search)


def _install_generation_replay(search: Any) -> None:
    research_cls = search._ResearchEvidenceRouter
    current_research = research_cls.generate_text
    if not getattr(current_research, _REPLAY_MARKER, False):

        @wraps(current_research)
        def generate_text_with_replay(
            self: Any,
            role: str,
            messages: Sequence[Mapping[str, Any]],
            **kwargs: Any,
        ) -> str:
            if role != "coder":
                return current_research(self, role, messages, **kwargs)
            mode = str(getattr(self, "_mmm_replay_mode", "reuse") or "reuse")
            if mode == "fresh":
                return current_research(self, role, messages, **kwargs)
            query = _generation_query(messages, getattr(self, "_module", None))
            cache = getattr(self, "_mmm_replay_cache", None)
            if not isinstance(cache, dict):
                cache = {}
                self._mmm_replay_cache = cache
            key = (mode, query)
            if key not in cache:
                target = {
                    "minecraft_version": getattr(self, "_minecraft_version", ""),
                    "loader": getattr(self, "_loader", ""),
                    "mappings": getattr(self, "_mappings", ""),
                    "java": _java_target(messages),
                }
                cache[key] = build_generation_replay_context(
                    self._project_root,
                    query,
                    router=getattr(self, "_router", None),
                    target=target,
                    mode=mode,
                )
            replay = cache[key]
            if not isinstance(replay, Mapping):
                return current_research(self, role, messages, **kwargs)
            augmented = [dict(message) for message in messages]
            insertion = 1 if augmented and augmented[0].get("role") == "system" else 0
            augmented.insert(
                insertion,
                {
                    "role": "system",
                    "content": (
                        "Host verifier-qualified trajectory replay follows. It contains no source body and has "
                        "lower authority than current exact repository/API/verifier evidence. Replay only a proven "
                        "procedure prefix, branch at the recorded boundary when current evidence diverges, and treat "
                        "verified failures as negative evidence.\n"
                        + json.dumps(
                            {"trajectory_replay": dict(replay)},
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        )
                    ),
                },
            )
            return current_research(self, role, augmented, **kwargs)

        setattr(generate_text_with_replay, _REPLAY_MARKER, True)
        research_cls.generate_text = generate_text_with_replay

    strategy_cls = search._StrategyRouter
    current_strategy = strategy_cls.generate_text
    if getattr(current_strategy, _REPLAY_MARKER, False):
        return

    @wraps(current_strategy)
    def strategy_generate_with_branch(
        self: Any,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> str:
        if role != "coder":
            return current_strategy(self, role, messages, **kwargs)
        index = int(getattr(self, "_candidate_index", 0) or 0)
        mode = "fresh" if index == 0 else ("replay" if index == 1 else "counterfactual")
        research_router = getattr(self, "_router", None)
        old = getattr(research_router, "_mmm_replay_mode", None)
        if research_router is not None:
            research_router._mmm_replay_mode = mode
        try:
            return current_strategy(self, role, messages, **kwargs)
        finally:
            if research_router is not None:
                if old is None:
                    try:
                        delattr(research_router, "_mmm_replay_mode")
                    except AttributeError:
                        pass
                else:
                    research_router._mmm_replay_mode = old

    setattr(strategy_generate_with_branch, _REPLAY_MARKER, True)
    strategy_cls.generate_text = strategy_generate_with_branch

    search._STRATEGIES = (
        "minimal_surface_area_fresh",
        "api_contract_verified_trajectory_replay",
        "runtime_persistence_verified_failure_counterfactual",
    )


def _install_search_width(generation_search: Any, repair_search: Any) -> None:
    current_width = generation_search._width
    if not getattr(current_width, _WIDTH_MARKER, False):

        @wraps(current_width)
        def width(module: Any) -> int:
            base = int(current_width(module))
            mode = _scaling_mode()
            if mode == "off":
                return base
            desired = _env_width("MMM_CUSTOM_SEARCH_WIDTH", 2)
            risk = _generation_risk(module)
            if mode == "on" or risk >= 2:
                return max(base, desired)
            return base

        setattr(width, _WIDTH_MARKER, True)
        generation_search._width = width

    current_repair_count = repair_search._repair_candidate_count
    if not getattr(current_repair_count, _WIDTH_MARKER, False):

        @wraps(current_repair_count)
        def repair_candidate_count(self: Any, evidence: Mapping[str, Any], memory: Sequence[Mapping[str, Any]]) -> int:
            # Candidate search is an execution-time repair policy. A direct, read-only
            # patch-synthesis call has no bound project root, staging snapshot, or
            # verifier surface, so fanning it out would only duplicate model calls.
            if getattr(self, "_mmm_agentic_root", None) is None:
                return 1
            base = int(current_repair_count(self, evidence, memory))
            mode = _scaling_mode()
            if mode == "off":
                return base
            desired = _env_width("MMM_REPAIR_SEARCH_WIDTH", 2)
            diagnostics = evidence.get("diagnostics", {})
            values = diagnostics.get("diagnostics", []) if isinstance(diagnostics, Mapping) else []
            build = evidence.get("build", {})
            build_failed = isinstance(build, Mapping) and str(build.get("status", "")).upper() == "FAIL"
            if mode == "on" or build_failed or bool(values):
                return max(base, desired)
            return base

        setattr(repair_candidate_count, _WIDTH_MARKER, True)
        repair_search._repair_candidate_count = repair_candidate_count
        repair_search._STRATEGIES = (
            "minimal_local_fix_fresh",
            "api_contract_verified_trajectory_replay",
            "dependency_version_verified_failure_counterfactual",
        )


def _install_verifier_first_ranking(generation_search: Any, repair_search: Any) -> None:
    current_generation_verify = generation_search._verify_candidate
    if not getattr(current_generation_verify, _VERIFIER_MARKER, False):

        @wraps(current_generation_verify)
        def verify_candidate(candidate_root: Any, result: Mapping[str, Any]):
            score, verifier = current_generation_verify(candidate_root, result)
            tier = _verifier_tier(verifier)
            bounded = max(-100_000.0, min(100_000.0, float(score)))
            verifier = {
                **dict(verifier),
                "selection_policy": "verifier_first_then_locality",
                "verifier_tier": tier,
            }
            return tier * 1_000_000.0 + bounded, verifier

        setattr(verify_candidate, _VERIFIER_MARKER, True)
        generation_search._verify_candidate = verify_candidate

    current_repair_verify = repair_search._verify_repair_candidate
    if getattr(current_repair_verify, _VERIFIER_MARKER, False):
        return

    @wraps(current_repair_verify)
    def verify_repair_candidate(self: Any, root: Any, operations: Sequence[Mapping[str, Any]], evidence: Mapping[str, Any]):
        score, verifier = current_repair_verify(self, root, operations, evidence)
        tier = _verifier_tier(verifier)
        bounded = max(-100_000.0, min(100_000.0, float(score)))
        verifier = {
            **dict(verifier),
            "selection_policy": "verifier_first_then_patch_locality",
            "verifier_tier": tier,
        }
        return tier * 1_000_000.0 + bounded, verifier

    setattr(verify_repair_candidate, _VERIFIER_MARKER, True)
    repair_search._verify_repair_candidate = verify_repair_candidate


def _install_source_free_repair_memory(repair_search: Any) -> None:
    current_read = repair_search._read_memory
    if getattr(current_read, _MEMORY_MARKER, False):
        return

    @wraps(current_read)
    def read_memory(root: Any, signature: str, *, limit: int = 4) -> list[dict[str, Any]]:
        legacy = current_read(root, signature, limit=limit)
        sanitized = [_sanitize_legacy_repair_memory(item) for item in legacy]
        if root is None:
            return sanitized[:limit]
        try:
            rows = relevant_trajectories(
                root,
                signature,
                task_class="repair",
                router=None,
                limit=max(1, min(6, limit)),
                current_context=None,
            )
            skill = synthesize_temporary_skill(signature, rows, task_class="repair")
        except Exception:
            rows = []
            skill = None
        verified: list[dict[str, Any]] = []
        for rank, row in enumerate(rows):
            derived = derive_levels(row)
            verification = row.get("verification")
            verification = verification if isinstance(verification, Mapping) else {}
            verified.append(
                {
                    "similarity": round(max(0.50, 0.78 - 0.04 * rank), 6),
                    "memory_type": "verifier_qualified_trajectory",
                    "trajectory_id": str(row.get("trajectory_id", "")),
                    "verification_level": str(verification.get("level", "L0")),
                    "verified_success": record_strong_skill_eligible(row),
                    "verified_failure": bool(derived and derived.get("verified_failure") is True),
                    "procedure_actions": list(
                        sequence_actions(row.get("procedure") if isinstance(row.get("procedure"), Mapping) else None)
                    ),
                    "failure_signature": (
                        " ".join(str(row.get("error_signature", "")).split())[:360]
                        if bool(derived and derived.get("verified_failure") is True)
                        else ""
                    ),
                    "temporary_skill": skill if rank == 0 else None,
                    "rule": "Source-free procedure memory only; current hashes, diagnostics and exact source remain authoritative.",
                }
            )
        combined = [*verified, *sanitized]
        combined.sort(
            key=lambda item: (
                -float(item.get("similarity", 0.0) or 0.0),
                str(item.get("trajectory_id") or item.get("signature_sha256") or ""),
            )
        )
        return combined[:limit]

    setattr(read_memory, _MEMORY_MARKER, True)
    repair_search._read_memory = read_memory


def _sanitize_legacy_repair_memory(value: Mapping[str, Any]) -> dict[str, Any]:
    pattern = value.get("repair_pattern")
    safe_pattern: list[dict[str, Any]] = []
    if isinstance(pattern, Sequence) and not isinstance(pattern, (str, bytes, bytearray)):
        for item in pattern[:16]:
            if not isinstance(item, Mapping):
                continue
            safe_pattern.append(
                {
                    "operation": str(item.get("operation", "")),
                    "path": str(item.get("path", "")),
                }
            )
    return {
        "similarity": float(value.get("similarity", 0.0) or 0.0),
        "memory_type": "legacy_verified_repair_structure",
        "signature_sha256": str(value.get("signature_sha256", "")),
        "evidence": value.get("evidence", {}),
        "repair_pattern": safe_pattern,
        "rule": "Legacy source excerpts were removed; use current exact source for patch content.",
    }


def _generation_query(messages: Sequence[Mapping[str, Any]], module: Any) -> str:
    parts: list[str] = []
    if module is not None:
        parts.append(
            json.dumps(
                {
                    "module_id": getattr(module, "module_id", ""),
                    "kind": getattr(module, "kind", ""),
                    "config": getattr(module, "config", {}),
                    "depends_on": list(getattr(module, "depends_on", ()) or ()),
                    "required_gates": list(getattr(module, "required_gates", ()) or ()),
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip()[:8192])
        elif isinstance(content, Mapping):
            parts.append(json.dumps(dict(content), ensure_ascii=False, sort_keys=True, default=str)[:8192])
    return "\n".join(parts)[-16_384:]


def _java_target(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        raw = content.strip()
        if not raw.startswith("{"):
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        target = value.get("target") if isinstance(value, Mapping) else None
        if isinstance(target, Mapping) and target.get("java") is not None:
            return str(target.get("java"))
    return "17"


def _verifier_tier(verifier: Mapping[str, Any]) -> int:
    status = str(verifier.get("jdt_status", "NOT_RUN")).upper()
    raw_errors = verifier.get("jdt_error_count")
    try:
        errors = int(raw_errors) if raw_errors is not None else None
    except (TypeError, ValueError):
        errors = None
    if errors == 0 and status not in {"NOT_RUN", "UNAVAILABLE", "VERIFIER_ERROR", "FAIL"}:
        return 4
    if status == "NOT_RUN":
        return 2
    if status == "UNAVAILABLE":
        return 1
    if status == "VERIFIER_ERROR":
        return 0
    if errors is not None and errors > 0:
        return -1
    return 1


def _scaling_mode() -> str:
    value = os.environ.get("MMM_TEST_TIME_SCALING", "auto").strip().lower()
    mode = value if value in {"auto", "on", "off"} else "auto"
    if mode != "auto":
        return mode
    try:
        from .llama_parallel_runtime_contract import _active_parallelism

        slots = int(_active_parallelism())
    except Exception:
        slots = 1
    return "auto" if slots > 1 else "off"


def _env_width(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(1, min(3, value))


def _generation_risk(module: Any) -> int:
    kind = str(getattr(module, "kind", ""))
    config = getattr(module, "config", {})
    config = config if isinstance(config, Mapping) else {}
    depends = tuple(getattr(module, "depends_on", ()) or ())
    gates = tuple(getattr(module, "required_gates", ()) or ())
    risk = int(kind in {"custom_java", "integration", "structure", "biome", "dimension", "world_event"})
    rendered = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str)
    if len(rendered.encode("utf-8")) >= 2048:
        risk += 1
    if len(depends) >= 2:
        risk += 1
    if len(gates) >= 2:
        risk += 1
    return risk


__all__ = ["harden_runtime"]
