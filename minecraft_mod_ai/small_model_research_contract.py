from __future__ import annotations

import hashlib
import json
from functools import wraps
from typing import Any, Mapping, Sequence


_EVIDENCE_FIELDS = (
    "module_ids",
    "asset_ids",
    "audio_ids",
    "acceptance_tests",
)
_PRODUCTION_KEYS = frozenset(
    {
        "modules",
        "assets",
        "audio",
        "acceptance_tests",
        "completed_deliverables",
        "complete",
        "next_cursor",
    }
)
_AGENTIC_RISK_MARKERS = (
    "networking",
    "multiplayer",
    "custom_java",
    "integration",
    "dimension",
    "world_event",
    "ai_inference",
    "agent_tool_use",
    "speech",
    "migration",
    "persistence",
)


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def _recent_catalog_ids(request: Any, field: str) -> set[str]:
    if not isinstance(request, Mapping):
        return set()
    receipt = request.get(field)
    if not isinstance(receipt, Mapping):
        return set()
    return _string_set(receipt.get("recent_ids", ()))


def _dependency_export_ids(request: Any) -> set[str]:
    if not isinstance(request, Mapping):
        return set()
    exports = request.get("dependency_exports")
    if not isinstance(exports, Mapping):
        return set()
    result: set[str] = set()
    for values in exports.values():
        result.update(_string_set(values))
    return result


def _produced_ids(page: Mapping[str, Any], field: str, id_field: str) -> set[str]:
    result: set[str] = set()
    values = page.get(field)
    if not isinstance(values, list):
        return result
    for item in values:
        if not isinstance(item, Mapping):
            continue
        value = str(item.get(id_field, "")).strip()
        if value:
            result.add(value)
    return result


def _evidence_has_declared_reference(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return any(_string_set(value.get(field, ())) for field in _EVIDENCE_FIELDS)


def _evidence_is_grounded(
    value: Any,
    *,
    module_ids: set[str],
    asset_ids: set[str],
    audio_ids: set[str],
    acceptance_tests: set[str],
) -> bool:
    if not isinstance(value, Mapping):
        return False
    return bool(
        (_string_set(value.get("module_ids", ())) & module_ids)
        or (_string_set(value.get("asset_ids", ())) & asset_ids)
        or (_string_set(value.get("audio_ids", ())) & audio_ids)
        or (_string_set(value.get("acceptance_tests", ())) & acceptance_tests)
    )


def _sanitize_production_page(
    page: Mapping[str, Any],
    request: dict[str, Any] | str,
) -> dict[str, Any]:
    """Validate explicit evidence without weakening legacy host-owned bookkeeping."""
    result = dict(page)
    evidence_value = result.get("deliverable_evidence")
    if not isinstance(evidence_value, Mapping):
        return result

    completed = result.get("completed_deliverables")
    if not isinstance(completed, list):
        completed = []

    evidence = {
        str(key): dict(value)
        for key, value in evidence_value.items()
        if isinstance(key, str) and isinstance(value, Mapping)
    }
    result["deliverable_evidence"] = evidence

    current_modules = _produced_ids(result, "modules", "module_id")
    current_assets = _produced_ids(result, "assets", "asset_id")
    current_audio = _produced_ids(result, "audio", "sound_id")
    current_tests = _string_set(result.get("acceptance_tests", ()))

    valid_modules = (
        current_modules
        | _recent_catalog_ids(request, "known_module_catalog")
        | _dependency_export_ids(request)
    )
    valid_assets = current_assets | _recent_catalog_ids(
        request, "known_asset_catalog"
    )
    valid_audio = current_audio | _recent_catalog_ids(
        request, "known_audio_catalog"
    )

    supported: list[str] = []
    for raw in completed:
        deliverable = str(raw).strip()
        if not deliverable:
            continue
        if _evidence_is_grounded(
            evidence.get(deliverable),
            module_ids=valid_modules,
            asset_ids=valid_assets,
            audio_ids=valid_audio,
            acceptance_tests=current_tests,
        ):
            supported.append(deliverable)
    result["completed_deliverables"] = supported
    return result


def _is_production_decode(
    request: dict[str, Any] | str,
    expected_contracts: Sequence[frozenset[str]],
) -> bool:
    if not isinstance(request, Mapping) or "remaining_deliverables" not in request:
        return False
    return any(set(contract) == set(_PRODUCTION_KEYS) for contract in expected_contracts)


def _install_evidence_contract(complete_planner_module: Any) -> None:
    current = complete_planner_module._generate_json_page_with_repair
    if getattr(current, "_mmm_small_model_evidence_guard", False):
        return

    @wraps(current)
    def generate_with_evidence_guard(
        router: Any,
        *,
        system_prompt: str,
        request: dict[str, Any] | str,
        media_paths: Sequence[Any],
        expected_contracts: Sequence[frozenset[str]],
        stage: str,
    ) -> dict[str, Any]:
        # Production evidence is optional input to the host-owned bookkeeping layer.
        # Never widen the strict top-level contract here: doing so turns an optional
        # quality signal into a required model field and can create pointless repair
        # decodes for otherwise valid pages.
        page = current(
            router,
            system_prompt=system_prompt,
            request=request,
            media_paths=media_paths,
            expected_contracts=expected_contracts,
            stage=stage,
        )
        if not _is_production_decode(request, expected_contracts):
            return page
        if "deliverable_evidence" not in page:
            return page
        sanitized = _sanitize_production_page(page, request)
        sanitized.pop("deliverable_evidence", None)
        return sanitized

    generate_with_evidence_guard._mmm_small_model_evidence_guard = True  # type: ignore[attr-defined]
    complete_planner_module._generate_json_page_with_repair = generate_with_evidence_guard


def _install_evidence_aware_scoring(agentic_module: Any) -> None:
    current = agentic_module._score_plan_page
    if getattr(current, "_mmm_evidence_aware_plan_score", False):
        return

    @wraps(current)
    def score_with_evidence(page: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
        base_score, verifier = current(page)
        if "deliverable_evidence" not in page:
            return base_score, dict(verifier)

        completed = page.get("completed_deliverables")
        completed_values = (
            [str(item).strip() for item in completed if str(item).strip()]
            if isinstance(completed, list)
            else []
        )
        evidence = page.get("deliverable_evidence")
        evidence = evidence if isinstance(evidence, Mapping) else {}

        current_modules = _produced_ids(page, "modules", "module_id")
        current_assets = _produced_ids(page, "assets", "asset_id")
        current_audio = _produced_ids(page, "audio", "sound_id")
        current_tests = _string_set(page.get("acceptance_tests", ()))

        declared = 0
        grounded = 0
        unsupported = 0
        for deliverable in completed_values:
            item = evidence.get(deliverable)
            if _evidence_has_declared_reference(item):
                declared += 1
            else:
                unsupported += 1
            if _evidence_is_grounded(
                item,
                module_ids=current_modules,
                asset_ids=current_assets,
                audio_ids=current_audio,
                acceptance_tests=current_tests,
            ):
                grounded += 1

        score = base_score + 24.0 * grounded - 36.0 * unsupported
        details = {
            **dict(verifier),
            "declared_completion_evidence": declared,
            "grounded_completion_evidence": grounded,
            "unsupported_completion_claims": unsupported,
        }
        return score, details

    score_with_evidence._mmm_evidence_aware_plan_score = True  # type: ignore[attr-defined]
    agentic_module._score_plan_page = score_with_evidence


def _semantic_digest(value: Any) -> str:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        rendered = repr(value)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _semantic_planner_key(prompt: str, research_brief: Any) -> tuple[str, str]:
    return _semantic_digest(prompt), _semantic_digest(research_brief)


def _semantic_ecosystem_key(
    prompt: str,
    game_design: Any,
    research_brief: Any,
) -> tuple[str, str, str]:
    return (
        _semantic_digest(prompt),
        _semantic_digest(game_design),
        _semantic_digest(research_brief),
    )


def _install_semantic_single_flight(parallel_module: Any) -> None:
    """Join equivalent planner prefetches by content rather than object identity."""

    parallel_module._planner_key = _semantic_planner_key
    parallel_module._ecosystem_key = _semantic_ecosystem_key


def _maximal_planner_risk(request: Any, stage: str) -> bool:
    rendered = (
        request
        if isinstance(request, str)
        else json.dumps(request, ensure_ascii=False, sort_keys=True, default=str)
    )
    size_risk = len(rendered.encode("utf-8")) >= 12 * 1024
    lowered = (stage + "\n" + rendered[:24_000]).casefold()
    domain_risk = any(marker in lowered for marker in _AGENTIC_RISK_MARKERS)
    target_risk = False
    if isinstance(request, Mapping):
        targets = request.get("current_target_deliverables", ())
        target_risk = (
            isinstance(targets, Sequence)
            and not isinstance(targets, (str, bytes))
            and len(targets) >= 3
        )
    return bool(size_risk and domain_risk and target_risk)


def _install_trace_adaptive_search(agentic_module: Any) -> None:
    """Use verified workflow history as a bounded utility signal for Best-of-N."""

    current = agentic_module._planner_candidate_count
    if getattr(current, "_mmm_trace_adaptive_width", False):
        return

    @wraps(current)
    def candidate_count(request: Any, stage: str) -> int:
        base = int(current(request, stage))
        if agentic_module._mode() != "auto":
            return base
        width = agentic_module._env_int("MMM_PLAN_SEARCH_WIDTH", 2, maximum=3)
        try:
            from .small_model_agent_policy import planner_search_width_hint

            hint = planner_search_width_hint(request, stage, maximum=width)
        except Exception:
            hint = None
        if hint is None:
            return base
        if hint > base:
            return int(hint)
        if hint < base and not _maximal_planner_risk(request, stage):
            return int(hint)
        return base

    candidate_count._mmm_trace_adaptive_width = True  # type: ignore[attr-defined]
    candidate_count.__wrapped__ = current  # type: ignore[attr-defined]
    agentic_module._planner_candidate_count = candidate_count


def install() -> None:
    """Bind research-derived small-model amplification to the fully composed runtime."""
    from . import (
        agentic_optimization_contract,
        complete_planner,
        parallel_runtime_contract,
        scheduler_parallel_safety_contract,
        work_graph,
    )
    from .max_efficiency_runtime_contract import enhance_runtime
    from .small_model_agent_policy import enhance_planner

    _install_evidence_aware_scoring(agentic_optimization_contract)
    _install_evidence_contract(complete_planner)
    _install_semantic_single_flight(parallel_runtime_contract)
    _install_trace_adaptive_search(agentic_optimization_contract)
    enhance_planner(complete_planner)

    # Post-bootstrap is the first point where planner, scheduler, router and generator
    # safety wrappers are all final. Bind the throughput layer here so it can align the
    # real executor with native slots without import-time side effects or branch logic.
    enhance_runtime(
        work_graph_module=work_graph,
        scheduler_module=scheduler_parallel_safety_contract,
    )


__all__ = ["install"]
