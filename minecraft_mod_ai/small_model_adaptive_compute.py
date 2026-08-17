from __future__ import annotations
'Adaptive test-time compute for the frozen small central agent.\n\nThe central council/reviewer stack is valuable on hard requests, but paying the full\nensemble cost for every request wastes the limited local-model budget. This late\nhardener keeps existing planning/research authority intact and only changes when the\noptional advisory amplification is invoked:\n\n* clearly simple requests use the existing research/planner/verifier stack directly;\n* ordinary non-trivial requests keep the council, but a second adversarial reviewer\n  is requested only when the first review is uncertain or finds a material issue;\n* complex/high-risk requests retain the complete existing council and parallel reviews.\n\nNo model weights, schemas, validation rules, or execution authority are changed.\n'
import os
from contextvars import ContextVar
from functools import wraps
from typing import Any, Mapping, Sequence
_MARKER = '_mmm_adaptive_central_compute_v1'
_ACTIVE_POLICY: ContextVar[dict[str, Any] | None] = ContextVar('mmm_adaptive_central_compute_policy', default=None)
_SEVERITY = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
_HIGH_RISK_EVIDENCE = frozenset({'source_code', 'local_project', 'testing', 'runtime_behavior', 'performance', 'release', 'ai_inference', 'agent_tool_use', 'translation', 'model_runtime', 'dataset_provenance', 'consent_privacy', 'latency_budget'})
_COMPLEX_MARKERS = ('multiplayer', 'network', 'server', 'client', 'persistence', 'persist', 'migration', 'custom java', 'custom_java', 'integration', 'dimension', 'world event', 'world_event', 'ai inference', 'agent tool', '동기화', '멀티플레이', '네트워크', '서버', '클라이언트', '영속', '마이그레이션', '통합', '차원', '인공지능', '음성')

def _mode() -> str:
    value = os.environ.get('MMM_CENTRAL_TEST_TIME_COMPUTE', 'auto').strip().lower()
    return value if value in {'auto', 'lean', 'full'} else 'auto'

def _domains(brief: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = brief.get('domains', [])
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, Mapping)]

def _compute_policy(agentic_module: Any, prompt: str) -> dict[str, Any]:
    mode = _mode()
    if mode == 'lean':
        return {'schema_version': 'mmm/adaptive-central-compute-v1', 'tier': 'lean', 'score': 0, 'reason': 'environment_override', 'central_amplification': False, 'review_policy': 'disabled_with_central_amplification'}
    if mode == 'full':
        return {'schema_version': 'mmm/adaptive-central-compute-v1', 'tier': 'full', 'score': 99, 'reason': 'environment_override', 'central_amplification': True, 'review_policy': 'full_parallel'}
    try:
        brief = agentic_module.normalize_research_brief(prompt, {'title': 'adaptive central compute'})
    except Exception as exc:
        return {'schema_version': 'mmm/adaptive-central-compute-v1', 'tier': 'full', 'score': 99, 'reason': f'classification_unavailable:{type(exc).__name__}', 'central_amplification': True, 'review_policy': 'full_parallel'}
    domains = _domains(brief)
    score = 0
    reasons: list[str] = []
    prompt_bytes = len(prompt.encode('utf-8'))
    if prompt_bytes >= 5000:
        score += 3
        reasons.append('large_request')
    elif prompt_bytes >= 1800:
        score += 2
        reasons.append('medium_request')
    elif prompt_bytes >= 600:
        score += 1
        reasons.append('nontrivial_request')
    if len(domains) >= 3:
        score += 3
        reasons.append('multi_domain')
    elif len(domains) == 2:
        score += 2
        reasons.append('two_domains')
    unresolved = brief.get('unresolved_questions', [])
    if isinstance(unresolved, list) and unresolved:
        score += 2
        reasons.append('unresolved_questions')
    evidence_kinds: set[str] = set()
    dependency_edges = 0
    for domain in domains:
        kinds = domain.get('evidence_kinds', [])
        if isinstance(kinds, list):
            evidence_kinds.update((str(value) for value in kinds))
        depends_on = domain.get('depends_on', [])
        if isinstance(depends_on, list):
            dependency_edges += len(depends_on)
    risky = sorted(evidence_kinds & _HIGH_RISK_EVIDENCE)
    if risky:
        score += 2
        reasons.append('high_risk_evidence')
    if dependency_edges:
        score += 1
        reasons.append('research_dependencies')
    lowered = prompt.casefold()
    marker_hits = sum((1 for marker in _COMPLEX_MARKERS if marker in lowered))
    if marker_hits >= 2:
        score += 2
        reasons.append('cross_system_markers')
    elif marker_hits == 1:
        score += 1
        reasons.append('complexity_marker')
    boundaries = prompt.count('\n') + prompt.count(';') + prompt.count('；')
    if boundaries >= 5:
        score += 1
        reasons.append('multi_constraint_request')
    if score >= 4:
        tier = 'full'
        review_policy = 'full_parallel'
    elif score >= 2:
        tier = 'standard'
        review_policy = 'uncertainty_gated_second_reviewer'
    else:
        tier = 'lean'
        review_policy = 'disabled_with_central_amplification'
    return {'schema_version': 'mmm/adaptive-central-compute-v1', 'tier': tier, 'score': score, 'reason': ','.join(reasons) if reasons else 'cheap_host_features', 'central_amplification': tier != 'lean', 'review_policy': review_policy, 'domain_count': len(domains), 'prompt_bytes': prompt_bytes, 'high_risk_evidence': risky, 'dependency_edges': dependency_edges}

def _review_requires_expansion(review: Mapping[str, Any]) -> bool:
    severity = _SEVERITY.get(str(review.get('severity', 'none')), 0)
    try:
        confidence = float(review.get('confidence', 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    material_issue = any((isinstance(review.get(field), list) and bool(review.get(field)) for field in ('missing_requirements', 'unsupported_additions', 'contradictions', 'research_gaps')))
    return severity >= _SEVERITY['medium'] or confidence < 0.72 or material_issue

def _attach_receipt(agentic_module: Any, result: Any, policy: Mapping[str, Any]) -> Any:
    if not isinstance(result, Mapping):
        return result
    value = dict(result)
    method = dict(value.get('method', {})) if isinstance(value.get('method'), Mapping) else {}
    method['adaptive_test_time_compute'] = dict(policy)
    value['method'] = method
    try:
        value['research_sha256'] = agentic_module._json_sha256(value)
    except Exception:
        pass
    return value

def harden(agentic_module: Any, central_module: Any) -> None:
    """Install quality-first adaptive compute outside the existing central wrappers."""
    current_enabled = central_module._amplification_enabled
    if not getattr(current_enabled, _MARKER, False):

        @wraps(current_enabled)
        def amplification_enabled(agentic: Any, router: Any) -> bool:
            policy = _ACTIVE_POLICY.get()
            if isinstance(policy, Mapping) and policy.get('tier') == 'lean':
                return False
            return current_enabled(agentic, router)
        setattr(amplification_enabled, _MARKER, True)
        amplification_enabled.__wrapped__ = current_enabled
        central_module._amplification_enabled = amplification_enabled
    current_reviews = central_module._parallel_reviews
    if not getattr(current_reviews, _MARKER, False):

        @wraps(current_reviews)
        def adaptive_reviews(router: Any, payload: Mapping[str, Any], *, target: str, reviewers: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
            policy = _ACTIVE_POLICY.get()
            if not isinstance(policy, Mapping) or policy.get('tier') != 'standard' or len(reviewers) <= 1:
                return current_reviews(router, payload, target=target, reviewers=reviewers)
            first = current_reviews(router, payload, target=target, reviewers=reviewers[:1])
            if not first or _review_requires_expansion(first[0]):
                rest = current_reviews(router, payload, target=target, reviewers=reviewers[1:])
                return [*first, *rest]
            return first
        setattr(adaptive_reviews, _MARKER, True)
        adaptive_reviews.__wrapped__ = current_reviews
        central_module._parallel_reviews = adaptive_reviews
    current_collect = agentic_module.collect_pre_design_research
    if not getattr(current_collect, _MARKER, False):

        @wraps(current_collect)
        def collect(router: Any, prompt: str, *, trace_metadata=None):
            policy = _compute_policy(agentic_module, prompt)
            token = _ACTIVE_POLICY.set(policy)
            try:
                result = current_collect(router, prompt, trace_metadata=trace_metadata)
            finally:
                _ACTIVE_POLICY.reset(token)
            return _attach_receipt(agentic_module, result, policy)
        setattr(collect, _MARKER, True)
        collect.__wrapped__ = current_collect
        agentic_module.collect_pre_design_research = collect
    current_generate = agentic_module.generate_sectioned_game_design
    if not getattr(current_generate, _MARKER, False):

        @wraps(current_generate)
        def generate(game_design_module: Any, router: Any, prompt: str, *, media_paths=(), research: Mapping[str, Any], trace_metadata=None):
            policy = _compute_policy(agentic_module, prompt)
            token = _ACTIVE_POLICY.set(policy)
            try:
                return current_generate(game_design_module, router, prompt, media_paths=media_paths, research=research, trace_metadata=trace_metadata)
            finally:
                _ACTIVE_POLICY.reset(token)
        setattr(generate, _MARKER, True)
        generate.__wrapped__ = current_generate
        agentic_module.generate_sectioned_game_design = generate
__all__ = ['harden', '_compute_policy', '_review_requires_expansion']
