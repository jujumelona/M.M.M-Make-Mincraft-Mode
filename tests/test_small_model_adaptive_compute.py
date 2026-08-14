from __future__ import annotations

from types import SimpleNamespace

import minecraft_mod_ai.small_model_adaptive_compute as adaptive


def _brief(prompt: str, _design):
    kinds = ["minecraft_api", "dependency", "compatibility", "license"]
    domains = [
        {
            "domain_id": "request",
            "evidence_kinds": kinds,
            "depends_on": [],
        }
    ]
    if "multiplayer" in prompt.casefold():
        domains[0]["evidence_kinds"].extend(["runtime_behavior", "testing"])
    return {"domains": domains, "unresolved_questions": []}


def test_compute_policy_keeps_clearly_simple_request_lean(monkeypatch):
    monkeypatch.delenv("MMM_CENTRAL_TEST_TIME_COMPUTE", raising=False)
    agentic = SimpleNamespace(normalize_research_brief=_brief)

    policy = adaptive._compute_policy(agentic, "Add one decorative block.")

    assert policy["tier"] == "lean"
    assert policy["central_amplification"] is False


def test_compute_policy_escalates_cross_system_request(monkeypatch):
    monkeypatch.delenv("MMM_CENTRAL_TEST_TIME_COMPUTE", raising=False)
    agentic = SimpleNamespace(normalize_research_brief=_brief)

    policy = adaptive._compute_policy(
        agentic,
        "Add multiplayer server/client networking with persistence and runtime testing.",
    )

    assert policy["tier"] == "full"
    assert policy["central_amplification"] is True


def test_unknown_classifier_fails_quality_safe_to_full(monkeypatch):
    monkeypatch.delenv("MMM_CENTRAL_TEST_TIME_COMPUTE", raising=False)
    agentic = SimpleNamespace(
        normalize_research_brief=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("classifier unavailable")
        )
    )

    policy = adaptive._compute_policy(agentic, "simple")

    assert policy["tier"] == "full"
    assert policy["reason"].startswith("classification_unavailable:")


def test_standard_review_expands_only_on_issue_or_uncertainty(monkeypatch):
    monkeypatch.delenv("MMM_CENTRAL_TEST_TIME_COMPUTE", raising=False)
    calls = []

    def base_reviews(_router, _payload, *, target, reviewers):
        calls.append(tuple(item[0] for item in reviewers))
        reviewer = reviewers[0][0]
        return [
            {
                "reviewer": reviewer,
                "missing_requirements": [],
                "unsupported_additions": [],
                "contradictions": [],
                "research_gaps": [],
                "severity": "none",
                "confidence": 0.9,
            }
        ]

    central = SimpleNamespace(
        _amplification_enabled=lambda *_args: True,
        _parallel_reviews=base_reviews,
    )
    agentic = SimpleNamespace(
        normalize_research_brief=_brief,
        collect_pre_design_research=lambda _router, _prompt, trace_metadata=None: {
            "method": {}
        },
        generate_sectioned_game_design=lambda *_args, **_kwargs: {},
        _json_sha256=lambda _value: "sha256:test",
    )
    adaptive.harden(agentic, central)
    token = adaptive._ACTIVE_POLICY.set({"tier": "standard"})
    try:
        result = central._parallel_reviews(
            object(),
            {},
            target="candidate",
            reviewers=(("first", "a"), ("second", "b")),
        )
    finally:
        adaptive._ACTIVE_POLICY.reset(token)

    assert [item["reviewer"] for item in result] == ["first"]
    assert calls == [("first",)]


def test_standard_review_widens_after_material_issue(monkeypatch):
    monkeypatch.delenv("MMM_CENTRAL_TEST_TIME_COMPUTE", raising=False)
    calls = []

    def base_reviews(_router, _payload, *, target, reviewers):
        calls.append(tuple(item[0] for item in reviewers))
        reviewer = reviewers[0][0]
        issue = reviewer == "first"
        return [
            {
                "reviewer": reviewer,
                "missing_requirements": ["missing"] if issue else [],
                "unsupported_additions": [],
                "contradictions": [],
                "research_gaps": [],
                "severity": "medium" if issue else "none",
                "confidence": 0.9,
            }
        ]

    central = SimpleNamespace(
        _amplification_enabled=lambda *_args: True,
        _parallel_reviews=base_reviews,
    )
    agentic = SimpleNamespace(
        normalize_research_brief=_brief,
        collect_pre_design_research=lambda _router, _prompt, trace_metadata=None: {
            "method": {}
        },
        generate_sectioned_game_design=lambda *_args, **_kwargs: {},
        _json_sha256=lambda _value: "sha256:test",
    )
    adaptive.harden(agentic, central)
    token = adaptive._ACTIVE_POLICY.set({"tier": "standard"})
    try:
        result = central._parallel_reviews(
            object(),
            {},
            target="candidate",
            reviewers=(("first", "a"), ("second", "b")),
        )
    finally:
        adaptive._ACTIVE_POLICY.reset(token)

    assert [item["reviewer"] for item in result] == ["first", "second"]
    assert calls == [("first",), ("second",)]


def test_collect_context_disables_optional_central_stack_for_lean(monkeypatch):
    monkeypatch.delenv("MMM_CENTRAL_TEST_TIME_COMPUTE", raising=False)
    central = SimpleNamespace(
        _amplification_enabled=lambda *_args: True,
        _parallel_reviews=lambda *_args, **_kwargs: [],
    )

    def collect(_router, prompt, *, trace_metadata=None):
        return {
            "method": {
                "central_seen": central._amplification_enabled(agentic, object())
            }
        }

    agentic = SimpleNamespace(
        normalize_research_brief=_brief,
        collect_pre_design_research=collect,
        generate_sectioned_game_design=lambda *_args, **_kwargs: {},
        _json_sha256=lambda _value: "sha256:test",
    )
    adaptive.harden(agentic, central)

    result = agentic.collect_pre_design_research(object(), "Add one decorative block.")

    assert result["method"]["central_seen"] is False
    receipt = result["method"]["adaptive_test_time_compute"]
    assert receipt["tier"] == "lean"
