from __future__ import annotations

import threading
import time
from functools import wraps
from types import SimpleNamespace

from minecraft_mod_ai import agentic_optimization_contract as agentic
from minecraft_mod_ai import agentic_search_efficiency_contract as efficiency
from minecraft_mod_ai.agentic_search_efficiency_contract import install


def _risky_request() -> dict[str, object]:
    return {
        "current_target_deliverables": ["a", "b", "c", "d"],
        "scope": "custom_java networking integration persistence",
    }


def test_auto_planner_search_preserves_risk_width_when_slots_exist(monkeypatch) -> None:
    install(agentic)
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_PLAN_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "3")
    assert agentic._planner_candidate_count(_risky_request(), "production page") == 3


def test_auto_planner_search_does_not_duplicate_serial_decode(monkeypatch) -> None:
    install(agentic)
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_PLAN_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    assert agentic._planner_candidate_count(_risky_request(), "production page") == 1


def test_auto_planner_search_caps_breadth_to_native_slots(monkeypatch) -> None:
    install(agentic)
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_PLAN_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    assert agentic._planner_candidate_count(_risky_request(), "production page") == 2


def test_explicit_agentic_search_on_keeps_requested_width(monkeypatch) -> None:
    install(agentic)
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "on")
    monkeypatch.setenv("MMM_PLAN_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    assert agentic._planner_candidate_count({}, "planner") == 3


def test_auto_repair_search_escalates_only_after_same_failure_repeats(monkeypatch) -> None:
    install(agentic)
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_REPAIR_SEARCH_WIDTH", "2")

    engine = SimpleNamespace()
    engine._signature = lambda evidence: "same-signature"
    evidence = {
        "diagnostics": {
            "diagnostics": [
                {"path": "A.java", "message": "error one"},
                {"path": "B.java", "message": "error two"},
            ]
        },
        "build": {"status": "FAIL", "error": "x" * 200},
    }

    first = agentic._repair_candidate_count(engine, evidence, ())
    second = agentic._repair_candidate_count(engine, evidence, ())
    assert first == 1
    assert second == 2


def test_parallel_planner_search_uses_slots_and_separate_durable_stages(monkeypatch) -> None:
    active = 0
    max_active = 0
    stages: list[str] = []
    guard = threading.Lock()

    def base_generate(
        _router,
        *,
        system_prompt,
        request,
        media_paths,
        expected_contracts,
        stage,
    ):
        nonlocal active, max_active
        del system_prompt, request, media_paths, expected_contracts
        with guard:
            active += 1
            max_active = max(max_active, active)
            stages.append(stage)
        time.sleep(0.04)
        with guard:
            active -= 1
        candidate = int(stage.split("search_candidate=", 1)[1].split("/", 1)[0])
        return {"quality": candidate}

    planner_module = SimpleNamespace(
        _generate_json_page_with_repair=base_generate,
        SpecValidationError=RuntimeError,
    )

    fake = SimpleNamespace()
    fake._planner_candidate_count = lambda _request, _stage: 3
    fake._repair_candidate_count = lambda *_args, **_kwargs: 1
    fake._mode = lambda: "auto"
    fake._env_int = lambda _name, default, **_kwargs: default
    fake._score_plan_page = lambda page: (float(page["quality"]), {})

    def core_installer(module) -> None:
        current = module._generate_json_page_with_repair

        @wraps(current)
        def sequential_search(router, **kwargs):
            return current(router, **kwargs)

        sequential_search._mmm_verifier_plan_search = True
        module._generate_json_page_with_repair = sequential_search

    fake._install_planner_search = core_installer

    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "3")
    efficiency.install(fake)
    fake._install_planner_search(planner_module)

    result = planner_module._generate_json_page_with_repair(
        SimpleNamespace(),
        system_prompt="system",
        request=_risky_request(),
        media_paths=(),
        expected_contracts=(frozenset({"quality"}),),
        stage="production page",
    )

    assert result == {"quality": 3}
    assert max_active >= 2
    assert len(stages) == 3
    assert len(set(stages)) == 3
    assert all("search_candidate=" in value for value in stages)
    assert getattr(
        planner_module._generate_json_page_with_repair,
        "_mmm_parallel_plan_search",
        False,
    ) is True


def test_first_native_search_primes_slot_count_before_width_decision(monkeypatch) -> None:
    config = SimpleNamespace(provider="local", adapter="llama_cpp")
    registry = SimpleNamespace(role=lambda _profile, _role: config)
    router = SimpleNamespace(registry=registry, profile="t4_local")

    monkeypatch.delenv("MMM_LLAMA_ACTIVE_PARALLEL", raising=False)
    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)

    from minecraft_mod_ai import llama_server_autotune

    calls = []

    def ensure_tuned_server(seen_config, request):
        calls.append((seen_config, request))
        monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
        return "http://127.0.0.1:18910/v1"

    monkeypatch.setattr(llama_server_autotune, "ensure_tuned_server", ensure_tuned_server)

    returned = efficiency._prime_native_slots(
        router,
        system_prompt="system",
        request={"current_target_deliverables": ["a", "b", "c"]},
        media_paths=(),
    )

    assert returned is config
    assert len(calls) == 1
    assert calls[0][0] is config
    assert calls[0][1].response_format == "json"
    assert efficiency._active_parallel_slots() == 2
