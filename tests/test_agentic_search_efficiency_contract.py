from __future__ import annotations

import json
import threading
import time
from contextvars import ContextVar
from functools import wraps
from types import SimpleNamespace

from minecraft_mod_ai import agentic_optimization_contract as agentic
from minecraft_mod_ai import agentic_search_efficiency_contract as efficiency
from minecraft_mod_ai import platform_repair_target_contract as platform_repair
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
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")

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


def test_auto_repair_search_never_duplicates_serial_native_decode(monkeypatch) -> None:
    install(agentic)
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_REPAIR_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")

    engine = SimpleNamespace(_signature=lambda _evidence: "same-signature")
    evidence = {"diagnostics": {}, "build": {"status": "FAIL", "error": "x" * 200}}

    assert agentic._repair_candidate_count(engine, evidence, ()) == 1
    assert agentic._repair_candidate_count(engine, evidence, ()) == 1
    assert agentic._repair_candidate_count(engine, evidence, ()) == 1


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
    fake._install_repair_search_and_memory = lambda _module: None

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


def test_parallel_repair_search_preserves_context_and_commits_only_winner_scope(
    monkeypatch,
) -> None:
    active = 0
    max_active = 0
    seen_targets: list[str] = []
    seen_scopes: list[tuple[str, ...]] = []
    guard = threading.Lock()
    repair_target: ContextVar[str] = ContextVar("test_repair_target", default="missing")

    class Engine:
        def __init__(self) -> None:
            self.router = SimpleNamespace()
            self._mmm_last_java_paths = ("sentinel.java",)

        def _signature(self, _evidence):
            return "same-signature"

        def _request_patch(self, evidence, context):
            nonlocal active, max_active
            del evidence
            candidate = int(context["agentic_candidate"]["index"])
            with guard:
                active += 1
                max_active = max(max_active, active)
                seen_targets.append(repair_target.get())
                seen_scopes.append(tuple(self._mmm_last_java_paths))
            time.sleep(0.04)
            with guard:
                active -= 1
            return [
                {
                    "operation": "create",
                    "path": f"src/main/java/Candidate{candidate}.java",
                    "content": f"final class Candidate{candidate} {{}}",
                }
            ]

    repair_module = SimpleNamespace(
        RepairEngine=Engine,
        RepairEngineError=RuntimeError,
    )

    fake = SimpleNamespace()
    fake._planner_candidate_count = lambda _request, _stage: 1
    fake._repair_candidate_count = lambda *_args, **_kwargs: 3
    fake._mode = lambda: "on"
    fake._env_int = lambda name, default, **_kwargs: (
        3 if name == "MMM_REPAIR_SEARCH_WIDTH" else default
    )
    fake._score_plan_page = lambda _page: (0.0, {})
    fake._STRATEGIES = (
        "minimal_local_fix",
        "api_contract_conservative_fix",
        "dependency_and_version_conservative_fix",
    )
    fake._read_memory = lambda *_args, **_kwargs: []
    fake._compact_evidence = lambda _evidence: {}
    fake._repair_pattern = lambda _operations: []
    fake._json_size = lambda value: len(json.dumps(value, sort_keys=True))
    fake._verify_repair_candidate = lambda _self, _root, operations, _evidence: (
        float(int(operations[0]["path"].split("Candidate", 1)[1].split(".", 1)[0])),
        {"jdt_status": "PASS", "jdt_error_count": 0},
    )

    def planner_installer(_module) -> None:
        return None

    def repair_installer(module) -> None:
        current = module.RepairEngine._request_patch

        @wraps(current)
        def sequential_search(self, evidence, context):
            return current(self, evidence, context)

        sequential_search._mmm_verifier_repair_search = True
        module.RepairEngine._request_patch = sequential_search

    fake._install_planner_search = planner_installer
    fake._install_repair_search_and_memory = repair_installer

    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "on")
    monkeypatch.setenv("MMM_REPAIR_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "3")
    efficiency.install(fake)
    fake._install_repair_search_and_memory(repair_module)

    engine = Engine()
    token = repair_target.set("fabric-1.21.1")
    try:
        result = engine._request_patch({}, {})
    finally:
        repair_target.reset(token)

    assert result[0]["path"].endswith("Candidate2.java")
    assert max_active >= 2
    assert seen_targets == ["fabric-1.21.1"] * 3
    assert seen_scopes == [("sentinel.java",)] * 3
    assert engine._mmm_last_java_paths == ("src/main/java/Candidate2.java",)
    assert engine._mmm_agentic_last_search["candidate_workers"] == 3
    assert getattr(engine._request_patch, "_mmm_parallel_repair_search", False) is True


def test_platform_candidate_request_does_not_mutate_progressive_scope() -> None:
    class Router:
        def generate_text(self, *_args, **_kwargs):
            return json.dumps(
                {
                    "operations": [
                        {
                            "operation": "create",
                            "path": "src/main/java/NewFix.java",
                            "content": "final class NewFix {}",
                        }
                    ]
                }
            )

    class Engine:
        def __init__(self) -> None:
            self.router = Router()
            self.policy = SimpleNamespace(max_patch_bytes=1024 * 1024)
            self._mmm_last_java_paths = ("winner-only.java",)

        def _request_patch(self, _evidence, _context):
            raise AssertionError("placeholder must be replaced")

    module = SimpleNamespace(
        RepairEngine=Engine,
        RepairEngineError=RuntimeError,
        _extract_json=json.loads,
    )
    platform_repair._install_dynamic_patch_request(module)

    adapter = SimpleNamespace(
        minecraft_version="1.21.1",
        loader="fabric",
        yarn_mappings="1.21.1+build.3",
        java_version="21",
        fabric_loader="0.16.10",
        fabric_api="0.116.4+1.21.1",
        fabric_loom="1.9.2",
        gradle="8.10.2",
    )
    token = platform_repair._ACTIVE_REPAIR_TARGET.set(adapter)
    try:
        engine = Engine()
        operations = engine._request_patch({}, {})
    finally:
        platform_repair._ACTIVE_REPAIR_TARGET.reset(token)

    assert operations[0]["path"] == "src/main/java/NewFix.java"
    assert engine._mmm_last_java_paths == ("winner-only.java",)
    assert getattr(engine._request_patch, "_mmm_defers_repair_scope_commit", False) is True


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


def test_first_native_repair_search_primes_coder_slots(monkeypatch) -> None:
    config = SimpleNamespace(provider="local", adapter="llama_cpp")
    roles: list[str] = []

    def role(_profile, selected_role):
        roles.append(selected_role)
        return config

    router = SimpleNamespace(
        registry=SimpleNamespace(role=role),
        profile="t4_local",
    )
    monkeypatch.delenv("MMM_LLAMA_ACTIVE_PARALLEL", raising=False)
    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)

    from minecraft_mod_ai import llama_server_autotune

    calls = []

    def ensure_tuned_server(seen_config, request):
        calls.append((seen_config, request))
        monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "3")
        return "http://127.0.0.1:18910/v1"

    monkeypatch.setattr(llama_server_autotune, "ensure_tuned_server", ensure_tuned_server)

    returned = efficiency._prime_native_repair_slots(
        router,
        evidence={"diagnostics": {"diagnostics": []}},
        context={"files": []},
    )

    assert returned is config
    assert roles == ["coder"]
    assert len(calls) == 1
    assert calls[0][0] is config
    assert calls[0][1].response_format == "json"
    assert efficiency._active_parallel_slots() == 3
