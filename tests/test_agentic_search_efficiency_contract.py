from __future__ import annotations

import json
import threading
import time
from contextvars import ContextVar
from functools import wraps
from types import SimpleNamespace

from minecraft_mod_ai import agentic_optimization_contract as agentic
from minecraft_mod_ai import agentic_search_efficiency_contract as efficiency
from minecraft_mod_ai.agentic_search_efficiency_contract import install


def _complex_failure() -> dict:
    return {
        "diagnostics": {
            "diagnostics": [
                {"path": "A.java", "message": "error one"},
                {"path": "B.java", "message": "error two"},
            ]
        },
        "build": {"status": "FAIL", "error": "x" * 200},
    }


def test_auto_repair_search_uses_native_slots_for_complex_failure(monkeypatch) -> None:
    install(agentic)
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_REPAIR_SEARCH_WIDTH", "2")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    engine = SimpleNamespace(_signature=lambda _evidence: "same-signature")
    assert agentic._repair_candidate_count(engine, _complex_failure(), ()) == 2


def test_explicit_repair_search_is_owned_by_efficiency_policy(monkeypatch) -> None:
    install(agentic)
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "on")
    monkeypatch.setenv("MMM_REPAIR_SEARCH_WIDTH", "2")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    monkeypatch.setattr(agentic, "_mode", lambda: "auto")
    engine = SimpleNamespace(_signature=lambda _evidence: "same-signature")
    assert agentic._repair_candidate_count(engine, _complex_failure(), ()) == 2
    assert (
        getattr(agentic._repair_candidate_count, "_mmm_failure_gated_search_epoch", "")
        == "mmm/failure-gated-search-v3"
    )


def test_installed_repair_policy_isolated_from_mutable_module_helpers(monkeypatch) -> None:
    install(agentic)
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "on")
    monkeypatch.setenv("MMM_REPAIR_SEARCH_WIDTH", "2")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    monkeypatch.setattr(efficiency, "_search_mode", lambda: "off")
    monkeypatch.setattr(efficiency, "_repair_search_width", lambda: 1)
    monkeypatch.setattr(efficiency, "_active_parallel_slots", lambda: 1)
    engine = SimpleNamespace(_signature=lambda _evidence: "same-signature")
    assert agentic._repair_candidate_count(engine, _complex_failure(), ()) == 2


def test_auto_repair_search_does_not_duplicate_serial_decode(monkeypatch) -> None:
    install(agentic)
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_REPAIR_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    engine = SimpleNamespace(_signature=lambda _evidence: "same-signature")
    evidence = {
        "diagnostics": {},
        "build": {"status": "FAIL", "error": "x" * 200},
    }
    assert agentic._repair_candidate_count(engine, evidence, ()) == 1


def test_parallel_repair_search_preserves_context_and_commits_only_winner(monkeypatch) -> None:
    active = 0
    max_active = 0
    seen_targets: list[str] = []
    guard = threading.Lock()
    repair_target: ContextVar[str] = ContextVar("repair_target", default="missing")

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
            time.sleep(0.03)
            with guard:
                active -= 1
            return [
                {
                    "operation": "create",
                    "path": f"src/main/java/Candidate{candidate}.java",
                    "content": f"final class Candidate{candidate} {{}}",
                }
            ]

    repair_module = SimpleNamespace(RepairEngine=Engine, RepairEngineError=RuntimeError)
    fake = SimpleNamespace()
    fake._repair_candidate_count = lambda *_args, **_kwargs: 3
    fake._mode = lambda: "on"
    fake._env_int = lambda name, default, **_kwargs: (
        3 if name == "MMM_REPAIR_SEARCH_WIDTH" else default
    )
    fake._STRATEGIES = ("minimal", "contract", "dependency")
    fake._read_memory = lambda *_args, **_kwargs: []
    fake._compact_evidence = lambda _evidence: {}
    fake._repair_pattern = lambda _operations: []
    fake._json_size = lambda value: len(json.dumps(value, sort_keys=True))
    fake._verify_repair_candidate = lambda _self, _root, operations, _evidence: (
        float(int(operations[0]["path"].split("Candidate", 1)[1].split(".", 1)[0])),
        {"status": "PASS"},
    )

    def repair_installer(module) -> None:
        current = module.RepairEngine._request_patch

        @wraps(current)
        def verified(self, evidence, context):
            return current(self, evidence, context)

        verified._mmm_verifier_repair_search = True
        verified.__wrapped__ = current
        module.RepairEngine._request_patch = verified

    fake._install_repair_search_and_memory = repair_installer
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "on")
    monkeypatch.setenv("MMM_REPAIR_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "3")
    efficiency.install(fake)
    fake._install_repair_search_and_memory(repair_module)

    token = repair_target.set("fabric-1.21.1")
    try:
        engine = Engine()
        result = engine._request_patch({}, {})
    finally:
        repair_target.reset(token)

    assert result[0]["path"].endswith("Candidate2.java")
    assert max_active >= 2
    assert seen_targets == ["fabric-1.21.1"] * 3
    assert engine._mmm_last_java_paths == ("src/main/java/Candidate2.java",)
    assert engine._mmm_agentic_last_search["candidate_workers"] == 3


def test_first_native_repair_search_primes_coder_slots(monkeypatch) -> None:
    config = SimpleNamespace(provider="local", adapter="llama_cpp")
    roles: list[str] = []

    def role(_profile, selected_role):
        roles.append(selected_role)
        return config

    router = SimpleNamespace(registry=SimpleNamespace(role=role), profile="t4_local")
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
    assert calls[0][1].response_format == "json"
    assert efficiency._active_parallel_slots() == 3
