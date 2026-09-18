from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import agentic_optimization_contract as contract


def test_repair_candidate_jdt_is_not_in_default_compile_backed_path(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("MMM_REPAIR_CANDIDATE_JDT", raising=False)

    def forbidden_diagnostics(*_args, **_kwargs):
        raise AssertionError("candidate ranking must not start JDT by default")

    monkeypatch.setattr(contract, "run_diagnostics", forbidden_diagnostics)

    score, verifier = contract._verify_repair_candidate(
        SimpleNamespace(),
        tmp_path,
        (
            {
                "operation": "replace",
                "path": "src/main/java/demo/Example.java",
                "content": "package demo; final class Example {}\n",
            },
        ),
        {
            "diagnostics": {
                "status": "DEFERRED_TO_POST_BUILD",
                "diagnostics": {},
            },
            "build": {"status": "FAIL", "commands": []},
        },
    )

    assert isinstance(score, float)
    assert verifier["jdt_status"] == "NOT_RUN"
    assert verifier["jdt_error_count"] is None
