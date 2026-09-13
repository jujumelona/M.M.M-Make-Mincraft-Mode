from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import active_repair_verifier_contract as contract


def _evaluation(
    score: float,
    index: int,
    *,
    jdt_error_count: int = 1,
    jdt_status: str = "FAIL",
):
    return (
        score,
        index,
        (),
        {
            "jdt_error_count": jdt_error_count,
            "jdt_status": jdt_status,
        },
    )


def test_ambiguity_policy_is_explicit_and_fail_closed(monkeypatch) -> None:
    evaluations = [
        _evaluation(200.0, 0),
        _evaluation(0.0, 1),
    ]

    monkeypatch.setenv("MMM_ACTIVE_REPAIR_VERIFIER", "off")
    assert contract._ambiguous(evaluations) is False

    monkeypatch.setenv("MMM_ACTIVE_REPAIR_VERIFIER", "on")
    assert contract._ambiguous(evaluations) is True

    monkeypatch.setenv("MMM_ACTIVE_REPAIR_VERIFIER", "auto")
    monkeypatch.setenv("MMM_ACTIVE_REPAIR_SCORE_MARGIN", "10")
    assert contract._ambiguous(evaluations) is True


def test_evidence_seed_retains_only_failure_oracle_structure() -> None:
    evidence = {
        "diagnostics": {
            "diagnostics": [
                {
                    "code": "compiler.err.cant.resolve",
                    "message": "  cannot   resolve   symbol  ",
                    "severity": 2,
                    "ignored": "not copied",
                }
            ]
        },
        "build": {
            "status": "FAIL",
            "commands": [
                {"name": "clean_build", "exit_code": 1, "timed_out": False},
                {"name": "successful_probe", "exit_code": 0, "timed_out": False},
                {"name": "gametest", "exit_code": 0, "timed_out": True},
            ],
        },
        "unrelated": {"large": "payload"},
    }

    seed = contract._evidence_seed(evidence)

    assert seed == {
        "schema_version": "mmm/counterexample-seed-v1",
        "diagnostics": [
            {
                "code": "compiler.err.cant.resolve",
                "message": "cannot resolve symbol",
                "severity": 2,
            }
        ],
        "failing_commands": ["clean_build", "gametest"],
        "build_status": "FAIL",
    }


def test_install_wraps_candidate_verifier_once_and_attaches_seed() -> None:
    calls: list[tuple[object, object, object, object]] = []

    def verify(self, root, operations, evidence):
        calls.append((self, root, operations, evidence))
        return 17.0, {"jdt_status": "FAIL", "jdt_error_count": 1}

    optimization_module = SimpleNamespace(
        _verify_repair_candidate=verify,
        _mmm_active_candidate_discriminator=None,
    )

    contract.install(optimization_module)
    wrapped = optimization_module._verify_repair_candidate
    contract.install(optimization_module)

    assert optimization_module._verify_repair_candidate is wrapped
    assert callable(optimization_module._mmm_active_candidate_discriminator)

    evidence = {
        "diagnostics": {
            "diagnostics": [
                {"code": "E1", "message": "broken", "severity": 2},
            ]
        },
        "build": {
            "status": "FAIL",
            "commands": [
                {"name": "clean_build", "exit_code": 1, "timed_out": False},
            ],
        },
    }
    owner = object()
    score, verifier = wrapped(owner, None, (), evidence)

    assert score == 17.0
    assert calls == [(owner, None, (), evidence)]
    assert verifier["jdt_status"] == "FAIL"
    assert verifier["counterexample_seed"]["schema_version"] == "mmm/counterexample-seed-v1"
    assert verifier["counterexample_seed"]["failing_commands"] == ["clean_build"]


def test_discriminator_does_not_run_without_project_root(monkeypatch) -> None:
    monkeypatch.setenv("MMM_ACTIVE_REPAIR_VERIFIER", "on")

    def verify(self, root, operations, evidence):
        return 0.0, dict(evidence)

    optimization_module = SimpleNamespace(
        _verify_repair_candidate=verify,
        _mmm_active_candidate_discriminator=None,
    )
    contract.install(optimization_module)

    evaluations = [
        _evaluation(10.0, 0),
        _evaluation(9.0, 1),
    ]
    result = optimization_module._mmm_active_candidate_discriminator(None, evaluations)

    assert result == evaluations
