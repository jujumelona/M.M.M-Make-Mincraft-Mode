from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.execution_feedback_semantic_convergence_installation import (
    _semantic_install_run_context,
    _verifier_infrastructure_failure,
)
from minecraft_mod_ai.project_java_diagnostics_installation import (
    ProjectJavaResolutionError,
    _infer_project_java_major,
    _missing_release_major,
)


def test_jdt_unavailable_is_non_source_repairable_and_extracts_release() -> None:
    failure = _verifier_infrastructure_failure(
        {
            "tool": "java_diagnostics",
            "result": {
                "code": "JDT_DIAGNOSTICS_UNAVAILABLE",
                "error": "error: release 25 is not found in the system",
            },
        }
    )

    assert failure is not None
    assert failure["code"] == "JDT_DIAGNOSTICS_UNAVAILABLE"
    assert failure["required_java"] == 25
    assert str(failure["fingerprint"]).startswith("sha256:")


def test_normal_source_diagnostic_is_not_infrastructure_failure() -> None:
    assert (
        _verifier_infrastructure_failure(
            {
                "diagnostics": [
                    {
                        "path": "src/main/java/example/DebugToken.java",
                        "message": "cannot find symbol: variable missingToken",
                        "severity": "error",
                    }
                ]
            }
        )
        is None
    )


def test_project_java_major_is_read_from_gradle_toolchain(tmp_path) -> None:
    (tmp_path / "build.gradle").write_text(
        """
        java {
            toolchain {
                languageVersion = JavaLanguageVersion.of(25)
            }
        }
        """,
        encoding="utf-8",
    )

    assert _infer_project_java_major(tmp_path) == 25


def test_conflicting_project_java_requirements_fail_closed(tmp_path) -> None:
    (tmp_path / "build.gradle").write_text(
        "java { toolchain { languageVersion = JavaLanguageVersion.of(25) } }\n",
        encoding="utf-8",
    )
    (tmp_path / "gradle.properties").write_text("java_version=21\n", encoding="utf-8")

    with pytest.raises(ProjectJavaResolutionError, match="Conflicting project Java"):
        _infer_project_java_major(tmp_path)


def test_missing_release_is_extracted_from_exception_chain() -> None:
    try:
        try:
            raise RuntimeError("error: release 25 is not found in the system")
        except RuntimeError as exc:
            raise ValueError("JDT import failed") from exc
    except ValueError as outer:
        assert _missing_release_major(outer) == 25


class _CompleteProductionError(RuntimeError):
    pass


@dataclass(frozen=True)
class _Options:
    resume: bool = False


class _Ledger:
    def __init__(self) -> None:
        self.invalidations = 0

    def invalidate_execution_feedback(self, feedback):
        self.invalidations += 1
        return {
            "feedback_fingerprint": "source-fingerprint",
            "global_replan_required": False,
            "impacted_generation_node_ids": ["generate-debug-token"],
        }


class _Orchestrator:
    def __init__(self, ledger: _Ledger) -> None:
        self._mmm_feedback_ledger = ledger
        self.calls = 0

    def _open_run(self, run_name, plan, *, resume):
        return None, self._mmm_feedback_ledger, False

    def execute(self, *args, **kwargs):
        self.calls += 1
        raise _CompleteProductionError("verification failed")


def test_verifier_infrastructure_failure_never_invalidates_generation(monkeypatch) -> None:
    ledger = _Ledger()

    feedback_module = SimpleNamespace(
        _latest_failed_feedback=lambda current_ledger: {
            "tool": "java_diagnostics",
            "code": "VERIFIER_UNAVAILABLE",
            "message": "JDT diagnostics are unavailable: release 25 is not found in the system",
        }
    )
    orchestrator_module = SimpleNamespace(
        CompleteProductionOrchestrator=_Orchestrator,
        CompleteProductionError=_CompleteProductionError,
        CompleteExecutionOptions=_Options,
    )

    # Patch only these local fake classes; production classes are untouched.
    _semantic_install_run_context(feedback_module, orchestrator_module)
    orchestrator = _Orchestrator(ledger)

    with pytest.raises(_CompleteProductionError, match="verification failed"):
        orchestrator.execute(options=_Options(resume=True))

    assert orchestrator.calls == 1
    assert ledger.invalidations == 0
