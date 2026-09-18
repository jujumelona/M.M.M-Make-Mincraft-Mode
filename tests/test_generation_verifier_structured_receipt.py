from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.generation_verifier_resilience import run_generation_verifier
from minecraft_mod_ai.validation_diagnostic_contract import diagnostic_errors


class _RuntimeModule:
    AgentToolRuntimeError = RuntimeError

    @staticmethod
    def _discover_model_project_root(workspace_root):
        return workspace_root, "."

    @staticmethod
    def _sanitize_observation(value):
        return value

    @staticmethod
    def _bounded_result(_value):
        raise AssertionError(
            "generation verifier receipts must not pass through generic result truncation"
        )


class _LargeDiagnosticService:
    def diagnostics(self, root, *, timeout_seconds, full_scan):
        del root, timeout_seconds, full_scan
        padding = "x" * (20 * 1024)
        return {
            "schema_version": "mmm/java-diagnostics-v3",
            "complete": True,
            "session_id": "session-1",
            "model_id": "model-1",
            "error_count": 1,
            "warning_count": 0,
            "verification_backend": "jdt_core",
            "diagnostics": {
                "file:///workspace/src/main/java/example/DebugToken.java": [
                    {
                        "severity": 1,
                        "code": 268435846,
                        "line": 3,
                        "message": "The import net.minecraft.item cannot be resolved",
                        "evidence": padding,
                    }
                ]
            },
        }

    def close(self):
        return None


def test_large_generation_diagnostics_remain_structured_and_actionable(tmp_path):
    runtime = SimpleNamespace(workspace_root=tmp_path)

    result = run_generation_verifier(
        runtime,
        {"timeout_seconds": 90},
        runtime_module=_RuntimeModule,
        java_service_factory=_LargeDiagnosticService,
    )

    assert result["complete"] is True
    assert result["error_count"] == 1
    assert "diagnostics" in result
    assert result["_mmm_observation"] == {
        "trust": "untrusted_data_only",
        "sanitized": True,
        "truncated": False,
    }

    errors = diagnostic_errors(result)
    assert len(errors) == 1
    assert errors[0].get("code") != "JDT_DIAGNOSTICS_UNAVAILABLE"
    assert errors[0]["message"] == "The import net.minecraft.item cannot be resolved"


class _UnavailableService:
    def diagnostics(self, root, *, timeout_seconds, full_scan):
        del root, timeout_seconds, full_scan
        raise TimeoutError("JDT owner deadline exceeded")

    def close(self):
        return None


def test_unavailable_jdt_receipt_remains_structured_and_untruncated(tmp_path):
    runtime = SimpleNamespace(workspace_root=tmp_path)
    result = run_generation_verifier(
        runtime,
        {"timeout_seconds": 90},
        runtime_module=_RuntimeModule,
        java_service_factory=_UnavailableService,
    )

    assert result["status"] == "UNAVAILABLE"
    assert result["available"] is False
    assert result["verification_backend"] == "jdt_core"
    assert result["_mmm_observation"]["truncated"] is False
    errors = diagnostic_errors(result)
    assert len(errors) == 1
    assert errors[0]["code"] == "JDT_DIAGNOSTICS_UNAVAILABLE"
