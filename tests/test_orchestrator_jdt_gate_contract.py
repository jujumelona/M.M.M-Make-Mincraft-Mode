from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.orchestrator_jdt_gate_contract import install


class _FakeJavaLanguageService:
    def diagnostics(self, *args, **kwargs):
        return {
            "schema_version": "mmm/java-diagnostics-v2",
            "error_count": 1,
            "warning_count": 1,
            "diagnostics": {
                "file:///A.java": [
                    {"severity": 1, "message": "compile error"},
                    {"severity": 2, "message": "warning"},
                ]
            },
        }


def test_orchestrator_gate_flattens_only_blocking_errors() -> None:
    module = SimpleNamespace(JavaLanguageService=_FakeJavaLanguageService)
    install(module)
    receipt = module.JavaLanguageService().diagnostics("unused")

    assert receipt["diagnostics"] == [
        {"severity": 1, "message": "compile error"}
    ]
    assert receipt["diagnostics_by_uri"]["file:///A.java"] == [
        {"severity": 1, "message": "compile error"},
        {"severity": 2, "message": "warning"},
    ]
    assert receipt["error_count"] == 1
    assert receipt["warning_count"] == 1
