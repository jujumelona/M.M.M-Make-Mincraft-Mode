from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.mcp_repair_diagnostic_shape_contract import install


def test_mcp_repair_diagnostic_text_flattens_uri_groups() -> None:
    module = SimpleNamespace(
        _diagnostic_text=lambda evidence: "legacy",
    )
    install(module)
    text = module._diagnostic_text(
        {
            "diagnostics": {
                "diagnostics": {
                    "file:///Example.java": [
                        {
                            "severity": 1,
                            "message": "Cannot resolve net.minecraft.ExampleType",
                            "code": "compiler.err.cant.resolve",
                        }
                    ]
                }
            },
            "build": {"error": "Gradle failed"},
        }
    )

    assert "net.minecraft.ExampleType" in text
    assert "compiler.err.cant.resolve" in text
    assert "Gradle failed" in text
