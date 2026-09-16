from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.model_adapters import ModelConfigurationError
from minecraft_mod_ai.verification_pending_result_installation import install


class _Module:
    module_id = "space-mode"
    kind = "custom_java"
    required_gates = ("CustomGate",)


def _generator_module(message: str):
    class Generator:
        def generate(self, *args, **kwargs):
            raise ModelConfigurationError(message)

    module = SimpleNamespace(CustomModuleGenerator=Generator)
    install(module)
    return module


@pytest.mark.parametrize(
    ("message", "code"),
    (
        ("VERIFIER_UNAVAILABLE: no healthy verifier remains", "VERIFIER_UNAVAILABLE"),
        (
            "VERIFIER_RECOVERY_UNAVAILABLE: recovery evidence unavailable",
            "VERIFIER_RECOVERY_UNAVAILABLE",
        ),
    ),
)
def test_resumable_generation_returns_structured_pending_verification(
    message: str,
    code: str,
) -> None:
    module = _generator_module(message)
    result = module.CustomModuleGenerator().generate_resumable(module=_Module())

    assert result["status"] == "VERIFICATION_PENDING"
    assert result["resumable"] is True
    assert result["release_ready"] is False
    assert result["verification"] == {
        "status": "UNAVAILABLE",
        "phase": "VERIFY",
        "verified": False,
        "reason_code": code,
        "detail": message,
    }
    assert result["generation_checkpoint"]["status"] == "PRESERVED_FOR_RESUME"
    assert result["required_gates"] == ["JDT", "Gradle", "GameTest", "CustomGate"]


def test_production_generate_remains_fail_closed() -> None:
    module = _generator_module("VERIFIER_UNAVAILABLE: no verifier")
    generator = module.CustomModuleGenerator()

    with pytest.raises(ModelConfigurationError, match="VERIFIER_UNAVAILABLE"):
        generator.generate(module=_Module())


def test_resumable_generation_does_not_swallow_other_configuration_errors() -> None:
    module = _generator_module("MUTATION_TARGET_DRIFT: wrong file")

    with pytest.raises(ModelConfigurationError, match="MUTATION_TARGET_DRIFT"):
        module.CustomModuleGenerator().generate_resumable(module=_Module())
