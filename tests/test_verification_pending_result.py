from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters import ModelConfigurationError
from minecraft_mod_ai.verification_pending_result import generate_resumable


class _Module:
    module_id = "space-mode"
    kind = "custom_java"
    required_gates = ("CustomGate",)


def _generator(message: str):
    class Generator:
        def generate(self, *args, **kwargs):
            raise ModelConfigurationError(message)

    return Generator()


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
    result = generate_resumable(_generator(message), module=_Module())

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
    generator = _generator("VERIFIER_UNAVAILABLE: no verifier")

    with pytest.raises(ModelConfigurationError, match="VERIFIER_UNAVAILABLE"):
        generator.generate(module=_Module())


def test_resumable_generation_does_not_swallow_other_configuration_errors() -> None:
    with pytest.raises(ModelConfigurationError, match="MUTATION_TARGET_DRIFT"):
        generate_resumable(
            _generator("MUTATION_TARGET_DRIFT: wrong file"),
            module=_Module(),
        )
