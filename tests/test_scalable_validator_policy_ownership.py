from __future__ import annotations

from minecraft_mod_ai import scalable_validator as scalable_validator_module


class _CountingPolicy:
    def __init__(self) -> None:
        self.validate_calls = 0

    def validate(self) -> None:
        self.validate_calls += 1


class _ProjectValidatorStub:
    def __init__(self, *, policy=None) -> None:
        self.policy = policy
        self.policy.validate()

    def validate(self, root, spec):
        return (root, spec)


def test_scalable_validator_does_not_revalidate_policy_before_delegate(monkeypatch, tmp_path) -> None:
    policy = _CountingPolicy()
    monkeypatch.setattr(
        scalable_validator_module,
        "ProjectValidator",
        _ProjectValidatorStub,
    )

    validator = scalable_validator_module.ScalableProjectValidator(policy=policy)

    assert policy.validate_calls == 1
    assert validator.policy is policy
    assert validator.validator.policy is policy

    spec = object()
    assert validator.validate(tmp_path, spec) == (tmp_path, spec)
