from __future__ import annotations

"""Late-bootstrap bridge for the validation fingerprint performance contract.

Broad cross-module research hotpath monkeypatch composition is retired.  The validation
fingerprint cache remains a supported exact-input optimization and is installed here
until that implementation is folded directly into validation_execution_contract.
"""


def install() -> None:
    from . import validation_execution_contract
    from .research_validation_fingerprint_performance import harden

    harden(validation_execution_contract)


__all__ = ["install"]
