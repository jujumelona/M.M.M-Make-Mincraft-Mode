from __future__ import annotations

import pytest

from minecraft_mod_ai.production_contract import (
    ProductionContractError,
    compile_production_contract,
)
from minecraft_mod_ai.quality_evidence import compile_quality_evidence

_PROPOSAL_HASH = "sha256:" + "a" * 64


def _compiled(design: dict[str, object]):
    return compile_production_contract(
        requested_prompt="Add a weather compass.",
        game_design=design,
        modules=(
            {
                "module_id": "weather_compass",
                "kind": "custom_java",
                "config": {},
                "depends_on": [],
                "required_gates": [],
            },
        ),
        acceptance_tests=("The weather compass loads.",),
    )


def _quality(contract: dict[str, object], design: dict[str, object]):
    return compile_quality_evidence(
        contract,
        _PROPOSAL_HASH,
        game_design=design,
        source_validation=None,
        build_report=None,
        jar_validation=None,
    )


def test_quality_binding_accepts_host_outline_added_after_contract_compile() -> None:
    design: dict[str, object] = {
        "title": "Weather compass",
        "systems": [{"goal": "Show the current weather state."}],
    }
    compiled = _compiled(design)
    decorated = {
        **design,
        "production_outline": [
            {
                "batch_id": "requested_features",
                "scope": "Implement the host-owned batch.",
                "deliverables": ["weather_compass_complete"],
            }
        ],
    }

    assert _quality(compiled.contract, decorated) == {}


def test_quality_binding_still_rejects_real_design_mutation() -> None:
    design: dict[str, object] = {"title": "Weather compass"}
    compiled = _compiled(design)
    mutated = {
        **design,
        "title": "Changed after approval",
        "production_outline": [{"batch_id": "requested_features"}],
    }

    with pytest.raises(ProductionContractError, match="game_design does not match"):
        _quality(compiled.contract, mutated)
