from collections.abc import Mapping

from minecraft_mod_ai.host_version_catalog import load_host_catalog
from minecraft_mod_ai.populate_version_artifact_rules import make_implementation


def test_make_implementation_preserves_declared_executor_type():
    implementation = make_implementation(
        "minecraft/item/requirement",
        "1.21.5",
        executor_type="canonical_contract_validator",
        validator_profile="semantic_contract",
    )
    assert implementation["executor_type"] == "canonical_contract_validator"


def test_packaged_host_api_symbols_are_structured_contracts():
    _, bundles = load_host_catalog()
    required = {"owner", "name", "descriptor", "kind", "static", "side", "namespace"}
    assert bundles
    for context in bundles:
        symbols = context.to_dict()["host_facts"]["api_symbols"]
        assert symbols
        for name, symbol in symbols.items():
            assert isinstance(symbol, Mapping), (context.minecraft, name, symbol)
            assert required <= set(symbol), (context.minecraft, name, symbol)
