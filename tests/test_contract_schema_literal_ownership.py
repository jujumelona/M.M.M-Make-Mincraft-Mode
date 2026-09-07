from __future__ import annotations

from minecraft_mod_ai.contract_schema_catalog import duplicate_contract_schema_ids


def test_versioned_contract_schema_ids_have_one_runtime_literal_owner() -> None:
    duplicates = duplicate_contract_schema_ids()
    assert not duplicates, "duplicate contract schema literal owners: " + repr(duplicates)
