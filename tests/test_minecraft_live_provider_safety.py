from __future__ import annotations

import json

from minecraft_mod_ai import platform_catalog


def test_fabric_provider_uses_only_host_coherent_reviewed_bundle() -> None:
    provider = platform_catalog.provider_for_loader("fabric")

    assert provider.provider_id == "host-coherent-version-catalog-v1"

    adapter = provider.resolve("1.21.11")
    adapter.validate()

    assert adapter.minecraft_version == "1.21.11"
    assert adapter.loader == "fabric"
    assert adapter.host_facts_json

    host_facts = json.loads(adapter.host_facts_json)
    assert host_facts["host_revision"]
    assert isinstance(host_facts["capabilities"], dict)
    assert isinstance(host_facts["api_symbols"], dict)
    assert isinstance(host_facts["artifact_rules"], dict)
    assert isinstance(host_facts["leaf_bindings"], dict)
