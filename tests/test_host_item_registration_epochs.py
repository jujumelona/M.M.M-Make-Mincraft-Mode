from minecraft_mod_ai.host_item_registration import item_registration_epoch


def _epoch(version: str) -> dict:
    return item_registration_epoch(version)


def test_item_registration_epoch_before_1_21_2_is_direct_resource_location():
    epoch = _epoch("1.21.1")
    assert epoch["id"] == "direct_resource_location"
    assert epoch["resource_identifier_owner"] == "net.minecraft.resources.ResourceLocation"
    assert epoch["requires_resource_key"] is False
    assert epoch["requires_set_id"] is False


def test_item_registration_epoch_1_21_2_through_1_21_10_is_keyed_resource_location():
    for version in ("1.21.2", "1.21.4", "1.21.5", "1.21.10"):
        epoch = _epoch(version)
        assert epoch["id"] == "keyed_resource_location"
        assert epoch["resource_identifier_owner"] == "net.minecraft.resources.ResourceLocation"
        assert epoch["resource_identifier_factory"] == "fromNamespaceAndPath"
        assert epoch["requires_resource_key"] is True
        assert epoch["requires_set_id"] is True


def test_item_registration_epoch_1_21_11_and_later_is_keyed_identifier():
    for version in ("1.21.11", "26.1.2"):
        epoch = _epoch(version)
        assert epoch["id"] == "keyed_identifier"
        assert epoch["resource_identifier_owner"] == "net.minecraft.resources.Identifier"
        assert epoch["resource_identifier_factory"] == "fromNamespaceAndPath"
        assert epoch["requires_resource_key"] is True
        assert epoch["requires_set_id"] is True


def test_item_registration_epoch_uses_mojang_mapped_registry_owners():
    epoch = _epoch("1.21.5")
    assert epoch["registry_owner"] == "net.minecraft.core.Registry"
    assert epoch["builtin_registries_owner"] == "net.minecraft.core.registries.BuiltInRegistries"
    assert epoch["registries_owner"] == "net.minecraft.core.registries.Registries"
    assert epoch["resource_key_owner"] == "net.minecraft.resources.ResourceKey"


def test_item_registration_epoch_returns_fresh_host_fact_mapping():
    first = _epoch("1.21.5")
    first["id"] = "mutated"
    assert _epoch("1.21.5")["id"] == "keyed_resource_location"
