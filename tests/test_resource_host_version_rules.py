from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.resource_contracts import _host_requires_resource_template


def _context(extra_templates: list[str]):
    return SimpleNamespace(
        facts={
            "leaf_bindings": {
                "minecraft/item/model": {
                    "implementation": {"extra_templates": extra_templates}
                }
            }
        }
    )


def test_client_item_requirement_comes_from_host_leaf_binding() -> None:
    assert _host_requires_resource_template(
        _context(["minecraft/resource/item/client_item"]),
        leaf="minecraft/item/model",
        template_id="minecraft/resource/item/client_item",
    )
    assert not _host_requires_resource_template(
        _context([]),
        leaf="minecraft/item/model",
        template_id="minecraft/resource/item/client_item",
    )


def test_missing_host_leaf_binding_fails_closed() -> None:
    with pytest.raises(ValueError, match="HOST resource leaf binding unavailable"):
        _host_requires_resource_template(
            SimpleNamespace(facts={}),
            leaf="minecraft/item/model",
            template_id="minecraft/resource/item/client_item",
        )
