import json

import pytest

from minecraft_mod_ai.atomic_slot_executor import _bounded_context
from minecraft_mod_ai.implementation_template_renderer import render_template
from minecraft_mod_ai.resolved_version_context import ResolvedVersionContext, VersionContextError


class _ResolvedFixture:
    context_id = "sha256:fixture-context"

    def to_dict(self):
        return {
            "context_id": self.context_id,
            "source": "HOST",
            "target": {
                "adapter_id": "fixture-adapter",
                "edition": "java",
                "loader": "fabric",
                "minecraft_version": "26.2",
                "java_version": "25",
                "yarn_mappings": "",
                "mappings_kind": "",
                "mappings_version": "",
                "fabric_loader": "fixture-loader",
                "fabric_api": "fixture-api",
                "fabric_loom": "fixture-loom",
                "gradle": "fixture-gradle",
                "gradle_sha256": "a" * 64,
                "data_pack_version": "88.0",
                "resource_pack_version": "69.0",
                "resource_pack_format": 69,
                "release_metadata_url": "https://www.minecraft.net/fixture",
                "source_api_family": "fixture-family",
                "deterministic_module_kinds": ["REGISTER_ITEM"],
            },
            "host_facts": {},
        }


@pytest.fixture
def resolved_context(monkeypatch):
    fixture = _ResolvedFixture()
    monkeypatch.setattr(
        ResolvedVersionContext,
        "from_dict",
        classmethod(lambda cls, value: fixture),
    )
    return fixture


def test_renderer_prefills_complete_resolved_target(resolved_context):
    template = {
        "id": "fixture/version-prefill",
        "inputs": {
            "loader": {"type": "string", "required": True},
            "gradle_sha256": {"type": "string", "required": True},
            "release_metadata_url": {"type": "string", "required": True},
            "source_api_family": {"type": "string", "required": True},
        },
        "render": {
            "language": "text",
            "body": "{{loader}}|{{gradle_sha256}}|{{release_metadata_url}}|{{source_api_family}}",
        },
    }

    rendered = render_template(
        template,
        {
            "resolved_version_context": {"synthetic": True},
            "unrelated_project_value": "must-not-enter-template-contract",
        },
    )

    assert rendered == (
        "fabric|"
        + "a" * 64
        + "|https://www.minecraft.net/fixture|fixture-family"
    )


def test_renderer_rejects_override_of_any_resolved_target_field(resolved_context):
    template = {
        "id": "fixture/version-conflict",
        "inputs": {"gradle_sha256": {"type": "string", "required": True}},
        "render": {"language": "text", "body": "{{gradle_sha256}}"},
    }

    with pytest.raises(VersionContextError) as error:
        render_template(
            template,
            {
                "resolved_version_context": {"synthetic": True},
                "gradle_sha256": "caller-invented-value",
            },
        )

    assert error.value.diagnostic["code"] == "HOST_FACT_OVERRIDE"
    assert error.value.diagnostic["field"] == "gradle_sha256"
    assert error.value.diagnostic["expected"] == "a" * 64
    assert error.value.diagnostic["actual"] == "caller-invented-value"


def test_renderer_prefills_derived_version_facts(resolved_context):
    template = {
        "id": "fixture/derived-version-prefill",
        "inputs": {
            "mappings_applicable": {"type": "boolean", "required": True},
            "pack_versions": {"type": "object", "required": True},
        },
        "render": {
            "language": "json",
            "body": {
                "mappings_applicable": "{{mappings_applicable}}",
                "pack_versions": "{{pack_versions}}",
            },
        },
    }

    rendered = render_template(
        template,
        {"resolved_version_context": {"synthetic": True}},
    )

    assert '"mappings_applicable": false' in rendered
    assert '"data": "88.0"' in rendered
    assert '"resource": "69.0"' in rendered
    assert '"resource_major": 69' in rendered


def test_atomic_slot_context_gets_prefill_without_full_host_snapshot(resolved_context):
    payload = json.loads(
        _bounded_context(
            {
                "resolved_version_context": {"synthetic": True},
                "mod_id": "demo",
            }
        )
    )

    assert "resolved_version_context" not in payload
    assert payload["mod_id"] == "demo"
    assert payload["minecraft_version"] == "26.2"
    assert payload["loader"] == "fabric"
    assert payload["java_version"] == "25"
    assert payload["gradle_sha256"] == "a" * 64
    assert payload["release_metadata_url"] == "https://www.minecraft.net/fixture"
    assert payload["mappings_applicable"] is False
    assert payload["pack_versions"] == {
        "data": "88.0",
        "resource": "69.0",
        "resource_major": 69,
    }
