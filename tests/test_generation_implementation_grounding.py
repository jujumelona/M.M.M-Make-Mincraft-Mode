from __future__ import annotations

import json
from types import MappingProxyType, SimpleNamespace

from minecraft_mod_ai import generation_implementation_grounding as grounding
from minecraft_mod_ai.grounding_policy import host_baseline_evidence_ready


class _Context:
    context_id = "sha256:context"

    def require_leaf_binding(self, leaf_id):
        if leaf_id != "minecraft/item/registry":
            raise ValueError("not admitted")
        return {
            "state": "not_reviewed",
            "implementation": {
                "implementation_id": "fabric/item/register_keyed",
                "executor_type": "deterministic_renderer",
                "implementation_sha256": "sha256:impl",
                "validator_profile": "java_syntax",
                "validator_sha256": "sha256:validator",
                "input_schema_sha256": "sha256:input",
                "output_schema_sha256": "sha256:output",
                "authority_sha256": "sha256:authority",
                "template": "fabric/item/register_keyed",
                "prerequisite_templates": ["fabric/item/key_identifier"],
            },
        }

    def require_fact(self, category, name):
        assert category == "api_symbols"
        symbols = {
            "register_item": {
                "owner": "net.minecraft.core.Registry",
                "name": "register",
                "descriptor": "(...)Item",
                "kind": "method",
                "static": True,
            },
            "builtin_item_registry": {
                "owner": "net.minecraft.core.registries.BuiltInRegistries",
                "name": "ITEM",
                "descriptor": "Lnet/minecraft/core/DefaultedRegistry;",
                "kind": "field",
                "static": True,
            },
            "resource_key_create": {
                "owner": "net.minecraft.resources.ResourceKey",
                "name": "create",
                "descriptor": "(...)ResourceKey",
                "kind": "method",
                "static": True,
            },
            "registries_item": {
                "owner": "net.minecraft.core.registries.Registries",
                "name": "ITEM",
                "descriptor": "Lnet/minecraft/resources/ResourceKey;",
                "kind": "field",
                "static": True,
            },
            "identifier_factory": {
                "owner": "net.minecraft.resources.Identifier",
                "name": "fromNamespaceAndPath",
                "descriptor": "(...)Identifier",
                "kind": "method",
                "static": True,
            },
            "item_stacks_to": {
                "owner": "net.minecraft.world.item.Item$Properties",
                "name": "stacksTo",
                "descriptor": "(I)Item$Properties",
                "kind": "method",
                "static": False,
            },
        }
        symbol = symbols[name]
        return MappingProxyType(
            {
                **symbol,
                "side": "common",
                "namespace": "minecraft",
                "metadata": MappingProxyType({"source": "host_catalog"}),
            }
        )

    @property
    def api_symbols(self):
        names = (
            "register_item",
            "builtin_item_registry",
            "resource_key_create",
            "registries_item",
            "identifier_factory",
            "item_stacks_to",
        )
        return MappingProxyType(
            {name: self.require_fact("api_symbols", name) for name in names}
        )

    def admit_template(self, template):
        assert template["id"] in {
            "fabric/item/register_keyed",
            "fabric/item/key_identifier",
        }
        return {
            "template_sha256": "sha256:fixture",
            "required_symbols": (
                ["register_item"]
                if template["id"] == "fabric/item/register_keyed"
                else ["resource_key_create", "identifier_factory"]
            ),
            "requires_capabilities": ["REGISTER_ITEM"],
        }


class _Target:
    minecraft_version = "26.2"
    loader = "fabric"
    mappings_applicable = False
    version_context = _Context()


def _module():
    return SimpleNamespace(
        kind="custom_java",
        config={
            "semantic_kind": "item",
            "implementation_responsibilities": ["registry"],
            "evidence_task": {"task_id": "debug_token"},
        },
    )


def test_generation_grounding_projects_only_explicit_host_responsibility(monkeypatch):
    monkeypatch.setattr(grounding, "host_target", lambda _version: _Target())
    monkeypatch.setattr(
        grounding,
        "steps_for_artifact",
        lambda _kind: (
            SimpleNamespace(
                template_id="minecraft/item/registry",
                outcome="Implement registry",
                host_requirements=(
                    ("capabilities", ("REGISTER_ITEM",)),
                    ("schemas", ()),
                    ("symbols", ("register_item",)),
                ),
                validators=("java_parse",),
                postconditions=("registry matches target",),
            ),
            SimpleNamespace(
                template_id="minecraft/item/integration",
                outcome="Implement integration",
                host_requirements=(
                    ("capabilities", ("REGISTER_ITEM",)),
                    ("schemas", ()),
                    ("symbols", ("register_item",)),
                ),
                validators=("java_parse",),
                postconditions=("integrated",),
            ),
        ),
    )
    monkeypatch.setattr(
        grounding,
        "load_template",
        lambda template_id: {
            "id": template_id,
            "requires": ["mod_id", "registry_path"],
            "dependencies": [],
            "target": {"operation": "JAVA_PATCH"},
            "render": {
                "language": "java",
                "body": (
                    "Registry.register(BuiltInRegistries.ITEM, KEY, new Item(new Item.Properties().setId(KEY)));"
                    if template_id == "fabric/item/register_keyed"
                    else "ResourceKey.create(Registries.ITEM, Identifier.fromNamespaceAndPath(...));"
                ),
            },
        },
    )

    result = grounding.build_generation_implementation_grounding(
        _module(),
        minecraft_version="26.2",
    )

    assert result is not None
    assert result["artifact_kind"] == "item"
    assert result["responsibilities"] == ["registry"]
    assert result["selected_fact_count"] == 1
    assert len(result["facts"]) == 1
    fact = result["facts"][0]
    assert fact["responsibility"] == "minecraft/item/registry"
    assert fact["registration_state"] == "not_reviewed"
    assert [item["template_id"] for item in fact["templates"]] == [
        "fabric/item/register_keyed",
        "fabric/item/key_identifier",
    ]
    assert "BuiltInRegistries.ITEM" in fact["templates"][0]["render_body"]
    assert set(fact["templates"][0]["symbol_usage"]) == {
        "register_item",
        "builtin_item_registry",
    }
    assert set(fact["templates"][1]["symbol_usage"]) == {
        "resource_key_create",
        "registries_item",
        "identifier_factory",
    }
    assert "builtin_item_registry" not in fact["templates"][1]["symbol_usage"]
    assert "registries_item" not in fact["templates"][0]["symbol_usage"]
    assert "receiver/member/argument" in fact["templates"][0]["topology_policy"]
    assert "never interchange" in fact["call_topology_policy"]
    assert set(fact["api_symbols"]) == {
        "register_item",
        "builtin_item_registry",
        "resource_key_create",
        "registries_item",
        "identifier_factory",
        "item_stacks_to",
    }
    assert set(fact["required_imports"]) == {
        "net.minecraft.core.Registry",
        "net.minecraft.core.registries.BuiltInRegistries",
        "net.minecraft.resources.ResourceKey",
        "net.minecraft.core.registries.Registries",
        "net.minecraft.resources.Identifier",
        "net.minecraft.world.item.Item",
    }
    assert "do not substitute Yarn" in fact["import_policy"]
    assert result["policy"]["model_must_not_substitute_api_names"] is True
    assert result["policy"]["model_must_preserve_template_call_topology"] is True
    assert isinstance(fact["api_symbols"]["register_item"], dict)
    assert fact["api_symbols"]["register_item"]["metadata"] == {"source": "host_catalog"}
    json.dumps(result, ensure_ascii=False)



def test_real_26_2_item_registry_grounding_has_complete_native_import_authority() -> None:
    result = grounding.build_generation_implementation_grounding(
        _module(),
        minecraft_version="26.2",
    )

    assert result is not None
    assert result["minecraft_version"] == "26.2"
    fact = result["facts"][0]
    assert fact["responsibility"] == "minecraft/item/registry"
    assert set(fact["required_imports"]) == {
        "net.minecraft.core.Registry",
        "net.minecraft.core.registries.BuiltInRegistries",
        "net.minecraft.resources.ResourceKey",
        "net.minecraft.core.registries.Registries",
        "net.minecraft.resources.Identifier",
        "net.minecraft.world.item.Item",
    }
    assert all(
        not owner.startswith(("net.minecraft.item.", "net.minecraft.registry.", "net.minecraft.util."))
        for owner in fact["required_imports"]
    )
    templates = {item["template_id"]: item for item in fact["templates"]}
    assert "builtin_item_registry" in templates["fabric/item/register_keyed"]["symbol_usage"]
    assert "registries_item" not in templates["fabric/item/register_keyed"]["symbol_usage"]
    assert "registries_item" in templates["fabric/item/key_identifier"]["symbol_usage"]
    assert "builtin_item_registry" not in templates["fabric/item/key_identifier"]["symbol_usage"]
    json.dumps(result, ensure_ascii=False)

def test_generation_grounding_requires_explicit_structural_responsibility() -> None:
    module = SimpleNamespace(
        kind="custom_java",
        config={"semantic_kind": "item", "evidence_task": {"task_id": "x"}},
    )
    assert (
        grounding.build_generation_implementation_grounding(
            module,
            minecraft_version="26.2",
        )
        is None
    )


def test_host_implementation_contract_satisfies_fresh_task_baseline() -> None:
    fresh_capsule = {
        "schema_version": "mmm/small-model-task-capsule",
        "reuse_action": "fresh",
        "mutation_target": {"path": "src/main/java/demo/Thing.java"},
    }
    host_grounding = {
        "schema_version": "mmm/host-owned-coder-grounding-v1",
        "policy": {
            "resolved_before_first_coder_decode": True,
            "baseline_grounding_owned_by_host": True,
            "baseline_grounding_optional_for_model": False,
            "model_tool_choice_required_for_baseline": False,
        },
        "evidence_bindings": {
            "project_exact_rag": {
                "receipt": {
                    "observation_count": 1,
                    "project_sha256": "sha256:project",
                    "observations_sha256": "sha256:observations",
                }
            },
            "approved_research_rag": {
                "receipt": {"selected_fact_count": 0}
            },
            "implementation_contract": {
                "receipt": {
                    "selected_fact_count": 1,
                    "grounding_sha256": "sha256:grounding",
                    "context_id": "sha256:context",
                }
            },
        },
    }
    messages = (
        {"role": "developer", "content": json.dumps(fresh_capsule)},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "phase": "implement_module",
                    "workspace_project_root": ".",
                    "host_grounding": host_grounding,
                }
            ),
        },
    )

    assert host_baseline_evidence_ready(messages) is True
