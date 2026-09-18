from __future__ import annotations

import json
from types import SimpleNamespace

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
        return {
            "owner": "net.minecraft.core.Registry",
            "name": "register",
            "descriptor": "(...)Item",
            "kind": "method",
            "static": True,
            "side": "common",
            "namespace": "minecraft",
        }

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
    assert set(fact["api_symbols"]) == {
        "register_item",
        "resource_key_create",
        "identifier_factory",
    }
    assert result["policy"]["model_must_not_substitute_api_names"] is True


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
