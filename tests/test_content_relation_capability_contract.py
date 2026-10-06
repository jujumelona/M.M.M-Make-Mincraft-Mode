from __future__ import annotations

from minecraft_mod_ai import design_record_runtime
from minecraft_mod_ai.content_design_contract import (
    relation_type_supported_for_content_pair,
)


def test_relation_capability_contract_reserves_contains_for_tag_sources() -> None:
    assert relation_type_supported_for_content_pair(
        "contains", "registry_tag", "item"
    )
    assert relation_type_supported_for_content_pair(
        "contains", "registry_tag", "block"
    )
    assert relation_type_supported_for_content_pair(
        "contains", "registry_tag", "entity"
    )
    assert not relation_type_supported_for_content_pair(
        "contains", "item", "registry_tag"
    )


def test_relation_capability_contract_matches_special_lowering_shapes() -> None:
    assert relation_type_supported_for_content_pair(
        "consumes", "crafting_recipe", "item"
    )
    assert relation_type_supported_for_content_pair(
        "produces", "smelting_recipe", "item"
    )
    assert relation_type_supported_for_content_pair(
        "key_A", "crafting_recipe", "item"
    )
    assert not relation_type_supported_for_content_pair(
        "key_A", "smelting_recipe", "item"
    )
    assert relation_type_supported_for_content_pair("drops", "block", "item")
    assert not relation_type_supported_for_content_pair("drops", "block", "block")
    assert relation_type_supported_for_content_pair("opens", "item", "gui")
    assert not relation_type_supported_for_content_pair("opens", "item", "block")


def test_relation_runtime_binds_contains_to_the_valid_direction(monkeypatch) -> None:
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    def serial_map(router, jobs, run_one, **kwargs):
        del router, kwargs
        return [run_one(job) for job in jobs]

    def fake_run_bounded(
        router,
        identifier,
        *,
        context,
        progress,
        checkpoint,
        **kwargs,
    ):
        del router, progress, checkpoint, kwargs
        assert identifier == "design/content_relation"
        source_id = context["source_id"]
        target_id = context["target_id"]
        allowed = tuple(context["allowed_relation_types"])
        calls.append((source_id, target_id, allowed))

        if source_id == "tag_valid_ships" and target_id == "ship_blueprint":
            assert allowed == ("contains",)
            return {"records": [{"relation_type": "contains"}]}
        assert "contains" not in allowed
        return {"records": []}

    monkeypatch.setattr(design_record_runtime, "deterministic_model_map", serial_map)
    monkeypatch.setattr(
        design_record_runtime,
        "run_bounded_record_template",
        fake_run_bounded,
    )

    records, _ = design_record_runtime._run_relations(
        object(),
        "design/content_relation",
        {
            "entity_ids": ["ship_blueprint", "tag_valid_ships"],
            "entities": [
                {
                    "entity_id": "ship_blueprint",
                    "kind": "item",
                    "implementation_obligations": [],
                },
                {
                    "entity_id": "tag_valid_ships",
                    "kind": "registry_tag",
                    "implementation_obligations": [],
                },
            ],
        },
        {},
        None,
    )

    assert records == [
        {
            "relation_type": "contains",
            "source_id": "tag_valid_ships",
            "target_id": "ship_blueprint",
        }
    ]
    assert calls
