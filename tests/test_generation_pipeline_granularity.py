from __future__ import annotations

from minecraft_mod_ai import generation_concurrency_safety as safety
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.scale_policy import ScalePolicy
from minecraft_mod_ai.work_graph import _module_shards


def _entity(module_id: str) -> ProductionModule:
    return ProductionModule(
        module_id=module_id,
        kind="entity",
        config={},
        depends_on=(),
        required_gates=(),
    )


def test_default_entity_pipeline_exposes_each_entity_as_its_own_work_node(monkeypatch):
    monkeypatch.delenv("MMM_ENTITY_PIPELINE_SHARD_SIZE", raising=False)
    safety._configure_pipeline_granularity()

    shards = list(
        _module_shards(
            (_entity("alpha"), _entity("beta"), _entity("gamma")),
            policy=ScalePolicy(),
        )
    )

    entity_shards = [members for stage, members in shards if stage == "entity"]
    assert len(entity_shards) == 3
    assert [members[0].module_id for members in entity_shards] == ["alpha", "beta", "gamma"]
    assert all(len(members) == 1 for members in entity_shards)


def test_explicit_entity_batching_is_still_respected(monkeypatch):
    monkeypatch.setenv("MMM_ENTITY_PIPELINE_SHARD_SIZE", "2")

    shards = list(
        _module_shards(
            (_entity("alpha"), _entity("beta"), _entity("gamma")),
            policy=ScalePolicy(),
        )
    )

    entity_shards = [members for stage, members in shards if stage == "entity"]
    assert [len(members) for members in entity_shards] == [2, 1]
