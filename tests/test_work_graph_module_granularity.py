from __future__ import annotations

from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.scale_policy import ScalePolicy
from minecraft_mod_ai.work_graph import _module_shards


def _module(module_id: str, kind: str, *, depends_on=()):
    return ProductionModule(
        module_id=module_id,
        kind=kind,
        config={},
        depends_on=tuple(depends_on),
    )


def test_content_defaults_to_one_module_per_work_node(monkeypatch):
    monkeypatch.delenv("MMM_CONTENT_PIPELINE_SHARD_SIZE", raising=False)
    shards = list(
        _module_shards(
            (_module("alpha_item", "item"), _module("beta_item", "item")),
            policy=ScalePolicy(),
            deterministic_module_kinds=frozenset({"item"}),
        )
    )
    assert [(stage, [m.module_id for m in members]) for stage, members in shards] == [
        ("content", ["alpha_item"]),
        ("content", ["beta_item"]),
    ]


def test_entity_defaults_to_one_module_per_work_node(monkeypatch):
    monkeypatch.delenv("MMM_ENTITY_PIPELINE_SHARD_SIZE", raising=False)
    shards = list(
        _module_shards(
            (_module("alpha_entity", "entity"), _module("beta_entity", "entity")),
            policy=ScalePolicy(),
        )
    )
    assert all(len(members) == 1 for stage, members in shards if stage == "entity")


def test_explicit_shard_override_can_trade_parallelism_for_batching(monkeypatch):
    monkeypatch.setenv("MMM_CONTENT_PIPELINE_SHARD_SIZE", "2")
    shards = list(
        _module_shards(
            (_module("alpha_item", "item"), _module("beta_item", "item")),
            policy=ScalePolicy(),
            deterministic_module_kinds=frozenset({"item"}),
        )
    )
    assert len(shards) == 1
    assert [m.module_id for m in shards[0][1]] == ["alpha_item", "beta_item"]
