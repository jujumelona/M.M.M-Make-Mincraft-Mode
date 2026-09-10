from __future__ import annotations

import inspect

from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.extended_content_generator import generate_extended_content
from minecraft_mod_ai.generation_concurrency_safety import _builtin_shared_anchors
from minecraft_mod_ai.scale_policy import ScalePolicy
from minecraft_mod_ai.work_graph import _module_shards


def _shards(monkeypatch, modules):
    for name in (
        'MMM_CONTENT_PIPELINE_SHARD_SIZE',
        'MMM_SYSTEM_PIPELINE_SHARD_SIZE',
        'MMM_ENTITY_PIPELINE_SHARD_SIZE',
    ):
        monkeypatch.delenv(name, raising=False)
    return list(_module_shards(modules, policy=ScalePolicy()))


def test_builtin_content_system_entity_are_singleton_dag_nodes_by_default(monkeypatch):
    modules = (
        ProductionModule('ruby_sword', 'weapon'),
        ProductionModule('ruby_armor', 'armor'),
        ProductionModule('quest_alpha', 'quest'),
        ProductionModule('shop_alpha', 'shop'),
        ProductionModule('wolf_alpha', 'entity'),
        ProductionModule('wolf_beta', 'entity'),
    )
    shards = _shards(monkeypatch, modules)
    relevant = [(stage, members) for stage, members in shards if stage in {'content', 'system', 'entity'}]
    assert len(relevant) == len(modules)
    assert all(len(members) == 1 for _, members in relevant)


def test_builtin_anchor_scope_is_module_or_system_pack_not_stage_global():
    sword = ProductionModule('ruby_sword', 'weapon')
    armor = ProductionModule('ruby_armor', 'armor')
    quest_a = ProductionModule('quest_a', 'quest')
    quest_b = ProductionModule('quest_b', 'quest')
    shop = ProductionModule('shop_a', 'shop')
    assert _builtin_shared_anchors(sword, 'content') != _builtin_shared_anchors(armor, 'content')
    assert _builtin_shared_anchors(quest_a, 'system') == _builtin_shared_anchors(quest_b, 'system')
    assert _builtin_shared_anchors(quest_a, 'system') != _builtin_shared_anchors(shop, 'system')


def test_extended_content_no_long_function_wide_serialization_wrapper():
    source = inspect.getsource(generate_extended_content)
    assert '_serialized_extended_content' not in source
    assert 'with project_write_lock(info.root):' in source
