from __future__ import annotations

import inspect

from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator
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


def test_builtin_content_system_entity_use_bounded_stage_aggregation_by_default(monkeypatch):
    modules = (
        ProductionModule('ruby_sword', 'weapon'),
        ProductionModule('ruby_armor', 'armor'),
        ProductionModule('quest_alpha', 'quest'),
        ProductionModule('shop_alpha', 'shop'),
        ProductionModule('wolf_alpha', 'entity'),
        ProductionModule('wolf_beta', 'entity'),
    )
    shards = _shards(monkeypatch, modules)
    relevant = [
        (stage, tuple(module.module_id for module in members))
        for stage, members in shards
        if stage in {'content', 'system', 'entity'}
    ]

    # Deterministic content is bounded by ScalePolicy.java_shard_size, system keeps
    # one module per shard by default, and entity generation uses bounded pairs.
    assert relevant == [
        ('content', ('ruby_sword', 'ruby_armor')),
        ('system', ('quest_alpha',)),
        ('system', ('shop_alpha',)),
        ('entity', ('wolf_alpha', 'wolf_beta')),
    ]
    assert sum(len(members) for _, members in relevant) == len(modules)


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


def test_blockbench_review_uses_dedicated_parallel_lane():
    source = inspect.getsource(CompleteProductionOrchestrator._execute_generation_work)
    assert "thread_name_prefix='blockbench_review'" in source
    assert 'review_pool.submit(' in source
    assert 'blockbench_receipts.append(run_named_checkpoint' not in source
    assert 'MMM_BLOCKBENCH_REVIEW_WORKERS' in source
