from __future__ import annotations

from pathlib import Path

WORK = Path("minecraft_mod_ai/work_graph.py")
SCHED = Path("minecraft_mod_ai/scheduler_parallel_safety_contract.py")
LANE_TEST = Path("tests/test_generation_lane_parallelism_contract.py")
LLAMA_TEST = Path("tests/test_llama_parallel_runtime_contract.py")
BOOT_TEST = Path("tests/test_runtime_bootstrap_clean.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise SystemExit(f"missing anchor: {label}")
    return text.replace(old, new, 1)


def replace_function(text: str, name: str, replacement: str) -> str:
    marker = f"def {name}("
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"missing function: {name}")
    next_pos = text.find("\ndef ", start + len(marker))
    if next_pos < 0:
        raise SystemExit(f"cannot find end of function: {name}")
    return text[:start] + replacement.rstrip() + "\n\n" + text[next_pos + 1 :]


def remove_function(text: str, name: str) -> str:
    marker = f"def {name}("
    start = text.find(marker)
    if start < 0:
        return text
    next_pos = text.find("\ndef ", start + len(marker))
    if next_pos < 0:
        raise SystemExit(f"cannot remove function: {name}")
    return text[:start] + text[next_pos + 1 :]


def patch_work_graph() -> None:
    text = WORK.read_text(encoding="utf-8")
    text = remove_function(text, "_content_shard_is_cpu_safe")
    text = remove_function(text, "_entity_pipeline_shard_size")

    node = '''def _node(
    node_id: str,
    stage: str,
    dependencies: Iterable[str],
    payload: dict[str, Any],
) -> WorkNode:
    """Classify execution lanes by actual mutation safety, not CPU/GPU cost alone."""

    kind = str(payload.get("kind", ""))
    gen_stage = str(payload.get("generation_stage", ""))
    if "resource_class" in payload:
        res_class = str(payload["resource_class"])
    elif kind == "asset-shard":
        res_class = "image_gpu"
    elif kind == "audio-finalize":
        res_class = "commit"
    elif kind == "module-shard" and gen_stage == "custom":
        res_class = "llm"
    elif kind == "module-shard" and gen_stage in {
        "content",
        "system",
        "entity",
        "audio-binding",
    }:
        # These generators can update shared Java/resource registries. Until they
        # produce isolated patches before commit, running them concurrently creates
        # real SHA-precondition races across stages.
        res_class = "commit"
    elif stage.startswith("validate:"):
        res_class = "commit"
    else:
        res_class = "cpu_io"

    payload_copy = dict(payload)
    payload_copy["resource_class"] = res_class
    normalized_dependencies = tuple(sorted(set(dependencies)))
    body = {
        "node_id": node_id,
        "stage": stage,
        "dependencies": normalized_dependencies,
        "payload": payload_copy,
    }
    return WorkNode(
        node_id=node_id,
        stage=stage,
        input_hash=_hash_json(body),
        dependencies=normalized_dependencies,
        payload=payload_copy,
        resource_class=res_class,
    )
'''
    text = replace_function(text, "_node", node)
    text = replace_once(
        text,
        '''        if stage == "entity":
            return _entity_pipeline_shard_size(policy)
''',
        '''        if stage == "entity":
            return max(1, int(policy.entity_shard_size))
''',
        "entity shard ceiling",
    )
    compile(text, str(WORK), "exec")
    WORK.write_text(text, encoding="utf-8")


def patch_scheduler() -> None:
    text = SCHED.read_text(encoding="utf-8")

    # Per-stage locks permit unsafe cross-stage writes to the same initializer.
    # All shared source mutations belong to the one project commit lane instead.
    start = text.find("_STAGE_WRITE_LOCKS = {")
    if start >= 0:
        boundary = text.find("_INDEX_COMMIT_LOCK", start)
        if boundary < 0:
            raise SystemExit("index commit lock boundary missing")
        text = text[:start] + text[boundary:]

    text = remove_function(text, "_stage_write_lock")

    marker = "            stage_lock = _stage_write_lock(node)\n"
    if marker in text:
        block_start = text.index(marker)
        block_end_marker = "            if not isinstance(receipt, dict):\n"
        block_end = text.find(block_end_marker, block_start)
        if block_end < 0:
            raise SystemExit("run_work_node receipt boundary missing")
        replacement_block = '''            if (
                node.resource_class == "commit"
                and shared_index is not None
                and hasattr(shared_index, "root")
            ):
                with project_write_lock(shared_index.root):
                    receipt = action()
            else:
                receipt = action()
'''
        text = text[:block_start] + replacement_block + text[block_end:]

    if "stage_lock = _stage_write_lock(node)" in text:
        raise SystemExit("stage-local write lock still owns generation")
    text = text.replace('    "_stage_write_lock",\n', "")
    compile(text, str(SCHED), "exec")
    SCHED.write_text(text, encoding="utf-8")


def patch_lane_tests() -> None:
    text = LANE_TEST.read_text(encoding="utf-8")
    text = text.replace(
        'def test_deterministic_generation_domains_use_cpu_lane() -> None:',
        'def test_shared_generation_domains_use_commit_lane() -> None:',
        1,
    )
    text = text.replace(
        '    assert content.resource_class == "cpu_io"\n    assert system.resource_class == "cpu_io"\n    assert entity.resource_class == "cpu_io"\n',
        '    assert content.resource_class == "commit"\n    assert system.resource_class == "commit"\n    assert entity.resource_class == "commit"\n',
        1,
    )
    text = text.replace(
        'def test_builtin_sidecar_integration_is_deterministic_cpu_work() -> None:',
        'def test_builtin_sidecar_integration_uses_shared_commit_lane() -> None:',
        1,
    )
    text = text.replace(
        '    assert node.resource_class == "cpu_io"\n\n\ndef test_stage_write_locks_are_domain_local_not_global()',
        '    assert node.resource_class == "commit"\n\n\ndef test_shared_generation_does_not_depend_on_stage_local_wrapper_locks()',
        1,
    )
    old_lock_body = '''    content_lock = safety._stage_write_lock(content)
    system_lock = safety._stage_write_lock(system)
    entity_lock = safety._stage_write_lock(entity)
    assert content_lock is not None
    assert system_lock is not None
    assert entity_lock is not None
    assert len({id(content_lock), id(system_lock), id(entity_lock)}) == 3
'''
    new_lock_body = '''    assert content.resource_class == "commit"
    assert system.resource_class == "commit"
    assert entity.resource_class == "commit"
    assert not hasattr(safety, "_stage_write_lock")
'''
    text = replace_once(text, old_lock_body, new_lock_body, "stage lock test")
    old_entity = '''def test_entities_use_small_pipeline_shards(monkeypatch) -> None:
    monkeypatch.delenv("MMM_ENTITY_PIPELINE_SHARD_SIZE", raising=False)
    modules = tuple(
        ProductionModule(
            module_id=f"entity_{index}",
            kind="entity",
            config={},
        )
        for index in range(5)
    )
    shards = list(work_graph_module._module_shards(modules, policy=ScalePolicy()))
    assert [stage for stage, _ in shards] == ["entity", "entity", "entity"]
    assert [len(members) for _, members in shards] == [2, 2, 1]
'''
    new_entity = '''def test_entities_use_policy_bounded_shards_without_row_explosion() -> None:
    modules = tuple(
        ProductionModule(
            module_id=f"entity_{index}",
            kind="entity",
            config={},
        )
        for index in range(5)
    )
    shards = list(work_graph_module._module_shards(modules, policy=ScalePolicy())
    assert [stage for stage, _ in shards] == ["entity"]
    assert [len(members) for _, members in shards] == [5]
'''
    text = replace_once(text, old_entity, new_entity, "entity shard test")
    LANE_TEST.write_text(text, encoding="utf-8")


def patch_llama_test() -> None:
    text = LLAMA_TEST.read_text(encoding="utf-8")
    text = text.replace('"name": "inspect_project",', '"name": "search_code_rag",', 1)
    LLAMA_TEST.write_text(text, encoding="utf-8")


def patch_bootstrap_test() -> None:
    text = BOOT_TEST.read_text(encoding="utf-8")
    text = text.replace('            "install_execution_efficiency(",\n', "", 1)
    anchor = '    assert "agent_tool_calling_contract" not in source\n'
    insertion = anchor + '    assert "execution_efficiency_contract" not in source\n'
    if 'assert "execution_efficiency_contract" not in source' not in text:
        text = replace_once(text, anchor, insertion, "bootstrap cleanup assertion")
    BOOT_TEST.write_text(text, encoding="utf-8")


def main() -> None:
    patch_work_graph()
    patch_scheduler()
    patch_lane_tests()
    patch_llama_test()
    patch_bootstrap_test()
    for path in (WORK, SCHED, LANE_TEST, LLAMA_TEST, BOOT_TEST):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


if __name__ == "__main__":
    main()
