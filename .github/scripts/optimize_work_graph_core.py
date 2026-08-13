from __future__ import annotations

from pathlib import Path

WORK = Path("minecraft_mod_ai/work_graph.py")
SCHED = Path("minecraft_mod_ai/scheduler_parallel_safety_contract.py")
BOOT = Path("minecraft_mod_ai/runtime_bootstrap.py")
EFF = Path("minecraft_mod_ai/execution_efficiency_contract.py")
EFF_TEST = Path("tests/test_execution_efficiency_contract.py")
LANE_TEST = Path("tests/test_generation_lane_parallelism_contract.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise SystemExit(f"missing anchor: {label}")
    return text.replace(old, new, 1)


def replace_function(text: str, name: str, replacement: str, *, indent: str = "") -> str:
    marker = f"{indent}def {name}("
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"missing function: {name}")
    search_from = start + len(marker)
    next_marker = f"\n{indent}def "
    next_pos = text.find(next_marker, search_from)
    if next_pos < 0:
        next_pos = text.find(f"\n{indent}class ", search_from)
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
    text = replace_once(
        text,
        "import hashlib\nimport json\nimport sqlite3\nimport time\nfrom dataclasses import asdict, dataclass\n",
        "import hashlib\nimport json\nimport math\nimport os\nimport sqlite3\nimport time\nfrom collections import Counter\nfrom dataclasses import asdict, dataclass\n",
        "work graph imports",
    )

    node = '''def _content_shard_is_cpu_safe(payload: dict[str, Any]) -> bool:
    members = payload.get("members")
    if not isinstance(members, list):
        return False
    for member in members:
        if not isinstance(member, dict):
            return False
        if str(member.get("kind", "")) != "integration":
            continue
        config = member.get("config")
        if not isinstance(config, dict):
            return False
        if str(config.get("integration_type", "")) != "mmm_local_ai_sidecar":
            return False
    return True


def _node(
    node_id: str,
    stage: str,
    dependencies: Iterable[str],
    payload: dict[str, Any],
) -> WorkNode:
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
    elif kind == "module-shard" and gen_stage in {"system", "entity"}:
        res_class = "cpu_io"
    elif kind == "module-shard" and gen_stage == "content":
        res_class = "cpu_io" if _content_shard_is_cpu_safe(payload) else "commit"
    elif kind == "module-shard" and gen_stage == "audio-binding":
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

    shards = '''def _active_llm_slots() -> int:
    raw = os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "1").strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 1


def _entity_pipeline_shard_size(policy: ScalePolicy) -> int:
    upper = max(1, int(policy.entity_shard_size))
    raw = os.environ.get("MMM_ENTITY_PIPELINE_SHARD_SIZE", "").strip()
    try:
        requested = int(raw) if raw else 2
    except ValueError:
        requested = 2
    return max(1, min(upper, requested))


def _module_shards(
    modules: Sequence[ProductionModule],
    *,
    policy: ScalePolicy,
) -> Iterator[tuple[str, tuple[ProductionModule, ...]]]:
    """Emit dependency-ready bounded shards without bootstrap monkey patches.

    Custom shards expose the configured native LLM slots for small waves while
    retaining the Java shard ceiling for large waves. Entity shards stay small so
    deterministic generation can overlap downstream review without exploding every
    module into its own durable DAG row.
    """

    staged = [(item, _module_stage(item)) for item in modules]
    stage_counts = Counter(stage for _, stage in staged)
    groups: list[dict[str, Any]] = []
    module_group: dict[str, int] = {}
    open_by_key: dict[tuple[str, frozenset[int]], int] = {}

    def shard_size_for(stage: str) -> int:
        if stage == "entity":
            return _entity_pipeline_shard_size(policy)
        if stage == "custom":
            count = max(1, int(stage_counts[stage]))
            slots = min(_active_llm_slots(), count)
            return min(
                max(1, int(policy.java_shard_size)),
                max(1, math.ceil(count / slots)),
            )
        return max(1, int(policy.java_shard_size))

    for item, stage in staged:
        missing = [
            dependency
            for dependency in item.depends_on
            if dependency not in module_group
        ]
        if missing:
            raise WorkGraphError(
                "Module sharding requires topological order; unresolved dependencies for "
                f"{item.module_id}: {missing[:4]}"
            )

        shard_size = shard_size_for(stage)
        dependency_groups = {
            module_group[dependency] for dependency in item.depends_on
        }
        candidates: set[int] = set()

        exact_key = (stage, frozenset(dependency_groups))
        exact = open_by_key.get(exact_key)
        if exact is not None and len(groups[exact]["members"]) < shard_size:
            candidates.add(exact)

        for index in dependency_groups:
            group = groups[index]
            if group["stage"] != stage or len(group["members"]) >= shard_size:
                continue
            if (dependency_groups - {index}).issubset(group["external_groups"]):
                candidates.add(index)

        chosen = max(candidates) if candidates else None
        if chosen is None:
            chosen = len(groups)
            external_groups = set(dependency_groups)
            groups.append(
                {
                    "stage": stage,
                    "members": [],
                    "external_groups": external_groups,
                    "first_order": len(module_group),
                }
            )
            open_by_key[(stage, frozenset(external_groups))] = chosen

        group = groups[chosen]
        group["members"].append(item)
        module_group[item.module_id] = chosen

        group_key = (str(group["stage"]), frozenset(group["external_groups"]))
        if len(group["members"]) >= shard_size:
            if open_by_key.get(group_key) == chosen:
                open_by_key.pop(group_key, None)
        else:
            previous = open_by_key.get(group_key)
            if previous is None or chosen > previous:
                open_by_key[group_key] = chosen

    dependents: dict[int, list[int]] = {index: [] for index in range(len(groups))}
    indegree = [0] * len(groups)
    for index, group in enumerate(groups):
        dependencies = sorted(set(group["external_groups"]))
        indegree[index] = len(dependencies)
        for dependency in dependencies:
            dependents[dependency].append(index)

    ready = sorted(
        (index for index, degree in enumerate(indegree) if degree == 0),
        key=lambda index: int(groups[index]["first_order"]),
    )
    emitted = 0
    while ready:
        next_ready: set[int] = set()
        for index in ready:
            group = groups[index]
            yield str(group["stage"]), tuple(group["members"])
            emitted += 1
            for dependent in dependents[index]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    next_ready.add(dependent)
        ready = sorted(
            next_ready,
            key=lambda index: int(groups[index]["first_order"]),
        )

    if emitted != len(groups):
        raise WorkGraphError("Module shard dependency graph contains a cycle.")
'''
    text = replace_function(text, "_module_shards", shards)
    compile(text, str(WORK), "exec")
    WORK.write_text(text, encoding="utf-8")


def patch_scheduler_contract() -> None:
    text = SCHED.read_text(encoding="utf-8")
    for name in (
        "_pipeline_shard_size",
        "_content_node_is_cpu_safe",
        "_install_pipeline_shards",
        "_install_generation_lanes",
    ):
        text = remove_function(text, name)
    text = text.replace("    _install_pipeline_shards(work_graph_module)\n", "")
    text = text.replace("    _install_generation_lanes(work_graph_module)\n", "")
    text = text.replace('    "_content_node_is_cpu_safe",\n', "")
    compile(text, str(SCHED), "exec")
    SCHED.write_text(text, encoding="utf-8")


def patch_bootstrap() -> None:
    text = BOOT.read_text(encoding="utf-8")
    text = text.replace(
        "    from .execution_efficiency_contract import install as install_execution_efficiency\n",
        "",
        1,
    )
    text = text.replace(
        "    install_execution_efficiency(work_graph_module=work_graph)\n",
        "",
        1,
    )
    if "install_execution_efficiency" in text:
        raise SystemExit("execution efficiency monkey patch still referenced")
    compile(text, str(BOOT), "exec")
    BOOT.write_text(text, encoding="utf-8")


def patch_tests() -> None:
    text = EFF_TEST.read_text(encoding="utf-8")
    text = text.replace(
        "from minecraft_mod_ai import complete_planner, execution_efficiency_contract, work_graph\n",
        "from minecraft_mod_ai import complete_planner, work_graph\n",
        1,
    )
    text = text.replace(
        "source = inspect.getsource(execution_efficiency_contract._dependency_wave_shards)",
        "source = inspect.getsource(work_graph._module_shards)",
        1,
    )
    EFF_TEST.write_text(text, encoding="utf-8")

    text = LANE_TEST.read_text(encoding="utf-8")
    old = '''def test_custom_modules_are_released_one_per_dag_node(monkeypatch) -> None:\n    monkeypatch.delenv("MMM_CUSTOM_PIPELINE_SHARD_SIZE", raising=False)\n    modules = tuple(\n        ProductionModule(\n            module_id=f"custom_{index}",\n            kind="custom_java",\n            config={"summary": f"custom {index}"},\n        )\n        for index in range(4)\n    )\n    shards = list(work_graph_module._module_shards(modules, policy=ScalePolicy()))\n    assert [stage for stage, _ in shards] == ["custom"] * 4\n    assert [len(members) for _, members in shards] == [1, 1, 1, 1]\n'''
    new = '''def test_custom_modules_use_bounded_llm_shards_without_dag_row_explosion(monkeypatch) -> None:\n    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")\n    modules = tuple(\n        ProductionModule(\n            module_id=f"custom_{index}",\n            kind="custom_java",\n            config={"summary": f"custom {index}"},\n        )\n        for index in range(4)\n    )\n    shards = list(work_graph_module._module_shards(modules, policy=ScalePolicy()))\n    assert [stage for stage, _ in shards] == ["custom"]\n    assert [len(members) for _, members in shards] == [4]\n'''
    text = replace_once(text, old, new, "custom shard contract test")
    LANE_TEST.write_text(text, encoding="utf-8")


def main() -> None:
    patch_work_graph()
    patch_scheduler_contract()
    patch_bootstrap()
    patch_tests()
    if EFF.exists():
        EFF.unlink()
    for path in (WORK, SCHED, BOOT, EFF_TEST, LANE_TEST):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


if __name__ == "__main__":
    main()
