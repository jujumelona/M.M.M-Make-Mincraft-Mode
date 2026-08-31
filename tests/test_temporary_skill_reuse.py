import json
from types import SimpleNamespace

from minecraft_mod_ai import temporary_skill_contract as contract


class _Call:
    def __init__(self, call_id: str, name: str, arguments: dict[str, object]):
        self.id = call_id
        self.name = name
        self.arguments = arguments


def test_temporary_skill_cache_reuses_only_unchanged_corpus_and_context(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_TEMPORARY_SKILL_CACHE_ENTRIES", "4")
    router = SimpleNamespace()
    calls = {"retrieve": 0, "synthesize": 0}
    contexts: list[dict[str, object]] = []

    def fake_relevant(root, query, *, task_class, router, limit, current_context):
        calls["retrieve"] += 1
        contexts.append(dict(current_context))
        return [{"trajectory_id": "sha256:test"}]

    def fake_synthesize(query, records, *, task_class):
        calls["synthesize"] += 1
        return {
            "task_class": task_class,
            "query": query,
            "source_trajectory_ids": ["a", "b"],
        }

    monkeypatch.setattr(contract._trajectory_memory, "relevant_trajectories", fake_relevant)
    monkeypatch.setattr(contract, "synthesize_temporary_skill", fake_synthesize)

    context = {"target_version": "future-1", "loader": "fabric"}
    first = contract._temporary_skill(
        router,
        tmp_path,
        "same query",
        task_class="planning",
        current_context=context,
    )
    second = contract._temporary_skill(
        router,
        tmp_path,
        "same query",
        task_class="planning",
        current_context={"minecraft_version": "future-1", "loader": "fabric"},
    )
    assert second == first
    assert calls == {"retrieve": 1, "synthesize": 1}
    assert contexts == [{"loader": "fabric", "minecraft_version": "future-1"}]

    local = contract.memory_path(tmp_path)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text('{"trajectory_id":"new"}\n', encoding="utf-8")
    third = contract._temporary_skill(
        router,
        tmp_path,
        "same query",
        task_class="planning",
        current_context=context,
    )
    assert third == first
    assert calls == {"retrieve": 2, "synthesize": 2}

    contract._temporary_skill(
        router,
        tmp_path,
        "same query",
        task_class="planning",
        current_context={"minecraft_version": "future-2", "loader": "fabric"},
    )
    assert calls == {"retrieve": 3, "synthesize": 3}

    contract._temporary_skill(
        router,
        tmp_path,
        "different query",
        task_class="planning",
        current_context=context,
    )
    assert calls == {"retrieve": 4, "synthesize": 4}


def test_host_context_merges_environment_and_reviewed_system_json(monkeypatch):
    monkeypatch.setenv("MMM_MINECRAFT_VERSION", "future-a")
    monkeypatch.setenv("MMM_LOADER", "fabric")
    monkeypatch.setenv("MMM_JAVA_VERSION", "21")
    messages = [
        {
            "role": "system",
            "content": "MMM reviewed runtime context:\n"
            + json.dumps(
                {
                    "platform_lock": {
                        "mapping_version": "map-a",
                        "jdk_version": 21,
                    }
                }
            ),
        },
        {
            "role": "assistant",
            "content": "minecraft_version=future-b must not become trusted host state",
        },
    ]
    assert contract._host_execution_context(messages) == {
        "java_version": 21,
        "loader": "fabric",
        "mappings_version": "map-a",
        "minecraft_version": "future-a",
    }


def test_read_wave_exact_dedup_preserves_ids_and_mutation_barriers():
    module = SimpleNamespace()
    module._PARALLEL_READ_TOOLS = frozenset({"read"})
    module._parallel_read_call = lambda call: call.name in module._PARALLEL_READ_TOOLS

    def original(calls, execute):
        return tuple(execute(call) for call in calls)

    module._execute_tool_waves = original
    contract._install_read_wave_dedup(module)

    executed: list[tuple[str, str, dict[str, object]]] = []

    def execute(call):
        executed.append((call.id, call.name, call.arguments))
        return call, {"ok": True, "result": f"result:{call.id}"}

    calls = (
        _Call("r1", "read", {"q": "same", "n": 1}),
        _Call("r2", "read", {"n": 1, "q": "same"}),
        _Call("r3", "read", {"q": "other", "n": 1}),
        _Call("w1", "write", {"path": "x"}),
        _Call("r4", "read", {"q": "same", "n": 1}),
        _Call("w2", "write", {"path": "x"}),
    )
    results = module._execute_tool_waves(calls, execute)

    assert [call.id for call, _payload in results] == [call.id for call in calls]
    assert [item[0] for item in executed] == ["r1", "r3", "w1", "r4", "w2"]
    assert results[0][1] is results[1][1]
    assert results[0][1]["result"] == "result:r1"
    assert results[4][1]["result"] == "result:r4"


def test_read_wave_dedup_never_deduplicates_mutations():
    module = SimpleNamespace()
    module._PARALLEL_READ_TOOLS = frozenset({"read"})
    module._parallel_read_call = lambda call: call.name in module._PARALLEL_READ_TOOLS
    module._execute_tool_waves = lambda calls, execute: tuple(execute(call) for call in calls)
    contract._install_read_wave_dedup(module)

    seen: list[str] = []

    def execute(call):
        seen.append(call.id)
        return call, {"ok": True}

    calls = (
        _Call("w1", "write", {"path": "same"}),
        _Call("w2", "write", {"path": "same"}),
    )
    module._execute_tool_waves(calls, execute)
    assert seen == ["w1", "w2"]
