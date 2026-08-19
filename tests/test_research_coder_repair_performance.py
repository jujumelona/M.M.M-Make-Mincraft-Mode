from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import custom_generation_search_contract as custom_search
from minecraft_mod_ai import research_code_context
from minecraft_mod_ai import research_coder_repair_performance as performance
from minecraft_mod_ai import research_coder_repair_reuse as reuse
from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator
from minecraft_mod_ai.repair_engine import RepairEngine


def _unit(path: str, package: str, *, imports=(), types=()):
    return SimpleNamespace(
        path=path,
        package=package,
        imports=tuple(imports),
        types=tuple(types),
        methods=(),
    )


def test_runtime_wires_semantic_reuse_and_performance_hardening() -> None:
    assert getattr(
        research_code_context.ResearchCodeContext._query_paths,
        performance._MARKER,
        False,
    )
    assert getattr(
        research_code_context.ResearchCodeContext.evolve_from_generation,
        reuse._MARKER,
        False,
    )
    assert getattr(CustomModuleGenerator.generate, reuse._MARKER, False)
    assert getattr(RepairEngine._context, reuse._MARKER, False)
    assert getattr(RepairEngine._context, "_mmm_narrow_diagnostic_repair_rag", False)
    assert getattr(reuse._dependency_neighborhood_query, performance._MARKER, False)
    assert getattr(reuse._read_log_tail, performance._MARKER, False)
    assert getattr(reuse._persist_research_receipt, performance._MARKER, False)
    assert custom_search._evolution_state_budget is reuse._bounded_evolution_state_budget


def test_dependency_neighborhood_index_is_reused_without_rescanning_units() -> None:
    target = "src/main/java/demo/Target.java"
    helper = "src/main/java/demo/Helper.java"
    caller = "src/main/java/demo/Caller.java"
    context = SimpleNamespace(
        units={
            target: _unit(
                target,
                "demo",
                imports=("demo.Helper",),
                types=("Target",),
            ),
            helper: _unit(helper, "demo", types=("Helper",)),
            caller: _unit(
                caller,
                "demo",
                imports=("demo.Target",),
                types=("Caller",),
            ),
        },
        index=SimpleNamespace(
            files=[SimpleNamespace(path="build.gradle"), SimpleNamespace(path=target)]
        ),
    )

    first = performance._dependency_neighborhood_query(
        research_code_context,
        context,
        "Target dependency API",
        None,
    )
    cached = getattr(context, performance._INDEX_ATTR)
    second = performance._dependency_neighborhood_query(
        research_code_context,
        context,
        "Target dependency API",
        None,
    )

    assert getattr(context, performance._INDEX_ATTR) is cached
    assert first == second
    assert target in first
    assert helper in first
    assert caller in first
    assert "build.gradle" in first


def test_dependency_lane_does_not_increase_canonical_query_path_fanout() -> None:
    class FakeContext:
        def _query_paths(self, query, plan_step):
            del query, plan_step
            return (
                "exact query",
                "repository dependency neighborhood direct reverse shared contracts",
                "capability plan",
                "action symbols",
                "repository API dependency",
                "known repository vocabulary broad fallback",
            )

    module = SimpleNamespace(ResearchCodeContext=FakeContext)
    performance._install_query_path_budget(module)
    paths = FakeContext()._query_paths("query", None)

    assert len(paths) == 5
    assert paths[0] == "exact query"
    assert "repository dependency neighborhood" in paths[1]
    assert all("known repository vocabulary" not in path for path in paths)


def test_build_log_signature_reads_only_a_bounded_tail(tmp_path: Path) -> None:
    log = tmp_path / "gradle.log"
    log.write_text(
        "START_MARKER\n" + ("x" * 200_000) + "\nEND_MARKER\n",
        encoding="utf-8",
    )

    tail = performance._bounded_log_tail({"log_path": str(log)})

    assert "END_MARKER" in tail
    assert "START_MARKER" not in tail
    assert len(tail) <= performance._LOG_TEXT_CHARS


def test_receipt_lock_pool_uses_distinct_project_locks() -> None:
    pool = performance._ProjectLockPool()
    with pool.hold("project-a"):
        first = pool._entries["project-a"][0]
        with pool.hold("project-b"):
            second = pool._entries["project-b"][0]
            assert first is not second
    assert not pool._entries
