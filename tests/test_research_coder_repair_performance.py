from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import custom_generation_search_contract as custom_search
from minecraft_mod_ai import research_code_context
from minecraft_mod_ai import research_code_context_performance as context_performance
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


def test_runtime_wires_single_repository_research_hardener() -> None:
    cls = research_code_context.ResearchCodeContext
    for function in (
        cls._entry_points,
        cls._expand_partial_graph,
        cls.evolve_from_generation,
        research_code_context._retrieval_metrics,
        research_code_context._adaptive_weights,
    ):
        assert getattr(function, context_performance._MARKER, False)

    assert getattr(cls._entry_points, "_mmm_semantic_entry_filter_v1", False)
    assert getattr(cls._expand_partial_graph, "_mmm_two_hop_graph_v1", False)
    assert getattr(cls.evolve_from_generation, "_mmm_generation_fixed_point_v1", False)


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


def test_repository_research_exposes_exactly_eight_weighted_signals() -> None:
    quality = research_code_context.QualityVector(
        correctness=0.8,
        efficiency=0.7,
        security=0.9,
        maintainability=0.8,
        complexity_fit=0.7,
        readability=0.8,
        stepwise_clarity=0.9,
    )
    metrics = research_code_context._retrieval_metrics(
        "plan API Target",
        "class Target { void register() {} }",
        path="src/main/java/demo/Target.java",
        symbols=("Target", "register"),
        graph_hop=0,
        quality=quality,
        target_plan="locate -> register -> validate",
        example_plan="locate -> register -> validate",
    )
    weights = research_code_context._adaptive_weights("plan API Target", metrics)

    assert "plan_alignment" not in metrics
    assert "plan_alignment" not in weights
    assert "quality" in metrics
    assert "quality" in weights
    assert len(weights) == 8
    assert sum(weights.values()) == pytest.approx(1.0)


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

    tail = performance._bounded_log_tail(
        {
            "stderr": "OLD_STDERR\n" + ("y" * 20_000),
            "log_path": str(log),
        }
    )

    assert "END_MARKER" in tail
    assert "START_MARKER" not in tail
    assert "OLD_STDERR" not in tail
    assert len(tail) <= performance._LOG_TEXT_CHARS


def test_receipt_lock_pool_uses_distinct_project_locks() -> None:
    pool = performance._ProjectLockPool()
    with pool.hold("project-a"):
        first = pool._entries["project-a"][0]
        with pool.hold("project-b"):
            second = pool._entries["project-b"][0]
            assert first is not second
    assert not pool._entries


def test_receipt_lock_pool_preserves_exceptions_and_cleans_up() -> None:
    pool = performance._ProjectLockPool()
    with pytest.raises(RuntimeError, match="boom"):
        with pool.hold("project-a"):
            raise RuntimeError("boom")
    assert not pool._entries
