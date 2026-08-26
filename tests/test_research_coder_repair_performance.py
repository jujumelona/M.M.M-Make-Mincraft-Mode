from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import custom_generation_research as custom_research
from minecraft_mod_ai import research_code_context
from minecraft_mod_ai import research_code_context_performance as context_performance
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


def _dependency_context() -> SimpleNamespace:
    target = "src/main/java/demo/Target.java"
    helper = "src/main/java/demo/Helper.java"
    caller = "src/main/java/demo/Caller.java"
    contract = "src/main/java/demo/api/TargetContract.java"
    return SimpleNamespace(
        byte_budget=16 * 1024,
        knowledge_terms=Counter({"vocabulary": 1}),
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
            contract: _unit(contract, "demo.api", types=("TargetContract",)),
        },
        index=SimpleNamespace(
            files=[SimpleNamespace(path="build.gradle"), SimpleNamespace(path=target)]
        ),
    )


def test_runtime_keeps_scoring_single_owned_and_only_hardens_fixed_point() -> None:
    cls = research_code_context.ResearchCodeContext
    assert getattr(cls.evolve_from_generation, context_performance._MARKER, False)
    assert not getattr(research_code_context._retrieval_metrics, context_performance._MARKER, False)
    assert not getattr(research_code_context._adaptive_weights, context_performance._MARKER, False)
    assert not getattr(cls._entry_points, context_performance._MARKER, False)
    assert getattr(cls.evolve_from_generation, "_mmm_generation_fixed_point_v1", False)
    assert not getattr(cls._expand_partial_graph, context_performance._MARKER, False)


def test_runtime_wires_single_coder_repair_reuse_owner_without_round_override() -> None:
    cls = research_code_context.ResearchCodeContext
    assert getattr(cls._query_paths, reuse._MARKER, False)
    assert not getattr(cls.evolve_from_generation, reuse._MARKER, False)
    assert getattr(CustomModuleGenerator.generate, reuse._MARKER, False)
    assert getattr(RepairEngine._context, reuse._MARKER, False)
    assert getattr(RepairEngine._context, "_mmm_narrow_diagnostic_repair_rag", False)
    assert custom_research._evolution_state_budget.__module__.endswith(
        "custom_generation_research"
    )


def test_native_evolution_state_budget_is_not_forced_to_two(monkeypatch) -> None:
    monkeypatch.setenv("MMM_CODE_RESEARCH_EVOLUTION_STATES", "8")
    assert custom_research._evolution_state_budget() == 8


def test_repository_research_exposes_eight_weighted_signals_plus_quality_gate() -> None:
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

    assert "plan_alignment" in metrics
    assert "plan_alignment" in weights
    assert "quality" in metrics
    assert "quality" not in weights
    assert len(weights) == 8
    assert sum(weights.values()) == pytest.approx(1.0)


def test_dependency_neighborhood_index_is_reused_without_rescanning_units() -> None:
    context = _dependency_context()

    first = reuse._dependency_neighborhood_query(
        research_code_context,
        context,
        "Target dependency API",
        None,
    )
    cached = getattr(context, reuse._DEPENDENCY_INDEX_ATTR)
    second = reuse._dependency_neighborhood_query(
        research_code_context,
        context,
        "Target dependency API",
        None,
    )

    assert getattr(context, reuse._DEPENDENCY_INDEX_ATTR) is cached
    assert first == second
    assert "src/main/java/demo/Target.java" in first
    assert "src/main/java/demo/Helper.java" in first
    assert "src/main/java/demo/Caller.java" in first
    assert "src/main/java/demo/api/TargetContract.java" in first
    assert "build.gradle" in first
    assert "contract_token_to_paths" in cached


def test_dependency_lane_replaces_broad_fallback_without_a_fixed_path_cap() -> None:
    context = _dependency_context()
    plan_step = SimpleNamespace(
        capability="Target API",
        algorithmic_plan="locate -> bind -> validate",
        action="bind_dependency",
        required_symbols=("Target",),
    )
    wrapped = research_code_context.ResearchCodeContext._query_paths
    baseline = wrapped.__wrapped__(context, "Target dependency API", plan_step)

    paths = wrapped(context, "Target dependency API", plan_step)

    assert len(paths) == len(baseline)
    assert paths[0] == "Target dependency API"
    assert "repository dependency neighborhood" in paths[1]
    assert all("known repository vocabulary" not in path for path in paths)


def test_dependency_query_uses_byte_budget_not_fixed_path_count() -> None:
    context = _dependency_context()
    context.byte_budget = 64 * 1024
    for index in range(32):
        path = f"src/main/java/demo/api/TargetContract{index}.java"
        context.units[path] = _unit(path, "demo.api", types=(f"TargetContract{index}",))

    query = reuse._dependency_neighborhood_query(
        research_code_context,
        context,
        "Target dependency API contract",
        None,
    )

    assert "TargetContract31.java" in query


def test_build_log_signature_reads_only_a_bounded_tail(tmp_path: Path) -> None:
    log = tmp_path / "gradle.log"
    log.write_text(
        "START_MARKER\n" + ("x" * 200_000) + "\nEND_MARKER\n",
        encoding="utf-8",
    )

    tail = reuse._read_log_tail(
        {
            "stderr": "OLD_STDERR\n" + ("y" * 20_000),
            "log_path": str(log),
        }
    )

    assert "END_MARKER" in tail
    assert "START_MARKER" not in tail
    assert "OLD_STDERR" not in tail
    assert len(tail) <= reuse._LOG_TEXT_CHARS


def test_receipt_lock_pool_uses_distinct_project_locks() -> None:
    pool = reuse._ProjectLockPool()
    with pool.hold("project-a"):
        first = pool._entries["project-a"][0]
        with pool.hold("project-b"):
            second = pool._entries["project-b"][0]
            assert first is not second
    assert not pool._entries


def test_receipt_lock_pool_preserves_exceptions_and_cleans_up() -> None:
    pool = reuse._ProjectLockPool()
    with pytest.raises(RuntimeError, match="boom"), pool.hold("project-a"):
        raise RuntimeError("boom")
    assert not pool._entries
