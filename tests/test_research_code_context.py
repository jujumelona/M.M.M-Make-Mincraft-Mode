from __future__ import annotations

import json
from dataclasses import dataclass, field

from minecraft_mod_ai import research_code_context as research
from minecraft_mod_ai.project_index import ProjectIndex

_TEST_MINECRAFT_VERSION = "mmm-test-target"
_TEST_LOADER = "fabric"
_TEST_MAPPINGS = "mmm-test-target+test-mappings"


@dataclass
class _Module:
    kind: str = "custom_java"
    config: dict = field(default_factory=lambda: {"feature": "tick compute service"})
    depends_on: tuple[str, ...] = ("service",)
    required_gates: tuple[str, ...] = ("GameTest",)


class _Router:
    def rerank(self, query, documents):
        query_tokens = set(str(query).casefold().split())
        scores = []
        for document in documents:
            tokens = set(str(document).casefold().split())
            scores.append(float(len(query_tokens & tokens)))
        return scores


def _repo(tmp_path):
    entry = tmp_path / "src/main/java/example/Entry.java"
    service = tmp_path / "src/main/java/example/Service.java"
    unrelated = tmp_path / "src/main/java/example/Unused.java"
    for path in (entry, service, unrelated):
        path.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(
        """
package example;

public final class Entry {
    public void tick() {
        Service service = new Service();
        service.compute();
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    service.write_text(
        """
package example;

public final class Service {
    public int compute() {
        return normalize(41);
    }

    private int normalize(int value) {
        return value + 1;
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    unrelated.write_text(
        """
package example;

public final class Unused {
    public void unrelated() {
        System.out.println("unused");
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "gradle.properties").write_text(
        "\n".join(
            (
                f"minecraft_version={_TEST_MINECRAFT_VERSION}",
                f"yarn_mappings={_TEST_MAPPINGS}",
                "loader_version=test-loader",
                "fabric_version=test-api",
                "loom_version=test-loom",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return ProjectIndex(tmp_path)


def _context(tmp_path):
    index = _repo(tmp_path)
    return research.ResearchCodeContext(
        tmp_path,
        project_index=index,
        router=_Router(),
        module=_Module(),
        minecraft_version=_TEST_MINECRAFT_VERSION,
        loader=_TEST_LOADER,
        mappings=_TEST_MAPPINGS,
        byte_budget=32 * 1024,
    )


def test_plan_as_query_decomposes_module_and_uses_algorithmic_steps(tmp_path) -> None:
    context = _context(tmp_path)
    assert len(context.plan) >= 4
    assert all("locate existing contract" in step.algorithmic_plan for step in context.plan)
    assert all(step.algorithmic_plan in step.query for step in context.plan)
    assert {step.action for step in context.plan} >= {
        "integrate_module",
        "implement_feature",
        "bind_dependency",
        "satisfy_gate",
    }


def test_dynamic_partial_call_graph_reaches_calls_without_global_unrelated_nodes(tmp_path) -> None:
    context = _context(tmp_path)
    entries = context._entry_points("tick")
    graph = context._expand_partial_graph(entries)
    names = {symbol.name for symbol, _hop in graph}
    assert "tick" in names
    assert "compute" in names
    assert "normalize" in names
    assert "unrelated" not in names
    assert max(hop for _symbol, hop in graph) <= 2


def test_adaptive_retrieval_computes_complementary_metrics(tmp_path) -> None:
    context = _context(tmp_path)
    compute = context.symbols_by_name["compute"][0]
    evidence = context._symbol_evidence(
        compute,
        query="Service compute dependency API validate",
        graph_hop=1,
    )
    assert evidence is not None
    assert set(evidence.metrics) == {
        "lexical",
        "semantic",
        "path",
        "symbol",
        "dependency",
        "structure",
        "call_graph",
        "plan_alignment",
        "quality",
    }
    weights = research._adaptive_weights("Service.compute dependency API")
    assert set(weights) == set(evidence.metrics) - {"quality"}
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["dependency"] > weights["lexical"]


def test_quality_aware_retrieval_penalizes_unsafe_unreadable_examples() -> None:
    clean = """
public final class Counter {
    public int next(int value) {
        int updated = value + 1;
        return updated;
    }
}
""".strip()
    unsafe = """
public class X {
    public void x() throws Exception {
        Runtime.getRuntime().exec("curl bad");
        TODO();
    }
}
""".strip()
    clean_quality = research._quality(clean, path="src/main/java/example/Counter.java")
    unsafe_quality = research._quality(unsafe, path="src/main/java/example/X.java")
    assert clean_quality.security > unsafe_quality.security
    assert clean_quality.correctness > unsafe_quality.correctness
    assert clean_quality.coqu_ir > unsafe_quality.coqu_ir
    assert clean_quality.example_quality > unsafe_quality.example_quality


def test_dependency_monitor_rejects_unknown_packages_coordinates_repositories_and_target_drift(tmp_path) -> None:
    _repo(tmp_path)
    monitor = research.DependencyMonitor(
        tmp_path,
        minecraft_version=_TEST_MINECRAFT_VERSION,
        loader=_TEST_LOADER,
        mappings=_TEST_MAPPINGS,
    )
    bad = json.dumps(
        {
            "operations": [
                {
                    "operation": "replace",
                    "path": "build.gradle",
                    "content": (
                        'repositories { maven { url "https://evil.example/repo" } }\n'
                        'dependencies {\n'
                        ' implementation "evil.fake:nonexistent:1.0"\n'
                        ' implementation "net.fabricmc.fabric-api:fabric-api:999.0"\n'
                        '}'
                    ),
                },
                {
                    "operation": "replace",
                    "path": "gradle.properties",
                    "content": "minecraft_version=wrong-target\n",
                },
            ]
        }
    )
    violations = monitor.validate_model_output(bad)
    values = {(item.kind, item.value) for item in violations}
    assert ("package", "evil.fake:nonexistent") in values
    assert ("coordinate", "net.fabricmc.fabric-api:fabric-api:999.0") in values
    assert ("repository", "https://evil.example/repo") in values
    assert ("target_property", "minecraft_version=wrong-target") in values

    good = json.dumps(
        {
            "operations": [
                {
                    "operation": "replace",
                    "path": "build.gradle",
                    "content": (
                        'repositories { maven { url "https://maven.fabricmc.net" } }\n'
                        'dependencies { implementation '
                        '"net.fabricmc.fabric-api:fabric-api:test-api" }'
                    ),
                }
            ]
        }
    )
    assert monitor.validate_model_output(good) == ()


def test_generation_query_evolution_reaches_fixed_point(tmp_path) -> None:
    context = _context(tmp_path)
    context.initial_bundle()
    before = len(context.query_history)
    draft = json.dumps(
        {
            "operations": [
                {
                    "operation": "edit",
                    "path": "src/main/java/example/Entry.java",
                    "replacements": [
                        {
                            "old": "service.compute();",
                            "new": "service.compute(); service.normalize(41);",
                        }
                    ],
                }
            ],
            "runtime_tests": ["GameTest"],
            "complete": True,
            "next_cursor": "",
        }
    )
    context.evolve_from_generation(draft)
    after_first = len(context.query_history)
    context.evolve_from_generation(draft)
    assert after_first > before
    assert len(context.query_history) == after_first


def test_bundle_records_all_research_mechanisms_and_is_bounded(tmp_path) -> None:
    context = _context(tmp_path)
    bundle = context.initial_bundle()
    methods = set(bundle["research_methods"])
    assert {
        "capir_compositional_subtask_retrieval",
        "perc_plan_as_query",
        "repocoder_iterative_retrieval_generation",
        "evor_query_and_knowledge_evolution",
        "coret_semantics_structure_dependency",
        "dyretriever_on_demand_partial_dependency_graph",
        "coderag_multi_path_bestfit",
        "aircoder_query_adaptive_metric_fusion",
        "rar_two_step_docs_examples",
        "docprompting_docs_before_generation",
        "coquir_quality_aware_retrieval",
        "example_quality_multi_aspect_selection",
        "packmonitor_authoritative_package_admission",
    } <= methods
    monitor = bundle["dependency_monitor"]
    assert monitor["zero_unknown_packages_in_accepted_patch"] is True
    assert monitor["zero_unknown_literal_coordinates_in_accepted_patch"] is True
    assert monitor["zero_unknown_repositories_in_accepted_patch"] is True
    assert bundle["evidence_count"] <= bundle["total_evidence_count"]
    assert len(
        json.dumps(
            bundle,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ) <= 32 * 1024
