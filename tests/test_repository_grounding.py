from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai.project_index import ProjectIndex
from minecraft_mod_ai.repository_grounding import (
    build_repair_repository_context,
    build_repository_observation_ledger,
)


class _Router:
    def embed(self, texts):
        return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]

    def rerank(self, query, documents):
        return [1.0 if "register" in document.lower() else 0.0 for document in documents]


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "mod"
    java = root / "src/main/java/example"
    java.mkdir(parents=True)
    (java / "Main.java").write_text(
        """
package example;
public final class Main {
    public static void registerNetwork() {
        Network.sync();
    }
}
""".strip() + "\n",
        encoding="utf-8",
    )
    (java / "Network.java").write_text(
        """
package example;
public final class Network {
    public static void sync() {
        PacketCodec.register();
    }
}
""".strip() + "\n",
        encoding="utf-8",
    )
    return root


def test_grounding_is_exact_and_region_ranked(tmp_path: Path) -> None:
    root = _root(tmp_path)
    ledger = build_repository_observation_ledger(
        _Router(),
        ProjectIndex(root),
        query="which registry API should register network packet",
        byte_budget=16 * 1024,
    )
    assert ledger["schema_version"] == "mmm/source-observation-ledger-v2"
    assert ledger["records"]
    receipt = ledger["receipt"]
    assert receipt["retrieval_route"] == "api"
    assert receipt["semantic_used"] is False
    assert receipt["rerank_used"] is False
    assert receipt["baseline_anchor_count"] >= 1
    assert receipt["policy"]["line_ranked_context"] is True
    assert receipt["policy"]["global_contract_anchors_before_ranked_regions"] is True
    for record in ledger["records"]:
        assert record["sha256"].startswith("sha256:")
        assert record["content_end_bytes"] >= record["content_start_bytes"]
        assert record["start_line"] >= 1
        assert record["text"]


def test_global_exact_anchor_lane_preserves_distant_contracts(tmp_path: Path) -> None:
    root = tmp_path / "mod"
    java = root / "src/main/java/example"
    java.mkdir(parents=True)
    size = 320

    def write_fixed(name: str, body: str) -> None:
        raw = body.encode("utf-8")
        assert len(raw) < size
        (java / name).write_bytes(raw + b" " * (size - len(raw)))

    write_fixed(
        "A0000.java",
        "package example; // required contract FIRST_PAGE_SOURCE_FACT\n"
        'final class A0000 { static final String HOOK = "crossFileHook"; }\n',
    )
    for index in range(1, 60):
        write_fixed(
            f"A{index:04d}.java",
            f"package example; final class A{index:04d} {{ "
            'static final String HOOK = "crossFileHook"; }\n',
        )
    write_fixed(
        "Z9999.java",
        "package example; // required contract HIGH_INDEX_SOURCE_SENTINEL\n"
        'final class Z9999 { static final String HOOK = "crossFileHook"; }\n',
    )

    ledger = build_repository_observation_ledger(
        _Router(),
        ProjectIndex(root),
        query="crossFileHook",
        byte_budget=4096,
    )
    text = "\n".join(str(item.get("text", "")) for item in ledger["records"])
    assert "FIRST_PAGE_SOURCE_FACT" in text
    assert "HIGH_INDEX_SOURCE_SENTINEL" in text
    assert ledger["receipt"]["source_page_count"] > 1


def test_greenfield_project_has_valid_zero_source_grounding(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    ledger = build_repository_observation_ledger(
        _Router(),
        ProjectIndex(root),
        query="implement a new custom module",
        byte_budget=4096,
    )
    assert ledger["records"] == []
    assert ledger["receipt"]["observation_count"] == 0
    assert ledger["receipt"]["source_page_count"] == 0
    assert ledger["receipt"]["project_sha256"].startswith("sha256:")
    assert ledger["receipt"]["observations_sha256"].startswith("sha256:")
    assert ledger["receipt"]["policy"]["greenfield_zero_source_is_valid"] is True


def test_repair_context_reuses_same_explorer_contract(tmp_path: Path) -> None:
    root = _root(tmp_path)
    context = build_repair_repository_context(
        _Router(),
        ProjectIndex(root),
        query="cannot find symbol PacketCodec",
        diagnostic_paths=["src/main/java/example/Network.java"],
        byte_budget=16 * 1024,
    )
    assert context["schema_version"] == "mmm/repair-repository-context-v1"
    assert context["retrieval_receipt"]["retrieval_route"] == "trace"
    assert context["relevant"]["selected_region_count"] >= 1
    assert any(
        item["path"] == "src/main/java/example/Network.java"
        for item in context["relevant"]["files"]
    )


def test_runtime_contract_installs_generation_and_repair_grounding() -> None:
    from minecraft_mod_ai import custom_module_generator, repair_engine

    assert getattr(
        custom_module_generator._collect_initial_observations,
        "__mmm_repository_grounding_live_context__",
        False,
    )
    assert getattr(
        repair_engine.RepairEngine._context,
        "__mmm_repository_grounding_live_context__",
        False,
    )
