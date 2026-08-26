from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai.project_index import ProjectIndex
from minecraft_mod_ai.repository_explorer import (
    RepositoryExplorer,
    classify_exploration_route,
)


class _Router:
    def __init__(self) -> None:
        self.embed_calls = 0
        self.rerank_calls = 0

    def embed(self, texts):
        self.embed_calls += 1
        return [[float(len(str(text)) % 7 + 1), 1.0] for text in texts]

    def rerank(self, query, documents):
        self.rerank_calls += 1
        return [float("register" in document.lower()) for document in documents]


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "mod"
    java = root / "src/main/java/example"
    java.mkdir(parents=True)
    (java / "Registry.java").write_text(
        """
package example;
public final class Registry {
    public static void registerItems() {
        Items.bootstrap();
        Network.sync();
    }
}
""".strip() + "\n",
        encoding="utf-8",
    )
    (java / "Items.java").write_text(
        """
package example;
public final class Items {
    public static void bootstrap() {
        Helper.validate();
        System.out.println("items");
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
    (java / "Unrelated.java").write_text(
        """
package example;
public final class Unrelated {
    public static void paint() {
        System.out.println("decorative");
    }
}
""".strip() + "\n",
        encoding="utf-8",
    )
    return root


def test_route_selection_is_task_adaptive() -> None:
    assert classify_exploration_route("cannot find symbol Registry") == "trace"
    assert classify_exploration_route("rename registry and find callers") == "ripple"
    assert classify_exploration_route("which API registry callback should I use") == "api"
    assert classify_exploration_route("implement save sync update flow") == "procedural"
    assert classify_exploration_route("exact_identifier") == "lexical"


def test_graph_localization_expands_calls(tmp_path: Path) -> None:
    root = _project(tmp_path)
    result = RepositoryExplorer(ProjectIndex(root)).explore(
        "change registerItems and dependency callers",
        line_budget=80,
        semantic=False,
        rerank=False,
    )
    paths = {region.path for region in result.regions}
    assert "src/main/java/example/Registry.java" in paths
    assert (
        "src/main/java/example/Items.java" in paths
        or "src/main/java/example/Network.java" in paths
    )
    assert result.graph_edges_considered > 0


def test_line_budget_is_hard_and_region_level(tmp_path: Path) -> None:
    root = _project(tmp_path)
    result = RepositoryExplorer(ProjectIndex(root)).explore(
        "register items network packet codec",
        line_budget=5,
        semantic=False,
        rerank=False,
    )
    assert result.lines_selected <= 5
    assert result.regions
    assert all(region.start_line >= 1 for region in result.regions)


def test_procedural_retrieval_favors_matching_flow(tmp_path: Path) -> None:
    root = _project(tmp_path)
    result = RepositoryExplorer(ProjectIndex(root)).explore(
        "implement register sync validate flow",
        line_budget=40,
        semantic=False,
        rerank=False,
    )
    symbols = [region.symbol for region in result.regions[:4]]
    assert "registerItems" in symbols or "bootstrap" in symbols or "sync" in symbols
    assert "paint" not in symbols[:1]


def test_semantic_and_reranker_are_selective(tmp_path: Path) -> None:
    root = _project(tmp_path)
    router = _Router()
    result = RepositoryExplorer(ProjectIndex(root), router=router).explore(
        "which registry API should register packet callback",
        line_budget=40,
    )
    assert result.semantic_used is True
    assert result.rerank_used is True
    assert router.embed_calls == 1
    assert router.rerank_calls == 1

    router2 = _Router()
    result2 = RepositoryExplorer(ProjectIndex(root), router=router2).explore(
        "Unrelated",
        line_budget=20,
    )
    assert result2.semantic_used is False
    assert result2.rerank_used is False
    assert router2.embed_calls == 0
    assert router2.rerank_calls == 0


def test_diagnostic_path_is_promoted_without_semantic_search(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = "src/main/java/example/Network.java"
    result = RepositoryExplorer(ProjectIndex(root)).explore(
        "compile failure",
        diagnostic_paths=[target],
        line_budget=20,
        semantic=False,
        rerank=False,
    )
    assert result.route == "trace"
    assert result.regions[0].path == target
    assert result.regions[0].diagnostic_score > 0
