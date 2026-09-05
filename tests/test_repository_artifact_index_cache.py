from __future__ import annotations

from minecraft_mod_ai.repository_artifact_index import (
    RepositoryArtifactIndex,
    clear_repository_artifact_index_cache,
)


def _tree() -> list[dict[str, str]]:
    return [
        {
            "path": "src/main/java/demo/reference/TradeService.java",
            "sha": "b" * 40,
            "type": "blob",
        }
    ]


def test_immutable_repository_index_and_graph_are_reused() -> None:
    clear_repository_artifact_index_cache()
    calls: list[tuple[str, str]] = []
    source = b"package demo.reference; public class TradeService { public void trade() {} }"

    def fetch_blob(repository: str, blob_sha: str) -> bytes:
        calls.append((repository, blob_sha))
        return source

    first = RepositoryArtifactIndex.build_from_tree(
        "owner/reference-mod",
        "a" * 40,
        _tree(),
        blob_fetcher=fetch_blob,
    )
    first_graph = first.build_dependency_graph()

    second = RepositoryArtifactIndex.build_from_tree(
        "owner/reference-mod",
        "a" * 40,
        _tree(),
        blob_fetcher=fetch_blob,
    )
    second_graph = second.build_dependency_graph()

    assert second is first
    assert second_graph is first_graph
    assert calls == [("owner/reference-mod", "b" * 40)]
    assert first.metadata["index_cache_hits"] == 1
    assert first.metadata["graph_cache_hits"] == 1


def test_cache_identity_includes_tree_and_fetcher_provenance() -> None:
    clear_repository_artifact_index_cache()

    def fetch_one(_repository: str, _blob_sha: str) -> bytes:
        return b"package demo; public class One {}"

    def fetch_two(_repository: str, _blob_sha: str) -> bytes:
        return b"package demo; public class Two {}"

    first = RepositoryArtifactIndex.build_from_tree(
        "owner/reference-mod",
        "a" * 40,
        _tree(),
        blob_fetcher=fetch_one,
    )
    different_fetcher = RepositoryArtifactIndex.build_from_tree(
        "owner/reference-mod",
        "a" * 40,
        _tree(),
        blob_fetcher=fetch_two,
    )
    changed_tree = [
        {
            "path": "src/main/java/demo/reference/Other.java",
            "sha": "c" * 40,
            "type": "blob",
        }
    ]
    different_tree = RepositoryArtifactIndex.build_from_tree(
        "owner/reference-mod",
        "a" * 40,
        changed_tree,
        blob_fetcher=fetch_one,
    )

    assert different_fetcher is not first
    assert different_tree is not first


def test_dependency_graph_is_not_reused_for_a_different_target_context() -> None:
    clear_repository_artifact_index_cache()
    calls = 0
    source = b"package demo.reference; public class TradeService {}"

    def fetch_blob(_repository: str, _blob_sha: str) -> bytes:
        nonlocal calls
        calls += 1
        return source

    index = RepositoryArtifactIndex.build_from_tree(
        "owner/reference-mod",
        "a" * 40,
        _tree(),
        blob_fetcher=fetch_blob,
    )
    fabric_graph = index.build_dependency_graph(
        target_context={"loader": "fabric", "minecraft_version": "1.21.1"}
    )
    neoforge_graph = index.build_dependency_graph(
        target_context={"loader": "neoforge", "minecraft_version": "1.21.1"}
    )

    assert neoforge_graph is not fabric_graph
    assert calls == 1
