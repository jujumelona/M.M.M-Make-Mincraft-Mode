from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import capability_implementation_locator as locator
from minecraft_mod_ai import source_transplant as transplant


def _fake_locator_index(count: int):
    paths = [f"src/main/java/demo/TradeHandler{index}.java" for index in range(count)]
    return SimpleNamespace(
        registry_to_path={},
        fqcn_to_path={},
        symbol_to_paths={"TradeHandler": paths},
        method_to_paths={},
        api_call_to_paths={},
        resource_to_path={},
        files_by_path={path: {"path": path} for path in paths},
    )


def test_locator_does_not_drop_structural_seeds_after_eighth():
    found = locator.CapabilityImplementationLocator.locate_seeds(
        "trade",
        _fake_locator_index(17),
    )

    assert len(found) == 17
    assert {item.node_id for item in found} == set(_fake_locator_index(17).files_by_path)


def test_locator_explicit_operator_cap_remains_available():
    found = locator.CapabilityImplementationLocator.locate_seeds(
        "trade",
        _fake_locator_index(17),
        max_seeds=5,
    )
    assert len(found) == 5


def test_donor_tests_are_not_truncated_to_24():
    blobs = {
        f"src/test/java/demo/FeatureTest{index}.java": f"sha-{index}"
        for index in range(41)
    }
    tests = transplant._donor_test_paths(blobs)
    assert len(tests) == 41


class _ClosureGraph:
    def __init__(self, count: int):
        self.nodes = {f"node-{index}": object() for index in range(count)}

    def compute_directional_closures(self, seed_paths):
        assert tuple(seed_paths) == ("node-0",)
        return (tuple(self.nodes),)


def test_dependency_closure_is_not_truncated_to_64_files():
    selected = transplant._closure_paths(_ClosureGraph(93), ("node-0",))
    assert len(selected) == 93
    assert "node-92" in selected


def test_large_nontruncated_tree_is_not_rejected_by_local_file_count(monkeypatch):
    root_tree = "b" * 40
    entries = [
        {"path": f"src/main/java/demo/C{index}.java", "sha": f"blob-{index}", "type": "blob"}
        for index in range(20_137)
    ]

    def fake_json(client, url, *, params=None):
        del client
        if "/git/commits/" in url:
            return {"tree": {"sha": root_tree}}
        if url.endswith(f"/git/trees/{root_tree}") and params == {"recursive": "1"}:
            return {"truncated": False, "tree": entries}
        raise AssertionError((url, params))

    monkeypatch.setattr(transplant, "_github_json", fake_json)
    found = transplant._repository_tree_entries(object(), "owner/repo", "a" * 40)
    assert len(found) == len(entries)


def test_truncated_recursive_tree_falls_back_to_complete_subtree_walk(monkeypatch):
    root_tree = "b" * 40
    tree_src = "c" * 40
    tree_main = "d" * 40
    tree_test = "e" * 40
    calls: list[tuple[str, object]] = []

    def fake_json(client, url, *, params=None):
        del client
        calls.append((url, params))
        if "/git/commits/" in url:
            return {"tree": {"sha": root_tree}}
        if url.endswith(f"/git/trees/{root_tree}") and params == {"recursive": "1"}:
            return {"truncated": True, "tree": [{"path": "partial.java", "type": "blob", "sha": "partial"}]}
        if url.endswith(f"/git/trees/{root_tree}"):
            return {
                "truncated": False,
                "tree": [
                    {"path": "src", "type": "tree", "sha": tree_src},
                    {"path": "README.md", "type": "blob", "sha": "readme"},
                ],
            }
        if url.endswith(f"/git/trees/{tree_src}"):
            return {
                "truncated": False,
                "tree": [
                    {"path": "main", "type": "tree", "sha": tree_main},
                    {"path": "test", "type": "tree", "sha": tree_test},
                ],
            }
        if url.endswith(f"/git/trees/{tree_main}"):
            return {
                "truncated": False,
                "tree": [{"path": "Planet.java", "type": "blob", "sha": "planet"}],
            }
        if url.endswith(f"/git/trees/{tree_test}"):
            return {
                "truncated": False,
                "tree": [{"path": "PlanetTest.java", "type": "blob", "sha": "planet-test"}],
            }
        raise AssertionError((url, params))

    monkeypatch.setattr(transplant, "_github_json", fake_json)
    found = transplant._repository_tree_entries(object(), "owner/repo", "a" * 40)
    paths = {item["path"] for item in found}

    assert "README.md" in paths
    assert "src/main/Planet.java" in paths
    assert "src/test/PlanetTest.java" in paths
    assert "partial.java" not in paths
    assert len(calls) >= 6
