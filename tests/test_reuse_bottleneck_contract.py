from __future__ import annotations

from types import SimpleNamespace

import minecraft_mod_ai.reuse_build_verifier as build_verifier
from minecraft_mod_ai.artifact_dependency_graph import ArtifactDependencyGraph
from minecraft_mod_ai.reuse_build_verifier import BuildToolchainReceipt


def _attested_toolchain() -> BuildToolchainReceipt:
    return BuildToolchainReceipt(
        gradle_version="8.10.2",
        distribution_sha256="d" * 64,
        wrapper_sha256="w" * 64,
        java_version="21",
        loader="fabric",
        minecraft_version="1.21.1",
        wrapper_verified=True,
        distribution_verified=True,
        target_matrix_verified=True,
        toolchain_hash="sha256:" + "a" * 64,
    )


def _reset_build_cache() -> None:
    with build_verifier._BUILD_CACHE_LOCK:
        build_verifier._BUILD_CACHE.clear()
        build_verifier._BUILD_INFLIGHT.clear()


def _install_fake_attested_workspace(monkeypatch, tmp_path):
    gradlew = tmp_path / "gradlew"
    gradlew.write_text("#!/bin/sh\n", encoding="utf-8")
    source = tmp_path / "src" / "main" / "java" / "demo" / "Example.java"
    source.parent.mkdir(parents=True)
    source.write_text("package demo; class Example {}\n", encoding="utf-8")
    monkeypatch.setattr(
        build_verifier,
        "_find_gradle_wrapper",
        lambda _ws: (gradlew, "gradle_wrapper"),
    )
    monkeypatch.setattr(
        build_verifier,
        "_inspect_build_toolchain",
        lambda _ws: _attested_toolchain(),
    )
    return source


def test_identical_compile_proof_runs_gradle_once(monkeypatch, tmp_path) -> None:
    _reset_build_cache()
    _install_fake_attested_workspace(monkeypatch, tmp_path)
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **_kwargs):
        calls.append(tuple(command))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(build_verifier.subprocess, "run", fake_run)

    first = build_verifier.verify_scratch_workspace_build(tmp_path)
    second = build_verifier.verify_scratch_workspace_build(tmp_path)

    assert first.compile_passed is True
    assert second.compile_passed is True
    assert len(calls) == 1
    assert "compileJava" in calls[0]


def test_test_upgrade_reuses_cached_compile_stage(monkeypatch, tmp_path) -> None:
    _reset_build_cache()
    _install_fake_attested_workspace(monkeypatch, tmp_path)
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **_kwargs):
        calls.append(tuple(command))
        if "test" in command:
            return SimpleNamespace(
                returncode=0,
                stdout="1 tests completed, 0 failed",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(build_verifier.subprocess, "run", fake_run)

    compile_only = build_verifier.verify_scratch_workspace_build(tmp_path)
    with_tests = build_verifier.verify_scratch_workspace_build(
        tmp_path,
        run_tests=True,
    )

    assert compile_only.compile_passed is True
    assert with_tests.compile_passed is True
    assert with_tests.tests_passed is True
    assert len(calls) == 2
    assert "compileJava" in calls[0]
    assert "test" in calls[1]


def test_changed_input_invalidates_compile_cache(monkeypatch, tmp_path) -> None:
    _reset_build_cache()
    source = _install_fake_attested_workspace(monkeypatch, tmp_path)
    calls = 0

    def fake_run(_command, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(build_verifier.subprocess, "run", fake_run)

    build_verifier.verify_scratch_workspace_build(tmp_path)
    source.write_text("package demo; class Example { int changed; }\n", encoding="utf-8")
    build_verifier.verify_scratch_workspace_build(tmp_path)

    assert calls == 2


def test_failed_compile_is_not_persistently_cached(monkeypatch, tmp_path) -> None:
    _reset_build_cache()
    _install_fake_attested_workspace(monkeypatch, tmp_path)
    calls = 0

    def fake_run(_command, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=1, stdout="", stderr="compile failed")

    monkeypatch.setattr(build_verifier.subprocess, "run", fake_run)

    first = build_verifier.verify_scratch_workspace_build(tmp_path)
    second = build_verifier.verify_scratch_workspace_build(tmp_path)

    assert first.compile_passed is False
    assert second.compile_passed is False
    assert calls == 2


def test_artifact_graph_symbol_matching_is_identifier_exact() -> None:
    files = {
        "src/main/java/demo/Foo.java": "package demo; class Foo {}",
        "src/main/java/demo/Foobar.java": "package demo; class Foobar {}",
        "src/main/java/demo/Use.java": "package demo; class Use { Foobar value; }",
    }

    graph = ArtifactDependencyGraph.build_from_files(files)
    use_edges = graph.adjacency["src/main/java/demo/Use.java"]

    assert "src/main/java/demo/Foobar.java" in use_edges
    assert "src/main/java/demo/Foo.java" not in use_edges


def test_unseeded_directional_closures_start_only_from_scc_roots() -> None:
    graph = ArtifactDependencyGraph()
    from minecraft_mod_ai.artifact_dependency_graph import ArtifactKind, ArtifactNode

    for node_id in ("A", "B", "C", "D"):
        graph.add_node(ArtifactNode(id=node_id, kind=ArtifactKind.OTHER))
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_edge("D", "C")

    closures = {tuple(sorted(component)) for component in graph.compute_directional_closures()}

    assert closures == {("A", "B", "C"), ("C", "D")}
