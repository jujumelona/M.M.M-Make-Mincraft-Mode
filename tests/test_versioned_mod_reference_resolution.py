from __future__ import annotations

import httpx

from minecraft_mod_ai import custom_generation_research
from minecraft_mod_ai.versioned_mod_reference_catalog import (
    ReferenceFamily,
    baseline_families,
    exact_version_in_text,
    task_families,
)
from minecraft_mod_ai.versioned_mod_reference_resolver import ExactReferenceResolver
from minecraft_mod_ai.versioned_reference_context_installation import install
from minecraft_mod_ai.versioned_research_context import VersionedResearchCodeContext


def _resolver() -> ExactReferenceResolver:
    return ExactReferenceResolver(
        minecraft_version="1.20.1",
        loader="fabric",
        mappings="1.20.1+build.10",
        java_version=17,
        fabric_loader="0.15.11",
        fabric_api="0.92.2+1.20.1",
    )


def _family() -> ReferenceFamily:
    return ReferenceFamily(
        repository="example/reference-mod",
        loaders=frozenset({"fabric"}),
        baseline_priority=50,
        capabilities=("automation", "machine", "자동화", "기계"),
    )


def _not_found(url: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", url)
    response = httpx.Response(404, request=request)
    return httpx.HTTPStatusError("not found", request=request, response=response)


def test_version_boundary_accepts_mc_branch_and_rejects_neighbor() -> None:
    assert exact_version_in_text("mc1.20.1/fabric/dev", "1.20.1")
    assert exact_version_in_text("release-1.20.1", "1.20.1")
    assert not exact_version_in_text("release-1.20.10", "1.20.1")
    assert not exact_version_in_text("release-1.20.2", "1.20.1")


def test_catalog_keeps_baseline_and_task_donor_selection_separate() -> None:
    baseline = baseline_families("fabric", limit=2)
    assert len(baseline) == 2
    assert all(family.loaders == frozenset({"fabric"}) for family in baseline)

    excluded = frozenset(family.repository for family in baseline)
    donors = task_families(
        "fabric",
        "산업 자동화 기계 에너지 레시피 처리",
        exclude=excluded,
        limit=3,
    )
    assert donors
    assert all(family.repository not in excluded for family in donors)
    assert any("automation" in family.capabilities for family in donors)


def test_ref_listing_is_bounded_paginated_deduplicated_and_cached() -> None:
    resolver = _resolver()
    calls: list[str] = []

    branch_page_1 = [
        {"name": "main" if index == 0 else f"branch-{index}", "commit": {"sha": f"b{index}"}}
        for index in range(100)
    ]
    branch_page_2 = [
        {"name": "mc1.20.1/fabric/dev", "commit": {"sha": "exact-branch"}}
    ]
    tag_page_1 = [
        {"name": "1.20.1-release", "commit": {"sha": "exact-tag"}},
        {"name": "main", "commit": {"sha": "duplicate-name"}},
    ]

    def fake_json(url: str):
        calls.append(url)
        if "/branches?per_page=100&page=1" in url:
            return branch_page_1
        if "/branches?per_page=100&page=2" in url:
            return branch_page_2
        if "/tags?per_page=100&page=1" in url:
            return tag_page_1
        raise AssertionError(f"unexpected URL: {url}")

    resolver._json = fake_json  # type: ignore[method-assign]
    try:
        first = resolver._refs("example/reference-mod", "main")
        call_count = len(calls)
        second = resolver._refs("example/reference-mod", "main")
    finally:
        resolver.close()

    assert ("mc1.20.1/fabric/dev", "exact-branch") in first
    assert ("1.20.1-release", "exact-tag") in first
    assert [item for item in first if item[0] == "main"] == [("main", "b0")]
    assert second == first
    assert len(calls) == call_count


def test_annotated_tag_object_is_dereferenced_to_immutable_commit() -> None:
    resolver = _resolver()
    calls: list[str] = []

    def fake_json(url: str):
        calls.append(url)
        if url.endswith("/git/commits/annotated-tag-object"):
            raise _not_found(url)
        if url.endswith("/git/tags/annotated-tag-object"):
            return {"object": {"type": "commit", "sha": "immutable-commit"}}
        if url.endswith("/git/commits/immutable-commit"):
            return {"sha": "immutable-commit", "tree": {"sha": "tree-sha"}}
        raise AssertionError(f"unexpected URL: {url}")

    resolver._json = fake_json  # type: ignore[method-assign]
    try:
        commit = resolver._commit_object("example/reference-mod", "annotated-tag-object")
        cached = resolver._commit_object("example/reference-mod", "annotated-tag-object")
    finally:
        resolver.close()

    assert commit is not None
    assert commit["sha"] == "immutable-commit"
    assert cached == commit
    assert calls.count(
        "https://api.github.com/repos/example/reference-mod/git/commits/annotated-tag-object"
    ) == 1


def test_license_is_resolved_at_selected_commit_not_default_branch() -> None:
    resolver = _resolver()
    calls: list[str] = []

    def fake_json(url: str):
        calls.append(url)
        assert url.endswith("/license?ref=immutable-commit")
        return {"license": {"spdx_id": "MIT"}}

    resolver._json = fake_json  # type: ignore[method-assign]
    try:
        first = resolver._license_at_commit("example/reference-mod", "immutable-commit")
        second = resolver._license_at_commit("example/reference-mod", "immutable-commit")
    finally:
        resolver.close()

    assert first == second == "MIT"
    assert len(calls) == 1


def test_metadata_version_match_is_hard_filter_even_when_ref_name_matches() -> None:
    resolver = _resolver()
    metadata = """
### gradle.properties
minecraft_version=1.20.10
loader_version=0.15.11
yarn_mappings=1.20.10+build.1
java_version=17
### fabric.mod.json
{"depends":{"fabricloader":">=0.15.0"}}
"""
    try:
        proof = resolver._compatibility(metadata, _family())
    finally:
        resolver.close()

    assert proof.minecraft_exact is False
    assert proof.admitted is False


def test_hot_path_installer_points_coder_engine_at_versioned_context() -> None:
    install()
    assert custom_generation_research.ResearchCodeContext is VersionedResearchCodeContext
