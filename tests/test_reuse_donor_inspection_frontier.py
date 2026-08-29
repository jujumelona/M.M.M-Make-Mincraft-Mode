from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import reuse_planner
from minecraft_mod_ai.source_transplant import DonorFile, DonorSlice


def _donor(repository: str, capability: str) -> DonorSlice:
    return DonorSlice(
        capability=capability,
        repository=repository,
        commit_sha="a" * 40,
        license_id="MIT",
        source_url=f"https://github.com/{repository}",
        target_compatibility="adapt",
        files=(
            DonorFile(
                path="src/main/java/example/Feature.java",
                blob_sha="b" * 40,
                sha256="sha256:" + "c" * 64,
                size_bytes=128,
                symbols=("example.Feature",),
            ),
        ),
        seed_files=("src/main/java/example/Feature.java",),
        source_symbols=("example.Feature",),
        required_dependencies=(),
        donor_tests=(),
        confidence=0.8,
        adaptation_cost=1.0,
        closure_complete=True,
    )


def test_repository_frontier_is_inspected_by_source_slice_gate(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def inspect_repository_slice(*, repository, capability, adapter, discovery_client):
        assert adapter.loader == "fabric"
        assert discovery_client is client
        calls.append((repository, capability))
        return _donor(repository, capability) if repository == "example/good" else None

    monkeypatch.setattr(reuse_planner, "inspect_repository_slice", inspect_repository_slice)
    client = object()
    adapter = SimpleNamespace(loader="fabric")

    donors = reuse_planner._discover_donor_candidates(
        "space.travel",
        adapter,
        client,
        ("example/rejected", "example/good"),
    )

    assert set(calls) == {
        ("example/rejected", "space.travel"),
        ("example/good", "space.travel"),
    }
    assert tuple(item.repository for item in donors) == ("example/good",)
    assert donors[0].license_id == "MIT"
    assert donors[0].closure_complete is True
