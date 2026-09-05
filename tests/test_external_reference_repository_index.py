from __future__ import annotations

from types import SimpleNamespace

import pytest

import minecraft_mod_ai.source_transplant as source_transplant
from minecraft_mod_ai.repository_artifact_index import RepositoryArtifactIndex


def test_kotlin_repository_index_reads_package_object_and_fun_symbols() -> None:
    source = b"""package demo.reference

object TradeEngine {
    fun barter(player: String): Boolean = true
}
"""
    index = RepositoryArtifactIndex.build_from_tree(
        "owner/kotlin-mod",
        "a" * 40,
        [
            {
                "path": "src/main/kotlin/demo/reference/TradeEngine.kt",
                "sha": "b" * 40,
                "type": "blob",
            }
        ],
        blob_fetcher=lambda _repo, _sha: source,
    )

    graph = index.build_dependency_graph()

    path = "src/main/kotlin/demo/reference/TradeEngine.kt"
    assert index.fqcn_to_path["demo.reference.TradeEngine"] == path
    assert path in index.symbol_to_paths["TradeEngine"]
    assert path in index.method_to_paths["barter"]
    assert path in graph.nodes


def test_kotlin_only_repository_can_reach_donor_slice_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = "src/main/kotlin/demo/reference/TradeEngine.kt"
    blob_sha = "b" * 40
    source = b"""package demo.reference

object TradeEngine {
    fun tradeOffer(): Int = 1
}
"""
    snapshot = {
        "commit_sha": "a" * 40,
        "license_id": "MIT",
        "source_url": "https://github.com/owner/kotlin-mod",
        "blobs": {path: blob_sha},
    }

    class _Client:
        def close(self) -> None:
            return None

    monkeypatch.setattr(
        source_transplant,
        "_repository_snapshot",
        lambda _repository, _discovery: snapshot,
    )
    monkeypatch.setattr(source_transplant, "_github_client", lambda _token: _Client())
    monkeypatch.setattr(
        source_transplant,
        "_build_metadata_text",
        lambda *_args, **_kwargs: "minecraft_version=1.21.1\nfabricloader\n",
    )
    monkeypatch.setattr(
        source_transplant,
        "_fetch_blob_bytes",
        lambda _client, _repository, sha: source if sha == blob_sha else b"",
    )

    adapter = SimpleNamespace(loader="fabric", minecraft_version="1.21.1")
    discovery = SimpleNamespace(github_token="")
    donor = source_transplant.inspect_repository_slice(
        repository="owner/kotlin-mod",
        capability="trade offer",
        adapter=adapter,
        discovery_client=discovery,
    )

    assert donor is not None
    assert donor.repository == "owner/kotlin-mod"
    assert donor.target_compatibility == "exact"
    assert path in {item.path for item in donor.files}
    assert "TradeEngine" in donor.source_symbols
    assert "tradeOffer" in donor.source_symbols


def test_neoforge_metadata_filename_is_loader_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_path = "src/main/resources/META-INF/neoforge.mods.toml"
    blobs = {metadata_path: "c" * 40}
    metadata = b'modLoader="javafml"\nloaderVersion="[4,)"\nminecraft_version=1.21.1\n'

    monkeypatch.setattr(
        source_transplant,
        "_fetch_blob_bytes",
        lambda _client, _repository, _sha: metadata,
    )
    text = source_transplant._build_metadata_text(
        object(),
        repository="owner/neoforge-mod",
        blobs=blobs,
    )
    evidence = source_transplant._target_compatibility_evidence(
        text,
        adapter=SimpleNamespace(loader="neoforge", minecraft_version="1.21.1"),
    )

    assert "MMM_METADATA_PATH src/main/resources/META-INF/neoforge.mods.toml" in text
    assert evidence.loader == "neoforge"
    assert evidence.minecraft_version == "1.21.1"
    assert evidence.status == "metadata_exact"
