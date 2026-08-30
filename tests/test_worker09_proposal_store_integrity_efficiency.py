from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import minecraft_mod_ai.proposal_store as proposal_store
from minecraft_mod_ai.complete_spec import ProductionModule, complete_proposal_from_parts
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner


def _approved_proposal(module_count: int = 33):
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan(
        "Create exactly one frost item."
    )
    proposal = complete_proposal_from_parts(
        requested_prompt="Create a sharded test plan.",
        base_proposal=base,
        game_design={"title": "Worker 09", "lore": "x" * 100_000},
        modules=tuple(
            ProductionModule(f"module_{index:05d}", "item")
            for index in range(module_count)
        ),
        acceptance_tests=("Every requested module is registered.",),
    )
    return proposal.approve(proposal.calculate_hash())


def test_every_advertised_collection_section_is_reachable(tmp_path: Path) -> None:
    index = proposal_store.write_sharded_complete_proposal(
        _approved_proposal(9),
        tmp_path / "complete-proposal.json",
        shard_size=2,
        part_size_bytes=16 * 1024,
    )

    expected = {"modules": 9, "assets": 0, "acceptance_tests": 1}
    for section, total in expected.items():
        page = proposal_store.read_sharded_complete_proposal_section(
            index, section, limit=2
        )
        assert page["section"] == section
        assert page["total_count"] == total
        assert page["returned"] == min(2, total)


def test_chunked_json_reader_reads_each_chunk_once(tmp_path: Path) -> None:
    index = proposal_store.write_sharded_complete_proposal(
        _approved_proposal(3),
        tmp_path / "complete-proposal.json",
        shard_size=2,
        part_size_bytes=16 * 1024,
    )
    raw = json.loads(index.read_text(encoding="utf-8"))
    descriptor = raw["game_design"]
    reads: Counter[str] = Counter()

    def read_part(relative: str) -> bytes:
        reads[relative] += 1
        return proposal_store._read_file_part(index.parent, relative)

    decoded = proposal_store._read_json_part(descriptor, read_part)

    assert decoded["title"] == "Worker 09"
    assert sum(reads.values()) == descriptor["chunk_count"]
    assert set(reads.values()) == {1}


def test_validated_manifest_cache_avoids_revalidating_every_page(
    tmp_path: Path, monkeypatch
) -> None:
    index = proposal_store.write_sharded_complete_proposal(
        _approved_proposal(300),
        tmp_path / "complete-proposal.json",
        shard_size=1,
        part_size_bytes=16 * 1024,
    )
    raw = json.loads(index.read_text(encoding="utf-8"))
    proposal_store._MANIFEST_CACHE.clear()
    real_read_bytes = Path.read_bytes
    manifest_reads = 0

    def counted_read_bytes(path: Path) -> bytes:
        nonlocal manifest_reads
        if "modules-manifest-" in path.name:
            manifest_reads += 1
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    first = proposal_store.read_sharded_complete_proposal_section(
        index, "modules", limit=1
    )
    reads_after_first = manifest_reads
    proposal_store.read_sharded_complete_proposal_section(
        index, "modules", cursor=first["next_cursor"], limit=1
    )

    assert reads_after_first == raw["modules"]["manifest_count"]
    assert manifest_reads == reads_after_first


def test_full_collection_load_does_not_read_manifests_twice(
    tmp_path: Path, monkeypatch
) -> None:
    index = proposal_store.write_sharded_complete_proposal(
        _approved_proposal(300),
        tmp_path / "complete-proposal.json",
        shard_size=1,
        part_size_bytes=16 * 1024,
    )
    raw = json.loads(index.read_text(encoding="utf-8"))
    proposal_store._MANIFEST_CACHE.clear()
    real_read_bytes = Path.read_bytes
    manifest_reads = 0

    def counted_read_bytes(path: Path) -> bytes:
        nonlocal manifest_reads
        if "modules-manifest-" in path.name:
            manifest_reads += 1
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    loaded = proposal_store.load_sharded_complete_proposal(index)

    assert len(loaded.modules) == 300
    assert manifest_reads == raw["modules"]["manifest_count"]
