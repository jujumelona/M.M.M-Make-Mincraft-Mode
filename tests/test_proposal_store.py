import json
import os
import zipfile
from pathlib import Path

import pytest

from minecraft_mod_ai.complete_spec import (
    ProductionModule,
    complete_proposal_from_parts,
)
from minecraft_mod_ai.json_stream import (
    CanonicalJsonError,
    StreamingJsonDecodeError,
    iter_canonical_json,
    parse_json_byte_chunks,
)
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.proposal_store import (
    load_sharded_complete_proposal,
    load_sharded_complete_proposal_from_zip,
    read_sharded_complete_proposal_section,
    write_sharded_complete_proposal,
)
from minecraft_mod_ai.spec import SpecValidationError, canonical_json


def _large_proposal(module_count: int=103, *, game_design: dict | None=None):
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan('Create exactly one frost item.')
    proposal = complete_proposal_from_parts(requested_prompt='Create a large deterministic content pack.', base_proposal=base, game_design=game_design or {'title': 'Sharded'}, modules=tuple(ProductionModule(f'module_{index:05d}', 'item') for index in range(module_count)), acceptance_tests=('Every requested module is registered.',))
    return proposal.approve(proposal.calculate_hash())

def test_complete_proposal_storage_scales_by_adding_bounded_shards(tmp_path: Path) -> None:
    proposal = _large_proposal()
    index = write_sharded_complete_proposal(proposal, tmp_path / 'complete-proposal.json', shard_size=8)
    raw = json.loads(index.read_text(encoding='utf-8'))
    assert raw['schema_version'] == 'mmm/complete-proposal-index-v3'
    assert raw['modules']['count'] == 103
    assert raw['modules']['shard_count'] == 13
    assert raw['modules']['manifest_count'] == 1
    assert index.stat().st_size < 16000
    loaded = load_sharded_complete_proposal(index)
    assert loaded.calculate_hash() == proposal.calculate_hash()
    assert len(loaded.modules) == 103

def test_complete_proposal_storage_rejects_a_changed_shard(tmp_path: Path) -> None:
    index = write_sharded_complete_proposal(_large_proposal(17), tmp_path / 'complete-proposal.json', shard_size=4)
    raw = json.loads(index.read_text(encoding='utf-8'))
    shard = _first_collection_chunk(index, raw['modules'])
    shard.write_text('[]', encoding='utf-8')
    with pytest.raises(SpecValidationError, match='chunk (size|hash) mismatch'):
        load_sharded_complete_proposal(index)

def test_index_swap_failure_keeps_previous_immutable_version_readable(tmp_path: Path, monkeypatch) -> None:
    index = write_sharded_complete_proposal(_large_proposal(9), tmp_path / 'complete-proposal.json', shard_size=3)
    previous_index = index.read_bytes()
    previous_raw = json.loads(previous_index)
    previous_shard = _first_collection_chunk(index, previous_raw['modules'])
    previous_shard_bytes = previous_shard.read_bytes()
    previous_hash = load_sharded_complete_proposal(index).calculate_hash()
    real_replace = os.replace

    def fail_index_swap(source, destination) -> None:
        if Path(destination).resolve() == index:
            raise OSError('simulated index swap failure')
        real_replace(source, destination)
    monkeypatch.setattr('minecraft_mod_ai.proposal_store.os.replace', fail_index_swap)
    with pytest.raises(OSError, match='simulated index swap failure'):
        write_sharded_complete_proposal(_large_proposal(12), index, shard_size=3)
    assert index.read_bytes() == previous_index
    assert previous_shard.read_bytes() == previous_shard_bytes
    assert load_sharded_complete_proposal(index).calculate_hash() == previous_hash

def test_large_game_design_uses_fixed_size_files_and_round_trips(tmp_path: Path) -> None:
    part_size = 32 * 1024
    lore = '한글 "quoted" line\\path\n' + '0123456789abcdef' * 16384 + ' 🐉'
    proposal = _large_proposal(33, game_design={'title': 'Large streamed design', 'lore': lore, 'nested': {'chapters': [lore[:100000], lore[100000:]]}})
    index = write_sharded_complete_proposal(proposal, tmp_path / 'complete-proposal.json', shard_size=4, part_size_bytes=part_size)
    raw = json.loads(index.read_text(encoding='utf-8'))
    version_root = (index.parent / Path(*raw['game_design']['path_prefix'].split('/'))).parent
    stored_files = [path for path in version_root.rglob('*') if path.is_file()]
    assert raw['game_design']['chunk_count'] > 8
    assert stored_files
    assert max(path.stat().st_size for path in stored_files) <= part_size
    assert index.stat().st_size <= part_size
    loaded = load_sharded_complete_proposal(index)
    assert loaded.game_design == proposal.game_design
    assert loaded.calculate_hash() == proposal.calculate_hash()

def test_root_index_stays_bounded_across_many_collection_shards(tmp_path: Path) -> None:
    index = write_sharded_complete_proposal(_large_proposal(300), tmp_path / 'complete-proposal.json', shard_size=1, part_size_bytes=32 * 1024)
    raw = json.loads(index.read_text(encoding='utf-8'))
    assert raw['modules']['shard_count'] == 300
    assert raw['modules']['manifest_count'] == 3
    assert 'shards' not in raw['modules']
    assert index.stat().st_size < 16000
    assert len(load_sharded_complete_proposal(index).modules) == 300

def test_complete_proposal_collection_pages_resume_without_inline_monolith(tmp_path: Path) -> None:
    index = write_sharded_complete_proposal(_large_proposal(257), tmp_path / 'complete-proposal.json', shard_size=7, part_size_bytes=32 * 1024)
    cursor = ''
    observed: list[str] = []
    page_count = 0
    while True:
        page = read_sharded_complete_proposal_section(index, 'modules', cursor=cursor, limit=31)
        observed.extend(item['module_id'] for item in page['items'])
        page_count += 1
        cursor = page['next_cursor']
        if not cursor:
            break
    assert page_count > 1
    assert observed == [f'module_{index:05d}' for index in range(257)]
    assert page['remaining'] == 0
    assert page['total_count'] == 257

def test_complete_proposal_page_cursor_is_bound_to_section_and_hash(tmp_path: Path) -> None:
    index = write_sharded_complete_proposal(_large_proposal(9), tmp_path / 'complete-proposal.json', shard_size=2)
    first = read_sharded_complete_proposal_section(index, 'modules', limit=1)
    with pytest.raises(SpecValidationError, match='does not match this proposal section'):
        read_sharded_complete_proposal_section(index, 'assets', cursor=first['next_cursor'], limit=1)

def test_nested_game_design_list_can_be_read_as_bounded_pages(tmp_path: Path) -> None:
    index = write_sharded_complete_proposal(_large_proposal(3, game_design={'title': 'Paged design', 'production_outline': [{'scope': f'Chapter {value}'} for value in range(53)]}), tmp_path / 'complete-proposal.json', shard_size=2)
    page = read_sharded_complete_proposal_section(index, 'game_design.production_outline', limit=10)
    assert len(page['items']) == 10
    assert page['items'][0] == {'scope': 'Chapter 0'}
    assert page['remaining'] == 43
    assert page['next_cursor']

def test_v2_chunked_store_loads_from_release_zip(tmp_path: Path) -> None:
    proposal = _large_proposal(19, game_design={'title': 'ZIP', 'lore': 'z' * 70000})
    tree = tmp_path / 'tree'
    index = write_sharded_complete_proposal(proposal, tree / 'META-INF/mmm-complete-proposal.json', shard_size=3, part_size_bytes=16 * 1024)
    archive_path = tmp_path / 'proposal.zip'
    with zipfile.ZipFile(archive_path, 'w') as archive:
        for path in sorted(tree.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(tree).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        loaded = load_sharded_complete_proposal_from_zip(archive, index.relative_to(tree).as_posix())
    assert loaded.game_design == proposal.game_design
    assert loaded.calculate_hash() == proposal.calculate_hash()

def test_streaming_json_matches_canonical_json_across_byte_boundaries() -> None:
    value = {'emoji': '🐉', 'escaped': '"quote"\\path\nline', 'numbers': [-1, 0, 1.25, 6.02e+23], 'values': [True, False, None]}
    encoded = ''.join(iter_canonical_json(value)).encode('utf-8')
    assert encoded.decode('utf-8') == canonical_json(value)
    assert parse_json_byte_chunks(bytes((byte,)) for byte in encoded) == value

def test_streaming_json_combines_surrogate_pairs_and_rejects_lone_values() -> None:
    assert parse_json_byte_chunks([b'"\\uD83D', b'\\uDC09"']) == '🐉'
    assert ''.join(iter_canonical_json('\ud83d\udc09')) == '"🐉"'
    for encoded in (b'"\\uD800"', b'"\\uDC00"', b'"\\uD800\\u0041"'):
        with pytest.raises(StreamingJsonDecodeError):
            parse_json_byte_chunks([encoded])
    for value in ('\ud800', '\udc00', '\ud800A'):
        with pytest.raises(CanonicalJsonError):
            ''.join(iter_canonical_json(value))

def _first_collection_chunk(index: Path, collection: dict) -> Path:
    manifest_prefix = Path(*collection['manifest_prefix'].split('/'))
    manifest = index.parent / manifest_prefix.with_name(manifest_prefix.name + '-00000000.json')
    descriptor = json.loads(manifest.read_text(encoding='utf-8'))['shards'][0]
    data_prefix = manifest.parent / descriptor['path_prefix']
    return data_prefix.with_name(data_prefix.name + '.chunk-00000000.jsonpart')
