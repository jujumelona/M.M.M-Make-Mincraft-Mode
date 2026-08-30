from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import shutil
import threading
import uuid
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from .complete_spec import CompleteProposal
from .json_stream import (
    StreamingJsonDecodeError,
    iter_canonical_json,
    parse_json_byte_chunks,
)
from .scale_policy import ScalePolicy
from .spec import SpecValidationError, canonical_json

INDEX_SCHEMA = 'mmm/complete-proposal-index-v3'
CHUNKED_JSON_ENCODING = 'mmm/canonical-json-chunks-v1'
COLLECTION_FORMAT = 'mmm/numbered-collection-manifests-v1'
DEFAULT_PART_SIZE_BYTES = 1024 * 1024
MIN_PART_SIZE_BYTES = 16 * 1024
_COLLECTION_MANIFEST_SHARDS = 128
_MANIFEST_CACHE_MAX_ENTRIES = 4
_PAGE_CURSOR = re.compile('^p_(\\d+)_(\\d+)_(\\d+)_(\\d+)_(\\d+)_([0-9a-f]{32})$')
MAX_PAGE_ITEMS = 1000
DEFAULT_PAGE_SIZE_BYTES = 256 * 1024
MIN_PAGE_SIZE_BYTES = 8 * 1024
MAX_PAGE_SIZE_BYTES = 4 * 1024 * 1024
_PAGE_METADATA_RESERVE_BYTES = 4 * 1024
_T = TypeVar('_T')
_MANIFEST_CACHE: OrderedDict[
    tuple[str, int, int, int, str],
    tuple[tuple[dict[str, Any], ...], ...],
] = OrderedDict()
_MANIFEST_CACHE_LOCK = threading.RLock()


def write_sharded_complete_proposal(proposal: CompleteProposal, index_path: str | Path, *, shard_size: int, policy: ScalePolicy | None=None, part_size_bytes: int=DEFAULT_PART_SIZE_BYTES) -> Path:
    """Persist a proposal as bounded, hash-addressed JSON chunks and shards.

    ``shard_size`` bounds the number of collection items per logical shard.
    ``part_size_bytes`` independently bounds every JSON data file. Collection
    descriptors are stored in fixed-fanout numbered manifests, so the root
    index stays bounded as the number of modules grows.
    """
    if type(shard_size) is not int or shard_size < 1:
        raise ValueError('shard_size must be a positive integer.')
    if type(part_size_bytes) is not int or part_size_bytes < MIN_PART_SIZE_BYTES:
        raise ValueError(f'part_size_bytes must be an integer of at least {MIN_PART_SIZE_BYTES}.')
    if policy is not None:
        part_size_bytes = min(part_size_bytes, policy.max_single_file_bytes)
        if part_size_bytes < MIN_PART_SIZE_BYTES:
            raise ValueError('Configured per-file policy is too small for proposal shards.')
    proposal.validate(policy=policy)
    proposal_hash = proposal.approval_hash or proposal.calculate_hash()
    index = Path(index_path).expanduser().resolve()
    index.parent.mkdir(parents=True, exist_ok=True)
    parts_root_name = f'{index.stem}.d'
    parts_root = index.parent / parts_root_name
    if parts_root.exists():
        if parts_root.is_symlink() or not parts_root.is_dir():
            raise SpecValidationError(f'Proposal shard path is unsafe: {parts_root}')
    else:
        parts_root.mkdir()
    version_name = f"v-{proposal_hash.removeprefix('sha256:')[:24]}-{uuid.uuid4().hex}"
    version_parts = parts_root / version_name
    staging = parts_root / f'.tmp-{uuid.uuid4().hex}'
    temporary_index: Path | None = None
    staging.mkdir(parents=False)
    try:
        metadata = {'schema_version': proposal.schema_version, 'proposal_version': proposal.proposal_version, 'status': proposal.status.value, 'requested_prompt': proposal.requested_prompt, 'external_runtime_required': proposal.external_runtime_required, 'existing_input_sha256': proposal.existing_input_sha256, 'approval_hash': proposal.approval_hash}
        payload: dict[str, Any] = {'schema_version': INDEX_SCHEMA, 'proposal_hash': proposal_hash, 'metadata': _write_chunked_json(staging, 'metadata', metadata, part_size_bytes=part_size_bytes), 'base_proposal': _write_chunked_json(staging, 'base-proposal', proposal.base_proposal.to_dict(), part_size_bytes=part_size_bytes), 'game_design': _write_chunked_json(staging, 'game-design', proposal.game_design, part_size_bytes=part_size_bytes), 'modules': _write_collection_v2(staging, 'modules', (asdict(value) for value in proposal.modules), shard_size, part_size_bytes=part_size_bytes), 'assets': _write_collection_v2(staging, 'assets', (asdict(value) for value in proposal.assets), shard_size, part_size_bytes=part_size_bytes), 'acceptance_tests': _write_collection_v2(staging, 'acceptance-tests', proposal.acceptance_tests, shard_size, part_size_bytes=part_size_bytes)}
        _prefix_part_paths(payload, f'{parts_root_name}/{version_name}')
        _sync_directory(staging)
        os.replace(staging, version_parts)
        _sync_directory(parts_root)
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
        if len(rendered.encode('utf-8')) > part_size_bytes:
            raise SpecValidationError('Proposal root index exceeded the configured part size.')
        temporary_index = index.with_name(f'.{index.name}.tmp-{uuid.uuid4().hex}')
        _write_bytes_durable(temporary_index, rendered.encode('utf-8'))
        os.replace(temporary_index, index)
        _sync_directory(index.parent)
        return index
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if temporary_index is not None and temporary_index.exists():
            temporary_index.unlink()


def load_sharded_complete_proposal(index_path: str | Path) -> CompleteProposal:
    index = Path(index_path).expanduser().resolve()
    raw = json.loads(index.read_text(encoding='utf-8'))
    return complete_proposal_from_index(raw, lambda relative: _read_file_part(index.parent, relative))


def read_sharded_complete_proposal_section(index_path: str | Path, section: str, *, cursor: str='', limit: int=100, max_bytes: int=DEFAULT_PAGE_SIZE_BYTES, cursor_key: bytes | None=None) -> dict[str, Any]:
    """Read one bounded page from an immutable complete-proposal store.

    Collection cursors record the exact manifest, shard and item position.  A
    page therefore resumes without loading or transferring every earlier item.
    The per-call limit protects the MCP transport; it never limits the number
    of items that may exist in the proposal.
    """
    if type(limit) is not int or not 1 <= limit <= MAX_PAGE_ITEMS:
        raise SpecValidationError(f'limit must be between 1 and {MAX_PAGE_ITEMS}.')
    if type(max_bytes) is not int or not MIN_PAGE_SIZE_BYTES <= max_bytes <= MAX_PAGE_SIZE_BYTES:
        raise SpecValidationError(f'max_bytes must be between {MIN_PAGE_SIZE_BYTES} and {MAX_PAGE_SIZE_BYTES}.')
    if not isinstance(section, str) or not section.strip() or len(section) > 256:
        raise SpecValidationError('section must not be empty.')
    if cursor_key is not None and (not isinstance(cursor_key, bytes) or len(cursor_key) < 16):
        raise SpecValidationError('cursor_key must contain at least 16 bytes.')
    selected = section.strip()
    index = Path(index_path).expanduser().resolve()
    try:
        raw = json.loads(index.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpecValidationError('Complete proposal index is missing or invalid.') from exc
    required = {'schema_version', 'proposal_hash', 'metadata', 'base_proposal', 'game_design', 'modules', 'assets', 'acceptance_tests'}
    if not isinstance(raw, dict) or set(raw) != required or raw.get('schema_version') != INDEX_SCHEMA:
        raise SpecValidationError('Complete proposal shard index fields are invalid.')
    proposal_hash = str(raw['proposal_hash'])

    def read_part(relative):
        return _read_file_part(index.parent, relative)
    if selected == 'overview':
        if cursor:
            raise SpecValidationError('overview does not accept a cursor.')
        metadata = _read_json_part(raw['metadata'], read_part)
        if not isinstance(metadata, dict):
            raise SpecValidationError('Complete proposal metadata part must be an object.')
        requested_prompt = metadata.get('requested_prompt', '')
        metadata_summary = {key: value for key, value in metadata.items() if key != 'requested_prompt'}
        metadata_summary['requested_prompt_chars'] = len(requested_prompt) if isinstance(requested_prompt, str) else 0
        metadata_summary['requested_prompt_sha256'] = 'sha256:' + hashlib.sha256((requested_prompt if isinstance(requested_prompt, str) else '').encode('utf-8')).hexdigest()
        return _bounded_page_result({'schema_version': 'mmm/complete-plan-section-v1', 'proposal_hash': proposal_hash, 'section': selected, 'items': [{'metadata': metadata_summary, 'counts': _proposal_collection_counts(raw), 'available_sections': _available_sections(raw)}], 'returned': 1, 'item_fragment': None, 'total_count': 1, 'next_cursor': '', 'remaining': 0, 'max_bytes': max_bytes}, max_bytes=max_bytes)
    if selected == 'metadata':
        metadata = _read_json_part(raw['metadata'], read_part)
        return _read_plain_value_page(metadata, proposal_hash=proposal_hash, section=selected, cursor=cursor, limit=limit, max_bytes=max_bytes, cursor_key=cursor_key)
    if selected.startswith('game_design.'):
        field_name = selected.removeprefix('game_design.')
        if not field_name or '.' in field_name:
            raise SpecValidationError(f'Unknown complete proposal section: {selected!r}')
        design = _read_json_part(raw['game_design'], read_part)
        if not isinstance(design, dict) or field_name not in design:
            raise SpecValidationError(f'Unknown complete proposal section: {selected!r}')
        return _read_plain_value_page(design[field_name], proposal_hash=proposal_hash, section=selected, cursor=cursor, limit=limit, max_bytes=max_bytes, cursor_key=cursor_key)
    if selected in {'game_design', 'base_proposal'}:
        value = _read_json_part(raw[selected], read_part)
        return _read_plain_value_page(value, proposal_hash=proposal_hash, section=selected, cursor=cursor, limit=limit, max_bytes=max_bytes, cursor_key=cursor_key)
    if selected in {'modules', 'assets', 'acceptance_tests'}:
        return _read_collection_page(raw[selected], read_part, proposal_hash=proposal_hash, section=selected, cursor=cursor, limit=limit, max_bytes=max_bytes, cursor_key=cursor_key)
    raise SpecValidationError(f'Unknown complete proposal section: {selected!r}')


def load_sharded_complete_proposal_from_zip(archive: zipfile.ZipFile, index_name: str) -> CompleteProposal:
    raw = json.loads(archive.read(index_name).decode('utf-8'))
    base = PurePosixPath(index_name).parent

    def read(relative: str) -> bytes:
        normalized = _safe_relative(relative)
        name = (base / normalized).as_posix()
        return archive.read(name)
    return complete_proposal_from_index(raw, read)


def complete_proposal_from_index(raw: dict[str, Any], read_part: Callable[[str], bytes]) -> CompleteProposal:
    required = {'schema_version', 'proposal_hash', 'metadata', 'base_proposal', 'game_design', 'modules', 'assets', 'acceptance_tests'}
    if not isinstance(raw, dict) or set(raw) != required:
        raise SpecValidationError('Complete proposal shard index fields are invalid.')
    if raw['schema_version'] != INDEX_SCHEMA:
        raise SpecValidationError('Unsupported complete proposal shard index.')
    metadata = _read_json_part(raw['metadata'], read_part)
    base_proposal = _read_json_part(raw['base_proposal'], read_part)
    game_design = _read_json_part(raw['game_design'], read_part)
    modules = _read_collection(raw['modules'], read_part)
    assets = _read_collection(raw['assets'], read_part)
    acceptance_tests = _read_collection(raw['acceptance_tests'], read_part)
    if not isinstance(metadata, dict):
        raise SpecValidationError('Complete proposal metadata part must be an object.')
    expected_metadata = {'schema_version', 'proposal_version', 'status', 'requested_prompt', 'external_runtime_required', 'existing_input_sha256', 'approval_hash'}
    if set(metadata) != expected_metadata:
        raise SpecValidationError('Complete proposal metadata fields are invalid.')
    proposal = CompleteProposal.from_dict({**metadata, 'base_proposal': base_proposal, 'game_design': game_design, 'modules': modules, 'assets': assets, 'acceptance_tests': acceptance_tests})
    actual_hash = proposal.approval_hash or proposal.calculate_hash()
    if raw['proposal_hash'] != actual_hash:
        raise SpecValidationError('Complete proposal shard root hash is invalid.')
    return proposal


def _proposal_collection_counts(raw: dict[str, Any]) -> dict[str, int]:
    return {name: _collection_count(raw[name]) for name in ('modules', 'assets', 'acceptance_tests')}


def _available_sections(raw: dict[str, Any]) -> list[str]:
    return ['overview', 'metadata', 'game_design', 'base_proposal', 'modules', 'assets', 'acceptance_tests']


def _collection_count(value: Any) -> int:
    if not isinstance(value, dict):
        raise SpecValidationError('Collection shard index fields are invalid.')
    count = value.get('count')
    if type(count) is not int or count < 0:
        raise SpecValidationError('Collection shard index values are invalid.')
    return count


def _read_plain_value_page(value: Any, *, proposal_hash: str, section: str, cursor: str, limit: int, max_bytes: int, cursor_key: bytes | None) -> dict[str, Any]:
    offset, manifest_index, shard_index, item_index, fragment_offset = _decode_page_cursor(cursor, proposal_hash=proposal_hash, section=section, cursor_key=cursor_key)
    if any((manifest_index, shard_index, item_index)):
        raise SpecValidationError('Plain-value page cursor is invalid.')
    if isinstance(value, dict):
        sequence = [{'key': key, 'value': value[key]} for key in sorted(value)]
    elif isinstance(value, list):
        sequence = value
    else:
        sequence = [value]
    if offset > len(sequence):
        raise SpecValidationError('Page cursor exceeds the section length.')
    items: list[Any] = []
    fragment: dict[str, Any] | None = None
    payload_budget = max_bytes - _PAGE_METADATA_RESERVE_BYTES
    payload_used = 2
    while offset < len(sequence) and len(items) < limit:
        encoded = canonical_json(sequence[offset]).encode('utf-8')
        if fragment_offset or len(encoded) > payload_budget:
            if items:
                break
            fragment, next_fragment_offset, fragment_complete = _item_fragment(encoded, offset=fragment_offset, payload_budget=payload_budget)
            if fragment_complete:
                offset += 1
                fragment_offset = 0
            else:
                fragment_offset = next_fragment_offset
            break
        item_cost = len(encoded) + (1 if items else 0)
        if items and payload_used + item_cost > payload_budget:
            break
        items.append(sequence[offset])
        payload_used += item_cost
        offset += 1
        fragment_offset = 0
    next_cursor = _encode_page_cursor(proposal_hash=proposal_hash, section=section, offset=offset, manifest_index=0, shard_index=0, item_index=0, fragment_offset=fragment_offset, cursor_key=cursor_key) if offset < len(sequence) else ''
    return _bounded_page_result({'schema_version': 'mmm/complete-plan-section-v1', 'proposal_hash': proposal_hash, 'section': section, 'items': items, 'returned': len(items), 'item_fragment': fragment, 'total_count': len(sequence), 'next_cursor': next_cursor, 'remaining': len(sequence) - offset, 'max_bytes': max_bytes}, max_bytes=max_bytes)


def _read_collection_page(value: Any, read_part: Callable[[str], bytes], *, proposal_hash: str, section: str, cursor: str, limit: int, max_bytes: int, cursor_key: bytes | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecValidationError('Collection shard index fields are invalid.')
    offset, manifest_index, shard_index, item_index, fragment_offset = _decode_page_cursor(cursor, proposal_hash=proposal_hash, section=section, cursor_key=cursor_key)
    if set(value) == {'count', 'shards'}:
        count = _collection_count(value)
        shards = value['shards']
        if not isinstance(shards, list) or manifest_index != 0:
            raise SpecValidationError('Collection shard index values are invalid.')
        manifests = (tuple(shards),) if shards else ()
        manifest_parent = PurePosixPath('.')
    else:
        count, manifest_prefix, manifests = _validated_v2_collection_manifests(value, read_part)
        manifest_parent = manifest_prefix.parent
    manifest_count = len(manifests)

    def load_manifest(index: int) -> tuple[tuple[dict[str, Any], ...], PurePosixPath]:
        if not 0 <= index < manifest_count:
            return ((), manifest_parent)
        return (manifests[index], manifest_parent)

    if offset > count:
        raise SpecValidationError('Page cursor exceeds the section length.')
    items: list[Any] = []
    fragment: dict[str, Any] | None = None
    payload_budget = max_bytes - _PAGE_METADATA_RESERVE_BYTES
    payload_used = 2
    stop_page = False
    while len(items) < limit and manifest_index < manifest_count:
        descriptors, descriptor_parent = load_manifest(manifest_index)
        if shard_index > len(descriptors):
            raise SpecValidationError('Page cursor shard position is invalid.')
        if shard_index == len(descriptors):
            manifest_index += 1
            shard_index = 0
            item_index = 0
            fragment_offset = 0
            continue
        descriptor = descriptors[shard_index]
        if descriptor_parent == PurePosixPath('.'):
            qualified = descriptor
        else:
            qualified = _qualify_descriptor(descriptor, descriptor_parent)
        wrapper = _read_json_part(qualified, read_part)
        if not isinstance(wrapper, dict) or set(wrapper) != {'items'} or (not isinstance(wrapper['items'], list)) or (not wrapper['items']):
            raise SpecValidationError('Collection shard must be a non-empty list.')
        shard_items = wrapper['items']
        if item_index > len(shard_items):
            raise SpecValidationError('Page cursor item position is invalid.')
        if item_index == len(shard_items):
            shard_index += 1
            item_index = 0
            fragment_offset = 0
            continue
        while item_index < len(shard_items) and len(items) < limit:
            item = shard_items[item_index]
            encoded = canonical_json(item).encode('utf-8')
            if fragment_offset or len(encoded) > payload_budget:
                if items:
                    stop_page = True
                    break
                fragment, next_fragment_offset, fragment_complete = _item_fragment(encoded, offset=fragment_offset, payload_budget=payload_budget)
                if fragment_complete:
                    item_index += 1
                    offset += 1
                    fragment_offset = 0
                else:
                    fragment_offset = next_fragment_offset
                stop_page = True
                break
            item_cost = len(encoded) + (1 if items else 0)
            if items and payload_used + item_cost > payload_budget:
                stop_page = True
                break
            items.append(item)
            payload_used += item_cost
            item_index += 1
            offset += 1
            fragment_offset = 0
        if item_index == len(shard_items):
            shard_index += 1
            item_index = 0
            fragment_offset = 0
        if stop_page:
            break
    if offset > count:
        raise SpecValidationError('Collection page contains more items than its index.')
    if offset < count and manifest_index >= manifest_count:
        raise SpecValidationError('Collection shard count does not match its index.')
    next_cursor = _encode_page_cursor(proposal_hash=proposal_hash, section=section, offset=offset, manifest_index=manifest_index, shard_index=shard_index, item_index=item_index, fragment_offset=fragment_offset, cursor_key=cursor_key) if offset < count else ''
    return _bounded_page_result({'schema_version': 'mmm/complete-plan-section-v1', 'proposal_hash': proposal_hash, 'section': section, 'items': items, 'returned': len(items), 'item_fragment': fragment, 'total_count': count, 'next_cursor': next_cursor, 'remaining': count - offset, 'max_bytes': max_bytes}, max_bytes=max_bytes)


def _validated_v2_collection_manifests(value: dict[str, Any], read_part: Callable[[str], bytes]) -> tuple[int, PurePosixPath, tuple[tuple[dict[str, Any], ...], ...]]:
    required = {'format', 'count', 'shard_count', 'manifest_count', 'manifest_prefix', 'manifest_sha256'}
    if set(value) != required or value['format'] != COLLECTION_FORMAT:
        raise SpecValidationError('Collection shard index fields are invalid.')
    count = _collection_count(value)
    shard_count = value['shard_count']
    manifest_count = value['manifest_count']
    if type(shard_count) is not int or shard_count < 0 or type(manifest_count) is not int or manifest_count < 0 or manifest_count > shard_count or shard_count > count or ((count == 0) != (shard_count == 0)) or ((shard_count == 0) != (manifest_count == 0)):
        raise SpecValidationError('Collection shard and manifest counts are inconsistent.')
    manifest_prefix = _safe_relative(str(value['manifest_prefix']))
    expected_hash = str(value['manifest_sha256'])
    cache_key = (expected_hash, count, shard_count, manifest_count, manifest_prefix.as_posix())
    with _MANIFEST_CACHE_LOCK:
        cached = _MANIFEST_CACHE.get(cache_key)
        if cached is not None:
            _MANIFEST_CACHE.move_to_end(cache_key)
            return (count, manifest_prefix, cached)

    manifest_parent = manifest_prefix.parent
    manifest_name = manifest_prefix.name
    digest = hashlib.sha256()
    observed_shards = 0
    manifests: list[tuple[dict[str, Any], ...]] = []
    for index in range(manifest_count):
        path = (manifest_parent / f'{manifest_name}-{index:08d}.json').as_posix()
        data = read_part(path)
        digest.update(data)
        payload = _decode_collection_manifest(path, data)
        descriptors = tuple(payload['shards'])
        manifests.append(descriptors)
        observed_shards += len(descriptors)
    if 'sha256:' + digest.hexdigest() != expected_hash:
        raise SpecValidationError('Collection manifest hash does not match its index.')
    if observed_shards != shard_count:
        raise SpecValidationError('Collection manifest shard count does not match its index.')
    frozen = tuple(manifests)
    with _MANIFEST_CACHE_LOCK:
        _MANIFEST_CACHE[cache_key] = frozen
        _MANIFEST_CACHE.move_to_end(cache_key)
        while len(_MANIFEST_CACHE) > _MANIFEST_CACHE_MAX_ENTRIES:
            _MANIFEST_CACHE.popitem(last=False)
    return (count, manifest_prefix, frozen)


def _decode_collection_manifest(path: str, data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpecValidationError(f'Collection manifest is invalid JSON: {path}') from exc
    if not isinstance(payload, dict) or set(payload) != {'shards'} or (not isinstance(payload['shards'], list)) or (not payload['shards']) or (len(payload['shards']) > _COLLECTION_MANIFEST_SHARDS):
        raise SpecValidationError(f'Collection manifest fields are invalid: {path}')
    return payload


def _encode_page_cursor(*, proposal_hash: str, section: str, offset: int, manifest_index: int, shard_index: int, item_index: int, fragment_offset: int, cursor_key: bytes | None) -> str:
    payload = f'{proposal_hash}\x00{section}\x00{offset}\x00{manifest_index}\x00{shard_index}\x00{item_index}\x00{fragment_offset}'
    key = cursor_key or hashlib.sha256(f'mmm-local-page-cursor\x00{proposal_hash}'.encode()).digest()
    checksum = hmac.new(key, payload.encode('utf-8'), hashlib.sha256).hexdigest()[:32]
    return f'p_{offset}_{manifest_index}_{shard_index}_{item_index}_{fragment_offset}_{checksum}'


def _decode_page_cursor(cursor: str, *, proposal_hash: str, section: str, cursor_key: bytes | None) -> tuple[int, int, int, int, int]:
    if cursor == '':
        return (0, 0, 0, 0, 0)
    if not isinstance(cursor, str) or len(cursor) > 512:
        raise SpecValidationError('Page cursor is invalid.')
    match = _PAGE_CURSOR.fullmatch(cursor)
    if match is None:
        raise SpecValidationError('Page cursor is invalid.')
    try:
        offset, manifest_index, shard_index, item_index, fragment_offset = (int(value) for value in match.groups()[:5])
    except ValueError as exc:
        raise SpecValidationError('Page cursor is invalid.') from exc
    expected = _encode_page_cursor(proposal_hash=proposal_hash, section=section, offset=offset, manifest_index=manifest_index, shard_index=shard_index, item_index=item_index, fragment_offset=fragment_offset, cursor_key=cursor_key)
    if not hmac.compare_digest(expected, cursor):
        raise SpecValidationError('Page cursor does not match this proposal section.')
    return (offset, manifest_index, shard_index, item_index, fragment_offset)


def _item_fragment(encoded: bytes, *, offset: int, payload_budget: int) -> tuple[dict[str, Any], int, bool]:
    if offset < 0 or offset >= len(encoded):
        raise SpecValidationError('Page cursor fragment position is invalid.')
    raw_capacity = max(1, (payload_budget - 1024) * 3 // 4)
    end = min(len(encoded), offset + raw_capacity)
    fragment = {'schema_version': 'mmm/canonical-json-item-fragment-v1', 'encoding': 'base64', 'content_type': 'application/json', 'item_sha256': 'sha256:' + hashlib.sha256(encoded).hexdigest(), 'offset_bytes': offset, 'total_bytes': len(encoded), 'data': base64.b64encode(encoded[offset:end]).decode('ascii'), 'complete': end == len(encoded)}
    return (fragment, end, end == len(encoded))


def _bounded_page_result(result: dict[str, Any], *, max_bytes: int) -> dict[str, Any]:
    encoded = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(encoded) > max_bytes:
        raise SpecValidationError('Complete proposal page exceeded its transport byte budget.')
    return result


def _write_collection_v2(root: Path, prefix: str, values: Sequence[Any] | Iterable[Any], shard_size: int, *, part_size_bytes: int) -> dict[str, Any]:
    manifest_page: list[dict[str, Any]] = []
    manifest_digest = hashlib.sha256()
    manifest_count = 0
    shard_count = 0
    page: list[Any] = []
    count = 0

    def flush_manifest() -> None:
        nonlocal manifest_count
        if not manifest_page:
            return
        rendered = canonical_json({'shards': manifest_page}).encode('utf-8')
        if len(rendered) > part_size_bytes:
            raise SpecValidationError('Proposal collection manifest exceeded the configured part size.')
        _write_bytes_durable(root / f'{prefix}-manifest-{manifest_count:08d}.json', rendered)
        manifest_digest.update(rendered)
        manifest_page.clear()
        manifest_count += 1

    def flush_page() -> None:
        nonlocal page, shard_count
        if not page:
            return
        descriptor = _write_chunked_json(root, f'{prefix}-shard-{shard_count:08d}', {'items': page}, part_size_bytes=part_size_bytes)
        candidate_manifest = canonical_json({'shards': [*manifest_page, descriptor]}).encode('utf-8')
        if manifest_page and len(candidate_manifest) > part_size_bytes:
            flush_manifest()
            candidate_manifest = canonical_json({'shards': [descriptor]}).encode('utf-8')
        if len(candidate_manifest) > part_size_bytes:
            raise SpecValidationError('One proposal shard descriptor exceeds the configured part size.')
        manifest_page.append(descriptor)
        page = []
        shard_count += 1
        if len(manifest_page) == _COLLECTION_MANIFEST_SHARDS:
            flush_manifest()
    for item in values:
        page.append(item)
        count += 1
        if len(page) == shard_size:
            flush_page()
    flush_page()
    flush_manifest()
    return {'format': COLLECTION_FORMAT, 'count': count, 'shard_count': shard_count, 'manifest_count': manifest_count, 'manifest_prefix': f'{prefix}-manifest', 'manifest_sha256': 'sha256:' + manifest_digest.hexdigest()}


def _read_collection(value: Any, read_part: Callable[[str], bytes]) -> list[Any]:
    if not isinstance(value, dict):
        raise SpecValidationError('Collection shard index fields are invalid.')
    if set(value) == {'count', 'shards'}:
        return _read_collection_v1(value, read_part)
    count, manifest_prefix, manifests = _validated_v2_collection_manifests(value, read_part)
    manifest_parent = manifest_prefix.parent
    result: list[Any] = []
    for descriptors in manifests:
        for descriptor in descriptors:
            page_wrapper = _read_json_part(_qualify_descriptor(descriptor, manifest_parent), read_part)
            if not isinstance(page_wrapper, dict) or set(page_wrapper) != {'items'} or (not isinstance(page_wrapper['items'], list)) or (not page_wrapper['items']):
                raise SpecValidationError('Collection shard must be a non-empty list.')
            result.extend(page_wrapper['items'])
    if len(result) != count:
        raise SpecValidationError('Collection shard count does not match its index.')
    return result


def _read_collection_v1(value: dict[str, Any], read_part: Callable[[str], bytes]) -> list[Any]:
    count = value['count']
    shards = value['shards']
    if type(count) is not int or count < 0 or (not isinstance(shards, list)):
        raise SpecValidationError('Collection shard index values are invalid.')
    result: list[Any] = []
    for descriptor in shards:
        page_wrapper = _read_json_part(descriptor, read_part)
        if not isinstance(page_wrapper, dict) or set(page_wrapper) != {'items'} or (not isinstance(page_wrapper['items'], list)) or (not page_wrapper['items']):
            raise SpecValidationError('Collection shard must be a non-empty list.')
        result.extend(page_wrapper['items'])
    if len(result) != count:
        raise SpecValidationError('Collection shard count does not match its index.')
    return result


def _write_chunked_json(root: Path, prefix: str, value: Any, *, part_size_bytes: int) -> dict[str, Any]:
    _safe_relative(prefix)
    digest = hashlib.sha256()
    pending = bytearray()
    chunk_count = 0
    size_bytes = 0

    def flush() -> None:
        nonlocal chunk_count
        if not pending:
            return
        _write_bytes_durable(root / f'{prefix}.chunk-{chunk_count:08d}.jsonpart', bytes(pending))
        pending.clear()
        chunk_count += 1
    for text in iter_canonical_json(value):
        encoded = text.encode('utf-8')
        offset = 0
        while offset < len(encoded):
            available = part_size_bytes - len(pending)
            piece = encoded[offset:offset + available]
            pending.extend(piece)
            digest.update(piece)
            size_bytes += len(piece)
            offset += len(piece)
            if len(pending) == part_size_bytes:
                flush()
    flush()
    if chunk_count == 0:
        raise SpecValidationError('Proposal JSON value encoded to no chunks.')
    return {'encoding': CHUNKED_JSON_ENCODING, 'path_prefix': prefix, 'chunk_count': chunk_count, 'size_bytes': size_bytes, 'sha256': 'sha256:' + digest.hexdigest()}


def _write_bytes_durable(path: Path, value: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    """Best-effort directory sync; unsupported on some Windows filesystems."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _read_json_part(descriptor: Any, read_part: Callable[[str], bytes]) -> Any:
    if not isinstance(descriptor, dict):
        raise SpecValidationError('Proposal shard descriptor fields are invalid.')
    if set(descriptor) == {'path', 'sha256'}:
        relative = str(descriptor['path'])
        expected = str(descriptor['sha256'])
        data = read_part(relative)
        actual = 'sha256:' + hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise SpecValidationError(f'Proposal shard hash mismatch: {relative}')
        try:
            return json.loads(data.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SpecValidationError(f'Proposal shard is invalid JSON: {relative}') from exc
    required = {'encoding', 'path_prefix', 'chunk_count', 'size_bytes', 'sha256'}
    if set(descriptor) != required or descriptor['encoding'] != CHUNKED_JSON_ENCODING:
        raise SpecValidationError('Proposal shard descriptor fields are invalid.')
    prefix = _safe_relative(str(descriptor['path_prefix']))
    chunk_count = descriptor['chunk_count']
    size_bytes = descriptor['size_bytes']
    expected = str(descriptor['sha256'])
    if type(chunk_count) is not int or chunk_count < 1 or type(size_bytes) is not int or (size_bytes < 1) or (chunk_count > size_bytes):
        raise SpecValidationError('Proposal chunk descriptor values are invalid.')
    digest = hashlib.sha256()
    prefix_parent = prefix.parent
    prefix_name = prefix.name
    observed_size = 0

    def chunks() -> Iterable[bytes]:
        nonlocal observed_size
        for index in range(chunk_count):
            relative = (prefix_parent / f'{prefix_name}.chunk-{index:08d}.jsonpart').as_posix()
            data = read_part(relative)
            digest.update(data)
            observed_size += len(data)
            yield data

    try:
        parsed = parse_json_byte_chunks(chunks())
    except StreamingJsonDecodeError as exc:
        raise SpecValidationError(f'Proposal chunked JSON is invalid: {prefix.as_posix()}') from exc
    if observed_size != size_bytes:
        raise SpecValidationError(f'Proposal chunk size mismatch: {prefix.as_posix()}')
    actual = 'sha256:' + digest.hexdigest()
    if actual != expected:
        raise SpecValidationError(f'Proposal chunk hash mismatch: {prefix.as_posix()}')
    return parsed


def _read_file_part(root: Path, relative: str) -> bytes:
    normalized = _safe_relative(relative)
    target = (root / Path(*normalized.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise SpecValidationError('Proposal shard escaped its index directory.') from exc
    if not target.is_file() or target.is_symlink():
        raise SpecValidationError(f'Proposal shard is missing or unsafe: {relative}')
    return target.read_bytes()


def _safe_relative(value: str) -> PurePosixPath:
    normalized = PurePosixPath(value.replace('\\', '/'))
    if not value or normalized.is_absolute() or any(part in {'', '.', '..'} for part in normalized.parts):
        raise SpecValidationError(f'Unsafe proposal shard path: {value!r}')
    return normalized


def _qualify_descriptor(descriptor: Any, parent: PurePosixPath) -> dict[str, Any]:
    if not isinstance(descriptor, dict):
        raise SpecValidationError('Proposal shard descriptor must be an object.')
    qualified = dict(descriptor)
    if set(qualified) == {'path', 'sha256'}:
        relative = _safe_relative(str(qualified['path']))
        qualified['path'] = (parent / relative).as_posix()
        return qualified
    if qualified.get('encoding') == CHUNKED_JSON_ENCODING:
        relative = _safe_relative(str(qualified.get('path_prefix', '')))
        qualified['path_prefix'] = (parent / relative).as_posix()
        return qualified
    raise SpecValidationError('Proposal shard descriptor fields are invalid.')


def _prefix_part_paths(value: Any, prefix: str) -> None:
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256'}:
            value['path'] = f"{prefix}/{value['path']}"
            return
        if value.get('encoding') == CHUNKED_JSON_ENCODING:
            value['path_prefix'] = f"{prefix}/{value['path_prefix']}"
            return
        if value.get('format') == COLLECTION_FORMAT:
            value['manifest_prefix'] = f"{prefix}/{value['manifest_prefix']}"
            return
        for nested in value.values():
            _prefix_part_paths(nested, prefix)
    elif isinstance(value, list):
        for nested in value:
            _prefix_part_paths(nested, prefix)
