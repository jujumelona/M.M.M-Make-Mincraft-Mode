from __future__ import annotations

from pathlib import Path


source = Path("minecraft_mod_ai/agentic_optimization_contract.py")
text = source.read_text(encoding="utf-8")
start = text.index("def _memory_path(root: Path) -> Path:\n")
end = text.index("def _repair_candidate_count(", start)
replacement = r'''_REPAIR_MEMORY_MAX_ROWS = 256
_REPAIR_MEMORY_MAX_BYTES = 4 * 1024 * 1024


def _memory_path(root: Path) -> Path:
    resolved_root = root.expanduser().resolve()
    parent = resolved_root / '.minecraft_ai'
    if parent.is_symlink():
        raise RuntimeError('Repair memory state must not traverse symbolic links.')
    if parent.exists() and not parent.is_dir():
        raise RuntimeError('Repair memory state parent is not a directory.')
    try:
        parent.resolve(strict=False).relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise RuntimeError('Repair memory state escaped the configured project root.') from exc
    path = parent / 'repair-experience.jsonl'
    if path.is_symlink():
        raise RuntimeError('Repair memory file must not be a symbolic link.')
    if path.exists() and not path.is_file():
        raise RuntimeError('Repair memory state leaf is not a file.')
    return path


def _read_recent_memory_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags)
    try:
        with os.fdopen(fd, 'rb') as handle:
            fd = -1
            size = os.fstat(handle.fileno()).st_size
            start = max(0, size - _REPAIR_MEMORY_MAX_BYTES)
            handle.seek(start)
            payload = handle.read(_REPAIR_MEMORY_MAX_BYTES)
    finally:
        if fd >= 0:
            os.close(fd)
    if start:
        separator = payload.find(b'\n')
        payload = payload[separator + 1:] if separator >= 0 else b''
    rows: deque[dict[str, Any]] = deque(maxlen=_REPAIR_MEMORY_MAX_ROWS)
    for raw in payload.splitlines():
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return list(rows)


def _read_memory(root: Path, signature: str, *, limit: int=4) -> list[dict[str, Any]]:
    try:
        path = _memory_path(root)
        rows = _read_recent_memory_rows(path)
    except (OSError, RuntimeError):
        return []
    target = _tokens(signature)
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for row in rows:
        source_signature = str(row.get('signature', ''))
        values = _tokens(source_signature)
        if not target or not values:
            similarity = 0.0
        else:
            similarity = len(target & values) / max(1, len(target | values))
        ranked.append((similarity, str(row.get('experience_id', '')), row))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [{'similarity': round(score, 6), 'signature_sha256': row.get('signature_sha256', ''), 'evidence': row.get('evidence', {}), 'repair_pattern': row.get('repair_pattern', [])} for score, _identity, row in ranked[:limit] if score > 0.0]


def _repair_pattern(operations: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in operations:
        path = str(item.get('path', ''))
        content = item.get('content')
        replacements = item.get('replacements')
        excerpt = ''
        if isinstance(content, str):
            excerpt = content[:4096]
        elif replacements is not None:
            excerpt = json.dumps(replacements, ensure_ascii=False)[:4096]
        result.append({'operation': str(item.get('operation', '')), 'path': path, 'repair_excerpt': excerpt})
    return result[:16]


def _atomic_write_memory(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
            fd = -1
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _write_memory(root: Path, trace: Mapping[str, Any]) -> None:
    path = _memory_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _memory_path(root)
    body = {'schema_version': 'mmm/verified-repair-experience-v1', 'signature': trace.get('signature', ''), 'signature_sha256': _sha(str(trace.get('signature', ''))), 'evidence': trace.get('evidence', {}), 'repair_pattern': trace.get('repair_pattern', []), 'verifier': trace.get('winner_verifier', {})}
    body['experience_id'] = _sha(body)
    rows = _read_recent_memory_rows(path)
    if any(str(row.get('experience_id', '')) == body['experience_id'] for row in rows):
        return
    bounded = [*rows[-(_REPAIR_MEMORY_MAX_ROWS - 1):], body]
    _atomic_write_memory(path, bounded)

'''
source.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


tests = Path("tests/test_worker12_shared_core.py")
test_text = tests.read_text(encoding="utf-8")
marker = "def test_jdt_cache_fingerprint_tracks_canonical_diagnostic_policy() -> None:\n"
addition = r'''def test_repair_memory_rejects_symlinked_leaf_without_touching_target(tmp_path: Path) -> None:
    project = tmp_path / 'project'
    state = project / '.minecraft_ai'
    outside = tmp_path / 'outside.jsonl'
    project.mkdir()
    state.mkdir()
    outside.write_text('sentinel\n', encoding='utf-8')
    try:
        (state / 'repair-experience.jsonl').symlink_to(outside)
    except OSError as exc:
        pytest.skip(f'symlink creation unavailable: {exc}')
    with pytest.raises(RuntimeError, match='symbolic link'):
        agentic_optimization_contract._write_memory(project, {'signature': 'unsafe'})
    assert outside.read_text(encoding='utf-8') == 'sentinel\n'


def test_repair_memory_rejects_symlinked_state_parent(tmp_path: Path) -> None:
    project = tmp_path / 'project'
    outside = tmp_path / 'outside'
    project.mkdir()
    outside.mkdir()
    try:
        (project / '.minecraft_ai').symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f'symlink creation unavailable: {exc}')
    with pytest.raises(RuntimeError, match='symbolic link'):
        agentic_optimization_contract._write_memory(project, {'signature': 'unsafe'})


def test_repair_memory_compacts_to_bounded_recent_window(tmp_path: Path) -> None:
    project = tmp_path / 'project'
    path = project / '.minecraft_ai' / 'repair-experience.jsonl'
    path.parent.mkdir(parents=True)
    with path.open('w', encoding='utf-8') as handle:
        for index in range(400):
            handle.write(json.dumps({'experience_id': f'old-{index}', 'signature': f's-{index}'}) + '\n')
    agentic_optimization_contract._write_memory(
        project,
        {'signature': 'new-signature', 'evidence': {}, 'repair_pattern': [], 'winner_verifier': {}},
    )
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == agentic_optimization_contract._REPAIR_MEMORY_MAX_ROWS
    assert rows[-1]['signature'] == 'new-signature'
    assert rows[0]['experience_id'] == 'old-145'
'''
if addition.strip() not in test_text:
    if marker not in test_text:
        raise RuntimeError("worker12 repair-memory test insertion marker missing")
    test_text = test_text.replace(marker, addition + "\n\n" + marker, 1)
if "import json\n" not in test_text.split("\n", 12):
    test_text = test_text.replace("import os\n", "import json\nimport os\n", 1)
tests.write_text(test_text, encoding="utf-8")

print("worker12 repair-memory hardening v2 applied")
