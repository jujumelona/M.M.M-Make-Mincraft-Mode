from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        if new in text:
            return text
        raise SystemExit(f"{label}: old and new blocks are both missing")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one old block, found {count}")
    return text.replace(old, new, 1)


path = ROOT / "minecraft_mod_ai" / "agentic_optimization_contract.py"
text = path.read_text(encoding="utf-8")

text = replace_once(
    text,
    "import time\n",
    "import time\nimport threading\n",
    "repair memory lock import",
)

text = replace_once(
    text,
    "from .preference_training import PreferenceCandidate, PreferenceTraceStore\n",
    "from .preference_training import PreferenceCandidate, PreferenceTraceStore\n"
    "from .validation_diagnostic_contract import (\n"
    "    diagnostic_errors,\n"
    "    diagnostic_items,\n"
    "    run_diagnostics,\n"
    ")\n",
    "canonical diagnostic imports",
)

text = replace_once(
    text,
    "def _compact_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:\n"
    "    try:\n"
    "        from .repair_diagnostics_contract import flatten_diagnostics\n"
    "        diagnostics = flatten_diagnostics(evidence.get('diagnostics'))\n"
    "    except Exception:\n"
    "        raw = evidence.get('diagnostics', {})\n"
    "        diagnostics = raw.get('diagnostics', []) if isinstance(raw, Mapping) else []\n"
    "        diagnostics = diagnostics if isinstance(diagnostics, list) else []\n",
    "def _diagnostic_receipt(value: Any) -> Mapping[str, Any] | None:\n"
    "    if isinstance(value, Mapping):\n"
    "        return value\n"
    "    if isinstance(value, list):\n"
    "        return {'diagnostics': value}\n"
    "    return None\n\n"
    "def _compact_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:\n"
    "    diagnostics = diagnostic_items(_diagnostic_receipt(evidence.get('diagnostics')))\n",
    "compact evidence diagnostic authority",
)

text = replace_once(
    text,
    "    try:\n"
    "        from .repair_diagnostics_contract import diagnostic_errors\n"
    "        errors = diagnostic_errors(evidence.get('diagnostics'))\n"
    "    except Exception:\n"
    "        errors = []\n",
    "    errors = diagnostic_errors(_diagnostic_receipt(evidence.get('diagnostics')))\n",
    "candidate count diagnostic authority",
)

text = replace_once(
    text,
    "def _diagnostic_paths(evidence: Mapping[str, Any]) -> set[str]:\n"
    "    try:\n"
    "        from .repair_diagnostics_contract import flatten_diagnostics\n"
    "        values = flatten_diagnostics(evidence.get('diagnostics'))\n"
    "    except Exception:\n"
    "        values = []\n",
    "def _diagnostic_paths(evidence: Mapping[str, Any]) -> set[str]:\n"
    "    values = diagnostic_items(_diagnostic_receipt(evidence.get('diagnostics')))\n",
    "diagnostic path authority",
)

old_local_import = (
    "        from .source_patch import TransactionalSourcePatcher\n"
    "        from .validation_diagnostic_contract import diagnostic_errors, run_diagnostics\n"
)
if old_local_import in text:
    if text.count(old_local_import) != 1:
        raise SystemExit("remove verifier local diagnostic import: duplicate old block")
    text = text.replace(
        old_local_import,
        "        from .source_patch import TransactionalSourcePatcher\n",
        1,
    )
elif "        from .validation_diagnostic_contract import diagnostic_errors, run_diagnostics\n" in text:
    raise SystemExit("remove verifier local diagnostic import: unexpected placement")

old_memory_read = '''def _memory_path(root: Path) -> Path:
    return root / '.minecraft_ai' / 'repair-experience.jsonl'

def _read_memory(root: Path, signature: str, *, limit: int=4) -> list[dict[str, Any]]:
    path = _memory_path(root)
    if not path.is_file() or path.is_symlink():
        return []
    target = _tokens(signature)
    rows: deque[dict[str, Any]] = deque(maxlen=256)
    try:
        with path.open('r', encoding='utf-8') as handle:
            for raw in handle:
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        return []
    ranked: list[tuple[float, str, dict[str, Any]]] = []
'''
new_memory_read = '''_REPAIR_MEMORY_MAX_ROWS = 256
_REPAIR_MEMORY_MAX_BYTES = 32 * 1024 * 1024
_REPAIR_MEMORY_LOCK = threading.RLock()

def _memory_path(root: Path) -> Path:
    root = root.expanduser().resolve()
    state_dir = root / '.minecraft_ai'
    if state_dir.is_symlink():
        raise RuntimeError('Repair experience state must not traverse symbolic links.')
    if state_dir.exists() and not state_dir.is_dir():
        raise RuntimeError(f'Repair experience state is not a directory: {state_dir}')
    path = state_dir / 'repair-experience.jsonl'
    if path.is_symlink():
        raise RuntimeError('Repair experience log must not be a symbolic link.')
    if path.exists() and not path.is_file():
        raise RuntimeError(f'Repair experience state is not a file: {path}')
    try:
        path.resolve(strict=False).relative_to(root)
    except (OSError, ValueError) as exc:
        raise RuntimeError('Repair experience state escaped the project root.') from exc
    return path

def _recent_memory_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise RuntimeError('Repair experience log is not a regular project file.')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise RuntimeError(f'Unable to open repair experience log safely: {exc}') from exc
    with os.fdopen(descriptor, 'rb') as handle:
        size = os.fstat(handle.fileno()).st_size
        start = max(0, size - _REPAIR_MEMORY_MAX_BYTES)
        handle.seek(start)
        payload = handle.read(_REPAIR_MEMORY_MAX_BYTES)
    if start:
        boundary = payload.find(b'\n')
        payload = b'' if boundary < 0 else payload[boundary + 1:]
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
    path = _memory_path(root)
    target = _tokens(signature)
    rows = _recent_memory_rows(path)
    ranked: list[tuple[float, str, dict[str, Any]]] = []
'''
text = replace_once(text, old_memory_read, new_memory_read, "secure bounded repair memory read")

old_memory_write = '''def _write_memory(root: Path, trace: Mapping[str, Any]) -> None:
    path = _memory_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {'schema_version': 'mmm/verified-repair-experience-v1', 'signature': trace.get('signature', ''), 'signature_sha256': _sha(str(trace.get('signature', ''))), 'evidence': trace.get('evidence', {}), 'repair_pattern': trace.get('repair_pattern', []), 'verifier': trace.get('winner_verifier', {})}
    body['experience_id'] = _sha(body)
    existing: set[str] = set()
    if path.is_file():
        try:
            with path.open('r', encoding='utf-8') as handle:
                for raw in handle:
                    try:
                        value = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        existing.add(str(value.get('experience_id', '')))
        except OSError:
            pass
    if body['experience_id'] in existing:
        return
    with path.open('a', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(body, ensure_ascii=False, sort_keys=True) + '\n')
'''
new_memory_write = '''def _write_memory(root: Path, trace: Mapping[str, Any]) -> None:
    body = {'schema_version': 'mmm/verified-repair-experience-v1', 'signature': trace.get('signature', ''), 'signature_sha256': _sha(str(trace.get('signature', ''))), 'evidence': trace.get('evidence', {}), 'repair_pattern': trace.get('repair_pattern', []), 'verifier': trace.get('winner_verifier', {})}
    body['experience_id'] = _sha(body)
    with _REPAIR_MEMORY_LOCK:
        path = _memory_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path = _memory_path(root)
        rows = _recent_memory_rows(path)
        if any(str(row.get('experience_id', '')) == body['experience_id'] for row in rows):
            return
        rows = [*rows, body][-_REPAIR_MEMORY_MAX_ROWS:]
        rendered = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows)
        temporary = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
        try:
            with temporary.open('x', encoding='utf-8', newline='\n') as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            _memory_path(root)
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
'''
text = replace_once(text, old_memory_write, new_memory_write, "secure bounded repair memory write")

path.write_text(text, encoding="utf-8")


test_path = ROOT / "tests" / "test_worker12_shared_core.py"
tests = test_path.read_text(encoding="utf-8")
marker = "\ndef test_colab_profile_switch_removes_stale_remote_credentials(monkeypatch) -> None:\n"
new_tests = '''\n\ndef test_agentic_diagnostics_use_canonical_authority_without_legacy_adapter() -> None:\n    source = inspect.getsource(agentic_optimization_contract)\n    assert "repair_diagnostics_contract" not in source\n    receipt = {\n        "status": "AVAILABLE",\n        "diagnostics": {\n            "file:///src/main/java/A.java": [\n                {"severity": 1, "message": "compile error", "code": "E1"}\n            ]\n        },\n    }\n    compact = agentic_optimization_contract._compact_evidence(\n        {"diagnostics": receipt, "build": {"status": "PASS"}}\n    )\n    assert compact["diagnostics"] == [\n        {\n            "path": "file:///src/main/java/A.java",\n            "message": "compile error",\n            "code": "E1",\n            "severity": 1,\n        }\n    ]\n    assert agentic_optimization_contract._diagnostic_paths(\n        {"diagnostics": receipt}\n    ) == {"file:///src/main/java/A.java", "A.java"}\n'''
if "test_agentic_diagnostics_use_canonical_authority_without_legacy_adapter" not in tests:
    if marker not in tests:
        raise SystemExit("worker12 test insertion marker missing")
    tests = tests.replace(marker, new_tests + marker, 1)

memory_tests = '''\n\ndef test_repair_memory_rejects_symlinked_state_directory(tmp_path: Path) -> None:\n    outside = tmp_path / "outside"\n    outside.mkdir()\n    (tmp_path / ".minecraft_ai").symlink_to(outside, target_is_directory=True)\n    with pytest.raises(RuntimeError, match="symbolic"):\n        agentic_optimization_contract._read_memory(tmp_path, "compile error")\n    with pytest.raises(RuntimeError, match="symbolic"):\n        agentic_optimization_contract._write_memory(\n            tmp_path,\n            {"signature": "compile error", "winner_verifier": {"status": "PASS"}},\n        )\n    assert not (outside / "repair-experience.jsonl").exists()\n\n\ndef test_repair_memory_compacts_legacy_log_to_bounded_window(tmp_path: Path) -> None:\n    state = tmp_path / ".minecraft_ai"\n    state.mkdir()\n    path = state / "repair-experience.jsonl"\n    rows = [\n        {"experience_id": f"legacy-{index}", "signature": f"failure token {index}"}\n        for index in range(300)\n    ]\n    path.write_text(\n        "".join(__import__("json").dumps(row) + "\\n" for row in rows),\n        encoding="utf-8",\n    )\n    agentic_optimization_contract._write_memory(\n        tmp_path,\n        {\n            "signature": "fresh failure token",\n            "evidence": {},\n            "repair_pattern": [],\n            "winner_verifier": {"status": "PASS"},\n        },\n    )\n    compacted = [\n        __import__("json").loads(line)\n        for line in path.read_text(encoding="utf-8").splitlines()\n    ]\n    assert len(compacted) == agentic_optimization_contract._REPAIR_MEMORY_MAX_ROWS\n    assert compacted[0]["experience_id"] == "legacy-45"\n    assert compacted[-1]["signature"] == "fresh failure token"\n'''
if "test_repair_memory_rejects_symlinked_state_directory" not in tests:
    if marker not in tests:
        raise SystemExit("worker12 memory test insertion marker missing")
    tests = tests.replace(marker, memory_tests + marker, 1)

test_path.write_text(tests, encoding="utf-8")

print("worker12 native cleanup applied")
