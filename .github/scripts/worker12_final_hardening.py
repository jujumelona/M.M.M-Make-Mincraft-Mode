from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one patch anchor, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, addition: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if addition.strip() in text:
        return
    if marker not in text:
        raise RuntimeError(f"{path}: test insertion marker missing")
    target.write_text(text.replace(marker, addition + "\n\n" + marker, 1), encoding="utf-8")


replace_once(
    "minecraft_mod_ai/validation_diagnostic_contract.py",
    '''def _availability_error(receipt: Mapping[str, Any]) -> dict[str, Any] | None:\n    """Return a blocking diagnostic when a JDT receipt is not trustworthy."""\n\n    status = str(receipt.get("status") or "").strip().upper()\n    error = str(receipt.get("error") or "").strip()\n    raw = receipt.get("diagnostics")\n    malformed = "diagnostics" not in receipt or not isinstance(raw, (Mapping, list))\n    failed_status = bool(status and status not in _DIAGNOSTIC_SUCCESS_STATUSES)\n    if not (failed_status or error or malformed):\n        return None\n\n    details: list[str] = []\n    if status:\n        details.append(f"status={status}")\n    if error:\n        details.append(error)\n    if malformed:\n        details.append("diagnostics payload is missing or malformed")\n    return {\n        "severity": 1,\n        "source": "jdtls",\n        "code": "JDT_DIAGNOSTICS_UNAVAILABLE",\n        "message": "JDT diagnostics are unavailable: " + "; ".join(details),\n    }\n''',
    '''def _diagnostics_shape_error(raw: Any) -> str:\n    if isinstance(raw, Mapping):\n        for uri, group in raw.items():\n            if not isinstance(group, list):\n                return f"diagnostics group for {uri!r} is not a list"\n            if any(not isinstance(item, Mapping) for item in group):\n                return f"diagnostics group for {uri!r} contains a non-mapping item"\n        return ""\n    if isinstance(raw, list):\n        if any(not isinstance(item, Mapping) for item in raw):\n            return "diagnostics list contains a non-mapping item"\n        return ""\n    return "diagnostics payload is missing or malformed"\n\n\ndef _availability_error(receipt: Mapping[str, Any]) -> dict[str, Any] | None:\n    """Return a blocking diagnostic when a JDT receipt is not trustworthy."""\n\n    status = str(receipt.get("status") or "").strip().upper()\n    error = str(receipt.get("error") or "").strip()\n    malformed = (\n        "diagnostics payload is missing or malformed"\n        if "diagnostics" not in receipt\n        else _diagnostics_shape_error(receipt.get("diagnostics"))\n    )\n    failed_status = bool(status and status not in _DIAGNOSTIC_SUCCESS_STATUSES)\n    if not (failed_status or error or malformed):\n        return None\n\n    details: list[str] = []\n    if status:\n        details.append(f"status={status}")\n    if error:\n        details.append(error)\n    if malformed:\n        details.append(malformed)\n    return {\n        "severity": 1,\n        "source": "jdtls",\n        "code": "JDT_DIAGNOSTICS_UNAVAILABLE",\n        "message": "JDT diagnostics are unavailable: " + "; ".join(details),\n    }\n''',
)

replace_once(
    "minecraft_mod_ai/agentic_optimization_contract.py",
    '''        from .performance_final_contract import _clone_source_snapshot\n        from .repair_diagnostics_contract import diagnostic_errors\n        from .source_patch import TransactionalSourcePatcher\n        stage = _clone_source_snapshot(root)\n        TransactionalSourcePatcher(stage).apply([copy.deepcopy(dict(item)) for item in operations])\n        java_paths = tuple(sorted(str(item.get('path', '')).replace('\\\\', '/') for item in operations if str(item.get('path', '')).lower().endswith('.java')))\n        service = self.diagnostics_factory()\n        try:\n            diagnostics = service.diagnostics(stage, relative_files=java_paths or None, timeout_seconds=60)\n        except TypeError:\n            diagnostics = service.diagnostics(stage, timeout_seconds=60)\n''',
    '''        from .performance_final_contract import _clone_source_snapshot\n        from .repair_diagnostics_contract import diagnostic_errors\n        from .source_patch import TransactionalSourcePatcher\n        from .validation_diagnostic_contract import run_diagnostics\n        stage = _clone_source_snapshot(root)\n        TransactionalSourcePatcher(stage).apply([copy.deepcopy(dict(item)) for item in operations])\n        java_paths = tuple(sorted(str(item.get('path', '')).replace('\\\\', '/') for item in operations if str(item.get('path', '')).lower().endswith('.java')))\n        diagnostics = run_diagnostics(\n            self.diagnostics_factory,\n            stage,\n            relative_files=java_paths or None,\n            timeout_seconds=60,\n        )\n''',
)

replace_once(
    "minecraft_mod_ai/trajectory_memory.py",
    '''import hashlib\nimport json\nimport re\nimport sqlite3\n''',
    '''import hashlib\nimport json\nimport os\nimport re\nimport sqlite3\n''',
)

replace_once(
    "minecraft_mod_ai/trajectory_memory.py",
    '''def _memory_dir(base: str | Path) -> Path:\n    root = Path(base).expanduser().resolve()\n    current = root\n    for part in (".minecraft_ai", "trajectory-memory"):\n        current = current / part\n        if current.is_symlink():\n            raise RuntimeError("Trajectory memory state must not traverse symbolic links.")\n        if current.exists() and not current.is_dir():\n            raise RuntimeError("Trajectory memory state parent is not a directory.")\n        try:\n            current.resolve(strict=False).relative_to(root)\n        except (OSError, ValueError) as exc:\n            raise RuntimeError(\n                "Trajectory memory state escaped the configured project root."\n            ) from exc\n    return current\n\n\ndef memory_path(base: str | Path) -> Path:\n    return _memory_dir(base) / "verified-trajectories.jsonl"\n\n\ndef remote_cache_path(base: str | Path, task_class: str) -> Path:\n    safe = re.sub(r"[^a-z0-9_-]+", "-", task_class.casefold()).strip("-") or "general"\n    return _memory_dir(base) / "remote-cache" / f"{safe}.jsonl"\n\n\ndef _index_path(base: str | Path) -> Path:\n    return _memory_dir(base) / "trajectory-index.sqlite3"\n''',
    '''def _state_path(\n    base: str | Path,\n    *parts: str,\n    leaf_file: bool = False,\n) -> Path:\n    """Build a state path without allowing any component to redirect outside root."""\n\n    root = Path(base).expanduser().resolve()\n    current = root\n    for index, part in enumerate(parts):\n        current = current / part\n        is_leaf = index == len(parts) - 1\n        if current.is_symlink():\n            raise RuntimeError("Trajectory memory state must not traverse symbolic links.")\n        if current.exists():\n            expect_file = bool(leaf_file and is_leaf)\n            valid_type = current.is_file() if expect_file else current.is_dir()\n            if not valid_type:\n                kind = "file" if expect_file else "directory"\n                raise RuntimeError(f"Trajectory memory state is not a {kind}: {current}")\n        try:\n            current.resolve(strict=False).relative_to(root)\n        except (OSError, ValueError) as exc:\n            raise RuntimeError(\n                "Trajectory memory state escaped the configured project root."\n            ) from exc\n    return current\n\n\ndef _memory_dir(base: str | Path) -> Path:\n    return _state_path(base, ".minecraft_ai", "trajectory-memory")\n\n\ndef memory_path(base: str | Path) -> Path:\n    return _state_path(\n        base,\n        ".minecraft_ai",\n        "trajectory-memory",\n        "verified-trajectories.jsonl",\n        leaf_file=True,\n    )\n\n\ndef remote_cache_path(base: str | Path, task_class: str) -> Path:\n    safe = re.sub(r"[^a-z0-9_-]+", "-", task_class.casefold()).strip("-") or "general"\n    return _state_path(\n        base,\n        ".minecraft_ai",\n        "trajectory-memory",\n        "remote-cache",\n        f"{safe}.jsonl",\n        leaf_file=True,\n    )\n\n\ndef _index_path(base: str | Path) -> Path:\n    return _state_path(\n        base,\n        ".minecraft_ai",\n        "trajectory-memory",\n        "trajectory-index.sqlite3",\n        leaf_file=True,\n    )\n''',
)

replace_once(
    "minecraft_mod_ai/trajectory_memory.py",
    '''    source = source.expanduser().resolve()\n    previous = connection.execute(\n''',
    '''    source = source.expanduser()\n    if source.is_symlink():\n        raise RuntimeError("Trajectory source must not be a symbolic link.")\n    source = source.resolve()\n    previous = connection.execute(\n''',
)

replace_once(
    "minecraft_mod_ai/trajectory_memory.py",
    '''\ndef _append_jsonl_fallback(base: str | Path, row: Mapping[str, Any]) -> bool:\n''',
    '''\ndef _append_jsonl_line(path: Path, rendered: str) -> None:\n    """Append without following a leaf symlink when the platform supports it."""\n\n    if path.is_symlink():\n        raise RuntimeError("Trajectory memory file must not be a symbolic link.")\n    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)\n    fd = os.open(path, flags, 0o600)\n    try:\n        with os.fdopen(fd, "a", encoding="utf-8", newline="\\n") as handle:\n            fd = -1\n            handle.write(rendered)\n    finally:\n        if fd >= 0:\n            os.close(fd)\n\n\ndef _append_jsonl_fallback(base: str | Path, row: Mapping[str, Any]) -> bool:\n''',
)

replace_once(
    "minecraft_mod_ai/trajectory_memory.py",
    '''    path.parent.mkdir(parents=True, exist_ok=True)\n    with path.open("a", encoding="utf-8", newline="\\n") as handle:\n        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\\n")\n    return True\n''',
    '''    path.parent.mkdir(parents=True, exist_ok=True)\n    _append_jsonl_line(\n        path,\n        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\\n",\n    )\n    return True\n''',
)

replace_once(
    "minecraft_mod_ai/trajectory_memory.py",
    '''            path.parent.mkdir(parents=True, exist_ok=True)\n            rendered = json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\\n"\n            with path.open("a", encoding="utf-8", newline="\\n") as handle:\n                handle.write(rendered)\n\n            order = _last_source_order(connection) + 1\n''',
    '''            path.parent.mkdir(parents=True, exist_ok=True)\n            rendered = json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\\n"\n            _append_jsonl_line(path, rendered)\n\n            order = _last_source_order(connection) + 1\n''',
)

replace_once(
    "tools/colab_runtime_setup.py",
    '''def _validate_checkout(\n''',
    '''def _reset_inactive_profile_state(*, local_profile: bool) -> None:\n    """Quiesce runtime state owned by the profile being left."""\n\n    if not local_profile:\n        _shutdown_loaded_managed_llama_server()\n    _clear_inactive_profile_environment(local_profile=local_profile)\n\n\ndef _validate_checkout(\n''',
)

replace_once(
    "tools/colab_runtime_setup.py",
    '''    local_profile = _is_local_profile(profile)\n    _clear_inactive_profile_environment(local_profile=local_profile)\n''',
    '''    local_profile = _is_local_profile(profile)\n    _reset_inactive_profile_state(local_profile=local_profile)\n''',
)

replace_once(
    "tests/test_worker12_shared_core.py",
    '''import os\nfrom pathlib import Path\n''',
    '''import os\nimport sys\nfrom pathlib import Path\nfrom types import SimpleNamespace\n''',
)

replace_once(
    "tests/test_worker12_shared_core.py",
    '''    validation_diagnostic_contract,\n    validation_execution_contract,\n)\n''',
    '''    agentic_optimization_contract,\n    validation_diagnostic_contract,\n    validation_execution_contract,\n)\n''',
)

append_once(
    "tests/test_worker12_shared_core.py",
    "def test_jdt_operational_failure_becomes_unavailable_receipt() -> None:\n",
    '''def test_malformed_nested_jdt_payload_fails_closed() -> None:\n    grouped = diagnostic_errors(\n        {"status": "AVAILABLE", "diagnostics": {"file:///A.java": "not-a-list"}}\n    )\n    listed = diagnostic_errors(\n        {"status": "AVAILABLE", "diagnostics": [{"severity": 2}, "bad-item"]}\n    )\n    assert grouped[-1]["code"] == "JDT_DIAGNOSTICS_UNAVAILABLE"\n    assert "not a list" in grouped[-1]["message"]\n    assert listed[-1]["code"] == "JDT_DIAGNOSTICS_UNAVAILABLE"\n    assert "non-mapping" in listed[-1]["message"]\n''',
)

append_once(
    "tests/test_worker12_shared_core.py",
    "def test_colab_profile_switch_removes_stale_remote_credentials(monkeypatch) -> None:\n",
    '''def test_agentic_verifier_does_not_retry_programming_typeerror(\n    monkeypatch, tmp_path: Path\n) -> None:\n    root = tmp_path / "root"\n    stage = tmp_path / "stage"\n    root.mkdir()\n    stage.mkdir()\n\n    from minecraft_mod_ai import performance_final_contract\n\n    monkeypatch.setattr(\n        performance_final_contract,\n        "_clone_source_snapshot",\n        lambda _root: stage,\n    )\n    calls = 0\n\n    class Service:\n        def diagnostics(self, *_args, **_kwargs):\n            nonlocal calls\n            calls += 1\n            raise TypeError("internal programmer defect")\n\n    engine = SimpleNamespace(diagnostics_factory=lambda: Service())\n    _score, verifier = agentic_optimization_contract._verify_repair_candidate(\n        engine, root, [], {}\n    )\n    assert calls == 1\n    assert verifier["jdt_status"] == "VERIFIER_ERROR"\n    assert "programmer defect" in verifier["verifier_error"]\n''',
)

append_once(
    "tests/test_worker12_shared_core.py",
    "def test_colab_remote_profile_removes_stale_local_server_routing(monkeypatch) -> None:\n",
    '''def test_colab_remote_transition_stops_stale_managed_server(monkeypatch) -> None:\n    stopped = 0\n\n    def shutdown() -> None:\n        nonlocal stopped\n        stopped += 1\n\n    module = SimpleNamespace(\n        _MANAGED_URL="http://127.0.0.1:8123",\n        _shutdown_managed_server=shutdown,\n    )\n    monkeypatch.setitem(\n        sys.modules,\n        "minecraft_mod_ai.llama_server_autotune",\n        module,\n    )\n    monkeypatch.setenv("LLAMA_SERVER_URL", module._MANAGED_URL)\n    monkeypatch.setenv("MMM_LLAMA_SERVER_BIN", "/content/llama-server")\n\n    colab_runtime_setup._reset_inactive_profile_state(local_profile=False)\n\n    assert stopped == 1\n    assert "LLAMA_SERVER_URL" not in os.environ\n    assert "MMM_LLAMA_SERVER_BIN" not in os.environ\n''',
)

append_once(
    "tests/test_worker12_shared_core.py",
    "def test_jdt_cache_fingerprint_tracks_canonical_diagnostic_policy() -> None:\n",
    '''def test_trajectory_memory_rejects_symlinked_leaf_and_remote_cache_dir(\n    tmp_path: Path,\n) -> None:\n    project = tmp_path / "project"\n    state = project / ".minecraft_ai" / "trajectory-memory"\n    outside = tmp_path / "outside"\n    state.mkdir(parents=True)\n    outside.mkdir()\n    leaf_target = outside / "verified-trajectories.jsonl"\n    leaf_target.write_text("sentinel\\n", encoding="utf-8")\n    try:\n        (state / "verified-trajectories.jsonl").symlink_to(leaf_target)\n    except OSError as exc:\n        pytest.skip(f"symlink creation unavailable: {exc}")\n    with pytest.raises(RuntimeError, match="symbolic links"):\n        memory_path(project)\n\n    (state / "verified-trajectories.jsonl").unlink()\n    (state / "remote-cache").symlink_to(outside, target_is_directory=True)\n    with pytest.raises(RuntimeError, match="symbolic links"):\n        remote_cache_path(project, "repair")\n''',
)

print("worker12 final hardening patch applied")
