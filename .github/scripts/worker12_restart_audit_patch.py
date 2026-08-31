from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one anchor, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "minecraft_mod_ai/trajectory_memory.py",
    '''    identity = str(row.get("trajectory_id", ""))\n    recent: deque[str] = deque(maxlen=512)\n    if path.is_file() and not path.is_symlink():\n        try:\n            with path.open("r", encoding="utf-8") as handle:\n                for raw in handle:\n                    try:\n                        value = json.loads(raw)\n                    except json.JSONDecodeError:\n                        continue\n                    if isinstance(value, Mapping):\n                        recent.append(str(value.get("trajectory_id", "")))\n        except OSError:\n            return False\n    if identity in recent:\n        return False\n''',
    '''    identity = str(row.get("trajectory_id", ""))\n    if path.is_file() and not path.is_symlink():\n        try:\n            with path.open("r", encoding="utf-8") as handle:\n                for raw in handle:\n                    try:\n                        value = json.loads(raw)\n                    except json.JSONDecodeError:\n                        continue\n                    if (\n                        isinstance(value, Mapping)\n                        and str(value.get("trajectory_id", "")) == identity\n                    ):\n                        return False\n        except OSError:\n            return False\n''',
)

replace_once(
    "minecraft_mod_ai/agentic_optimization_contract.py",
    '''def _read_memory(root: Path, signature: str, *, limit: int=4) -> list[dict[str, Any]]:\n    path = _memory_path(root)\n    if not path.is_file() or path.is_symlink():\n        return []\n    target = _tokens(signature)\n    rows: deque[dict[str, Any]] = deque(maxlen=256)\n    try:\n        with path.open('r', encoding='utf-8') as handle:\n            for raw in handle:\n                try:\n                    value = json.loads(raw)\n                except json.JSONDecodeError:\n                    continue\n                if isinstance(value, dict):\n                    rows.append(value)\n    except OSError:\n        return []\n''',
    '''def _recent_jsonl_rows(path: Path, *, max_rows: int = 256) -> list[dict[str, Any]]:\n    if max_rows <= 0 or not path.is_file() or path.is_symlink():\n        return []\n    chunks: list[bytes] = []\n    newline_count = 0\n    try:\n        with path.open("rb") as handle:\n            handle.seek(0, os.SEEK_END)\n            position = handle.tell()\n            while position > 0 and newline_count <= max_rows:\n                size = min(64 * 1024, position)\n                position -= size\n                handle.seek(position)\n                chunk = handle.read(size)\n                chunks.append(chunk)\n                newline_count += chunk.count(b"\\n")\n    except OSError:\n        return []\n    rows: list[dict[str, Any]] = []\n    for raw in b"".join(reversed(chunks)).splitlines()[-max_rows:]:\n        try:\n            value = json.loads(raw.decode("utf-8"))\n        except (UnicodeDecodeError, json.JSONDecodeError):\n            continue\n        if isinstance(value, dict):\n            rows.append(value)\n    return rows\n\n\ndef _read_memory(root: Path, signature: str, *, limit: int=4) -> list[dict[str, Any]]:\n    path = _memory_path(root)\n    if not path.is_file() or path.is_symlink():\n        return []\n    target = _tokens(signature)\n    rows = _recent_jsonl_rows(path, max_rows=256)\n''',
)

replace_once(
    "minecraft_mod_ai/agentic_optimization_contract.py",
    '''    existing: set[str] = set()\n    if path.is_file():\n        try:\n            with path.open('r', encoding='utf-8') as handle:\n                for raw in handle:\n                    try:\n                        value = json.loads(raw)\n                    except json.JSONDecodeError:\n                        continue\n                    if isinstance(value, dict):\n                        existing.add(str(value.get('experience_id', '')))\n        except OSError:\n            pass\n    if body['experience_id'] in existing:\n        return\n''',
    '''    if path.is_file():\n        try:\n            with path.open('r', encoding='utf-8') as handle:\n                for raw in handle:\n                    try:\n                        value = json.loads(raw)\n                    except json.JSONDecodeError:\n                        continue\n                    if (\n                        isinstance(value, dict)\n                        and str(value.get('experience_id', '')) == body['experience_id']\n                    ):\n                        return\n        except OSError:\n            pass\n''',
)

replace_once(
    "minecraft_mod_ai/agentic_optimization_contract.py",
    '''                except BaseException as exc:\n                    errors.append(exc)\n                    continue\n''',
    '''                except Exception as exc:\n                    errors.append(exc)\n                    continue\n''',
)

replace_once(
    "tools/colab_runtime_setup.py",
    '''def _safe_remote_url(value: str) -> str:\n    parsed = urlsplit(value.strip())\n    if not parsed.scheme or not parsed.hostname:\n        return ""\n    host = parsed.hostname\n    if ":" in host and not host.startswith("["):\n        host = f"[{host}]"\n    if parsed.port is not None:\n        host = f"{host}:{parsed.port}"\n    return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))\n''',
    '''def _safe_remote_url(value: str) -> str:\n    parsed = urlsplit(value.strip())\n    if not parsed.scheme or not parsed.hostname:\n        return ""\n    try:\n        port = parsed.port\n    except ValueError:\n        return ""\n    host = parsed.hostname\n    if ":" in host and not host.startswith("["):\n        host = f"[{host}]"\n    if port is not None:\n        host = f"{host}:{port}"\n    return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))\n''',
)

replace_once(
    "tools/colab_runtime_setup.py",
    '''    request = {\n        "setup_api_version": SETUP_API_VERSION,\n        "repo_dir": str(Path(repo_dir).resolve()),\n        "used_commit": used_commit.strip(),\n        "model_profile": model_profile.strip(),\n        "save_to_google_drive": bool(save_to_google_drive),\n        "remote_base_url": remote_base_url.strip(),\n        "remote_text_model": remote_text_model.strip(),\n        "remote_image_model": remote_image_model.strip(),\n        "remote_speech_model": remote_speech_model.strip(),\n        "llama_server_source_ref": (\n            LLAMA_SERVER_SOURCE_REF if _is_local_profile(model_profile) else ""\n        ),\n    }\n''',
    '''    local_profile = _is_local_profile(model_profile)\n    request = {\n        "setup_api_version": SETUP_API_VERSION,\n        "repo_dir": str(Path(repo_dir).resolve()),\n        "used_commit": used_commit.strip(),\n        "model_profile": model_profile.strip(),\n        "save_to_google_drive": bool(save_to_google_drive),\n        "remote_base_url": "" if local_profile else remote_base_url.strip(),\n        "remote_text_model": "" if local_profile else remote_text_model.strip(),\n        "remote_image_model": "" if local_profile else remote_image_model.strip(),\n        "remote_speech_model": "" if local_profile else remote_speech_model.strip(),\n        "llama_server_source_ref": LLAMA_SERVER_SOURCE_REF if local_profile else "",\n    }\n''',
)

replace_once(
    "tools/colab_runtime_setup.py",
    '''    script_path = Path(__file__).resolve()\n    return {\n''',
    '''    script_path = Path(__file__).resolve()\n    local_profile = _is_local_profile(model_profile)\n    return {\n''',
)

replace_once(
    "tools/colab_runtime_setup.py",
    '''        "remote": {\n            "base_url": _safe_remote_url(remote_base_url),\n            "text_model": remote_text_model.strip(),\n            "image_model": remote_image_model.strip() or remote_text_model.strip(),\n            "speech_model": remote_speech_model.strip() or remote_text_model.strip(),\n        },\n''',
    '''        "remote": {\n            "base_url": "" if local_profile else _safe_remote_url(remote_base_url),\n            "text_model": "" if local_profile else remote_text_model.strip(),\n            "image_model": (\n                ""\n                if local_profile\n                else remote_image_model.strip() or remote_text_model.strip()\n            ),\n            "speech_model": (\n                ""\n                if local_profile\n                else remote_speech_model.strip() or remote_text_model.strip()\n            ),\n        },\n''',
)

replace_once(
    "tests/test_worker12_shared_core.py",
    '''    complete_orchestrator,\n    validation_checkpoint_policy,\n    agentic_optimization_contract,\n''',
    '''    complete_orchestrator,\n    validation_checkpoint_policy,\n    agentic_optimization_contract,\n    trajectory_memory,\n''',
)

append_anchor = '''def test_jdt_cache_fingerprint_tracks_canonical_diagnostic_policy() -> None:\n    modules = validation_checkpoint_policy._validation_modules("validate-jdt")\n    assert validation_diagnostic_contract in modules\n    assert all(module.__name__ != "minecraft_mod_ai.orchestrator_jdt_gate_contract" for module in modules)\n'''
append_tests = append_anchor + '''\n\ndef test_trajectory_fallback_dedupe_checks_full_log(tmp_path: Path) -> None:\n    path = trajectory_memory.memory_path(tmp_path)\n    path.parent.mkdir(parents=True, exist_ok=True)\n    original = "".join(\n        f'{{"trajectory_id": "id-{index}"}}\\n' for index in range(600)\n    )\n    path.write_text(original, encoding="utf-8")\n\n    assert (\n        trajectory_memory._append_jsonl_fallback(\n            tmp_path, {"trajectory_id": "id-1"}\n        )\n        is False\n    )\n    assert path.read_text(encoding="utf-8") == original\n\n\ndef test_agentic_memory_tail_reader_is_bounded_to_recent_rows(tmp_path: Path) -> None:\n    path = tmp_path / "repair-experience.jsonl"\n    path.write_text(\n        "".join(f'{{"index": {index}}}\\n' for index in range(400)),\n        encoding="utf-8",\n    )\n    rows = agentic_optimization_contract._recent_jsonl_rows(path, max_rows=256)\n    assert len(rows) == 256\n    assert rows[0]["index"] == 144\n    assert rows[-1]["index"] == 399\n\n\ndef test_agentic_candidate_search_does_not_retry_keyboard_interrupt(monkeypatch) -> None:\n    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "on")\n    monkeypatch.setenv("MMM_REPAIR_SEARCH_WIDTH", "3")\n    calls = 0\n\n    class Engine:\n        def _signature(self, _evidence):\n            return "signature"\n\n        def _request_patch(self, _evidence, _context):\n            nonlocal calls\n            calls += 1\n            raise KeyboardInterrupt("cancelled")\n\n        def repair(self, *_args, **_kwargs):\n            return {"status": "FAIL"}\n\n    module = SimpleNamespace(RepairEngine=Engine, RepairEngineError=RuntimeError)\n    agentic_optimization_contract._install_repair_search_and_memory(module)\n\n    with pytest.raises(KeyboardInterrupt, match="cancelled"):\n        Engine()._request_patch({}, {})\n    assert calls == 1\n\n\ndef test_local_colab_fingerprint_ignores_remote_only_inputs(tmp_path: Path) -> None:\n    common = {\n        "repo_dir": tmp_path,\n        "used_commit": "abc123",\n        "model_profile": "local_gpu",\n        "save_to_google_drive": False,\n    }\n    clean = colab_runtime_setup.setup_request_fingerprint(**common)\n    noisy = colab_runtime_setup.setup_request_fingerprint(\n        **common,\n        remote_base_url="https://example.invalid:not-a-port/v1",\n        remote_text_model="unused-text",\n        remote_image_model="unused-image",\n        remote_speech_model="unused-speech",\n    )\n    assert noisy == clean\n\n\ndef test_local_colab_receipt_does_not_parse_or_persist_remote_config(tmp_path: Path) -> None:\n    receipt = colab_runtime_setup._build_receipt(\n        repo_dir=tmp_path,\n        used_commit="abc123",\n        model_profile="local_gpu",\n        save_to_google_drive=False,\n        output_root=str(tmp_path),\n        remote_base_url="https://example.invalid:not-a-port/v1",\n        remote_text_model="unused-text",\n        remote_image_model="unused-image",\n        remote_speech_model="unused-speech",\n        setup_fingerprint="fingerprint",\n        torch=None,\n        llama_server_binary="",\n    )\n    assert receipt["remote"] == {\n        "base_url": "",\n        "text_model": "",\n        "image_model": "",\n        "speech_model": "",\n    }\n    assert colab_runtime_setup._safe_remote_url(\n        "https://example.invalid:not-a-port/v1"\n    ) == ""\n'''
replace_once("tests/test_worker12_shared_core.py", append_anchor, append_tests)
