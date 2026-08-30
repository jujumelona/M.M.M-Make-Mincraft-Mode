from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_exact(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_exact(
    "minecraft_mod_ai/forced_tool_execution_contract.py",
    '''import hashlib\nimport json\nimport threading\n''',
    '''import hashlib\nimport json\nimport os\nimport threading\nimport time\n''',
)
replace_exact(
    "minecraft_mod_ai/forced_tool_execution_contract.py",
    '''_NATIVE_PROBE_TOOL = "mmm_required_tool_probe"\n_NATIVE_PROBE_LOCK = threading.RLock()\n_NATIVE_PROBE_CACHE: dict[tuple[str, str], bool] = {}\n''',
    '''_NATIVE_PROBE_TOOL = "mmm_required_tool_probe"\n_NATIVE_PROBE_LOCK = threading.RLock()\n_NATIVE_PROBE_CACHE: dict[tuple[str, str], bool] = {}\n_NATIVE_PROBE_NEGATIVE_AT: dict[tuple[str, str], float] = {}\n_NATIVE_PROBE_TRANSIENT_AT: dict[tuple[str, str], float] = {}\n_NATIVE_PROBE_KEY_LOCKS: dict[tuple[str, str], threading.Lock] = {}\n_DEFAULT_NATIVE_NEGATIVE_TTL_SECONDS = 60.0\n_DEFAULT_NATIVE_TRANSIENT_COOLDOWN_SECONDS = 5.0\n\n\ndef _positive_seconds(name: str, default: float) -> float:\n    raw = os.environ.get(name, "").strip()\n    if not raw:\n        return default\n    try:\n        value = float(raw)\n    except ValueError:\n        return default\n    return value if value > 0 else default\n\n\ndef _native_probe_key_lock(key: tuple[str, str]) -> threading.Lock:\n    with _NATIVE_PROBE_LOCK:\n        lock = _NATIVE_PROBE_KEY_LOCKS.get(key)\n        if lock is None:\n            lock = threading.Lock()\n            _NATIVE_PROBE_KEY_LOCKS[key] = lock\n        return lock\n''',
)
old = '''def _native_required_supported(current: Any, adapter: Any, request: Any) -> bool:\n    key = _native_probe_key(adapter, request)\n    if key is None:\n        return False\n    with _NATIVE_PROBE_LOCK:\n        cached = _NATIVE_PROBE_CACHE.get(key)\n    if cached is not None:\n        return cached\n\n    supported = False\n    try:\n        turn = current(adapter, _native_probe_request(request))\n        if _contains_exact_call(turn, _NATIVE_PROBE_TOOL):\n            call = tuple(getattr(turn, "tool_calls", ()) or ())[0]\n            arguments = getattr(call, "arguments", {})\n            supported = isinstance(arguments, Mapping) and arguments.get("nonce") == "mmm"\n    except Exception:\n        supported = False\n\n    with _NATIVE_PROBE_LOCK:\n        _NATIVE_PROBE_CACHE[key] = supported\n    print(\n        "llama native forced-tool preflight:",\n        f" supported={'yes' if supported else 'no'}",\n        f" model={key[1]}",\n        flush=True,\n    )\n    return supported\n\n\ndef _native_probe_cache_key(adapter: Any, request: Any) -> tuple[str, str] | None:\n    return _native_probe_key(adapter, request)\n\n\ndef _mark_native_unsupported(adapter: Any, request: Any) -> None:\n    key = _native_probe_cache_key(adapter, request)\n    if key is not None:\n        with _NATIVE_PROBE_LOCK:\n            _NATIVE_PROBE_CACHE[key] = False\n'''
new = '''def _native_required_supported(current: Any, adapter: Any, request: Any) -> bool:\n    key = _native_probe_key(adapter, request)\n    if key is None:\n        return False\n    negative_ttl = _positive_seconds(\n        "MMM_LLAMA_NATIVE_TOOL_NEGATIVE_TTL_SECONDS",\n        _DEFAULT_NATIVE_NEGATIVE_TTL_SECONDS,\n    )\n    transient_cooldown = _positive_seconds(\n        "MMM_LLAMA_NATIVE_TOOL_TRANSIENT_COOLDOWN_SECONDS",\n        _DEFAULT_NATIVE_TRANSIENT_COOLDOWN_SECONDS,\n    )\n\n    # Serialize probes only per endpoint/model. Different local models remain concurrent,\n    # while duplicate simultaneous requests do not each launch the same capability decode.\n    with _native_probe_key_lock(key):\n        now = time.monotonic()\n        with _NATIVE_PROBE_LOCK:\n            cached = _NATIVE_PROBE_CACHE.get(key)\n            negative_at = _NATIVE_PROBE_NEGATIVE_AT.get(key)\n            transient_at = _NATIVE_PROBE_TRANSIENT_AT.get(key)\n            if cached is True:\n                return True\n            if cached is False and negative_at is not None:\n                if now - negative_at < negative_ttl:\n                    return False\n                _NATIVE_PROBE_CACHE.pop(key, None)\n                _NATIVE_PROBE_NEGATIVE_AT.pop(key, None)\n            elif cached is False:\n                # Reprobe legacy unbounded negative entries instead of inheriting a\n                # permanent false capability state.\n                _NATIVE_PROBE_CACHE.pop(key, None)\n            if transient_at is not None and now - transient_at < transient_cooldown:\n                return False\n\n        supported = False\n        try:\n            turn = current(adapter, _native_probe_request(request))\n            if _contains_exact_call(turn, _NATIVE_PROBE_TOOL):\n                call = next(iter(getattr(turn, "tool_calls", ()) or ()))\n                arguments = getattr(call, "arguments", {})\n                supported = (\n                    isinstance(arguments, Mapping)\n                    and arguments.get("nonce") == "mmm"\n                )\n        except Exception as exc:  # noqa: BLE001 - capability transport/protocol boundary\n            with _NATIVE_PROBE_LOCK:\n                if _native_protocol_failure(exc):\n                    _NATIVE_PROBE_CACHE[key] = False\n                    _NATIVE_PROBE_NEGATIVE_AT[key] = now\n                    _NATIVE_PROBE_TRANSIENT_AT.pop(key, None)\n                    reason = "protocol"\n                    retry_after = negative_ttl\n                else:\n                    _NATIVE_PROBE_CACHE.pop(key, None)\n                    _NATIVE_PROBE_TRANSIENT_AT[key] = now\n                    reason = "transient"\n                    retry_after = transient_cooldown\n            print(\n                "llama native forced-tool preflight:",\n                " supported=unknown" if reason == "transient" else " supported=no",\n                f" reason={reason}",\n                f" model={key[1]}",\n                f" retry_after={retry_after:.0f}s",\n                flush=True,\n            )\n            return False\n\n        with _NATIVE_PROBE_LOCK:\n            _NATIVE_PROBE_TRANSIENT_AT.pop(key, None)\n            _NATIVE_PROBE_CACHE[key] = supported\n            if supported:\n                _NATIVE_PROBE_NEGATIVE_AT.pop(key, None)\n            else:\n                _NATIVE_PROBE_NEGATIVE_AT[key] = now\n        print(\n            "llama native forced-tool preflight:",\n            f" supported={'yes' if supported else 'no'}",\n            f" model={key[1]}",\n            "" if supported else f" retry_after={negative_ttl:.0f}s",\n            sep="",\n            flush=True,\n        )\n        return supported\n\n\ndef _native_probe_cache_key(adapter: Any, request: Any) -> tuple[str, str] | None:\n    return _native_probe_key(adapter, request)\n\n\ndef _mark_native_unsupported(adapter: Any, request: Any) -> None:\n    key = _native_probe_cache_key(adapter, request)\n    if key is not None:\n        with _NATIVE_PROBE_LOCK:\n            _NATIVE_PROBE_CACHE[key] = False\n            _NATIVE_PROBE_NEGATIVE_AT[key] = time.monotonic()\n            _NATIVE_PROBE_TRANSIENT_AT.pop(key, None)\n'''
replace_exact("minecraft_mod_ai/forced_tool_execution_contract.py", old, new)

# Do not hide unexpected jsonschema implementation faults. Known invalid-schema/value
# failures are handled immediately above; anything else should fail loudly.
replace_exact(
    "minecraft_mod_ai/forced_tool_execution_contract.py",
    '''    except (SchemaError, TypeError, ValueError, KeyError, RecursionError):\n        return False\n    except Exception:\n        return False\n''',
    '''    except (SchemaError, TypeError, ValueError, KeyError, RecursionError):\n        return False\n''',
)
# Endpoint resolution is an adapter/backend boundary: inability to resolve a local URL
# simply means native capability probing is unavailable for this request.
replace_exact(
    "minecraft_mod_ai/forced_tool_execution_contract.py",
    '''    try:\n        endpoint = str(adapter._server_url(request)).strip().rstrip("/")\n    except Exception:\n        return None\n''',
    '''    try:\n        endpoint = str(adapter._server_url(request)).strip().rstrip("/")\n    except Exception:  # noqa: BLE001 - optional adapter endpoint boundary\n        return None\n''',
)

replace_exact(
    "minecraft_mod_ai/runtime_bootstrap.py",
    '''    from .llama_forced_tool_capability_contract import (\n        install as install_forced_tool_capability,\n    )\n''',
    "",
)
replace_exact(
    "minecraft_mod_ai/runtime_bootstrap.py",
    "    install_forced_tool_capability(forced_tool_execution_contract)\n",
    "",
)
replace_exact(
    "minecraft_mod_ai/runtime_bootstrap.py",
    "        forced_tool_execution_contract,\n",
    "",
)

shim = ROOT / "minecraft_mod_ai/llama_forced_tool_capability_contract.py"
if not shim.exists():
    raise SystemExit("forced-tool capability shim unexpectedly missing")
shim.unlink()

(ROOT / "tests/test_llama_forced_tool_capability_contract.py").write_text(
    '''from __future__ import annotations\n\nimport threading\nimport time\nfrom types import SimpleNamespace\n\nfrom minecraft_mod_ai import forced_tool_execution_contract as forced\n\n\ndef _reset() -> None:\n    with forced._NATIVE_PROBE_LOCK:\n        forced._NATIVE_PROBE_CACHE.clear()\n        forced._NATIVE_PROBE_NEGATIVE_AT.clear()\n        forced._NATIVE_PROBE_TRANSIENT_AT.clear()\n        forced._NATIVE_PROBE_KEY_LOCKS.clear()\n\n\ndef _adapter():\n    return SimpleNamespace(\n        config=SimpleNamespace(model_id="model"),\n        _server_url=lambda request: "http://127.0.0.1:8080",\n    )\n\n\ndef _success():\n    return SimpleNamespace(\n        tool_calls=(SimpleNamespace(name=forced._NATIVE_PROBE_TOOL, arguments={"nonce": "mmm"}),)\n    )\n\n\ndef test_transient_failure_cools_down_without_poisoning(monkeypatch) -> None:\n    _reset()\n    monkeypatch.setenv("MMM_LLAMA_NATIVE_TOOL_TRANSIENT_COOLDOWN_SECONDS", "5")\n    times = iter((0.0, 2.0, 6.0))\n    monkeypatch.setattr(forced.time, "monotonic", lambda: next(times))\n    calls = 0\n\n    def current(_adapter, _request):\n        nonlocal calls\n        calls += 1\n        if calls == 1:\n            raise RuntimeError("temporary transport failure")\n        return _success()\n\n    adapter, request = _adapter(), object()\n    assert forced._native_required_supported(current, adapter, request) is False\n    assert forced._NATIVE_PROBE_CACHE == {}\n    assert forced._native_required_supported(current, adapter, request) is False\n    assert calls == 1\n    assert forced._native_required_supported(current, adapter, request) is True\n    assert calls == 2\n\n\ndef test_protocol_negative_expires(monkeypatch) -> None:\n    _reset()\n    monkeypatch.setenv("MMM_LLAMA_NATIVE_TOOL_NEGATIVE_TTL_SECONDS", "60")\n    times = iter((0.0, 30.0, 61.0))\n    monkeypatch.setattr(forced.time, "monotonic", lambda: next(times))\n    monkeypatch.setattr(forced, "_native_protocol_failure", lambda exc: True)\n    calls = 0\n\n    def current(_adapter, _request):\n        nonlocal calls\n        calls += 1\n        if calls == 1:\n            raise RuntimeError("protocol rejected")\n        return _success()\n\n    adapter, request = _adapter(), object()\n    assert forced._native_required_supported(current, adapter, request) is False\n    assert forced._native_required_supported(current, adapter, request) is False\n    assert calls == 1\n    assert forced._native_required_supported(current, adapter, request) is True\n    assert calls == 2\n\n\ndef test_mark_native_unsupported_is_ttl_scoped(monkeypatch) -> None:\n    _reset()\n    monkeypatch.setattr(forced.time, "monotonic", lambda: 10.0)\n    adapter, request = _adapter(), object()\n    forced._mark_native_unsupported(adapter, request)\n    key = ("http://127.0.0.1:8080", "model")\n    assert forced._NATIVE_PROBE_CACHE[key] is False\n    assert forced._NATIVE_PROBE_NEGATIVE_AT[key] == 10.0\n\n\ndef test_concurrent_probe_is_deduplicated_per_endpoint() -> None:\n    _reset()\n    calls = 0\n    calls_lock = threading.Lock()\n    barrier = threading.Barrier(4)\n\n    def current(_adapter, _request):\n        nonlocal calls\n        with calls_lock:\n            calls += 1\n        time.sleep(0.03)\n        return _success()\n\n    adapter, request = _adapter(), object()\n    results: list[bool] = []\n\n    def run() -> None:\n        barrier.wait()\n        results.append(forced._native_required_supported(current, adapter, request))\n\n    threads = [threading.Thread(target=run) for _ in range(4)]\n    for thread in threads:\n        thread.start()\n    for thread in threads:\n        thread.join()\n\n    assert results == [True] * 4\n    assert calls == 1\n''',
    encoding="utf-8",
)

for base in (ROOT / "minecraft_mod_ai", ROOT / "tests"):
    for path in base.rglob("*.py"):
        if "llama_forced_tool_capability_contract" in path.read_text(encoding="utf-8"):
            raise SystemExit(f"stale forced capability shim reference: {path.relative_to(ROOT)}")

(ROOT / ".github/workflows/worker05-forced-capability-canonical-v1.yml").unlink(missing_ok=True)
Path(__file__).unlink(missing_ok=True)
