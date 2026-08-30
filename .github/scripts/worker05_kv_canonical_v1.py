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


def replace_expected(path: str, old: str, new: str, *, expected: int) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected:
        raise SystemExit(f"{path}: expected {expected} matches, found {count}")
    target.write_text(text.replace(old, new), encoding="utf-8")


# KV correctness is part of the decode-speed owner's semantics, not a runtime wrapper.
replace_exact(
    "minecraft_mod_ai/llama_decode_speed_contract.py",
    '_KV_SCHEMA_VERSION = "mmm/llama-kv-decode-speed-v2"\n',
    '_KV_SCHEMA_VERSION = "mmm/llama-kv-decode-speed-v3-precision-reference"\n_KV_PRECISION_RANK = {"f16": 0, "q8_0": 1, "q4_0": 2}\n',
)
replace_exact(
    "minecraft_mod_ai/llama_decode_speed_contract.py",
    '''def _kv_candidates() -> tuple[str, ...]:\n''',
    '''def _precision_reference_order(candidates: Any) -> tuple[str, ...]:\n    values = tuple(str(value) for value in candidates)\n    return tuple(\n        sorted(\n            values,\n            key=lambda value: (_KV_PRECISION_RANK.get(value, 100), value),\n        )\n    )\n\n\ndef _kv_candidates() -> tuple[str, ...]:\n''',
)
replace_exact(
    "minecraft_mod_ai/llama_decode_speed_contract.py",
    '        "candidates": list(candidates),\n',
    '        "candidates": list(_precision_reference_order(candidates)),\n',
)
replace_exact(
    "minecraft_mod_ai/llama_decode_speed_contract.py",
    '''def _probe_kv_types(\n    autotune: Any,\n    binary: str,\n    model_path: str,\n    config: Any,\n    request: Any,\n    candidates: tuple[str, ...],\n) -> tuple[str, list[dict[str, Any]]]:\n    bench_request = autotune._compact_benchmark_request(request)\n''',
    '''def _probe_kv_types(\n    autotune: Any,\n    binary: str,\n    model_path: str,\n    config: Any,\n    request: Any,\n    candidates: tuple[str, ...],\n) -> tuple[str, list[dict[str, Any]]]:\n    candidates = _precision_reference_order(candidates)\n    bench_request = autotune._compact_benchmark_request(request)\n''',
)

# Make cache parsing fail only for expected cache corruption/I/O conditions, rather than
# swallowing arbitrary programming errors. A non-object cache is simply invalid.
replace_exact(
    "minecraft_mod_ai/llama_decode_speed_contract.py",
    '''    try:\n        payload = json.loads(_kv_cache_path(autotune).read_text(encoding="utf-8"))\n    except Exception:\n        return None\n    selected = str(payload.get("selected", "")).strip().lower()\n''',
    '''    try:\n        payload = json.loads(_kv_cache_path(autotune).read_text(encoding="utf-8"))\n    except (OSError, UnicodeError, json.JSONDecodeError):\n        return None\n    if not isinstance(payload, dict):\n        return None\n    selected = str(payload.get("selected", "")).strip().lower()\n''',
)

# Probe loops intentionally isolate one hardware/runtime candidate from the next. Those
# boundaries must catch arbitrary backend failures, but make that policy explicit to Ruff.
replace_expected(
    "minecraft_mod_ai/llama_decode_speed_contract.py",
    "        except Exception as exc:\n",
    "        except Exception as exc:  # noqa: BLE001 - isolate optional benchmark candidate failures\n",
    expected=1,
)
replace_expected(
    "minecraft_mod_ai/llama_decode_speed_contract.py",
    "            except Exception as exc:\n",
    "            except Exception as exc:  # noqa: BLE001 - isolate optional KV candidate failures\n",
    expected=1,
)

# Runtime bootstrap no longer composes the deleted KV monkeypatch shim or imports its
# canonical owner merely for wrapper injection.
replace_exact(
    "minecraft_mod_ai/runtime_bootstrap.py",
    '        llama_decode_speed_contract,\n',
    "",
)
replace_exact(
    "minecraft_mod_ai/runtime_bootstrap.py",
    '    from .llama_kv_correctness_contract import install as install_kv_correctness\n',
    "",
)
replace_exact(
    "minecraft_mod_ai/runtime_bootstrap.py",
    '    install_kv_correctness(llama_decode_speed_contract)\n',
    "",
)
replace_exact(
    "minecraft_mod_ai/runtime_bootstrap.py",
    '    from .llama_completion_liveness_contract import install as install_completion_liveness\n',
    '''    from .llama_completion_liveness_contract import (\n        install as install_completion_liveness,\n    )\n''',
)

shim = ROOT / "minecraft_mod_ai/llama_kv_correctness_contract.py"
if not shim.exists():
    raise SystemExit("KV correctness shim unexpectedly missing before canonicalization")
shim.unlink()

(ROOT / "tests/test_llama_kv_correctness_contract.py").write_text(
    '''from __future__ import annotations\n\nimport os\nfrom pathlib import Path\nfrom types import SimpleNamespace\n\nfrom minecraft_mod_ai import llama_decode_speed_contract as decode_speed\n\n\ndef test_precision_reference_order_prioritizes_high_precision() -> None:\n    assert decode_speed._precision_reference_order(("q4_0", "f16", "q8_0")) == (\n        "f16",\n        "q8_0",\n        "q4_0",\n    )\n\n\ndef test_kv_schema_invalidates_pre_correctness_receipts() -> None:\n    assert (\n        decode_speed._KV_SCHEMA_VERSION\n        == "mmm/llama-kv-decode-speed-v3-precision-reference"\n    )\n\n\ndef test_kv_fingerprint_is_candidate_order_canonical(tmp_path: Path) -> None:\n    model = tmp_path / "model.gguf"\n    model.write_bytes(b"gguf-test-model")\n    autotune = SimpleNamespace(\n        _server_version=lambda _binary: "server-v1",\n        _hardware_identity=lambda: "hardware-v1",\n        _env_int=lambda _name, default: default,\n        _BENCHMARK_OUTPUT_TOKENS=8,\n    )\n    config = SimpleNamespace(\n        model_id="model",\n        extra={"gguf_filename": "model.gguf"},\n        max_context=4096,\n        max_new_tokens=8,\n    )\n    first = decode_speed._kv_fingerprint(\n        autotune, config, "llama-server", str(model), ("q4_0", "f16", "q8_0")\n    )\n    second = decode_speed._kv_fingerprint(\n        autotune, config, "llama-server", str(model), ("q8_0", "q4_0", "f16")\n    )\n    assert first == second\n\n\ndef test_kv_probe_uses_f16_as_semantic_reference_and_restores_env(monkeypatch) -> None:\n    starts: list[str] = []\n\n    class _Autotune:\n        _BENCHMARK_OUTPUT_TOKENS = 8\n        ProbeResult = SimpleNamespace\n\n        @staticmethod\n        def _compact_benchmark_request(request):\n            return request\n\n        @staticmethod\n        def _env_int(_name: str, default: int) -> int:\n            return default\n\n        @staticmethod\n        def _env_float(_name: str, default: float) -> float:\n            return default\n\n        @staticmethod\n        def _free_port(port: int) -> int:\n            return port\n\n        @staticmethod\n        def _start_server(_binary, _model_path, _config, _variant, _port):\n            kv = os.environ["MMM_KV_CACHE_QUANT"]\n            starts.append(kv)\n            return SimpleNamespace(kv=kv)\n\n        @staticmethod\n        def _wait_ready(process, port: int) -> str:\n            return f"http://127.0.0.1:{port}/{process.kv}"\n\n        @staticmethod\n        def _probe_server(_url, _request, *, max_tokens: int, variant):\n            kv = os.environ["MMM_KV_CACHE_QUANT"]\n            tps = {"f16": 10.0, "q8_0": 12.0, "q4_0": 14.0}[kv]\n            return SimpleNamespace(\n                variant=variant,\n                ok=True,\n                output_sha256="same-semantic-output",\n                predicted_tokens=max_tokens,\n                predicted_tps=tps,\n                prompt_tps=100.0,\n                elapsed_seconds=0.01,\n                error="",\n            )\n\n        @staticmethod\n        def _stop_server(_process) -> None:\n            return None\n\n    monkeypatch.setenv("MMM_KV_CACHE_QUANT", "q4_0")\n    config = SimpleNamespace(max_new_tokens=8)\n    selected, probes = decode_speed._probe_kv_types(\n        _Autotune(),\n        "llama-server",\n        "/tmp/model.gguf",\n        config,\n        SimpleNamespace(),\n        ("q4_0", "f16", "q8_0"),\n    )\n\n    assert starts == ["f16", "q8_0", "q4_0"]\n    assert [probe["kv"] for probe in probes] == starts\n    assert selected == "q4_0"\n    assert os.environ["MMM_KV_CACHE_QUANT"] == "q4_0"\n''',
    encoding="utf-8",
)

for base in (ROOT / "minecraft_mod_ai", ROOT / "tests"):
    for path in base.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "llama_kv_correctness_contract" in text:
            raise SystemExit(f"stale KV shim reference: {path.relative_to(ROOT)}")

# One-shot staging disappears in the same production commit.
(ROOT / ".github/workflows/worker05-kv-canonical-v1.yml").unlink(missing_ok=True)
Path(__file__).unlink(missing_ok=True)
