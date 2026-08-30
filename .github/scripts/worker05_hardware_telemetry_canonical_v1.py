from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARDWARE = "minecraft_mod_ai/llama_server_hardware_policy.py"
BOOTSTRAP = "minecraft_mod_ai/runtime_bootstrap.py"


def replace_exact(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_exact(
    HARDWARE,
    '''_TELEMETRY_TOTALS = {\n    "prompt_tokens": 0,\n    "output_tokens": 0,\n    "generation_seconds": 0.0,\n    "requests": 0,\n}\n''',
    '''_TELEMETRY_TOTALS = {\n    "prompt_tokens": 0,\n    "prompt_seconds": 0.0,\n    "output_tokens": 0,\n    "generation_seconds": 0.0,\n    "requests": 0,\n}\n''',
)
replace_exact(
    HARDWARE,
    '''def _server_origin(server_url: str) -> str:\n    value = server_url.rstrip("/")\n    return value.removesuffix("/v1")\n\n\ndef _parse_prometheus_metrics(text: str) -> dict[str, float]:\n''',
    '''def _server_origin(server_url: str) -> str:\n    value = server_url.rstrip("/")\n    return value.removesuffix("/v1")\n\n\ndef _auxiliary_native_telemetry_enabled() -> bool:\n    """Keep auxiliary /metrics and /slots requests off the default inference path."""\n\n    return os.environ.get("MMM_LLAMA_AUXILIARY_TELEMETRY", "").strip().casefold() in {\n        "1",\n        "true",\n        "yes",\n        "on",\n    }\n\n\ndef _parse_prometheus_metrics(text: str) -> dict[str, float]:\n''',
)
replace_exact(
    HARDWARE,
    '''    generation_seconds = max(\n        0.0,\n        float(after.get("tokens_predicted_seconds_total", 0.0))\n        - float(before.get("tokens_predicted_seconds_total", 0.0)),\n    )\n    with _TELEMETRY_LOCK:\n        _TELEMETRY_TOTALS["prompt_tokens"] += prompt\n        _TELEMETRY_TOTALS["output_tokens"] += output\n        _TELEMETRY_TOTALS["generation_seconds"] += generation_seconds\n        _TELEMETRY_TOTALS["requests"] += 1\n        cumulative = dict(_TELEMETRY_TOTALS)\n    return {\n        "prompt_tokens": prompt,\n        "output_tokens": output,\n        "generation_seconds": generation_seconds,\n        "cumulative_prompt_tokens": int(cumulative["prompt_tokens"]),\n        "cumulative_output_tokens": int(cumulative["output_tokens"]),\n        "cumulative_generation_seconds": float(cumulative["generation_seconds"]),\n        "cumulative_requests": int(cumulative["requests"]),\n    }\n''',
    '''    prompt_seconds = max(\n        0.0,\n        float(after.get("prompt_seconds_total", 0.0))\n        - float(before.get("prompt_seconds_total", 0.0)),\n    )\n    generation_seconds = max(\n        0.0,\n        float(after.get("tokens_predicted_seconds_total", 0.0))\n        - float(before.get("tokens_predicted_seconds_total", 0.0)),\n    )\n    with _TELEMETRY_LOCK:\n        _TELEMETRY_TOTALS["prompt_tokens"] += prompt\n        _TELEMETRY_TOTALS["prompt_seconds"] = (\n            float(_TELEMETRY_TOTALS.get("prompt_seconds", 0.0)) + prompt_seconds\n        )\n        _TELEMETRY_TOTALS["output_tokens"] += output\n        _TELEMETRY_TOTALS["generation_seconds"] += generation_seconds\n        _TELEMETRY_TOTALS["requests"] += 1\n        cumulative = dict(_TELEMETRY_TOTALS)\n    prompt_tps = prompt / prompt_seconds if prompt_seconds > 0 else 0.0\n    cumulative_prompt_seconds = float(cumulative["prompt_seconds"])\n    cumulative_prompt_tps = (\n        float(cumulative["prompt_tokens"]) / cumulative_prompt_seconds\n        if cumulative_prompt_seconds > 0\n        else 0.0\n    )\n    result = {\n        "prompt_tokens": prompt,\n        "prompt_seconds": prompt_seconds,\n        "prompt_tps": prompt_tps,\n        "output_tokens": output,\n        "generation_seconds": generation_seconds,\n        "cumulative_prompt_tokens": int(cumulative["prompt_tokens"]),\n        "cumulative_prompt_seconds": cumulative_prompt_seconds,\n        "cumulative_prompt_tps": cumulative_prompt_tps,\n        "cumulative_output_tokens": int(cumulative["output_tokens"]),\n        "cumulative_generation_seconds": float(cumulative["generation_seconds"]),\n        "cumulative_requests": int(cumulative["requests"]),\n    }\n    print(\n        "llama server: prefill complete",\n        f" prompt_tokens={prompt}",\n        f" prompt_seconds={prompt_seconds:.3f}",\n        f" prompt_tok_s={prompt_tps:.2f}",\n        f" cumulative_prompt_tok_s={cumulative_prompt_tps:.2f}",\n        sep="",\n        flush=True,\n    )\n    return result\n''',
)
replace_exact(
    HARDWARE,
    '''    _reject_tool_stream_request(adapter, request)\n    metrics_before: dict[str, float] | None = None\n''',
    '''    _reject_tool_stream_request(adapter, request)\n    auxiliary_telemetry = _auxiliary_native_telemetry_enabled()\n    metrics_before: dict[str, float] | None = None\n''',
)
replace_exact(
    HARDWARE,
    '''        metrics_before = _metrics_snapshot(client, server_url)\n        committed_at_start = _telemetry_totals()\n''',
    '''        if auxiliary_telemetry:\n            metrics_before = _metrics_snapshot(client, server_url)\n        committed_at_start = _telemetry_totals()\n''',
)
replace_exact(
    HARDWARE,
    '''                    slot = _slot_snapshot(client, server_url)\n''',
    '''                    slot = (\n                        _slot_snapshot(client, server_url)\n                        if auxiliary_telemetry\n                        else None\n                    )\n''',
)
replace_exact(
    HARDWARE,
    '''        metrics_after = _metrics_snapshot(client, server_url)\n        usage = _commit_metrics_delta(metrics_before, metrics_after)\n''',
    '''        metrics_after = (\n            _metrics_snapshot(client, server_url)\n            if metrics_before is not None\n            else None\n        )\n        usage = _commit_metrics_delta(metrics_before, metrics_after)\n''',
)

replace_exact(
    BOOTSTRAP,
    '''    from .llama_prefill_telemetry_contract import (\n        install as install_llama_prefill_telemetry,\n    )\n''',
    "",
)
replace_exact(
    BOOTSTRAP,
    "    install_llama_prefill_telemetry(llama_server_hardware_policy)\n",
    "",
)

shim = ROOT / "minecraft_mod_ai/llama_prefill_telemetry_contract.py"
if not shim.exists():
    raise SystemExit("prefill telemetry shim unexpectedly missing")
shim.unlink()

(ROOT / "tests/test_llama_prefill_telemetry_contract.py").write_text(
    '''from __future__ import annotations\n\nfrom minecraft_mod_ai import llama_server_hardware_policy as hardware\n\n\ndef test_prefill_telemetry_uses_native_prompt_counter_deltas(capsys, monkeypatch) -> None:\n    monkeypatch.setattr(\n        hardware,\n        "_TELEMETRY_TOTALS",\n        {\n            "prompt_tokens": 0,\n            "prompt_seconds": 0.0,\n            "output_tokens": 0,\n            "generation_seconds": 0.0,\n            "requests": 0,\n        },\n    )\n\n    result = hardware._commit_metrics_delta(\n        {\n            "prompt_tokens_total": 100.0,\n            "prompt_seconds_total": 2.0,\n            "tokens_predicted_total": 10.0,\n            "tokens_predicted_seconds_total": 1.0,\n        },\n        {\n            "prompt_tokens_total": 500.0,\n            "prompt_seconds_total": 4.0,\n            "tokens_predicted_total": 30.0,\n            "tokens_predicted_seconds_total": 2.0,\n        },\n    )\n\n    assert result is not None\n    assert result["prompt_tokens"] == 400\n    assert result["prompt_seconds"] == 2.0\n    assert result["prompt_tps"] == 200.0\n    assert result["cumulative_prompt_tps"] == 200.0\n    assert hardware._TELEMETRY_TOTALS["prompt_seconds"] == 2.0\n    assert "prompt_tok_s=200.00" in capsys.readouterr().out\n\n\ndef test_prefill_telemetry_is_owned_directly_by_hardware_policy() -> None:\n    assert not hasattr(hardware._commit_metrics_delta, "_mmm_prompt_prefill_telemetry_v1")\n''',
    encoding="utf-8",
)

test_path = ROOT / "tests/test_llama_server_hardware_policy.py"
test_text = test_path.read_text(encoding="utf-8")
anchor = '''def test_strict_server_generate_reuses_one_http_client_without_auxiliary_metrics(monkeypatch) -> None:\n    client = _FakeClient()\n'''
replacement = '''def test_strict_server_generate_reuses_one_http_client_without_auxiliary_metrics(monkeypatch) -> None:\n    monkeypatch.delenv("MMM_LLAMA_AUXILIARY_TELEMETRY", raising=False)\n    client = _FakeClient()\n'''
if test_text.count(anchor) != 1:
    raise SystemExit(f"test_llama_server_hardware_policy.py: no-aux anchor count={test_text.count(anchor)}")
test_text = test_text.replace(anchor, replacement, 1)
test_text += '''\n\ndef test_auxiliary_native_telemetry_is_explicit_opt_in(monkeypatch) -> None:\n    monkeypatch.delenv("MMM_LLAMA_AUXILIARY_TELEMETRY", raising=False)\n    assert hardware._auxiliary_native_telemetry_enabled() is False\n    monkeypatch.setenv("MMM_LLAMA_AUXILIARY_TELEMETRY", "true")\n    assert hardware._auxiliary_native_telemetry_enabled() is True\n'''
test_path.write_text(test_text, encoding="utf-8")

for base in (ROOT / "minecraft_mod_ai", ROOT / "tests"):
    for path in base.rglob("*.py"):
        if "llama_prefill_telemetry_contract" in path.read_text(encoding="utf-8"):
            raise SystemExit(f"stale prefill shim reference: {path.relative_to(ROOT)}")

(ROOT / ".github/workflows/worker05-hardware-telemetry-canonical-v1.yml").unlink(missing_ok=True)
Path(__file__).unlink(missing_ok=True)
