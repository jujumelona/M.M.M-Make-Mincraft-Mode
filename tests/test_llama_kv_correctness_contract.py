from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import llama_decode_speed_contract
from minecraft_mod_ai import llama_kv_correctness_contract as contract


def test_precision_reference_order_uses_highest_precision_first() -> None:
    assert contract._precision_reference_order(("q4_0", "q8_0", "f16")) == (
        "f16",
        "q8_0",
        "q4_0",
    )
    assert contract._precision_reference_order(("q4_0", "q8_0")) == (
        "q8_0",
        "q4_0",
    )


def test_install_reorders_probe_reference_without_changing_candidate_policy() -> None:
    calls: list[tuple[str, ...]] = []

    def probe(_autotune, _binary, _model_path, _config, _request, candidates):
        calls.append(tuple(candidates))
        return "q8_0", []

    fake = SimpleNamespace(_probe_kv_types=probe)
    contract.install(fake)
    result = fake._probe_kv_types(
        object(),
        "server",
        "model.gguf",
        object(),
        object(),
        ("q4_0", "q8_0", "f16"),
    )

    assert result == ("q8_0", [])
    assert calls == [("f16", "q8_0", "q4_0")]


def test_runtime_installs_precision_reference_on_kv_probe() -> None:
    assert getattr(
        llama_decode_speed_contract._probe_kv_types,
        "_mmm_kv_precision_reference_v1",
        False,
    )
