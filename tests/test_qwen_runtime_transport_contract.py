from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai import qwen_runtime_transport_contract as contract


def _config(model_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_id=model_id,
        max_context=262144,
        max_new_tokens=8192,
        extra={"gguf_filename": f"{model_id.split('/')[-1]}.gguf"},
    )


def _tuning_variant(
    name: str,
    spec_type: str = "none",
    draft_n_max: int = 0,
    *,
    ubatch: int = 0,
    parallel: int = 1,
    cache_reuse: int = 0,
) -> SimpleNamespace:
    """Represent staged tuning metadata without changing production ServerVariant."""

    return SimpleNamespace(
        name=name,
        spec_type=spec_type,
        draft_n_max=draft_n_max,
        ubatch=ubatch,
        parallel=parallel,
        cache_reuse=cache_reuse,
    )


def _tool_response(*, call_id: str, arguments: str) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "mmm_transport_probe",
                                "arguments": arguments,
                            },
                        }
                    ]
                }
            }
        ]
    }


def test_tool_signature_is_semantic_not_call_id_or_json_format() -> None:
    first = contract._tool_call_signature(
        _tool_response(call_id="call-a", arguments='{"value":7}')
    )
    second = contract._tool_call_signature(
        _tool_response(call_id="call-b", arguments='{ "value" : 7 }')
    )
    assert first
    assert first == second
    assert first == hashlib.sha256(
        b'{"arguments":{"value":7},"name":"mmm_transport_probe"}'
    ).hexdigest()


def test_tool_signature_rejects_wrong_tool_arguments() -> None:
    assert not contract._tool_call_signature(
        _tool_response(call_id="call-a", arguments='{"value":8}')
    )


def test_only_initial_speculation_candidates_get_tool_calibration() -> None:
    baseline = autotune.ServerVariant("baseline")
    mtp = autotune.ServerVariant("mtp-2", "draft-mtp", 2)
    assert contract._initial_calibration_variant(baseline)
    assert contract._initial_calibration_variant(mtp)
    assert not contract._initial_calibration_variant(
        _tuning_variant("mtp-2|ub1024", "draft-mtp", 2, ubatch=1024)
    )
    assert not contract._initial_calibration_variant(
        _tuning_variant("mtp-2|p2", "draft-mtp", 2, parallel=2)
    )
    assert not contract._initial_calibration_variant(
        _tuning_variant("mtp-2|cr64", "draft-mtp", 2, cache_reuse=64)
    )


class _FakeAutotune:
    def __init__(self) -> None:
        self.tool_calls = 0

        def probe_server(
            _base_url: str,
            _request: object,
            *,
            max_tokens: int,
            variant: object,
        ) -> SimpleNamespace:
            return SimpleNamespace(ok=True, max_tokens=max_tokens, variant=variant)

        self._probe_server = probe_server

        def run_variant(
            _binary: str,
            _model_path: str,
            _config: object,
            request: object,
            variant: object,
            **_kwargs: object,
        ) -> SimpleNamespace:
            self._probe_server(
                "http://127.0.0.1:8910/v1",
                request,
                max_tokens=1,
                variant=variant,
            )
            return self._probe_server(
                "http://127.0.0.1:8910/v1",
                request,
                max_tokens=64,
                variant=variant,
            )

        self._mmm_run_tuning_variant = run_variant


def test_initial_qwen_variant_requires_tool_probe(monkeypatch) -> None:
    fake = _FakeAutotune()

    def passing_probe(_base_url: str, _autotune: object) -> tuple[bool, str]:
        fake.tool_calls += 1
        return True, ""

    monkeypatch.setattr(contract, "_tool_probe", passing_probe)
    contract._install_tool_equivalence_policy(fake)
    variant = autotune.ServerVariant("mtp-2", "draft-mtp", 2)
    result = fake._mmm_run_tuning_variant(
        "llama-server",
        "/tmp/model.gguf",
        _config("unsloth/Qwen3.5-9B-MTP-GGUF"),
        object(),
        variant,
    )
    assert result.max_tokens == 64
    assert fake.tool_calls == 1


def test_bad_mtp_tool_transport_fails_variant_before_decode(monkeypatch) -> None:
    fake = _FakeAutotune()
    monkeypatch.setattr(
        contract,
        "_tool_probe",
        lambda _base_url, _autotune: (False, "canonical call mismatch"),
    )
    contract._install_tool_equivalence_policy(fake)
    variant = autotune.ServerVariant("mtp-4", "draft-mtp", 4)
    with pytest.raises(RuntimeError, match="native tool transport calibration failed"):
        fake._mmm_run_tuning_variant(
            "llama-server",
            "/tmp/model.gguf",
            _config("unsloth/Qwen3.6-27B-MTP-GGUF"),
            object(),
            variant,
        )


def test_non_qwen_and_neutral_refinements_do_not_add_tool_probe(monkeypatch) -> None:
    fake = _FakeAutotune()
    calls = 0

    def counted_probe(_base_url: str, _autotune: object) -> tuple[bool, str]:
        nonlocal calls
        calls += 1
        return True, ""

    monkeypatch.setattr(contract, "_tool_probe", counted_probe)
    contract._install_tool_equivalence_policy(fake)
    fake._mmm_run_tuning_variant(
        "llama-server",
        "/tmp/model.gguf",
        _config("other/model"),
        object(),
        autotune.ServerVariant("baseline"),
    )
    fake._mmm_run_tuning_variant(
        "llama-server",
        "/tmp/model.gguf",
        _config("unsloth/Qwen3.6-35B-A3B-MTP-GGUF"),
        object(),
        _tuning_variant("mtp-2|ub1024", "draft-mtp", 2, ubatch=1024),
    )
    assert calls == 0


def test_runtime_installs_zero_reload_tool_calibration() -> None:
    assert getattr(
        autotune._mmm_run_tuning_variant,
        "_mmm_qwen_tool_calibration_context_v2",
        False,
    )
    assert getattr(
        autotune._probe_server,
        "_mmm_qwen_tool_calibration_probe_v2",
        False,
    )
