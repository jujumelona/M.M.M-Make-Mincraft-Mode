from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai import llama_server_runtime_tuning as runtime_tuning
from minecraft_mod_ai import qwen_runtime_transport_contract as contract


def _config(model_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_id=model_id,
        max_context=262144,
        max_new_tokens=8192,
        extra={"gguf_filename": f"{model_id.split('/')[-1]}.gguf"},
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


def test_runtime_tuning_owns_extended_server_variant_after_bootstrap() -> None:
    assert autotune.ServerVariant is runtime_tuning.ServerVariant
    variant = runtime_tuning.ServerVariant(
        "mtp-2|ub1024|p2|cr64",
        "draft-mtp",
        2,
        ubatch=1024,
        parallel=2,
        cache_reuse=64,
        draft_p_min=0.8,
    )
    assert variant.ubatch == 1024
    assert variant.parallel == 2
    assert variant.cache_reuse == 64
    assert variant.draft_p_min == 0.8


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
    baseline = runtime_tuning.ServerVariant("baseline")
    mtp = runtime_tuning.ServerVariant("mtp-2", "draft-mtp", 2)
    assert contract._initial_calibration_variant(baseline)
    assert contract._initial_calibration_variant(mtp)
    assert not contract._initial_calibration_variant(
        runtime_tuning.ServerVariant(
            "mtp-2|ub1024", "draft-mtp", 2, ubatch=1024
        )
    )
    assert not contract._initial_calibration_variant(
        runtime_tuning.ServerVariant("mtp-2|p2", "draft-mtp", 2, parallel=2)
    )
    assert not contract._initial_calibration_variant(
        runtime_tuning.ServerVariant(
            "mtp-2|cr64", "draft-mtp", 2, cache_reuse=64
        )
    )


def test_qwen_mtp_skips_unsupported_parallel_refinement() -> None:
    selected = runtime_tuning.ServerVariant(
        "mtp-2|ub512",
        "draft-mtp",
        2,
        ubatch=512,
    )
    calls: list[object] = []

    def run_variant(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise AssertionError("Qwen MTP parallel refinement must not launch")

    result = runtime_tuning._run_parallel_stage(
        run_variant,
        binary="llama-server",
        model_path="/tmp/model.gguf",
        config=_config("unsloth/Qwen3.6-27B-MTP-GGUF"),
        benchmark_request=object(),
        selected=selected,
        probe_tokens=64,
        parallel_values=(1, 2, 4),
        minimum_gain=1.01,
        forced_parallel=4,
    )

    winner, parallel_winner, p1_probe, probes = result
    assert winner.spec_type == "draft-mtp"
    assert winner.parallel == 1
    assert parallel_winner is None
    assert p1_probe is None
    assert probes == ()
    assert calls == []


def test_qwen_mtp_final_launch_forces_one_slot_and_restores_operator_env(
    monkeypatch,
) -> None:
    seen: list[tuple[str, int, str]] = []

    class FakeAutotune:
        def __init__(self) -> None:
            def launch_selected(
                _binary: str,
                _model_path: str,
                _config: object,
                selected: object,
            ) -> str:
                seen.append(
                    (
                        os.environ.get("MMM_LLAMA_PARALLEL", ""),
                        int(getattr(selected, "parallel", 1)),
                        str(getattr(selected, "spec_type", "none")),
                    )
                )
                return "http://127.0.0.1:8910/v1"

            def fingerprint(*_args: object) -> str:
                return "base"

            self._launch_selected = launch_selected
            self._fingerprint = fingerprint

    fake = FakeAutotune()
    contract._install_mtp_single_slot_policy(fake)
    monkeypatch.setenv("MMM_LLAMA_PARALLEL", "4")

    mtp = runtime_tuning.ServerVariant(
        "mtp-2|p4",
        "draft-mtp",
        2,
        parallel=4,
    )
    fake._launch_selected(
        "llama-server",
        "/tmp/model.gguf",
        _config("unsloth/Qwen3.6-35B-A3B-MTP-GGUF"),
        mtp,
    )
    assert seen[-1] == ("1", 1, "draft-mtp")
    assert os.environ["MMM_LLAMA_PARALLEL"] == "4"

    baseline = runtime_tuning.ServerVariant("baseline|p4", parallel=4)
    fake._launch_selected(
        "llama-server",
        "/tmp/model.gguf",
        _config("unsloth/Qwen3.6-35B-A3B-MTP-GGUF"),
        baseline,
    )
    assert seen[-1] == ("4", 4, "none")


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
    variant = runtime_tuning.ServerVariant("mtp-2", "draft-mtp", 2)
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
    variant = runtime_tuning.ServerVariant("mtp-4", "draft-mtp", 4)
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
        runtime_tuning.ServerVariant("baseline"),
    )
    fake._mmm_run_tuning_variant(
        "llama-server",
        "/tmp/model.gguf",
        _config("unsloth/Qwen3.6-35B-A3B-MTP-GGUF"),
        object(),
        runtime_tuning.ServerVariant(
            "mtp-2|ub1024", "draft-mtp", 2, ubatch=1024
        ),
    )
    assert calls == 0


def test_runtime_installs_zero_reload_tool_calibration_and_single_slot_mtp() -> None:
    assert autotune.ServerVariant is runtime_tuning.ServerVariant
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
    assert getattr(
        runtime_tuning._run_parallel_stage,
        "_mmm_qwen_mtp_single_slot_stage_v1",
        False,
    )
    assert getattr(
        autotune._launch_selected,
        "_mmm_qwen_mtp_single_slot_launch_v1",
        False,
    )
