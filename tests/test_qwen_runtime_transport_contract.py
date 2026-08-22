from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace

import httpx
import pytest

from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai import llama_server_runtime_tuning as runtime_tuning
from minecraft_mod_ai import qwen_runtime_transport_contract as contract


def _config(model_id: str, *, qwen: bool = True) -> SimpleNamespace:
    extra = {
        "gguf_filename": f"{model_id.split('/')[-1]}.gguf",
        "mtp_widths": "1,2",
    }
    if qwen:
        extra["runtime_contract"] = "qwen"
    return SimpleNamespace(
        model_id=model_id,
        max_context=262144,
        max_new_tokens=8192,
        extra=extra,
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


def test_qwen35_tool_probe_uses_required_jinja_and_raw_host_parser(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "<think>check the exact argument</think>"
                                "<tool_call><function=mmm_transport_probe>"
                                "<parameter=value>7</parameter>"
                                "</function></tool_call>"
                            )
                        }
                    }
                ]
            }

    def post(url: str, *, json: dict, timeout: int) -> Response:
        captured.update(url=url, payload=dict(json), timeout=timeout)
        return Response()

    monkeypatch.setattr(httpx, "post", post)
    fake_autotune = SimpleNamespace(
        _env_int=lambda _name, default: default,
    )

    ok, error = contract._tool_probe(
        "http://127.0.0.1:8910/v1",
        fake_autotune,
        _config("unsloth/Qwen3.5-9B-MTP-GGUF"),
    )

    assert ok is True
    assert error == ""
    payload = captured["payload"]
    assert payload["tool_choice"] == "required"
    assert payload["temperature"] == 0.0
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["tools"][0]["function"]["name"] == "mmm_transport_probe"


def test_qwen35_tool_probe_rejects_server_parsed_tool_calls(monkeypatch) -> None:
    class Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return _tool_response(call_id="server-call", arguments='{"value":7}')

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: Response())

    ok, error = contract._tool_probe(
        "http://127.0.0.1:8910/v1",
        SimpleNamespace(_env_int=lambda _name, default: default),
        _config("unsloth/Qwen3.5-9B-MTP-GGUF"),
    )

    assert ok is False
    assert "server-parsed tool_calls" in error


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

    current = fake._fingerprint(
        _config("unsloth/Qwen3.5-9B-MTP-GGUF"),
        "llama-server",
        "/tmp/model.gguf",
    )
    legacy = hashlib.sha256(
        b'{"base":"base","qwen_mtp_parallel":"single-slot-v1"}'
    ).hexdigest()
    expected = hashlib.sha256(
        (
            '{"base":"base","qwen_mtp_parallel":"single-slot-v1",'
            f'"qwen_tool_transport":"{contract._TOOL_TRANSPORT_EPOCH}"}}'
        ).encode("utf-8")
    ).hexdigest()
    assert current == expected
    assert current != legacy


class _FakeAutotune:
    def __init__(self) -> None:
        self.tool_calls = 0
        self.benchmark_variant: object = runtime_tuning.ServerVariant("baseline")

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

        def benchmark(
            _binary: str,
            _model_path: str,
            _config: object,
            request: object,
            _fingerprint: str,
        ) -> SimpleNamespace:
            variant = self.benchmark_variant
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

        self._benchmark = benchmark


def test_main_staged_benchmark_requires_tool_probe(monkeypatch) -> None:
    fake = _FakeAutotune()
    fake.benchmark_variant = runtime_tuning.ServerVariant("mtp-2", "draft-mtp", 2)

    def passing_probe(
        _base_url: str, _autotune: object, _config_value: object
    ) -> tuple[bool, str]:
        fake.tool_calls += 1
        return True, ""

    monkeypatch.setattr(contract, "_tool_probe", passing_probe)
    contract._install_tool_equivalence_policy(fake)
    result = fake._benchmark(
        "llama-server",
        "/tmp/model.gguf",
        _config("unsloth/Qwen3.6-27B-MTP-GGUF"),
        object(),
        "fingerprint",
    )
    assert result.max_tokens == 64
    assert fake.tool_calls == 1


def test_initial_qwen_variant_requires_tool_probe(monkeypatch) -> None:
    fake = _FakeAutotune()

    def passing_probe(
        _base_url: str, _autotune: object, _config_value: object
    ) -> tuple[bool, str]:
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
        lambda _base_url, _autotune, _config_value: (
            False,
            "canonical call mismatch",
        ),
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

    def counted_probe(
        _base_url: str, _autotune: object, _config_value: object
    ) -> tuple[bool, str]:
        nonlocal calls
        calls += 1
        return True, ""

    monkeypatch.setattr(contract, "_tool_probe", counted_probe)
    contract._install_tool_equivalence_policy(fake)
    fake._mmm_run_tuning_variant(
        "llama-server",
        "/tmp/model.gguf",
        _config("other/model", qwen=False),
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


def test_model_name_without_registry_contract_is_not_qwen_runtime() -> None:
    config = _config("unsloth/Qwen3.6-27B-MTP-GGUF", qwen=False)
    assert contract._family(config) is None


def test_runtime_installs_zero_reload_tool_calibration_and_single_slot_mtp() -> None:
    assert autotune.ServerVariant is runtime_tuning.ServerVariant
    assert getattr(
        autotune._benchmark,
        "_mmm_qwen_tool_calibration_benchmark_v1",
        False,
    )
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
