from __future__ import annotations

import json

import pytest
from types import SimpleNamespace

from minecraft_mod_ai import complete_planner, llama_server_hardware_policy
from minecraft_mod_ai.model_adapters import AdapterConfig, GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter
from minecraft_mod_ai.planner_json_runtime_contract import _JSON_SCHEMA


class _Router:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def generate_text(
        self,
        role: str,
        messages,
        *,
        media_paths=(),
        response_format="text",
    ) -> str:
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "media_paths": media_paths,
                "response_format": response_format,
            }
        )
        if not self.responses:
            raise AssertionError("planner repair loop requested an unexpected extra response")
        return self.responses.pop(0)


def _request(*targets: str) -> dict[str, object]:
    selected = list(targets or ("entity_runtime",))
    return {
        "current_target_deliverable": selected[0],
        "current_target_deliverables": selected,
        "remaining_deliverables": selected,
        "total_remaining": len(selected),
        "contract": complete_planner._PRODUCTION_PAGE_CONTRACT,
    }


def _module(
    module_id: str = "entity_runtime",
    *,
    implements: tuple[str, ...] = (),
) -> dict[str, object]:
    value: dict[str, object] = {
        "module_id": module_id,
        "kind": "entity",
        "config": {},
        "depends_on": [],
        "required_gates": [],
    }
    if implements:
        value["implements_deliverables"] = list(implements)
    return value


def test_production_page_host_owns_bookkeeping_and_empty_collections() -> None:
    router = _Router(json.dumps({"modules": [_module()]}))

    page = complete_planner._generate_json_page_with_repair(
        router,
        system_prompt="Return the production page.",
        request=_request(),
        media_paths=(),
        expected_contracts=(frozenset(complete_planner._PRODUCTION_PAGE_CONTRACT),),
        stage="unit production page",
    )

    assert page["assets"] == []
    assert page["audio"] == []
    assert page["acceptance_tests"] == []
    assert page["completed_deliverables"] == ["entity_runtime"]
    assert page["complete"] is True
    assert page["next_cursor"] == ""
    assert len(router.calls) == 1


def test_production_page_derives_multi_target_progress_from_item_claims() -> None:
    router = _Router(
        json.dumps(
            {
                "modules": [
                    _module("entity_runtime_impl", implements=("entity_runtime",)),
                    _module("ui_runtime_impl", implements=("ui_runtime",)),
                ],
            }
        )
    )

    page = complete_planner._generate_json_page_with_repair(
        router,
        system_prompt="Return the production page.",
        request=_request("entity_runtime", "ui_runtime"),
        media_paths=(),
        expected_contracts=(frozenset(complete_planner._PRODUCTION_PAGE_CONTRACT),),
        stage="unit production page",
    )

    assert page["completed_deliverables"] == ["entity_runtime", "ui_runtime"]
    assert page["complete"] is True
    assert page["next_cursor"] == ""
    assert len(router.calls) == 1


def test_large_production_page_keeps_ai_requested_target_width() -> None:
    router = _Router(
        json.dumps(
            {
                "modules": [
                    _module("d1_impl", implements=("d1",)),
                    _module("d2_impl", implements=("d2",)),
                ]
            }
        )
    )

    page = complete_planner._generate_json_page_with_repair(
        router,
        system_prompt="Implement ALL four deliverables in this page.",
        request=_request("d1", "d2", "d3", "d4"),
        media_paths=(),
        expected_contracts=(frozenset(complete_planner._PRODUCTION_PAGE_CONTRACT),),
        stage="unit production page",
    )

    assert page["completed_deliverables"] == ["d1", "d2"]
    assert page["complete"] is False
    assert page["next_cursor"]
    assert len(router.calls) == 1
    first_user = router.calls[0]["messages"][1]["content"]
    assert '"current_target_deliverables": ["d1", "d2", "d3", "d4"]' in first_user
    first_system = router.calls[0]["messages"][0]["content"]
    assert "ACTIVE HOST PAGE WIDTH OVERRIDE" not in first_system


def test_production_retry_keeps_requested_targets_after_invalid_first_page() -> None:
    router = _Router(
        "not json",
        json.dumps(
            {
                "modules": [
                    _module("entity_runtime_impl", implements=("entity_runtime",))
                ]
            }
        ),
    )

    page = complete_planner._generate_json_page_with_repair(
        router,
        system_prompt="Implement ALL requested deliverables in this page.",
        request=_request("entity_runtime", "ui_runtime"),
        media_paths=("reference.png",),
        expected_contracts=(frozenset(complete_planner._PRODUCTION_PAGE_CONTRACT),),
        stage="unit production page",
    )

    assert page["completed_deliverables"] == ["entity_runtime"]
    assert page["complete"] is False
    assert page["next_cursor"]
    assert len(router.calls) == 2
    second_user = router.calls[1]["messages"][1]["content"]
    assert '"current_target_deliverables": ["entity_runtime", "ui_runtime"]' in second_user
    second_system = router.calls[1]["messages"][0]["content"]
    assert "CORRECTION:" in second_system
    assert router.calls[1]["media_paths"] == ()


def test_production_page_stops_after_one_full_page_retry() -> None:
    router = _Router(
        "not json",
        "still not json",
        json.dumps({"modules": [_module()]}),
    )

    with pytest.raises(
        complete_planner.SpecValidationError,
        match="failed after one page-local repair",
    ):
        complete_planner._generate_json_page_with_repair(
            router,
            system_prompt="Return the production page.",
            request=_request(),
            media_paths=(),
            expected_contracts=(frozenset(complete_planner._PRODUCTION_PAGE_CONTRACT),),
            stage="unit production page",
        )

    assert len(router.calls) == 2
    assert "CORRECTION:" in router.calls[1]["messages"][0]["content"]


def test_production_page_repairs_zero_progress_with_exact_host_diagnostic() -> None:
    first = json.dumps(
        {
            "modules": [],
            "assets": [],
            "audio": [],
            "acceptance_tests": [],
        }
    )
    router = _Router(first, json.dumps({"modules": [_module()]}))

    page = complete_planner._generate_json_page_with_repair(
        router,
        system_prompt="Return the production page.",
        request=_request(),
        media_paths=(),
        expected_contracts=(frozenset(complete_planner._PRODUCTION_PAGE_CONTRACT),),
        stage="unit production page",
    )

    assert page["completed_deliverables"] == ["entity_runtime"]
    assert len(router.calls) == 2
    repair_prompt = router.calls[1]["messages"][0]["content"]
    assert "made no host-verifiable deliverable progress" in repair_prompt
    assert "HOST JSON CONTRACT" in repair_prompt


def test_llama_server_payload_receives_decode_time_json_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"complete": {"type": "boolean"}},
        "required": ["complete"],
    }
    token = _JSON_SCHEMA.set(schema)
    try:
        payload = llama_server_hardware_policy._server_payload(
            SimpleNamespace(config=SimpleNamespace(max_new_tokens=64)),
            SimpleNamespace(
                messages=({"role": "user", "content": "x"},),
                response_format="json",
            ),
        )
    finally:
        _JSON_SCHEMA.reset(token)

    assert payload["response_format"] == {
        "type": "json_object",
        "schema": schema,
    }


def test_native_server_is_final_fail_closed_hot_path(monkeypatch) -> None:
    calls: list[str] = []
    server_url = "http://127.0.0.1:8910/v1"
    monkeypatch.setenv("LLAMA_SERVER_URL", server_url)
    monkeypatch.setattr(
        llama_server_hardware_policy,
        "_strict_server_generate",
        lambda adapter, request, url: calls.append(url) or "native-only",
    )

    adapter = LlamaCppAdapter(
        AdapterConfig(
            role="planner",
            adapter="llama_cpp",
            model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
            max_new_tokens=64,
        )
    )
    result = adapter.generate(
        GenerationRequest(
            messages=({"role": "user", "content": "x"},),
            response_format="json",
        )
    )

    assert result == "native-only"
    assert calls == [server_url]
    assert getattr(LlamaCppAdapter.generate, "_mmm_explicit_server_strict", False)
