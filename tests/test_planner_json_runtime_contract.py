from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai import complete_planner, llama_server_hardware_policy
from minecraft_mod_ai import colab_mtp_server
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
        return self.responses.pop(0)


def _request(target: str = "entity_runtime") -> dict[str, object]:
    return {
        "current_target_deliverable": target,
        "current_target_deliverables": [target],
        "remaining_deliverables": [target],
        "contract": complete_planner._PRODUCTION_PAGE_CONTRACT,
    }


def _module(module_id: str = "entity_runtime") -> dict[str, object]:
    return {
        "module_id": module_id,
        "kind": "entity",
        "config": {},
        "depends_on": [],
        "required_gates": [],
    }


def test_production_page_recovers_only_semantically_empty_collections() -> None:
    router = _Router(
        json.dumps(
            {
                "modules": [_module()],
                "completed_deliverables": ["entity_runtime"],
                "complete": True,
                "next_cursor": "",
            }
        )
    )

    page = complete_planner._generate_json_page_with_repair(
        router,
        system_prompt="Return the production page.",
        request=_request(),
        media_paths=(),
        expected_contracts=(
            frozenset(complete_planner._PRODUCTION_PAGE_CONTRACT),
        ),
        stage="unit production page",
    )

    assert page["assets"] == []
    assert page["audio"] == []
    assert page["acceptance_tests"] == []
    assert page["completed_deliverables"] == ["entity_runtime"]
    assert len(router.calls) == 1


def test_production_page_repairs_zero_progress_with_exact_host_diagnostic() -> None:
    first = json.dumps(
        {
            "modules": [_module()],
            "assets": [],
            "audio": [],
            "acceptance_tests": [],
            "completed_deliverables": [],
            "complete": True,
            "next_cursor": "",
        }
    )
    second = json.dumps(
        {
            "modules": [_module()],
            "assets": [],
            "audio": [],
            "acceptance_tests": [],
            "completed_deliverables": ["entity_runtime"],
            "complete": True,
            "next_cursor": "",
        }
    )
    router = _Router(first, second)

    page = complete_planner._generate_json_page_with_repair(
        router,
        system_prompt="Return the production page.",
        request=_request(),
        media_paths=(),
        expected_contracts=(
            frozenset(complete_planner._PRODUCTION_PAGE_CONTRACT),
        ),
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


def test_enabled_mtp_is_final_fail_closed_hot_path(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(colab_mtp_server, "colab_mtp_server_enabled", lambda: True)
    monkeypatch.setattr(colab_mtp_server, "colab_mtp_server_running", lambda: True)
    monkeypatch.setattr(
        llama_server_hardware_policy,
        "_strict_server_generate",
        lambda adapter, request, url: calls.append(url) or "mtp-only",
    )
    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)

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

    assert result == "mtp-only"
    assert calls == [colab_mtp_server.SERVER_API_URL]
    assert getattr(LlamaCppAdapter.generate, "_mmm_final_strict_mtp", False)
