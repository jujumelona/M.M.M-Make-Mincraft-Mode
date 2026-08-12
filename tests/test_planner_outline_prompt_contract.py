from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import complete_planner, llama_server_hardware_policy
from minecraft_mod_ai import planner_json_runtime_contract as runtime
from minecraft_mod_ai.planner_outline_prompt_contract import (
    _OUTLINE_MODE,
    _OUTLINE_TOKEN_CAP,
    install,
)
from minecraft_mod_ai.spec import SpecValidationError


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


def _outline_request() -> dict[str, object]:
    return {
        "known_batch_catalog": {
            "count": 0,
            "sha256": "0" * 64,
            "recent_ids": [],
        },
        "cursor": "",
        "contract": complete_planner._PRODUCTION_OUTLINE_CONTRACT,
    }


def _valid_outline() -> str:
    return json.dumps(
        {
            "production_batches": [
                {
                    "batch_id": "ui_interaction",
                    "scope": "Implement the UI and interaction layer.",
                    "depends_on_batches": [],
                    "deliverables": ["ui_screen", "interaction_handler"],
                    "exports": ["ui_screen"],
                }
            ],
            "complete": True,
            "next_cursor": "",
        }
    )


def test_outline_replaces_verbose_system_prompt_with_single_json_api_prompt() -> None:
    install(runtime)
    router = _Router(_valid_outline())
    page = complete_planner._generate_json_page_with_repair(
        router,
        system_prompt="VERBOSE_SENTINEL should never reach the outline model",
        request=_outline_request(),
        media_paths=(),
        expected_contracts=(frozenset(complete_planner._PRODUCTION_OUTLINE_CONTRACT),),
        stage="production outline continuation",
    )

    assert page["complete"] is True
    assert len(router.calls) == 1
    system = str(router.calls[0]["messages"][0]["content"])
    assert "VERBOSE_SENTINEL" not in system
    assert "Return exactly ONE JSON object" in system
    assert "at most TWO new batches" in system
    assert "Do not output Markdown fences" in system


def test_outline_server_payload_is_capped_below_planner_8k_budget() -> None:
    install(runtime)
    token = _OUTLINE_MODE.set(True)
    try:
        payload = llama_server_hardware_policy._server_payload(
            SimpleNamespace(config=SimpleNamespace(max_new_tokens=8192)),
            SimpleNamespace(
                messages=({"role": "user", "content": "x"},),
                response_format="json",
            ),
        )
    finally:
        _OUTLINE_MODE.reset(token)

    assert payload["max_tokens"] == _OUTLINE_TOKEN_CAP == 2048


def test_outline_has_only_one_bounded_repair() -> None:
    install(runtime)
    router = _Router("not-json", "still-not-json", _valid_outline())
    with pytest.raises(SpecValidationError, match="failed after 1 page-local repairs"):
        complete_planner._generate_json_page_with_repair(
            router,
            system_prompt="ignored verbose prompt",
            request=_outline_request(),
            media_paths=(),
            expected_contracts=(frozenset(complete_planner._PRODUCTION_OUTLINE_CONTRACT),),
            stage="production outline continuation",
        )
    assert len(router.calls) == 2
