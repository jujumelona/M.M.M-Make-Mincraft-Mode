from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai import complete_planner, llama_server_hardware_policy
from minecraft_mod_ai import planner_json_runtime_contract as runtime
from minecraft_mod_ai.planner_outline_prompt_contract import install


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


def test_outline_replaces_verbose_prompt_without_imposing_batch_limit() -> None:
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
    assert "NO fixed batch count and NO fixed page count" in system
    assert "Choose the page size yourself" in system
    assert "at most TWO" not in system


def test_outline_does_not_cap_the_selected_models_output_budget() -> None:
    install(runtime)
    payload = llama_server_hardware_policy._server_payload(
        SimpleNamespace(config=SimpleNamespace(max_new_tokens=8192)),
        SimpleNamespace(
            messages=({"role": "user", "content": "x"},),
            response_format="json",
        ),
    )

    assert payload["max_tokens"] == 8192


def test_outline_prompt_explicitly_allows_sequential_pages() -> None:
    install(runtime)
    router = _Router(_valid_outline())
    complete_planner._generate_json_page_with_repair(
        router,
        system_prompt="ignored",
        request=_outline_request(),
        media_paths=(),
        expected_contracts=(frozenset(complete_planner._PRODUCTION_OUTLINE_CONTRACT),),
        stage="production outline continuation",
    )
    system = str(router.calls[0]["messages"][0]["content"])
    assert "emit the next complete JSON page immediately" in system
    assert "consecutive pages of ONE outline" in system
