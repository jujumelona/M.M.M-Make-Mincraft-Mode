from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.game_design import (
    GameDesignPlanner,
    _generate_game_design_once,
    _lossless_request_pages,
    _planner_plugin_manifest,
    _system_prompt,
)
from minecraft_mod_ai.spec import SpecValidationError


class _Router:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[str, list[dict], dict]] = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append((role, list(messages), dict(kwargs)))
        return self.text


def _once(router: _Router, prompt: str = "Create a moon crystal item.") -> dict:
    return _generate_game_design_once(
        router,
        authoritative_prompt=prompt,
        media_paths=(),
        system_prompt=_system_prompt(),
    )


def test_game_design_model_fills_host_template_once() -> None:
    router = _Router(
        json.dumps(
            {
                "game_design": {
                    "title": "Moon Forge",
                    "pitch": "Craft moon relics.",
                    "core_loop": ["mine", "craft"],
                    "progression": ["ore", "relic"],
                    "combat": {},
                    "mod_context": {"vanilla_integration": ["crafting"]},
                    "modules": [
                        {
                            "plugin_id": "custom_weather",
                            "status": "blocked",
                            "reason": "seasonal weather",
                        }
                    ],
                    "assets": [
                        {
                            "id": "moon_crystal",
                            "kind": "item",
                            "brief": "blue crystal",
                        }
                    ],
                    "acceptance_tests": ["item is registered"],
                }
            }
        )
    )

    design = _once(router)

    assert len(router.calls) == 1
    assert design["title"] == "Moon Forge"
    assert design["modules"] == [
        {
            "plugin_id": "custom_weather",
            "status": "custom",
            "reason": "seasonal weather",
            "requirement_refs": [],
            "implementation_obligations": [],
        }
    ]
    assert design["assets"] == [
        {"id": "moon_crystal", "kind": "item", "brief": "blue crystal"}
    ]


def test_malformed_model_output_fails_readiness_without_retry() -> None:
    router = _Router("not json at all")

    with pytest.raises(SpecValidationError, match="design readiness failed"):
        _once(router, "Add a copper lantern.")

    assert len(router.calls) == 1


def test_host_merge_drops_unknown_and_invalid_nested_values() -> None:
    router = _Router(
        json.dumps(
            {
                "game_design": {
                    "title": "Sanitized",
                    "pitch": "Sanitize nested values without losing a viable design.",
                    "core_loop": ["collect", "craft"],
                    "progression": ["unlock weather tools"],
                    "acceptance_tests": ["valid nested entries remain"],
                    "unknown_control": "model must not own this",
                    "modules": [
                        "bad entry",
                        {
                            "plugin_id": "weather_system",
                            "status": "blocked",
                            "reason": "requested weather",
                            "model_only": "drop me",
                        },
                    ],
                    "assets": [
                        {
                            "id": "unsupported_asset",
                            "kind": "unsupported_media",
                            "brief": "must be rejected",
                        },
                        {
                            "id": "moon_crystal",
                            "kind": "item",
                            "prompt": "blue crystal",
                            "model_only": "drop me",
                        },
                    ],
                }
            }
        )
    )

    design = _once(router)

    assert len(router.calls) == 1
    assert "unknown_control" not in design
    assert design["modules"] == [
        {
            "plugin_id": "weather_system",
            "status": "custom",
            "reason": "requested weather",
            "requirement_refs": [],
            "implementation_obligations": [],
        }
    ]
    assert design["assets"] == [
        {"id": "moon_crystal", "kind": "item", "brief": "blue crystal"}
    ]


def test_game_design_prompt_declares_host_ownership_not_repair_loops() -> None:
    prompt = _system_prompt()
    manifest = _planner_plugin_manifest()

    assert "host-owned game_design template" in prompt
    assert "host owns platform selection, paging, required fields, validation and fallback" in prompt
    assert "repair" not in prompt.casefold()
    assert set(manifest) == {"product_scope", "standalone_map_generation", "plugins"}
    assert all(set(plugin) == {"plugin_id", "status"} for plugin in manifest["plugins"])


def test_request_paging_is_lossless() -> None:
    prompt = "first requirement " + ("bounded filler " * 100) + "last requirement"
    pages = _lossless_request_pages(prompt, max_json_text_bytes=128)

    assert len(pages) > 1
    assert "".join(pages) == prompt
    assert all(page for page in pages)


def test_empty_prompt_has_readable_error() -> None:
    with pytest.raises(SpecValidationError, match="프롬프트를 입력해 주세요"):
        GameDesignPlanner(_Router("{}")).plan("   ")
