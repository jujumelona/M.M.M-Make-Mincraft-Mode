from __future__ import annotations

import json

import pytest

from minecraft_mod_ai import evidence_request_guard as request_guard
from minecraft_mod_ai import game_design
from minecraft_mod_ai.spec import SpecValidationError


def _catalog(prompt: str, *requirement_ids: str) -> dict[str, object]:
    return {
        "schema_version": "mmm/approved-requirement-graph-v1",
        "purpose": prompt,
        "requirements": [
            {
                "requirement_id": requirement_id,
                "capability": f"gameplay.capability_{index}",
                "statement": prompt,
                "semantic_statement": f"semantic requirement {index}",
                "source_span": {"text": f"authored requirement {index}"},
                "acceptance": [f"requirement {index} is observable"],
            }
            for index, requirement_id in enumerate(requirement_ids, start=1)
        ],
    }


class _Router:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def generate_text(self, role: str, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.calls.append({"role": role, "messages": messages, "kwargs": kwargs})
        return json.dumps(self.payload, ensure_ascii=False)


def _design_payload(requirement_refs: list[str]) -> dict[str, object]:
    return {
        "game_design": {
            "title": "Crystal Portal",
            "pitch": "수정을 모아 포탈을 해금하는 진행형 모드",
            "core_loop": ["수정을 탐색하고 수집한다", "포탈 진행도를 올린다"],
            "progression": ["수정 수집", "포탈 해금", "새 지역 진입"],
            "combat": {},
            "mod_context": {},
            "modules": [
                {
                    "plugin_id": "crystal_portal",
                    "status": "custom",
                    "reason": "수집과 포탈 진행을 구현한다",
                    "requirement_refs": requirement_refs,
                    "implementation_obligations": [
                        "수정 수집 상태를 저장한다",
                        "해금 조건을 만족하면 포탈을 활성화한다",
                    ],
                }
            ],
            "assets": [],
            "acceptance_tests": ["수정 수집 후 포탈이 해금되는 것을 관찰한다"],
        }
    }


def test_active_response_schema_requires_complete_design_and_trace_fields() -> None:
    design_schema = game_design._GAME_DESIGN_RESPONSE_SCHEMA["properties"]["game_design"]
    assert set(game_design._GAME_DESIGN_FIELDS) <= set(design_schema["required"])
    module_schema = design_schema["properties"]["modules"]["items"]
    assert "requirement_refs" in module_schema["required"]
    assert "implementation_obligations" in module_schema["required"]
    assert module_schema["properties"]["requirement_refs"]["minItems"] == 1


def test_module_normalization_preserves_requirement_traceability() -> None:
    raw = _design_payload(["req_collect"])["game_design"]["modules"]
    normalized = game_design._modules(raw)

    assert normalized[0]["requirement_refs"] == ["req_collect"]
    assert normalized[0]["implementation_obligations"] == [
        "수정 수집 상태를 저장한다",
        "해금 조건을 만족하면 포탈을 활성화한다",
    ]


def test_host_skeleton_cannot_turn_partial_raw_output_into_accepted_design() -> None:
    prompt = "수정을 모아 포탈을 열 수 있게 해줘."
    router = _Router(
        {
            "game_design": {
                "title": "Partial",
                "pitch": "부분 응답",
            }
        }
    )
    token = request_guard._ACTIVE_REQUEST_CATALOG.set(
        (prompt, _catalog(prompt, "req_portal"))
    )
    try:
        with pytest.raises(SpecValidationError, match="core_loop is empty"):
            game_design._generate_game_design_once(
                router,
                authoritative_prompt=prompt,
                media_paths=(),
                system_prompt="design the requested mod",
            )
    finally:
        request_guard._ACTIVE_REQUEST_CATALOG.reset(token)

    assert len(router.calls) == 1
    schema = router.calls[0]["kwargs"]["response_schema"]
    assert schema["properties"]["game_design"]["properties"]["modules"]["minItems"] == 1


def test_single_page_active_design_binds_frozen_requirement_ids_before_planning() -> None:
    prompt = "수정을 모아 포탈을 열 수 있게 해줘."
    router = _Router(_design_payload(["req_collect", "req_portal"]))
    token = request_guard._ACTIVE_REQUEST_CATALOG.set(
        (prompt, _catalog(prompt, "req_collect", "req_portal"))
    )
    try:
        result = game_design._generate_game_design_once(
            router,
            authoritative_prompt=prompt,
            media_paths=(),
            system_prompt="design the requested mod",
        )
    finally:
        request_guard._ACTIVE_REQUEST_CATALOG.reset(token)

    binding = result["_requirement_design_bindings"]
    assert binding["requirement_ids"] == ["req_collect", "req_portal"]
    assert all(row["module_ids"] == ["crystal_portal"] for row in binding["bindings"])
    system_prompt = router.calls[0]["messages"][0]["content"]
    assert "FROZEN REQUIREMENT AUTHORITY" in system_prompt
    assert "req_collect" in system_prompt
    assert "req_portal" in system_prompt


def test_sharded_merge_requires_union_to_cover_every_frozen_requirement() -> None:
    prompt = "수정을 모으고 포탈을 열고 새 지역으로 이동하게 해줘."
    first = _design_payload(["req_collect"])["game_design"]
    second = _design_payload(["req_portal"])["game_design"]
    second = dict(second)
    second["modules"] = [
        {
            "plugin_id": "portal_travel",
            "status": "custom",
            "reason": "포탈과 지역 이동을 구현한다",
            "requirement_refs": ["req_portal"],
            "implementation_obligations": ["포탈 통과 시 대상 지역으로 이동시킨다"],
        }
    ]
    token = request_guard._ACTIVE_REQUEST_CATALOG.set(
        (prompt, _catalog(prompt, "req_collect", "req_portal", "req_travel"))
    )
    try:
        merged = game_design._merge_game_design_pages([first, second])
        with pytest.raises(SpecValidationError, match="req_travel"):
            game_design._validate_ready_design(prompt, merged)
    finally:
        request_guard._ACTIVE_REQUEST_CATALOG.reset(token)
