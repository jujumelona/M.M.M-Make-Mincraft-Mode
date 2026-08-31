from __future__ import annotations

import json

import pytest

from minecraft_mod_ai import agentic_research_game_design as design
from minecraft_mod_ai import evidence_first_planning as evidence
from minecraft_mod_ai import evidence_request_guard as request_guard
from minecraft_mod_ai.planner_design_readiness_contract import (
    _validate_design_coverage,
)
from minecraft_mod_ai.spec import SpecValidationError


class _RepairRouter:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def generate_text(self, role: str, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.calls.append({"role": role, "messages": messages, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("unexpected extra planner repair call")
        return json.dumps(self.responses.pop(0), ensure_ascii=False)


def _catalog(prompt: str, *requirement_ids: str) -> dict[str, object]:
    return {
        "requirements": [
            {
                "requirement_id": requirement_id,
                "capability": f"gameplay.{index}",
                "statement": prompt,
                "semantic_statement": f"semantic requirement {index}",
                "source_span": {"text": prompt},
                "observable_behavior": {
                    "given": "the world is ready",
                    "when": f"requirement {index} is exercised",
                    "then": f"requirement {index} is observable",
                },
                "acceptance": [f"requirement {index} is observable"],
            }
            for index, requirement_id in enumerate(requirement_ids, start=1)
        ]
    }


def _module_properties() -> dict[str, object]:
    for section_id, _fields, properties in design._SECTION_SPECS:
        if section_id == "modules_and_assets":
            return properties
    raise AssertionError("modules_and_assets section not found")


def test_runtime_installs_worker03_design_readiness_contract() -> None:
    assert getattr(design._generate_section, "__mmm_requirement_design_context__", False)
    # Deep-design execution is intentionally installed after Worker 03 and wraps the
    # requirement-aware message owner. Verify both layers rather than requiring the
    # outermost callable to duplicate the inner owner's marker.
    assert getattr(design._section_messages, "__mmm_deep_design_section_prompt__", False)
    assert getattr(
        getattr(design._section_messages, "__wrapped__", None),
        "__mmm_requirement_design_messages__",
        False,
    )
    assert getattr(
        design.generate_sectioned_game_design,
        "__mmm_requirement_design_coverage__",
        False,
    )
    assert getattr(evidence._semantic_spans, "__mmm_crlf_lossless__", False)


def test_empty_first_module_section_is_rejected_and_repaired() -> None:
    prompt = "플레이어가 수정 조각을 모으고 포탈을 연다."
    requirement_id = "req_crystal_portal"
    router = _RepairRouter(
        [
            {"section": {"modules": [], "assets": []}},
            {
                "section": {
                    "modules": [
                        {
                            "plugin_id": "crystal_portal",
                            "status": "custom",
                            "reason": "수정 수집과 포탈 진행을 구현한다.",
                            "requirement_refs": [requirement_id],
                            "implementation_obligations": [
                                "수정 조각 수집 상태와 포탈 해금 조건을 구현한다."
                            ],
                        }
                    ],
                    "assets": [],
                }
            },
        ]
    )
    token = request_guard._ACTIVE_REQUEST_CATALOG.set(
        (prompt, _catalog(prompt, requirement_id))
    )
    try:
        section = design._generate_section(
            router,
            prompt=prompt,
            section_id="modules_and_assets",
            fields=("modules", "assets"),
            properties=_module_properties(),
            research={},
            media_paths=(),
            trace_metadata={"test": "worker03"},
        )
    finally:
        request_guard._ACTIVE_REQUEST_CATALOG.reset(token)

    assert len(router.calls) == 2
    assert section["modules"][0]["requirement_refs"] == [requirement_id]
    first_payload = json.loads(router.calls[0]["messages"][1]["content"])
    assert first_payload["approved_requirements"][0]["requirement_id"] == requirement_id
    second_payload = json.loads(router.calls[1]["messages"][1]["content"])
    assert "modules must be non-empty" in second_payload["validator_error"]
    assert second_payload["previous_candidate"] == {"modules": [], "assets": []}


def test_missing_required_design_field_is_not_silently_synthesized() -> None:
    with pytest.raises(SpecValidationError, match="host fallback is not accepted"):
        design._validate_section_types(
            {"title": "Generated title", "pitch": "real pitch", "core_loop": ["loop"]},
            ("title", "pitch", "core_loop"),
        )

    with pytest.raises(SpecValidationError, match="core_loop must be a non-empty list"):
        design._validate_section_types(
            {"title": "real title", "pitch": "real pitch", "core_loop": []},
            ("title", "pitch", "core_loop"),
        )


def test_every_approved_requirement_needs_an_implementation_bearing_design_module() -> None:
    ledger = (
        {"requirement_id": "req_collect"},
        {"requirement_id": "req_portal"},
    )
    design_payload = {
        "modules": [
            {
                "plugin_id": "collection",
                "requirement_refs": ["req_collect"],
                "implementation_obligations": ["track collected fragments"],
            }
        ]
    }

    with pytest.raises(SpecValidationError, match="req_portal"):
        _validate_design_coverage(design_payload, ledger)


def test_requirement_design_binding_preserves_all_exact_ids_and_obligations() -> None:
    ledger = (
        {"requirement_id": "req_collect"},
        {"requirement_id": "req_portal"},
    )
    design_payload = {
        "modules": [
            {
                "plugin_id": "progression_core",
                "requirement_refs": ["req_collect", "req_portal"],
                "implementation_obligations": [
                    "persist collected fragments",
                    "unlock portal after the authored threshold",
                ],
            }
        ]
    }

    result = _validate_design_coverage(design_payload, ledger)
    binding = result["_requirement_design_bindings"]
    assert binding["requirement_ids"] == ["req_collect", "req_portal"]
    assert [row["requirement_id"] for row in binding["bindings"]] == [
        "req_collect",
        "req_portal",
    ]
    assert all(row["module_ids"] == ["progression_core"] for row in binding["bindings"])
    assert all(row["implementation_obligations"] for row in binding["bindings"])


def test_crlf_prompt_offsets_replay_exact_authored_text() -> None:
    prompt = "Players gather crystals.\r\nPlayers open a portal."
    spans = evidence._semantic_clause_spans(prompt)
    replayed = [prompt[start:end] for start, end in spans]

    assert replayed == ["Players gather crystals.", "Players open a portal."]
    assert spans[1][0] == prompt.index("Players open a portal.")
