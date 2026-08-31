from __future__ import annotations

import pytest

from minecraft_mod_ai import agentic_research_game_design as design
from minecraft_mod_ai import evidence_first_planning as evidence
from minecraft_mod_ai import evidence_request_guard as request_guard
from minecraft_mod_ai.planner_design_readiness_contract import _validate_design_coverage
from minecraft_mod_ai.spec import SpecValidationError


class _TextRouter:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def generate_text(self, role: str, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.calls.append({"role": role, "messages": messages, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("unexpected extra planner call")
        return self.responses.pop(0)


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


def test_runtime_uses_canonical_design_owner_without_readiness_wrappers() -> None:
    assert not getattr(design._generate_section, "__mmm_requirement_design_context__", False)
    assert not getattr(design._section_messages, "__mmm_requirement_design_messages__", False)
    assert not getattr(
        design.generate_sectioned_game_design,
        "__mmm_requirement_design_coverage__",
        False,
    )
    assert not hasattr(design._section_messages, "__wrapped__")
    assert getattr(evidence._semantic_spans, "__mmm_crlf_lossless__", False)


def test_requirement_module_section_is_text_native_and_generated_once() -> None:
    prompt = "플레이어가 수정 조각을 모으고 포탈을 연다."
    requirement_id = "req_crystal_portal"
    router = _TextRouter(
        [
            """## modules
- crystal_portal | custom | 수정 수집과 포탈 진행을 구현한다. | req_crystal_portal | 수정 조각 수집 상태를 저장한다; 포탈 해금 조건을 구현한다
## assets
- none
"""
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
            research={},
            media_paths=(),
            trace_metadata={"test": "worker03"},
        )
    finally:
        request_guard._ACTIVE_REQUEST_CATALOG.reset(token)

    assert len(router.calls) == 1
    call = router.calls[0]
    kwargs = call["kwargs"]
    assert kwargs["response_format"] == "text"
    assert kwargs["response_schema"] is None
    assert kwargs["enable_tools"] is False
    rendered = "\n".join(str(message["content"]) for message in call["messages"])
    assert "APPROVED REQUIREMENTS" in rendered
    assert requirement_id in rendered
    assert "requirement_refs" in rendered
    assert (
        "plugin_id | status | reason | requirement_refs | implementation_obligations"
        in rendered
    )
    assert section["modules"][0]["requirement_refs"] == [requirement_id]
    assert section["modules"][0]["implementation_obligations"] == [
        "수정 조각 수집 상태를 저장한다",
        "포탈 해금 조건을 구현한다",
    ]


def test_missing_requirement_module_fails_closed_without_model_repair_loop() -> None:
    prompt = "플레이어가 수정 조각을 모으고 포탈을 연다."
    requirement_id = "req_crystal_portal"
    router = _TextRouter(["""## modules
- none
## assets
- none
"""])
    token = request_guard._ACTIVE_REQUEST_CATALOG.set(
        (prompt, _catalog(prompt, requirement_id))
    )
    try:
        with pytest.raises(SpecValidationError, match="modules must be non-empty"):
            design._generate_section(
                router,
                prompt=prompt,
                section_id="modules_and_assets",
                fields=("modules", "assets"),
                research={},
                media_paths=(),
                trace_metadata={"test": "worker03-empty"},
            )
    finally:
        request_guard._ACTIVE_REQUEST_CATALOG.reset(token)

    assert len(router.calls) == 1


def test_required_design_fields_fail_closed_without_host_synthesis() -> None:
    with pytest.raises(SpecValidationError, match="core_loop must be a non-empty list"):
        design._validate_section_types(
            {"title": "real title", "pitch": "real pitch", "core_loop": []},
            ("title", "pitch", "core_loop"),
        )

    with pytest.raises(SpecValidationError, match="title must be a non-empty string"):
        design._validate_section_types(
            {"title": "", "pitch": "real pitch", "core_loop": ["loop"]},
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
                "status": "planning",
                "reason": "collect fragments",
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
                "status": "planning",
                "reason": "own authored progression",
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
