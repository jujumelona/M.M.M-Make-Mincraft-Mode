from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager

import pytest

import minecraft_mod_ai.complete_planner as complete_planner_module
import minecraft_mod_ai.game_design as game_design_module
from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner
from minecraft_mod_ai.game_design import GameDesignPlanner
from minecraft_mod_ai.spec import SpecValidationError


EARLY_REQUIREMENT = "EARLY_GRAVITY_LANTERN_REQUIREMENT"
LATE_REQUIREMENT = "LATE_TIDAL_COMPASS_REQUIREMENT"


def _page_design(text: str) -> dict[str, object]:
    markers = [
        marker
        for marker in (EARLY_REQUIREMENT, LATE_REQUIREMENT)
        if marker in text
    ]
    return {
        "game_design": {
            "title": "Bounded request page",
            "pitch": (
                "Preserve " + ", ".join(markers)
                if markers
                else "Preserve this bounded request page"
            ),
            "core_loop": [f"implement {marker}" for marker in markers],
            "progression": [],
            "combat": {},
            "mod_context": {},
            "modules": [
                {
                    "plugin_id": "custom",
                    "status": "custom",
                    "reason": marker,
                }
                for marker in markers
            ],
            "assets": [],
            "acceptance_tests": [
                f"{marker} is observable" for marker in markers
            ],
        }
    }


class _LosslessWorkflowRouter:
    def __init__(self) -> None:
        self.message_bytes: list[int] = []
        self.design_pages: list[dict[str, object]] = []
        self.outline_pages: list[dict[str, object]] = []
        self.expansion_scopes: list[str] = []
        self.session_events: list[str] = []

    @contextmanager
    def generation_session(self, role: str):
        self.session_events.append(f"enter:{role}")
        try:
            yield self
        finally:
            self.session_events.append(f"exit:{role}")

    def generate_text(self, role, messages, **kwargs):
        assert role == "planner"
        assert kwargs["response_format"] == "json"
        self.message_bytes.append(
            sum(len(message["content"].encode("utf-8")) for message in messages)
        )
        request = json.loads(messages[-1]["content"])
        if request.get("schema_version") == (
            "mmm/authoritative-request-page-v1"
        ):
            self.design_pages.append(request)
            return json.dumps(
                _page_design(request["authoritative_request_text"]),
                ensure_ascii=False,
            )
        if request.get("schema_version") == (
            "mmm/request-production-outline-page-v1"
        ):
            self.outline_pages.append(request)
            source = request["request_ingestion_page"]
            text = source["authoritative_request_text"]
            page_index = int(source["page_index"])
            marker = next(
                (
                    value
                    for value in (EARLY_REQUIREMENT, LATE_REQUIREMENT)
                    if value in text
                ),
                None,
            )
            batches = (
                [
                    {
                        "batch_id": "page_feature",
                        "scope": f"Implement {marker} exactly.",
                        "depends_on_batches": [],
                        "deliverables": [f"deliver {marker}"],
                        "exports": ["page_feature_module"],
                    }
                ]
                if marker is not None
                else []
            )
            return json.dumps(
                {
                    "production_batches": batches,
                    "complete": True,
                    "next_cursor": "",
                }
            )
        if "batch" in request and "remaining_deliverables" in request:
            batch = request["batch"]
            scope = str(batch["scope"])
            self.expansion_scopes.append(scope)
            return json.dumps(
                {
                    "modules": [
                        {
                            "module_id": batch["exports"][0],
                            "kind": "custom_java",
                            "config": {
                                "implementation": "custom",
                                "source_scope_sha256": hashlib.sha256(
                                    scope.encode("utf-8")
                                ).hexdigest(),
                                "observed_early": EARLY_REQUIREMENT in scope,
                                "observed_late": LATE_REQUIREMENT in scope,
                            },
                            # The host must materialize cross-page ordering even if
                            # the descriptive model omits these dependencies.
                            "depends_on": [],
                            "required_gates": [],
                        }
                    ],
                    "assets": [],
                    "audio": [],
                    "acceptance_tests": [
                        f"{batch['batch_id']} completes its exact request page"
                    ],
                    "completed_deliverables": request[
                        "remaining_deliverables"
                    ],
                    "complete": True,
                    "next_cursor": "",
                }
            )
        raise AssertionError(f"Unexpected planner request: {request}")


def _patch_research_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    brief = {
        "schema_version": "minecraft-mod-ai/research-brief-v1",
        "domains": [],
    }
    monkeypatch.setattr(
        game_design_module,
        "normalize_research_brief",
        lambda prompt, design: brief,
    )
    monkeypatch.setattr(
        complete_planner_module,
        "normalize_research_brief",
        lambda prompt, design: brief,
    )
    monkeypatch.setattr(
        complete_planner_module,
        "collect_technology_radar",
        lambda *args, **kwargs: {"requirements": []},
    )
    monkeypatch.setattr(
        complete_planner_module,
        "_retrieve_implementation_evidence",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        complete_planner_module,
        "collect_ecosystem_seed_bundle",
        lambda *args, **kwargs: None,
    )


def test_more_than_6144_word_request_is_losslessly_paged_through_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_research_to_empty(monkeypatch)
    prompt = (
        f"  \n{EARLY_REQUIREMENT}: lantern gravity must invert on redstone.\n"
        + "neutral_requirement_context " * 6_500
        + f"\n{LATE_REQUIREMENT}: compass must track the current tide.\n  "
    )
    assert len(prompt.split()) > 6_144
    router = _LosslessWorkflowRouter()

    proposal = CompleteGameDesignPlanner(router).plan(prompt)

    ingestion = proposal.game_design["_request_ingestion"]
    production_ingestion = proposal.game_design[
        "_request_production_ingestion"
    ]
    assert proposal.requested_prompt == prompt
    assert ingestion["prompt_sha256"] == hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()
    assert ingestion["page_count"] > 10
    assert production_ingestion["page_count"] == ingestion["page_count"]
    assert production_ingestion["batch_count"] == 2
    assert len(router.design_pages) == ingestion["page_count"]
    assert len(router.outline_pages) == ingestion["page_count"]
    assert "".join(
        page["authoritative_request_text"] for page in router.design_pages
    ) == prompt
    assert "".join(
        page["request_ingestion_page"]["authoritative_request_text"]
        for page in router.outline_pages
    ) == prompt

    rendered_design = json.dumps(proposal.game_design, ensure_ascii=False)
    rendered_outline = json.dumps(
        proposal.game_design["production_outline"], ensure_ascii=False
    )
    assert EARLY_REQUIREMENT in rendered_design
    assert LATE_REQUIREMENT in rendered_design
    assert EARLY_REQUIREMENT in rendered_outline
    assert LATE_REQUIREMENT in rendered_outline
    assert any(EARLY_REQUIREMENT in scope for scope in router.expansion_scopes)
    assert any(LATE_REQUIREMENT in scope for scope in router.expansion_scopes)

    outline = proposal.game_design["production_outline"]
    assert outline[0]["depends_on_batches"] == []
    assert all(item["depends_on_batches"] for item in outline[1:])
    gameplay_modules = [
        module
        for module in proposal.modules
        if module.kind == "custom_java"
    ]
    gameplay_ids = {module.module_id for module in gameplay_modules}
    assert not (set(gameplay_modules[0].depends_on) & gameplay_ids)
    assert all(
        set(module.depends_on) & gameplay_ids
        for module in gameplay_modules[1:]
    )
    assert any(module.config["observed_early"] for module in gameplay_modules)
    assert any(module.config["observed_late"] for module in gameplay_modules)

    # 6,500+ words increase the number of calls, not the size of any one call.
    assert max(router.message_bytes) < 20_000
    assert router.session_events == ["enter:planner", "exit:planner"]


class _MalformedSecondDesignPageRouter:
    def generate_text(self, role, messages, **kwargs):
        del role, kwargs
        request = json.loads(messages[-1]["content"])
        if request["page"]["page_index"] == 1:
            return '{"game_design":'
        return json.dumps(
            _page_design(request["authoritative_request_text"]),
            ensure_ascii=False,
        )


def test_malformed_large_request_page_fails_closed_after_local_repair() -> None:
    prompt = "first requirement " + ("bounded filler " * 500) + "last requirement"
    with pytest.raises(
        SpecValidationError,
        match=r"page 2/.*failed after one page-local repair",
    ):
        GameDesignPlanner(_MalformedSecondDesignPageRouter()).plan(prompt)


class _ValidDesignPageRouter:
    def generate_text(self, role, messages, **kwargs):
        del role, kwargs
        request = json.loads(messages[-1]["content"])
        return json.dumps(
            _page_design(request["authoritative_request_text"]),
            ensure_ascii=False,
        )


def test_large_request_research_classification_is_itself_losslessly_paged() -> None:
    prompt = "generic capability requirement " * 1_000

    design, proposal = GameDesignPlanner(_ValidDesignPageRouter()).plan(prompt)

    brief = design["_research_brief"]
    research_ingestion = brief["request_ingestion"]
    assert proposal.requested_prompt == prompt
    assert research_ingestion["page_count"] > 1
    assert research_ingestion["prompt_sha256"] == hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()
    assert len(brief["domains"]) >= research_ingestion["page_count"]
    assert len({domain["domain_id"] for domain in brief["domains"]}) == len(
        brief["domains"]
    )


def test_empty_prompt_has_readable_error() -> None:
    with pytest.raises(SpecValidationError, match="프롬프트를 입력해 주세요"):
        GameDesignPlanner(_MalformedSecondDesignPageRouter()).plan("   ")
