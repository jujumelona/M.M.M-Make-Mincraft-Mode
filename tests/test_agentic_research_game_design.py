from __future__ import annotations

import json
from pathlib import Path

import pytest

import minecraft_mod_ai.agentic_research_game_design as agentic
from minecraft_mod_ai import game_design
from minecraft_mod_ai.spec import SpecValidationError


class _SectionRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_text(
        self,
        role,
        messages,
        *,
        media_paths=(),
        response_format="text",
        response_schema=None,
        tool_stage=None,
        enable_tools=True,
    ):
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "media_paths": tuple(media_paths),
                "response_format": response_format,
                "response_schema": response_schema,
                "tool_stage": tool_stage,
                "enable_tools": enable_tools,
            }
        )
        text = str(messages[-1]["content"])
        marker = "\n\nFIELD\n"
        field = text.split(marker, 1)[1].split("\n", 1)[0].strip()
        bodies = {
            "title": "연구 기반 모드",
            "pitch": "검색 근거를 바탕으로 설계한다.",
            "core_loop": "- 탐색하고 상호작용한다",
            "progression": "- 기능을 단계적으로 해금한다",
            "combat": "none",
            "mod_context": "none",
            "modules": "none",
            "assets": "none",
            "acceptance_tests": "- 요청한 핵심 루프가 게임 내에서 동작한다",
            "art_direction": "none",
        }
        return bodies[field]

class _ResearchRouter:
    def __init__(self, evidence_ref: str = "forced_project_rag") -> None:
        self.calls: list[dict[str, object]] = []
        self.evidence_ref = evidence_ref

    def generate_text(self, role, messages, **kwargs):
        self.calls.append({"role": role, "messages": messages, **kwargs})
        return json.dumps(
            {
                "research_note": {
                    "domain_id": "request",
                    "claims": [
                        {
                            "claim": "요청은 target-neutral Minecraft 모드 기능 설계로 조사할 수 있다.",
                            "evidence_refs": [self.evidence_ref],
                        }
                    ],
                    "gaps": [],
                    "next_queries": ["정확한 API 이름은 target freeze 뒤 검증"],
                    "procedures": [],
                    "sufficient": True,
                }
            },
            ensure_ascii=False,
        )


def _deterministic_research() -> dict[str, object]:
    return {
        "official_rag": {"status": "deferred_until_platform_selected"},
        "technology_radar": {
            "status": "deferred_until_target_freeze",
            "target_frozen": False,
        },
        "forced_project_rag": {
            "schema_version": "mmm/forced-pre-design-rag-v2",
            "research_sha256": "sha256:forced",
            "domain_count": 1,
            "query_count": 1,
            "project_source_count": 6,
            "domains": [
                {
                    "domain_id": "request",
                    "queries": [{"query": "feature", "sources": [{"source_id": "fixture"}]}],
                }
            ],
        },
    }


def test_sectioned_game_design_uses_host_owned_field_compiler() -> None:
    router = _SectionRouter()
    research = {
        "research_brief": {"domains": []},
        "domain_notes": [],
        "deterministic": {},
        "errors": [],
    }
    result = agentic.generate_sectioned_game_design(
        game_design,
        router,
        "연구를 먼저 하고 모드를 설계해줘",
        research=research,
    )
    expected_fields = [
        field
        for _section_id, fields, _properties in agentic._SECTION_SPECS
        for field in fields
    ]
    assert len(router.calls) == len(expected_fields)
    assert all(call["role"] == "planner" for call in router.calls)
    assert all(call["response_format"] == "text" for call in router.calls)
    assert all(call["response_schema"] is None for call in router.calls)
    assert all(call["tool_stage"] == "game_design" for call in router.calls)
    assert all(call["enable_tools"] is False for call in router.calls)
    for field, call in zip(expected_fields, router.calls, strict=True):
        system = str(call["messages"][0]["content"])
        user = str(call["messages"][1]["content"])
        assert "No JSON" in system
        assert "Do not write the field name, a Markdown ## heading" in system
        assert f"\n\nFIELD\n{field}\n" in user
    assert result["title"] == "연구 기반 모드"
    assert result["core_loop"]
    assert result["acceptance_tests"]
    assert "art_direction" not in result


def test_missing_markdown_headings_can_never_abort_identity_section() -> None:
    class HeadinglessRouter:
        def generate_text(self, role, messages, **kwargs):
            del role, messages, kwargs
            return "계절 작물과 요리를 연결하는 플레이 경험"

    section = agentic._generate_section(
        HeadinglessRouter(),
        prompt="계절마다 다른 작물을 재배하고 요리하는 모드를 만들어줘.",
        section_id="identity_and_loop",
        fields=("title", "pitch", "core_loop"),
        research={},
        media_paths=(),
        trace_metadata=None,
    )
    assert set(section) == {"title", "pitch", "core_loop"}
    assert section["title"]
    assert section["pitch"]
    assert section["core_loop"]

def test_research_domain_legacy_facade_is_host_owned() -> None:
    class NeverModel:
        def __getattr__(self, name):
            raise AssertionError(f"model called: {name}")

    result = agentic._research_domain_with_agent(
        NeverModel(),
        prompt="기능을 조사해서 설계해줘",
        domain={"domain_id": "request", "objective": "요청 조사", "queries": ["minecraft mod feature"]},
        deterministic=_deterministic_research(),
        trace_metadata=None,
    )
    assert result["sufficient"] is True
    assert result["research_mode"] == "advisory_predesign"
    assert result["quality_contract"]["model_json"] is False


def test_sufficient_research_rejects_empty_and_invented_refs() -> None:
    note = {
        "domain_id": "request",
        "claims": [{"claim": "unsupported", "evidence_refs": []}],
        "gaps": [],
        "next_queries": [],
        "procedures": [],
        "sufficient": True,
    }
    with pytest.raises(SpecValidationError, match="no host-issued evidence_ref"):
        agentic._validate_sufficient_research(
            note,
            allowed_refs=frozenset({"forced_project_rag"}),
        )

    note["claims"][0]["evidence_refs"] = ["minecraft_api:invented"]
    with pytest.raises(SpecValidationError, match="unverified evidence_refs"):
        agentic._validate_sufficient_research(
            note,
            allowed_refs=frozenset({"forced_project_rag"}),
        )


def test_domain_slice_issues_refs_only_for_real_host_evidence() -> None:
    prompt_slice = agentic._domain_evidence_slice(
        "request",
        _deterministic_research(),
    )

    assert "evidence_ref" not in prompt_slice["official_rag"]
    assert "evidence_ref" not in prompt_slice["technology_radar"]
    assert prompt_slice["forced_project_rag"]["evidence_ref"] == "forced_project_rag"
    assert agentic._allowed_research_refs(prompt_slice) == frozenset(
        {"forced_project_rag"}
    )


def test_evidence_document_preserves_full_raw_and_bounds_every_page(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MMM_RESEARCH_DOCUMENT_DIR", str(tmp_path))
    huge_official = "official:" + ("A" * 45_000)
    huge_forced = "forced:" + ("B" * 55_000)
    evidence = {
        "official_rag": {
            "domain_id": "mk_combat",
            "queries": [{"query": "damage", "raw": huge_official}],
        },
        "forced_project_rag": {
            "domain_id": "mk_combat",
            "queries": [{"query": "bossbar", "raw": huge_forced}],
        },
        "technology_radar": {"status": "available"},
    }

    document = paged_rag._materialize_domain_evidence_document("mk_combat", evidence)
    raw = json.loads(Path(document["raw_path"]).read_text(encoding="utf-8"))
    pages = paged_rag._read_evidence_pages(document)

    assert raw == evidence
    assert document["page_count"] == len(pages)
    assert pages
    assert document["model_projection"] == "claim_bearing_source_bodies_only;raw_receipt_lossless"
    assert all(
        len(str(page.get("content", "")).encode("utf-8"))
        <= paged_rag._EVIDENCE_PAGE_CHARS
        for page in pages
    )
    oversized_records: dict[str, list[dict[str, object]]] = {}
    for page in pages:
        record_sha256 = page.get("record_sha256")
        if record_sha256:
            oversized_records.setdefault(str(record_sha256), []).append(page)
    reconstructed = []
    for fragments in oversized_records.values():
        fragments.sort(key=lambda item: int(item["part_index"]))
        assert [int(item["part_index"]) for item in fragments] == list(
            range(int(fragments[0]["part_count"]))
        )
        rendered = "".join(str(item["content"]) for item in fragments)
        reconstructed.append(rendered)
        assert paged_rag._sha256_text(rendered) == fragments[0]["record_sha256"]
    joined = "\n".join(reconstructed)
    assert huge_official in joined
    assert huge_forced in joined
    assert [page["page_index"] for page in pages] == list(range(len(pages)))
    assert all(page["page_count"] == len(pages) for page in pages)


def test_legacy_paged_entrypoint_uses_small_model_text_and_host_quote_verification(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MMM_RESEARCH_DOCUMENT_DIR", str(tmp_path / "evidence"))
    evidence = {
        "forced_project_rag": {
            "domain_id": "mk_combat",
            "queries": [
                {
                    "query": "damage",
                    "raw": "Minecraft Fabric combat damage registration example",
                }
            ],
        }
    }
    document = paged_rag._materialize_domain_evidence_document("mk_combat", evidence)
    calls: list[dict[str, object]] = []

    class Router:
        def generate_text(self, role, messages, **kwargs):
            assert role == "planner"
            calls.append({"messages": messages, **kwargs})
            rendered = str(messages[-1]["content"])
            assert rendered.startswith("OBJECTIVE\n")
            assert "\nSOURCE\n" in rendered
            assert "Minecraft Fabric combat damage registration example" in rendered
            return (
                "EVIDENCE\tMinecraft Fabric combat damage registration example"
                "\tUse a combat damage registration pattern."
            )

    result = paged_rag._research_document_domain(
        agentic,
        Router(),
        prompt="전투 기능을 정확한 근거로 설계해줘",
        domain={
            "domain_id": "mk_combat",
            "objective": "minecraft combat damage",
            "queries": ["minecraft combat damage"],
        },
        document=document,
        trace_metadata=None,
    )

    assert calls
    assert all(call["response_format"] == "text" for call in calls)
    assert all(call["response_schema"] is None for call in calls)
    assert all(call["enable_tools"] is False for call in calls)
    assert result["research_mode"] == "advisory_predesign"
    assert result["claims"]
    assert result["claims"][0]["support_quote"] == (
        "Minecraft Fabric combat damage registration example"
    )
    assert result["claims"][0]["support_verification"] == (
        "host_exact_quote_from_small_model_line"
    )
    assert result["quality_contract"]["model_json"] is False


def test_domain_slice_bounds_forced_receipt_without_materializing_document() -> None:
    huge_forced = "F" * 40_000
    forced = {
        "schema_version": "mmm/forced-pre-design-rag-v2",
        "research_sha256": "sha256:forced",
        "query_count": 1,
        "project_source_count": 6,
        "code_index_status": "not_indexed",
        "domains": [
            {
                "domain_id": "request",
                "queries": [{"query": "forced", "content": huge_forced}],
            }
        ],
    }
    prompt_slice = agentic._domain_evidence_slice(
        "request",
        {
            "official_rag": {
                "status": "deferred_until_platform_selected",
                "domains": [],
            },
            "forced_project_rag": forced,
        },
    )

    rendered = json.dumps(prompt_slice, ensure_ascii=False, sort_keys=True)
    assert huge_forced not in rendered
    assert set(prompt_slice) == {"official_rag", "forced_project_rag"}
    assert prompt_slice["forced_project_rag"]["research_sha256"] == "sha256:forced"
    assert prompt_slice["forced_project_rag"]["project_source_count"] == 6
    assert prompt_slice["forced_project_rag"]["evidence_ref"] == "forced_project_rag"
    assert "evidence_document" not in prompt_slice


def test_each_page_reader_prompt_stays_bounded(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MMM_RESEARCH_DOCUMENT_DIR", str(tmp_path))
    evidence = {
        "forced_project_rag": {
            "domain_id": "mk_entity",
            "queries": [{"query": "entity AI", "content": "Z" * 120_000}],
        }
    }
    document = paged_rag._materialize_domain_evidence_document("mk_entity", evidence)
    pages = paged_rag._read_evidence_pages(document)

    for page in pages:
        messages = paged_rag._research_page_messages(
            prompt="Build the requested mod without changing the authoritative scope.",
            domain={"domain_id": "mk_entity", "queries": ["entity AI"]},
            document=document,
            page=page,
        )
        assert len(messages) == 2
        assert len(messages[1]["content"]) < 20_000
        assert str(page["page_ref"]) in messages[1]["content"]


def test_runtime_binding_uses_native_research_first_owner() -> None:
    assert getattr(game_design.GameDesignPlanner.plan, "_mmm_host_owned_template", False)
    assert not hasattr(game_design._generate_game_design_once, "__wrapped__")
    assert not hasattr(game_design.GameDesignPlanner._plan_sharded_request, "__wrapped__")
