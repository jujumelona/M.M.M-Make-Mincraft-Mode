from __future__ import annotations

import json
from pathlib import Path

import minecraft_mod_ai.agentic_pre_design_rag as paged_rag
import minecraft_mod_ai.agentic_research_game_design as agentic
import minecraft_mod_ai.game_design as game_design


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
        required = tuple(response_schema["properties"]["section"]["required"])
        values = {
            "title": "연구 기반 모드",
            "pitch": "검색 근거를 바탕으로 설계한다.",
            "core_loop": ["탐색하고 상호작용한다"],
            "progression": ["기능을 단계적으로 해금한다"],
            "combat": {},
            "mod_context": {},
            "modules": [],
            "assets": [],
            "acceptance_tests": ["요청한 핵심 루프가 게임 내에서 동작한다"],
            "art_direction": {},
        }
        return json.dumps(
            {"section": {field: values[field] for field in required}},
            ensure_ascii=False,
        )


class _ResearchRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append({"role": role, "messages": messages, **kwargs})
        return json.dumps(
            {
                "research_note": {
                    "domain_id": "request",
                    "claims": [
                        {
                            "claim": "요청은 일반 Minecraft 모드 기능 설계다.",
                            "evidence_refs": ["official_rag:request"],
                        }
                    ],
                    "gaps": [],
                    "next_queries": [],
                    "sufficient": True,
                }
            },
            ensure_ascii=False,
        )


def test_sectioned_game_design_uses_four_small_schema_calls() -> None:
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

    assert len(router.calls) == 4
    assert all(call["role"] == "planner" for call in router.calls)
    assert all(call["response_format"] == "json" for call in router.calls)
    assert all(call["enable_tools"] is False for call in router.calls)
    assert all(call["response_schema"] for call in router.calls)
    assert result["title"] == "연구 기반 모드"
    assert result["core_loop"]
    assert result["acceptance_tests"]
    assert "art_direction" not in result


def test_research_domain_ledgers_pages_then_runs_one_bounded_synthesis(
    monkeypatch, tmp_path: Path
) -> None:
    router = _ResearchRouter()

    class _Trace:
        def __init__(self, *args, **kwargs):
            self.attempts = []

        def record_attempt(self, **kwargs):
            self.attempts.append(kwargs)

        def record_success(self, value):
            self.success = value

    monkeypatch.setenv("MMM_RESEARCH_DOCUMENT_DIR", str(tmp_path))
    monkeypatch.setattr(agentic, "PlannerStageTrace", _Trace)

    result = agentic._research_domain_with_agent(
        router,
        prompt="기능을 조사해서 설계해줘",
        domain={
            "domain_id": "request",
            "objective": "요청 조사",
            "requirements": ["기능"],
            "evidence_kinds": ["minecraft_api"],
            "queries": ["minecraft mod feature"],
            "providers": ["official_docs", "project_rag", "github"],
            "depends_on": [],
        },
        deterministic={"official_rag": {"domains": []}},
        trace_metadata=None,
    )

    assert result["sufficient"] is True
    assert len(router.calls) == 1
    synthesis_call = router.calls[0]
    assert all(call["tool_stage"] == "research" for call in router.calls)
    assert all(call["response_format"] == "json" for call in router.calls)
    assert all(call["response_schema"] == agentic._RESEARCH_NOTE_SCHEMA for call in router.calls)
    assert synthesis_call["enable_tools"] is False
    assert "evidence_document" in result
    assert result["evidence_ledger"]["record_count"] == 1


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
    assert document["model_projection"] == "lossless_ordered_utf8_fragments"
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


def test_domain_slice_persists_raw_forced_receipt_instead_of_inlining_it(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MMM_RESEARCH_DOCUMENT_DIR", str(tmp_path))
    huge_forced = "F" * 40_000
    forced = {
        "schema_version": "mmm/forced-pre-design-rag-v2",
        "query_count": 1,
        "domains": [
            {
                "domain_id": "request",
                "queries": [{"query": "forced", "content": huge_forced}],
            }
        ],
    }
    token = paged_rag._FORCED_RAG_CONTEXT.set(forced)
    try:
        prompt_slice = agentic._domain_evidence_slice(
            "request",
            {"official_rag": {"domains": []}},
        )
    finally:
        paged_rag._FORCED_RAG_CONTEXT.reset(token)

    rendered = json.dumps(prompt_slice, ensure_ascii=False, sort_keys=True)
    assert huge_forced not in rendered
    assert set(prompt_slice) == {"evidence_document"}

    document = prompt_slice["evidence_document"]
    raw_text = Path(document["raw_path"]).read_text(encoding="utf-8")
    assert huge_forced in raw_text


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


def test_runtime_binding_marks_game_design_as_research_first() -> None:
    assert getattr(
        game_design.GameDesignPlanner.plan,
        "_mmm_agentic_research_sectioned",
        False,
    )
