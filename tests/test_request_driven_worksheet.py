from __future__ import annotations

import json

from minecraft_mod_ai import authored_structured_design as design
from minecraft_mod_ai import planner_state_authoring
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
from minecraft_mod_ai.worksheet_atomic_chunker import (
    pack_section_concerns,
    worksheet_chunk_schema,
)


def test_section_counts_choose_inactive_concerns_before_field_authoring():
    class Router:
        def generate_text(self, role, messages, **kwargs):
            self.messages = messages
            return json.dumps({key: "1" if key == "outputs" else "0"
                               for key in kwargs["response_schema"]["properties"]})

    router = Router()
    counts = design._generate_section_record_counts(
        router, "A tool that teleports players", section="behavior_contract", completed={},
    )
    assert counts == {key: int(key == "outputs") for key in DETAIL_RECORDS["behavior_contract"]}
    assert "zero" in str(router.messages).lower()


def test_inactive_concern_never_calls_model():
    class Router:
        def generate_text(self, *args, **kwargs):
            raise AssertionError("Inactive worksheet concern must not invoke inference")

    pages = pack_section_concerns("behavior_contract")
    request = design._PlannerSectionRequest(
        router=Router(), prompt="teleport", section="behavior_contract", completed={},
        media_paths=(), chunks=pages, slots=1, record_counts={"actors": 0},
    )
    indexed = [(i, page) for i, page in enumerate(pages, 1) if page[0] == "actors"]
    rows, results = design._generate_concern_pages(request, "actors", indexed)
    assert rows == []
    assert results == [(i, {"actors": []}) for i, _ in indexed]


def test_actual_cardinality_packs_fewer_pages_without_truncating_sentences():
    counts = {key: 1 for key in DETAIL_RECORDS["behavior_contract"]}
    pages = pack_section_concerns("behavior_contract", record_counts=counts)
    assert len(pages) < len(pack_section_concerns("behavior_contract"))
    schema = worksheet_chunk_schema("behavior_contract", pages[0], record_counts=counts)
    fields = schema["properties"][pages[0][0]]["items"]["properties"]
    assert any(value.get("maxLength", 0) >= 160 for value in fields.values())


def test_state_authoring_receives_behavior_prerequisites(monkeypatch):
    captured = []

    def author(router, prompt, **kwargs):
        captured.append(prompt)
        return {kwargs["concern"]: []}

    monkeypatch.setattr(planner_state_authoring, "author_state_semantic_page", author)
    monkeypatch.setattr(planner_state_authoring, "author_state_field_page", author)
    for field in ("type", "default"):
        page_spec = next(p for p in pack_section_concerns("state_model")
                         if p[0] == "variables" and field in p.field_projection["variables"])
        page = design._PlannerPageRequest(
            section="state_model", chunk_index=1, chunk_count=1, concerns=page_spec,
            completed={"behavior_contract": {"outputs": "Successful launch consumes fuel"}},
            media_paths=(),
        )
        design._generate_authored_page_value(
            object(), "space travel", (), page,
            worksheet_chunk_schema("state_model", page_spec, record_counts={"variables": 1}),
            None,
        )
    assert len(captured) == 2
    assert all("Successful launch consumes fuel" in prompt for prompt in captured)
