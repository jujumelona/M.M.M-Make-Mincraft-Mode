from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai import research_synthesis_performance as synthesis


def _fixed_group(notes):
    groups = []
    current = []
    current_bytes = 2
    for note in notes:
        size = len(json.dumps(note, sort_keys=True).encode("utf-8"))
        if current and (len(current) >= 4 or current_bytes + size > 3_600):
            groups.append(current)
            current = []
            current_bytes = 2
        current.append(note)
        current_bytes += size + 1
    if current:
        groups.append(current)
    return groups


def _module():
    module = SimpleNamespace(
        _SYNTHESIS_INPUT_BYTES=3_600,
        _SYNTHESIS_GROUP_ITEMS=4,
        _EVIDENCE_PAGE_CHARS=1_800,
    )
    module._group_synthesis_notes = _fixed_group

    def hierarchy(_runtime, _router, *, page_notes, **_kwargs):
        return module._group_synthesis_notes(page_notes)

    module._hierarchical_synthesis = hierarchy
    return module


def _qwen_router():
    return SimpleNamespace(
        profile="Qwen3.5-9B_6GB",
        registry=SimpleNamespace(
            role=lambda _profile, _role: SimpleNamespace(
                max_context=32_768,
                max_input_tokens=0,
                max_new_tokens=8_192,
            )
        ),
    )


def _notes(count=56):
    return [
        {
            "domain_id": "mk_combat",
            "claims": [],
            "gaps": [],
            "next_queries": [],
            "sufficient": True,
            "evidence_fragment": {
                "page_ref": f"page:{index}",
                "content": "X" * 1_500,
            },
        }
        for index in range(count)
    ]


def test_qwen_context_collapses_many_lossless_pages() -> None:
    module = _module()
    synthesis.harden(module)
    notes = _notes()

    groups = module._hierarchical_synthesis(
        object(),
        _qwen_router(),
        page_notes=notes,
    )

    max_bytes, max_items = synthesis._planner_limits(_qwen_router(), module)
    assert max_bytes == 20_480
    assert max_items > 4
    assert len(groups) <= 7
    assert [item for group in groups for item in group] == notes
    assert all(synthesis._encoded_size(group) <= max_bytes for group in groups)


def test_context_budget_is_scoped_to_hierarchy_call() -> None:
    module = _module()
    synthesis.harden(module)
    notes = _notes(12)

    dynamic = module._hierarchical_synthesis(
        object(),
        _qwen_router(),
        page_notes=notes,
    )
    fallback = module._group_synthesis_notes(notes)

    assert len(dynamic) < len(fallback)
    assert fallback == _fixed_group(notes)


def test_live_runtime_context_caps_registry_budget(monkeypatch) -> None:
    module = _module()
    monkeypatch.setenv(
        "MMM_LLAMA_RUNTIME_RECEIPT",
        json.dumps({"context_per_slot": 16_384, "slots": 2}),
    )

    max_bytes, max_items = synthesis._planner_limits(_qwen_router(), module)

    assert max_bytes == 4_096
    assert max_items == 4


def test_unknown_router_preserves_conservative_fallback() -> None:
    module = _module()
    synthesis.harden(module)
    notes = _notes(12)

    groups = module._hierarchical_synthesis(
        object(),
        object(),
        page_notes=notes,
    )

    assert groups == _fixed_group(notes)
