from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import small_model_predesign_research as research


def _pages(count: int):
    return [
        {
            "page_ref": f"page:{index}",
            "content": f"evidence body quote-{index} useful implementation",
        }
        for index in range(count)
    ]


class _Registry:
    def role(self, profile, role):
        assert role == "planner"
        return SimpleNamespace(max_new_tokens=20)


class _ExactRouter:
    profile = "test"
    registry = _Registry()

    def __init__(self):
        self.calls = []

    def input_context_accounting(self, role, messages, **kwargs):
        count = sum(
            line.startswith("SOURCE PAGE_REF=")
            for line in messages[-1]["content"].splitlines()
        )
        return SimpleNamespace(input_tokens=count * 40, context_tokens=100)

    def generate_text(self, role, messages, **kwargs):
        self.calls.append(messages)
        lines = []
        for line in messages[-1]["content"].splitlines():
            if line.startswith("SOURCE PAGE_REF="):
                ref = line.split("=", 1)[1]
                index = ref.split(":", 1)[1]
                lines.append(f"EVIDENCE\t{ref}\tquote-{index}\tclaim-{index}")
        return "\n".join(lines)


def test_candidate_order_keeps_every_page_instead_of_fixed_fallback_cutoff():
    pages = _pages(7)
    chosen = research._candidate_pages(
        pages,
        {"objective": "unmatched", "queries": []},
    )
    assert [page["page_ref"] for page in chosen] == [
        page["page_ref"] for page in pages
    ]


def test_live_context_capacity_sets_batch_boundaries_not_page_count():
    router = _ExactRouter()
    batches, diagnostics = research._capacity_batches(
        router,
        domain={"objective": "implementation"},
        pages=_pages(5),
    )
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert not diagnostics


def test_batch_extraction_preserves_more_than_three_supported_claims():
    router = _ExactRouter()
    claims, diagnostics, calls = research._extract_batch(
        router,
        domain={"objective": "implementation"},
        pages=_pages(5),
    )
    assert calls == 1
    assert len(claims) == 5
    assert not diagnostics


def test_accounting_unavailable_keeps_all_pages_without_new_fixed_cap():
    class Router:
        profile = "test"
        registry = _Registry()

    batches, diagnostics = research._capacity_batches(
        Router(),
        domain={"objective": "implementation"},
        pages=_pages(9),
    )
    assert [len(batch) for batch in batches] == [9]
    assert (
        "exact_input_accounting_unavailable;all_pages_kept_in_one_batch"
        in diagnostics
    )
