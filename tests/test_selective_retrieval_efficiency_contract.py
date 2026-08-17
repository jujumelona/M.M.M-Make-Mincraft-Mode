from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai import custom_generation_search_contract as search_contract
from minecraft_mod_ai import custom_module_generator as generator_module


class _DirectRouter:
    def __init__(self) -> None:
        self.calls = 0

    def generate_text(self, role, messages, **kwargs):
        self.calls += 1
        assert role == "coder"
        assert kwargs.get("enable_tools") is False
        return '{"operations":[]}'


def test_structural_repair_bypasses_research_engine(tmp_path: Path) -> None:
    direct = _DirectRouter()
    wrapper = search_contract._ResearchEvidenceRouter(
        direct,
        owner=object(),
        project_root=tmp_path,
        module=None,
        minecraft_version="1.20.1",
        loader="fabric",
        mappings="1.20.1+build.10",
    )

    def forbidden_engine():
        raise AssertionError("structural repair must not enter ResearchCodeContext")

    wrapper._engine = forbidden_engine  # type: ignore[method-assign]
    result = wrapper.generate_text(
        "coder",
        [
            {
                "role": "system",
                "content": "You are repairing only the JSON/patch/precondition shape.",
            },
            {
                "role": "user",
                "content": "Correct that exact structural failure using supplied host evidence.",
            },
        ],
        response_format="json",
        enable_tools=False,
    )

    assert result == '{"operations":[]}'
    assert direct.calls == 1
    assert getattr(
        search_contract._ResearchEvidenceRouter.generate_text,
        "_mmm_structural_no_rag",
        False,
    )


def test_exact_source_anchor_payload_is_only_sent_on_first_page() -> None:
    records = []
    for index in range(12):
        sentinel = "FIRST_PAGE_SOURCE_FACT" if index == 0 else f"SOURCE_{index}"
        records.append(
            {
                "observation_id": f"obs-{index}",
                "path": f"src/main/java/example/A{index:02d}.java",
                "sha256": f"sha256:{index:064x}",
                "content_start_bytes": 0,
                "content_end_bytes": 700,
                "source_page_index": index,
                "kind": "exact_source_excerpt",
                "text": (
                    f"required contract {sentinel} crossFileHook "
                    + ("source-body-should-not-repeat " * 22)
                ),
            }
        )
    ledger = {
        "receipt": {
            "schema_version": "mmm/source-observation-receipt-v1",
            "project_sha256": "sha256:" + "a" * 64,
            "query_sha256": "sha256:" + "b" * 64,
        },
        "records": records,
    }

    pages = generator_module._observation_context_pages(
        ledger,
        query="crossFileHook navigation",
        byte_budget=4096,
    )

    assert len(pages) > 1
    assert pages[0]["global_anchor_payload"] == "exact_source"
    assert "source-body-should-not-repeat" in " ".join(
        item["text"] for item in pages[0]["global_anchors"]
    )
    for page in pages[1:]:
        assert page["global_anchor_payload"] == "compact_refs"
        joined = " ".join(item["text"] for item in page["global_anchors"])
        assert "FIRST_PAGE_SOURCE_FACT" in joined
        assert "source-body-should-not-repeat" not in joined
        assert page["policy"]["global_anchor_source_payload"] == "first_page_only"

    assert getattr(
        generator_module._observation_context_pages,
        "_mmm_first_page_anchor_payload",
        False,
    )
