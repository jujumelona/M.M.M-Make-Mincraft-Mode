from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing expected block in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def normalize_host_queries() -> None:
    path = "minecraft_mod_ai/authored_scope_research_contract.py"
    replace_once(
        path,
        '''        for query in queries:
            value = _query_text(query)
            if _is_english_retrieval_query(value) and value.casefold() not in {q.casefold() for q in cleaned}:
                cleaned.append(value)
            if len(cleaned) >= 5:
                break
''',
        '''        for query in queries:
            value = _query_text(query)
            if value and "minecraft" not in value.casefold():
                value = f"minecraft {value}"
            if _is_english_retrieval_query(value) and value.casefold() not in {q.casefold() for q in cleaned}:
                cleaned.append(value)
            if len(cleaned) >= 5:
                break
''',
    )


def replace_obsolete_json_synthesis_test() -> None:
    path = "tests/test_agentic_research_game_design.py"
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    start_marker = "def test_all_lossless_evidence_fragments_reach_bounded_synthesis(\n"
    end_marker = "def test_domain_slice_bounds_forced_receipt_without_materializing_document() -> None:\n"
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError("obsolete JSON synthesis test block was not found")
    replacement = '''def test_legacy_paged_entrypoint_uses_small_model_text_and_host_quote_verification(
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
            assert rendered.startswith("OBJECTIVE\\n")
            assert "\\nSOURCE\\n" in rendered
            assert "Minecraft Fabric combat damage registration example" in rendered
            return (
                "EVIDENCE\\tMinecraft Fabric combat damage registration example"
                "\\tUse a combat damage registration pattern."
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


'''
    text = text[:start] + replacement + text[end:]
    text = text.replace("from collections import Counter\n", "", 1)
    target.write_text(text, encoding="utf-8")


def main() -> None:
    normalize_host_queries()
    replace_obsolete_json_synthesis_test()


if __name__ == "__main__":
    main()
