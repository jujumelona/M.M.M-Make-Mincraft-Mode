from __future__ import annotations

from pathlib import Path

SOURCE = Path("minecraft_mod_ai/semantic_requirement_authority.py")
TEST = Path("tests/test_semantic_requirement_authority.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


source = SOURCE.read_text(encoding="utf-8")

source = replace_once(
    source,
    '                        "source_quote": {"type": "string", "minLength": 1},\n',
    "",
    "remove model-owned source_quote schema field",
)
source = replace_once(
    source,
    '                        "capability_id", "source_quote", "semantic_statement",\n',
    '                        "capability_id", "semantic_statement",\n',
    "remove source_quote from required schema fields",
)
source = replace_once(
    source,
    '                        "case dotted semantic identifier, never an opaque hash. source_quote must be "\n                        "the smallest unique verbatim substring of current_clause that grounds that "\n                        "requirement. Adjacent clauses are context only and may not become new output "\n',
    '                        "case dotted semantic identifier, never an opaque hash. Source provenance, "\n                        "source text, and source offsets are host-owned; do not copy or rewrite the "\n                        "authored clause. Adjacent clauses are context only and may not become new output "\n',
    "make provenance ownership explicit in prompt",
)
source = replace_once(
    source,
    '                                "source_quote": item.get("source_quote"),\n',
    '                                "source_quote": str(clause["text"]),\n',
    "ignore legacy model source_quote",
)
source = replace_once(
    source,
    '''        clause = clauses[clause_index]\n        quote = str(raw.get("source_quote") or "").strip()\n        receipt = _find_quote(clause, quote)\n        if receipt is None:\n            return None, _diagnostic(\n                "REQ_SOURCE_GROUNDING",\n                path + ".source_quote",\n                quote,\n                "one unique verbatim substring inside the assigned host clause",\n                path,\n            )\n\n''',
    '''        clause = clauses[clause_index]\n        # Source provenance is host-owned. The semantic model is not allowed to\n        # transcribe authored text because copy/spacing errors (especially in\n        # non-English prompts) must never invalidate otherwise-correct meaning.\n        quote = str(clause["text"])\n        receipt = (int(clause["char_start"]), int(clause["char_end"]))\n\n''',
    "replace model quote validation with host receipt",
)
source = replace_once(
    source,
    '''    for clause in clauses:\n        members = [\n            item\n            for item in normalized\n            if item["provenance_role"] == "explicit"\n            and item["source_clause_index"] == clause["clause_index"]\n        ]\n        if len(members) > 1 and all(\n            item["source_quote"] == clause["text"] for item in members\n        ):\n            return None, _diagnostic(\n                "REQ_SOURCE_AMBIGUITY",\n                "$.requirements",\n                {\n                    "source_clause_index": clause["clause_index"],\n                    "local_ids": [item["local_id"] for item in members],\n                },\n                "multiple independent meanings need distinct smallest verbatim source_quote grounding",\n                "$.requirements",\n            )\n\n''',
    '''    # Multiple semantic requirements may legitimately originate from one\n    # authored clause. They share the same host-owned source receipt; semantic\n    # identity is carried by capability_id/semantic_statement, not by forcing the\n    # model to manufacture distinct textual provenance spans.\n\n''',
    "remove model-copy ambiguity gate",
)
SOURCE.write_text(source, encoding="utf-8")


test = TEST.read_text(encoding="utf-8")
test = replace_once(
    test,
    '    assert "provenance_role" not in props\n    assert "depends_on" not in props\n',
    '    assert "provenance_role" not in props\n    assert "depends_on" not in props\n    assert "source_quote" not in props\n',
    "assert source quote is absent from tool schema",
)
test = replace_once(
    test,
    '''def test_multiple_meanings_in_one_clause_need_distinct_grounding_quotes():\n    prompt = "Players gather crystals, then trade crystals."\n    router = TextRouter([\n        {"requirements": [_item("resource.gathering", "gather crystals")]},\n        {"requirements": [_item("economy.trade", "trade crystals")]},\n    ])\n    catalog = build_approved_requirement_catalog(prompt, router)\n    assert len(catalog["requirements"]) == 2\n    assert {r["source_span"]["text"] for r in catalog["requirements"]} == {"gather crystals", "trade crystals"}\n\n\n''',
    '''def test_source_receipts_are_host_owned_even_when_model_quote_is_wrong():\n    prompt = "우주선을 부위마다 만들어서 만들 수 있고 조립할 수 있다."\n    wrong_copy = _item("vehicle.spacecraft.assembly", "우무선을 부위마다 만들어서 만들수있고")\n    router = TextRouter([{"requirements": [wrong_copy]}])\n\n    catalog = build_approved_requirement_catalog(prompt, router)\n\n    requirement = catalog["requirements"][0]\n    span = requirement["source_span"]\n    assert prompt[span["char_start"]:span["char_end"]] == span["text"]\n    assert span["text"] != wrong_copy["source_quote"]\n    assert requirement["capability"] == "vehicle.spacecraft.assembly"\n\n\ndef test_host_owned_provenance_allows_multiple_requirements_from_one_clause():\n    prompt = "Players can gather and trade crystals."\n    router = TextRouter([{\n        "requirements": [\n            _item("resource.gathering", "bad copy one"),\n            _item("economy.trade", "bad copy two"),\n        ]\n    }])\n\n    catalog = build_approved_requirement_catalog(prompt, router)\n\n    assert len(catalog["requirements"]) == 2\n    spans = [requirement["source_span"] for requirement in catalog["requirements"]]\n    assert spans[0] == spans[1]\n    assert prompt[spans[0]["char_start"]:spans[0]["char_end"]] == spans[0]["text"]\n\n\n''',
    "replace model-owned quote regression tests",
)
TEST.write_text(test, encoding="utf-8")
