from __future__ import annotations

from pathlib import Path


source = Path("minecraft_mod_ai/agent_security_contract.py")
text = source.read_text(encoding="utf-8")
start = text.index("def usable_rag_result(value: Any) -> bool:\n")
end = text.index("\ndef _receipt_warning_set(", start)
replacement = '''def usable_rag_result(value: Any) -> bool:\n    """Accept evidence only when receipt authority stays within one result object.\n\n    Receipts are authoritative when present. A positive but unscored receipt may pair\n    with concrete legacy evidence only when both are siblings in the same mapping;\n    receipt state from one nested result must never authorize evidence from another.\n    When no receipt exists anywhere, legacy evidence packs remain compatible.\n    """\n\n    found_receipt = False\n    usable_receipted_result = False\n    legacy_evidence = False\n\n    def visit(item: Any) -> None:\n        nonlocal found_receipt, usable_receipted_result, legacy_evidence\n        if isinstance(item, Mapping):\n            direct_evidence = any(\n                str(key).strip().lower() in _LEGACY_EVIDENCE_KEYS\n                and _nonempty_sequence(child)\n                for key, child in item.items()\n            )\n            if direct_evidence:\n                legacy_evidence = True\n\n            receipt = item.get("receipt")\n            if isinstance(receipt, Mapping):\n                found_receipt = True\n                if _usable_receipt(receipt) or (\n                    _positive_receipt(receipt) and direct_evidence\n                ):\n                    usable_receipted_result = True\n\n            for child in item.values():\n                visit(child)\n        elif isinstance(item, Sequence) and not isinstance(\n            item, (str, bytes, bytearray)\n        ):\n            for child in item:\n                visit(child)\n\n    visit(value)\n    if found_receipt:\n        return usable_receipted_result\n    return legacy_evidence\n'''
source.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


tests = Path("tests/test_agent_security_contract.py")
test_text = tests.read_text(encoding="utf-8")
marker = "def test_rag_gate_keeps_legacy_known_evidence_pack_compatibility() -> None:\n"
addition = '''def test_rag_gate_does_not_cross_join_receipt_and_evidence_from_siblings() -> None:\n    assert not usable_rag_result(\n        {\n            "siblings": [\n                {"receipt": {"result_count": 1}},\n                {"hits": [{"text": "unrelated sibling evidence"}]},\n            ]\n        }\n    )\n\n\ndef test_rag_gate_allows_positive_receipt_with_evidence_in_same_result() -> None:\n    assert usable_rag_result(\n        {\n            "receipt": {"result_count": 1},\n            "hits": [{"text": "same result evidence"}],\n        }\n    )\n'''
if addition.strip() not in test_text:
    if marker not in test_text:
        raise RuntimeError("worker12 RAG test insertion marker missing")
    test_text = test_text.replace(marker, addition + "\n\n" + marker, 1)
tests.write_text(test_text, encoding="utf-8")

print("worker12 RAG receipt-scope hardening applied")
