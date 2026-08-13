from pathlib import Path

path = Path("tests/test_production_page_durable_contract.py")
text = path.read_text(encoding="utf-8")
needle = '''    assert len(router.calls) == 2
    assert all(
        call["request"]["repair_mode"] in {"field_patch", "replacement"}
        for call in router.calls
    )
'''
replacement = '''    # The unchanged semantic validation state is already proven after the first
    # repair response, so a second identical LLM call would be wasted work.
    assert len(router.calls) == 1
    assert router.calls[0]["request"]["repair_mode"] == "field_patch"
'''
if needle not in text:
    # Keep the migration robust to line wrapping while still scoping it to this test.
    marker = "def test_repeated_invalid_model_output_stops_exact_cycle("
    start = text.find(marker)
    if start < 0:
        raise SystemExit("exact-cycle test not found")
    end = text.find("\ndef ", start + len(marker))
    if end < 0:
        end = len(text)
    section = text[start:end]
    if "assert len(router.calls) == 2" not in section:
        raise SystemExit("old exact-cycle call count not found")
    section = section.replace("assert len(router.calls) == 2", "assert len(router.calls) == 1", 1)
    text = text[:start] + section + text[end:]
else:
    text = text.replace(needle, replacement, 1)
path.write_text(text, encoding="utf-8")
print("exact no-progress call-count regression aligned")
