from __future__ import annotations

"""REG-025: source text must never silently lose authored semantic spans."""

from minecraft_mod_ai.semantic_source_fidelity import validate_semantic_source_partition


def _clause(text: str, *, start: int = 0) -> dict:
    return {
        "clause_index": 0,
        "char_start": start,
        "char_end": start + len(text),
        "text": text,
    }


def _node(start: int, end: int) -> dict:
    return {
        "source_clause_index": 0,
        "source_start": start,
        "source_end": end,
    }


def _codes(nodes, clauses):
    return {
        item["error_code"]
        for item in validate_semantic_source_partition(nodes, clauses)
    }


def test_empty_semantic_graph_cannot_claim_full_source_coverage():
    clause = _clause("Mine ore then launch")
    assert "REQ_SOURCE_PARTITION_GAP" in _codes([], [clause])


def test_partial_semantic_partition_reports_unowned_source_tail():
    text = "Mine ore then launch"
    assert "REQ_SOURCE_PARTITION_GAP" in _codes(
        [_node(0, 8)], [_clause(text)]
    )


def test_duplicate_semantic_ownership_is_detected_without_hiding_other_errors():
    text = "Mine ore then launch"
    codes = _codes(
        [_node(0, 10), _node(8, len(text))],
        [_clause(text)],
    )
    assert "REQ_SOURCE_PARTITION_OVERLAP" in codes


def test_exact_source_partition_covers_all_authored_characters():
    text = "Mine ore then launch"
    assert _codes(
        [_node(0, 8), _node(8, len(text))],
        [_clause(text)],
    ) == set()


def test_nonzero_absolute_source_offset_still_detects_dropped_text():
    text = "Build a ship and launch"
    start = 300
    assert "REQ_SOURCE_PARTITION_GAP" in _codes(
        [_node(start, start + 12)],
        [_clause(text, start=start)],
    )
