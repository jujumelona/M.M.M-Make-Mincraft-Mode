from __future__ import annotations

from jsonschema import Draft202012Validator

from minecraft_mod_ai.worksheet_atomic_chunker import (
    merge_worksheet_section_chunks,
    pack_section_concerns,
    worksheet_chunk_schema,
)


def _behavior_chunks(*, actors=None, inputs_marker=None):
    chunks = []
    for concerns in pack_section_concerns("behavior_contract"):
        chunk = {}
        if actors is not None and "actors" in concerns:
            chunk["actors"] = actors
        if inputs_marker is not None and "inputs" in concerns:
            chunk["inputs"] = inputs_marker
        if not any(
            isinstance(value, (list, dict)) and bool(value)
            for value in chunk.values()
        ):
            chunk["inapplicable_concerns"] = [
                {
                    "concern": concerns[0],
                    "reason": "Not required by this focused test fixture.",
                }
            ]
        chunks.append(chunk)
    return chunks


def test_model_facing_chunk_requires_complete_bounded_field_page() -> None:
    page = pack_section_concerns("behavior_contract")[0]
    schema = worksheet_chunk_schema(
        "behavior_contract",
        page,
        record_counts={"actors": 1},
    )
    validator = Draft202012Validator(schema)
    complete_record = {"name": "player", "role": "actor", "authority": "server"}
    # pack_section_concerns is allowed to split a concern into atomic field
    # pages. Validate exactly the projection of THIS page, not all fields of
    # the complete merged worksheet record.
    projected = tuple(page.field_projection["actors"])
    complete = {field: complete_record[field] for field in projected}
    assert validator.is_valid({"actors": [complete]})
    assert not validator.is_valid({"actors": [{}]})
    missing = {key: value for key, value in complete.items() if key != projected[0]}
    assert not validator.is_valid({"actors": [missing]})
    oversized = {**complete, projected[0]: "a" * 513}
    assert not validator.is_valid({"actors": [oversized]})


def test_merge_salvages_partial_chunk_and_host_fills_omissions() -> None:
    merged = merge_worksheet_section_chunks(
        "behavior_contract",
        _behavior_chunks(
            actors=[{"name": "player"}],
            inputs_marker="wrong scalar shape",
        ),
        set(),
    )

    actors = merged["specification"]["actors"]
    assert len(actors) == 1
    assert actors[0]["name"] == "player"
    assert actors[0]["role"]
    assert actors[0]["authority"]

    assert merged["specification"]["inputs"] == []
    inapplicable = {
        row["concern"]: row["reason"]
        for row in merged["specification"]["inapplicable_concerns"]
    }
    assert "inputs" in inapplicable
    assert inapplicable["inputs"]


def test_merge_combines_multiple_partial_records_for_one_concern() -> None:
    merged = merge_worksheet_section_chunks(
        "behavior_contract",
        _behavior_chunks(
            actors=[{"name": "player"}, {"name": "server"}],
        ),
        set(),
    )

    assert [row["name"] for row in merged["specification"]["actors"]] == [
        "player",
        "server",
    ]
