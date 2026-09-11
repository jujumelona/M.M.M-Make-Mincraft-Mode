from __future__ import annotations

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


def test_model_facing_chunk_schema_allows_partial_bounded_records() -> None:
    schema = worksheet_chunk_schema(
        "behavior_contract",
        ("actors", "inputs"),
        include_evidence=True,
    )

    assert schema["required"] == []
    assert schema["additionalProperties"] is False
    assert schema["properties"]["actors"]["items"]["required"] == []
    assert schema["properties"]["inputs"]["items"]["required"] == []
    name_schema = schema["properties"]["actors"]["items"]["properties"]["name"]
    assert name_schema["minLength"] == 1
    assert name_schema["maxLength"] == 256


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
