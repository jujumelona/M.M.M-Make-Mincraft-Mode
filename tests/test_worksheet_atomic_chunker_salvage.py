from __future__ import annotations

from minecraft_mod_ai.worksheet_atomic_chunker import (
    merge_worksheet_section_chunks,
    worksheet_chunk_schema,
)


def test_model_facing_chunk_schema_allows_partial_records() -> None:
    schema = worksheet_chunk_schema(
        "behavior_contract",
        ("actors", "inputs"),
        include_evidence=True,
    )

    assert schema["required"] == []
    assert schema["additionalProperties"] is False
    assert schema["properties"]["actors"]["items"]["required"] == []
    assert schema["properties"]["inputs"]["items"]["required"] == []
    assert "minLength" not in schema["properties"]["actors"]["items"]["properties"]["name"]


def test_merge_salvages_partial_chunk_and_host_fills_omissions() -> None:
    merged = merge_worksheet_section_chunks(
        "behavior_contract",
        [
            {
                "actors": [{"name": "player"}],
                "inputs": "wrong scalar shape",
                "adapter_metadata": {"ignored": True},
            }
        ],
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


def test_merge_combines_duplicate_concern_payloads_instead_of_rejecting_section() -> None:
    merged = merge_worksheet_section_chunks(
        "behavior_contract",
        [
            {"actors": [{"name": "player"}]},
            {"actors": {"name": "server"}},
        ],
        set(),
    )

    assert [row["name"] for row in merged["specification"]["actors"]] == [
        "player",
        "server",
    ]
