from __future__ import annotations

import minecraft_mod_ai.design_record_runtime as runtime


def test_relation_pages_cannot_author_duplicate_relation_type_siblings(monkeypatch) -> None:
    """A fixed entity pair must never author duplicate relation-type siblings."""

    monkeypatch.setattr(
        runtime,
        "deterministic_model_map",
        lambda _router, values, worker, **_kwargs: [worker(value) for value in values],
    )

    seen_relation_pages: list[tuple[str, ...]] = []

    def fake_selector(
        _router,
        _template,
        _normalized,
        *,
        response_schema,
        **_kwargs,
    ):
        properties = response_schema["properties"]
        assert all(child["type"] == "boolean" for child in properties.values())
        fields = tuple(properties)
        seen_relation_pages.append(fields)
        return {field: index < 2 for index, field in enumerate(fields)}

    monkeypatch.setattr(runtime, "_relation_selector", fake_selector)

    records, _ = runtime._run_relations(
        object(),
        "design/content_relation",
        {
            "requirement": "Two UI panels explicitly depend on each other in different ways.",
            "requirements": [],
            "design_contexts": [],
            "entity_ids": ["alpha", "beta"],
            "entities": [
                {"entity_id": "alpha", "kind": "gui", "implementation_obligations": []},
                {"entity_id": "beta", "kind": "gui", "implementation_obligations": []},
            ],
        },
        None,
        None,
    )

    by_pair: dict[tuple[str, str], list[str]] = {}
    for record in records:
        pair = (record["source_id"], record["target_id"])
        by_pair.setdefault(pair, []).append(record["relation_type"])

    assert set(by_pair) == {("alpha", "beta"), ("beta", "alpha")}
    for relation_types in by_pair.values():
        assert len(relation_types) == 2
        assert len(relation_types) == len(set(relation_types))

    # One boolean per host-owned edge makes duplicates structurally impossible;
    # there is no count request or a separate request for each selected type.
    assert len(seen_relation_pages) == 2
    assert all(len(page) == len(set(page)) for page in seen_relation_pages)
