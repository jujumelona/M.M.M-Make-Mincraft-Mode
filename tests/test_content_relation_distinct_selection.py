from __future__ import annotations

import minecraft_mod_ai.design_record_runtime as runtime


def test_relation_pair_selection_removes_previously_selected_types(monkeypatch) -> None:
    """A fixed entity pair must never author duplicate relation-type siblings."""

    monkeypatch.setattr(
        runtime,
        "deterministic_model_map",
        lambda _router, values, worker, **_kwargs: [worker(value) for value in values],
    )

    seen_relation_enums: list[tuple[str, ...]] = []

    def fake_selector(
        _router,
        _template,
        _normalized,
        *,
        response_schema,
        **_kwargs,
    ):
        properties = response_schema["properties"]
        if "count" in properties:
            return {"count": 2}
        relation_enum = tuple(properties["relation_type"]["enum"])
        seen_relation_enums.append(relation_enum)
        return {"relation_type": relation_enum[0]}

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

    # Each pair authors two members. The second member must see an enum with the
    # first choice physically removed, so duplicate output is impossible even if
    # the model always selects the first enum member.
    assert len(seen_relation_enums) == 4
    for first, second in zip(seen_relation_enums[::2], seen_relation_enums[1::2], strict=True):
        assert len(second) == len(first) - 1
        assert first[0] not in second
