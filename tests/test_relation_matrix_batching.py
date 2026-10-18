"""Every admissible edge is considered without an inference per ordered pair."""
import json

from minecraft_mod_ai import design_record_runtime as runtime


def test_relation_pages_preserve_all_candidates_and_resume_without_model_calls(monkeypatch):
    names = [f"panel_{i}" for i in range(12)]
    context = {
        "requirements": [], "design_contexts": [], "entity_ids": names,
        "entities": [{"entity_id": name, "kind": "gui",
                      "implementation_obligations": []} for name in names],
    }
    calls, observed, saved = [], [], {}

    def generate(_router, _role, messages, *, response_schema, **_kwargs):
        payload = json.loads(messages[-1]["content"])
        candidates = payload["relation_candidates"]
        calls.append(candidates)
        observed.extend((row["source_id"], row["target_id"], row["relation_type"])
                        for row in candidates)
        assert set(response_schema["required"]) == {row["id"] for row in candidates}
        return {row["id"]: (row["source_id"] == "panel_0"
                                and row["target_id"] == "panel_11"
                                and row["relation_type"] == "requires")
                for row in candidates}

    monkeypatch.setattr(runtime, "generate_fixed_template_value", generate)
    monkeypatch.setattr(runtime, "deterministic_model_map",
                        lambda _r, jobs, worker, **_kw: [worker(job) for job in jobs])
    records, _ = runtime._run_relations(
        object(), "design/content_relation", context, {},
        lambda key, value: saved.__setitem__(key, value),
    )
    expected = {
        (source, target, relation) for source in names for target in names
        if source != target for relation in runtime._relation_vocabulary()
        if runtime.relation_type_supported_for_content_pair(relation, "gui", "gui")
    }
    assert len(observed) == len(set(observed))
    assert set(observed) == expected
    assert records == [{"relation_type": "requires", "source_id": "panel_0",
                        "target_id": "panel_11"}]
    assert len(calls) < len(names) * (len(names) - 1)
    first_count = len(calls)
    resumed, _ = runtime._run_relations(
        object(), "design/content_relation", context, saved, None,
    )
    assert resumed == records
    assert len(calls) == first_count


def test_relation_page_rejects_unsolicited_edges(monkeypatch):
    from jsonschema import ValidationError
    import pytest

    monkeypatch.setattr(runtime, "generate_fixed_template_value",
                        lambda *_a, **_kw: {"invented_edge": True})
    monkeypatch.setattr(runtime, "deterministic_model_map",
                        lambda _r, jobs, worker, **_kw: [worker(job) for job in jobs])
    context = {"entity_ids": ["a", "b"], "entities": [
        {"entity_id": name, "kind": "gui", "implementation_obligations": []}
        for name in ("a", "b")], "requirements": [], "design_contexts": []}
    with pytest.raises(ValidationError):
        runtime._run_relations(object(), "design/content_relation", context, {}, None)
