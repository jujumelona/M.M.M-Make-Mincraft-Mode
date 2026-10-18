"""Relation authoring uses exhaustive host-owned candidate matrices."""
import json

from minecraft_mod_ai.design_record_runtime import run_record_template


class RelationRouter:
    def __init__(self):
        self.text_calls = []
        self.tool_calls = 0

    def generate_text(self, role, messages, **kwargs):
        assert role == "planner"
        assert kwargs["enable_tools"] is False
        schema = kwargs["response_schema"]
        self.text_calls.append({
            "messages": tuple(messages),
            "schema": schema,
            "ceiling": kwargs["output_token_ceiling"],
        })
        context = json.loads(messages[-1]["content"])
        candidates = context["relation_candidates"]
        assert set(schema["properties"]) == {row["id"] for row in candidates}
        assert all(schema["properties"][row["id"]]["type"] == "boolean"
                   for row in candidates)
        return json.dumps({
            row["id"]: (
                row["source_id"] == "source"
                and row["target_id"] == "target"
                and row["relation_type"] == "requires"
            )
            for row in candidates
        })

    def generate_tool_decision(self, *_args, **_kwargs):
        self.tool_calls += 1
        raise AssertionError("planner relation authoring must not use native tools")


def test_relation_cardinality_is_host_owned_exhaustive_matrix():
    router = RelationRouter()
    result = run_record_template(
        router,
        "design/content_relation",
        context={
            "requirement_id": "req_relations",
            "requirement": "source requires target",
            "entity_ids": ["source", "target"],
            "entities": [
                {"entity_id": name, "kind": "gui", "implementation_obligations": []}
                for name in ("source", "target")
            ],
            "requirements": [],
            "design_contexts": [],
        },
    )
    assert result["records"] == [
        {"relation_type": "requires", "source_id": "source", "target_id": "target"}
    ]
    assert router.tool_calls == 0
    assert router.text_calls
    assert all(int(call["ceiling"]) > 0 for call in router.text_calls)
    # The older count-then-one-edge approach is forbidden: host supplies every
    # closed candidate and the model decides booleans, never an unbounded count.
    assert all(
        set(call["schema"]["properties"]) != {"count"}
        for call in router.text_calls
    )
