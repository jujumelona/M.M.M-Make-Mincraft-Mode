import json

from minecraft_mod_ai.design_record_runtime import run_record_template


class RelationRouter:
    def __init__(self):
        self.text_calls = []
        self.tool_calls = 0
        self.edges = [
            "consumes",
            "produces",
            "contains",
            "drops",
            "requires",
        ]

    def generate_text(self, role, messages, **kwargs):
        assert role == "planner"
        assert kwargs["response_format"] == "json"
        assert kwargs["enable_tools"] is False
        assert kwargs["force_non_thinking"] is True
        schema = kwargs["response_schema"]
        self.text_calls.append(
            {
                "messages": tuple(messages),
                "schema": schema,
                "ceiling": kwargs["output_token_ceiling"],
            }
        )

        context = json.loads(messages[1]["content"])
        source_id = context["source_id"]
        target_id = context["target_id"]
        pair_edges = (
            self.edges
            if (source_id, target_id) == ("source", "target")
            else []
        )

        if set(schema["properties"]) == {"count"}:
            return json.dumps({"count": len(pair_edges)})

        assert set(schema["properties"]) == {"relation_type"}
        index = int(context["record_index"])
        return json.dumps({"relation_type": pair_edges[index]})

    def generate_tool_decision(self, *_args, **_kwargs):
        self.tool_calls += 1
        raise AssertionError("planner relation authoring must not use native tools")


def test_relation_cardinality_is_bounded_count_then_host_owned_ordinals():
    router = RelationRouter()
    result = run_record_template(
        router,
        "design/content_relation",
        context={
            "requirement_id": "req_relations",
            "requirement": "source has five explicit relations to target",
            "entity_ids": ["source", "target"],
        },
    )

    assert result["records"] == [
        {
            "relation_type": relation_type,
            "source_id": "source",
            "target_id": "target",
        }
        for relation_type in router.edges
    ]
    assert router.tool_calls == 0

    count_calls = [
        call
        for call in router.text_calls
        if set(call["schema"]["properties"]) == {"count"}
    ]
    record_calls = [
        call
        for call in router.text_calls
        if set(call["schema"]["properties"]) == {"relation_type"}
    ]
    assert len(count_calls) == 2
    assert len(record_calls) == len(router.edges)
    assert all(int(call["ceiling"]) > 0 for call in router.text_calls)
