import json

from minecraft_mod_ai.design_record_runtime import run_record_template
from minecraft_mod_ai.model_output_atomicity_contract import assert_atomic_model_schema


class RelationRouter:
    def __init__(self):
        self.calls = []
        self.edges = [
            "consumes",
            "produces",
            "contains",
            "drops",
            "requires",
        ]

    def generate_tool_decision(
        self, role, messages, *, tool_name, parameters, **kwargs
    ):
        self.calls.append(tool_name)
        assert_atomic_model_schema(parameters, surface=tool_name)
        context = json.loads(messages[-1]["content"])
        source_id = context["source_id"]
        target_id = context["target_id"]
        pair_edges = self.edges if (source_id, target_id) == ("source", "target") else []

        if tool_name == "submit_one_design_content_relation_count":
            return {"count": len(pair_edges)}
        if tool_name == "submit_one_design_content_relation":
            index = int(context["record_index"])
            assert context["record_count"] == len(pair_edges)
            return {"relation_type": pair_edges[index]}
        raise AssertionError(tool_name)


def test_relation_cardinality_is_host_owned_without_array_truncation():
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
        {"relation_type": relation_type, "source_id": "source", "target_id": "target"}
        for relation_type in router.edges
    ]
    assert router.calls.count("submit_one_design_content_relation_count") == 2
    assert router.calls.count("submit_one_design_content_relation") == 5
    assert "submit_one_design_relation_set" not in router.calls
