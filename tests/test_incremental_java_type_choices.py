"""Java type pages cannot confuse a declaration category with a value's type."""
from __future__ import annotations

import json

from test_incremental_java_recovery import _builder


def test_local_type_page_uses_types_instead_of_open_declaration_words():
    from minecraft_mod_ai.incremental_java_schema import SCHEMAS
    class Coder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            values = parameters["properties"]["type"].get("enum")
            assert values is not None, "the type page is still an unconstrained code string"
            assert "method" not in values
            assert "int" in values and "String" in values
            return {"type": "int"}
    assert _builder(Coder()).typed(SCHEMAS["local"], "local", "type") == {"type": "int"}


def test_type_lookup_retains_metadata_and_makes_actual_platform_types_selectable():
    from minecraft_mod_ai.incremental_java_schema import SCHEMAS
    class Coder:
        selected = False

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            if data_name.endswith("_field"):
                return {"type": "@lookup", "purpose": "Observe the target world", "static": False, "final": False}
            if data_name.endswith("_lookup"):
                return {"query": "example.World", "offset": 0}
            assert "example.World" in parameters["properties"]["type"]["enum"]
            assert task["current_type_request"]["field"] == "type"
            assert task["current_type_request"]["metadata"]["purpose"] == "Observe the target world"
            assert task["current_type_request"]["metadata"]["static"] is False
            return {"type": "example.World"}
    builder = _builder(Coder())
    builder.api._queries["example.World"] = ([], [])
    original_lookup = builder.api.lookup
    builder.api.lookup = lambda query, offset=0: ({"query": query, "offset": offset, "total": 1,
        "next_offset": None, "declarations": [{"kind": "CLASS", "owner": "example.World"}], "members": []}
        if query == "example.World" else original_lookup(query, offset))
    result = builder.typed(SCHEMAS["field"], "field", "type")
    assert result == {"type": "example.World", "purpose": "Observe the target world", "static": False, "final": False}


def test_type_selection_can_construct_a_generic_array_from_host_delimiters():
    class Coder:
        steps = iter(["@array", "@parameterized", "java.util.List", "String"])

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            if "count" in parameters["properties"]:
                return {"count": 1}
            return {"type": next(self.steps)}
    builder = _builder(Coder())
    builder.latest_api = {"declarations": [{"kind": "CLASS", "owner": "java.util.List", "type_parameters": ["E"]}]}
    assert builder.select_java_type("local_type") == "java.util.List<String>[]"
