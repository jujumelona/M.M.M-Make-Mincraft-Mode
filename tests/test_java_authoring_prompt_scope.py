"""Large host inventories must not become one small Java inference request."""
from __future__ import annotations

import json

from test_direct_feature_source import SourceCoder, _fixture, _run_java
from minecraft_mod_ai.first_pass_java_source import compose_first_pass_java


def _large_host_context():
    state_methods = [
        {"name": "readState_balance", "signature": "public static int readState_balance();"},
        {"name": "writeState_balance", "signature": "public static void writeState_balance(int value);"},
    ] + [{"name": f"readState_other_{i}",
          "signature": f"public static String readState_other_{i}();"} for i in range(2000)]
    selected = {"kind": "METHOD", "owner": "example.World", "name": "perform",
                "parameters": ["java.lang.String", "int"], "return_type": "boolean",
                "generic_parameters": ["java.lang.String", "int"]}
    return {
        "requirement": "원문 요구를 보존해서 새로운 동작을 구현한다",
        "original_feature_goal": "원문 요구를 보존해서 새로운 동작을 구현한다",
        "assigned_behaviors": [], "target": {"minecraft_version": "26.2", "java_version": 25},
        "inspected_target_api": {
            "target_api": [selected],
            "class_members": [{"verified_class": "example.World", "declaration": {
                "kind": "CLASS", "owner": "example.World", "name": "World"},
                "members": [selected, *({"kind": "METHOD", "owner": "example.World",
                    "name": f"unrelated_{i}", "parameters": [], "return_type": "void",
                    "binary_debug_metadata": "BINARY_METADATA_NOT_MODEL_INPUT" * 60}
                    for i in range(120))]}],
        },
        "materialized_prerequisite_api": [{
            "path": "src/main/java/example/State.java", "imports": [], "types": [{
                "owner": "example.State", "declaration": "public class State",
                "members": [row["signature"] for row in state_methods],
            }],
        }],
        "host_state_access": [{"owner": "example.State", "methods": state_methods}],
        "host_state_work_order_scope": {"declared_readable": ["balance", *(
            f"other_{i}" for i in range(2000))], "declared_writable": ["balance"]},
        "available_fabric_callbacks": [{
            "event_id": "example.RealEvents#ACTION", "event_owner": "example.RealEvents",
            "event_field": "ACTION", "event_interface": "example.RealEvents.Action",
            "method": "onAction", "parameters": ["example.Player", "java.lang.String"],
            "return_type": "void", "type_parameters": [], "throws": [],
        }],
        "retained_typed_behavior": {
            "functions": [{"name": "existingPurchase", "return_type": "boolean",
                           "parameters": [{"name": "item", "type": "String"}],
                           "body": [{"op": "existing_effect", "data": "RETAINED_BODY_NOT_MODEL_INPUT" * 12000}]}],
            "ui_owned_records": [{"action": "existingPurchase", "source_quote": "existing purchase panel"}],
        },
        "owned_runtime_records": [{"name": "selectedLocation", "type": "String"}],
    }


def test_every_direct_authoring_stage_uses_its_own_small_context_and_reaches_executable_java(tmp_path):
    behavior, spec, context = _fixture()
    behavior.update({"state_reads": ["balance"], "state_writes": ["balance"]})
    payload = _large_host_context()
    payload["assigned_behaviors"] = [behavior]
    payload["host_java_authoring"] = True
    context["host_state_access"] = payload["host_state_access"]
    context["host_state_work_order_scope"] = payload["host_state_work_order_scope"]
    spec["bindings"]["retained_typed_behavior_json"] = json.dumps(payload["retained_typed_behavior"])
    spec["bindings"]["shared_state_contract_json"] = json.dumps({"variables": [{
        "name": f"other_{i}", "type": "string", "initial": "unrelated state value",
        "description": "Original state declaration remains in the host"}
        for i in range(2000)]})
    raw_source = "RAW_PROJECT_SOURCE_NOT_MODEL_INPUT" * 16000
    original = json.dumps(payload)

    class BoundedCoder(SourceCoder):
        observations = []

        def _observe(self, messages, name):
            value = json.dumps(messages, ensure_ascii=False)
            size = len(value.encode())
            assert size < 32768, f"one Java action received {size} input bytes: {name}"
            assert "RAW_PROJECT_SOURCE_NOT_MODEL_INPUT" not in value
            assert "RETAINED_BODY_NOT_MODEL_INPUT" not in value
            assert "BINARY_METADATA_NOT_MODEL_INPUT" not in value
            self.observations.append((name, value))

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            self._observe(messages, data_name)
            task = json.loads(messages[-1]["content"])
            assert task["owned_runtime_records"] == payload["owned_runtime_records"]
            assert task["retained_typed_behavior"]["ui_owned_records"][0]["action"] == "existingPurchase"
            if data_name == "submit_java_construction_plan_count":
                combined = "\n".join(m["content"] for m in messages)
                assert behavior["trigger"] in combined
                assert payload["requirement"] in combined
            return super().generate_structured_data(role, messages, data_name=data_name, parameters=parameters)

        def generate_java_source(self, role, messages, *, data_name, source_frame):
            self._observe(messages, data_name)
            task = json.loads(messages[-1]["content"])
            assert task["available_fabric_callbacks"][0]["event_id"] == "example.RealEvents#ACTION"
            assert task["retained_typed_behavior"]["ui_owned_records"][0]["action"] == "existingPurchase"
            assert task["host_state_access"][0]["methods"] == payload["host_state_access"][0]["methods"][:2]
            assert "readState_other_" not in messages[-1]["content"]
            assert "perform" in messages[-1]["content"]
            return super().generate_java_source(role, messages, data_name=data_name, source_frame=source_frame)

    coder = BoundedCoder()
    source = compose_first_pass_java(
        coder, spec, [{"role": "user", "content": original}],
        construction_context=context, behaviors=[behavior],
        source_context=[{"path": "src/main/java/example/State.java", "source": raw_source}])
    _run_java(tmp_path, source)
    assert len(coder.body_tasks) == 4
    assert json.dumps(payload) == original
    assert len(context["host_state_access"][0]["methods"]) == 2002


def test_direct_compiler_repair_does_not_resend_retained_implementation_or_state_inventory():
    from minecraft_mod_ai.direct_feature_repair import repair_direct_feature_member

    behavior, _, _ = _fixture()
    behavior["state_reads"] = ["balance"]
    payload = _large_host_context()
    source = "package example; public class Feature {\nprivate void start() {missing();}\n}"

    class RepairCoder:
        def generate_java_source(self, role, messages, *, data_name, source_frame):
            value = json.dumps(messages, ensure_ascii=False)
            assert len(value.encode()) < 32768
            assert "RETAINED_BODY_NOT_MODEL_INPUT" not in value
            task = json.loads(messages[-1]["content"])
            assert task["assigned_behaviors"] == [behavior]
            assert task["host_state_access"][0]["methods"] == payload["host_state_access"][0]["methods"][:2]
            assert task["javac_diagnostic"] == "Feature.java:2: error: cannot find symbol"
            return source_frame["prefix"] + "java.lang.System.nanoTime();" + source_frame["suffix"]

    result = repair_direct_feature_member(
        RepairCoder(), source, "Feature.java:2: error: cannot find symbol",
        context={"filename": "Feature.java", "target_path": "src/main/java/example/Feature.java",
                 "symbols": {}, "owner": "example.Feature", "api_evidence": payload["inspected_target_api"],
                 "attempt": 1, "behaviors": [behavior], "state_access": payload["host_state_access"],
                 "prerequisite_api": payload["materialized_prerequisite_api"],
                 "implementation_context": payload})
    assert "missing();" not in result


def test_body_keeps_existing_typed_effect_ownership_when_projecting_function_interfaces():
    from minecraft_mod_ai.java_authoring_prompt import java_body_messages

    function = {"id": "purchase_materials", "covers": ["purchase_request", "consume_balance"],
                "parameters": [{"name": "amount", "type": "int"}], "return_type": "boolean",
                "body": [{"op": "existing_effect"}]}
    task = {"host_java_authoring": True,
            "assigned_behaviors": [{"behavior_id": "other_work", "intent": "Implement residual work"}],
            "retained_typed_behavior": {"functions": [function],
                                        "ui_owned_records": [{"action": "purchase_materials"}]}}
    frame = {"kind": "method", "prefix": "private void work() {\n", "suffix": "\n}"}
    messages = java_body_messages([{"role": "user", "content": json.dumps(task)}], frame)
    projected = json.loads(messages[-1]["content"])["retained_typed_behavior"]
    assert projected["functions"][0] == {
        "id": "purchase_materials", "covers": ["purchase_request", "consume_balance"],
        "parameters": [{"name": "amount", "type": "int"}], "return_type": "boolean"}
    assert projected["ui_owned_records"] == [{"action": "purchase_materials"}]
    assert "body" in function


def test_selected_factory_keeps_its_builder_dependency_interfaces():
    from minecraft_mod_ai.construction_file_api_evidence import scoped_body_api_records
    from minecraft_mod_ai.java_authoring_prompt import java_body_messages

    factory = {"owner": "example.Factory", "kind": "METHOD", "name": "create",
               "return_type": "example.Factory.Builder", "parameters": []}
    build = {"owner": "example.Factory.Builder", "kind": "METHOD", "name": "build",
             "return_type": "example.Product", "generic_return_type": "example.Product<T>",
             "parameters": ["java.lang.String"], "generic_parameters": ["java.lang.String"],
             "varargs": False, "signature": "public example.Product<T> build(String name)",
             "binary_debug_metadata": "DEBUG_ONLY"}
    evidence = {"target_api": [factory], "class_members": [
        {"verified_class": "example.Factory", "members": [factory]},
        {"verified_class": "example.Factory.Builder", "declaration": {
            "kind": "CLASS", "owner": "example.Factory.Builder", "enclosing_owner": "example.Factory"},
         "members": [build]},
    ]}
    frame = {"kind": "method", "prefix": "private void make() {\n", "suffix": "\n}"}
    for records in (evidence["class_members"], scoped_body_api_records(evidence)):
        task = {"host_java_authoring": True, "inspected_target_api": {**evidence, "class_members": records}}
        messages = java_body_messages([{"role": "user", "content": json.dumps(task)}], frame)
        result = json.loads(messages[-1]["content"])["inspected_target_api"]
        builder = next(row for row in result["class_members"]
                       if row["verified_class"] == "example.Factory.Builder")
        assert builder["members"] == [{k: v for k, v in build.items() if k != "binary_debug_metadata"}]


def test_declaration_metadata_sees_exact_override_and_super_constructor_contracts():
    from minecraft_mod_ai.first_pass_java_source import _one_structured_action

    override = {"kind": "METHOD", "owner": "example.BaseTask", "name": "execute",
                "access": 1025, "return_type": "boolean", "parameters": ["java.lang.String", "int"]}
    constructor = {"kind": "CONSTRUCTOR", "owner": "example.BaseTask", "name": "<init>",
                   "access": 4, "parameters": ["java.lang.String"], "return_type": "void"}
    task = {"host_java_authoring": True, "current_unit": {"kind": "class", "name": "CustomTask"},
            "requirement": "Implement the real task interface",
            "inspected_target_api": {"target_api": [{
                "kind": "CLASS", "owner": "example.BaseTask", "name": "BaseTask"}],
                "class_members": [{"verified_class": "example.BaseTask", "declaration": {
                    "kind": "CLASS", "owner": "example.BaseTask", "access": 1025},
                    "members": [constructor, override]}]},
            "materialized_prerequisite_api": [{"path": "example/LocalTask.java", "types": [{
                "owner": "example.LocalTask", "declaration": "public interface LocalTask",
                "members": ["boolean execute(String input, int amount);"]}]}]}

    class MetadataCoder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            value = json.dumps(messages)
            assert "<init>" in value
            assert "execute" in value
            assert "java.lang.String" in value
            assert "boolean execute(String input, int amount);" in value
            return {"count": 2} if data_name.endswith("_count") else {"visibility": "public"}

    assert _one_structured_action(
        MetadataCoder(), [{"role": "user", "content": json.dumps(task)}],
        name="submit_java_declaration_1", schema={"type": "object", "properties": {
            "visibility": {"type": "string"}}, "required": ["visibility"],
            "additionalProperties": False}) == {"visibility": "public"}
    for action in ("submit_java_declaration_1_methods_count",
                   "submit_java_declaration_1_methods_1_parameters_count"):
        assert _one_structured_action(
            MetadataCoder(), [{"role": "user", "content": json.dumps(task)}],
            name=action, schema={"type": "object", "properties": {"count": {"type": "integer"}},
                                 "required": ["count"], "additionalProperties": False}) == {"count": 2}


def test_actual_parser_retry_keeps_small_member_context_and_full_host_validation(monkeypatch):
    from types import SimpleNamespace
    from test_first_pass_java_data_transport import DataRouter
    from minecraft_mod_ai.java_authoring_prompt import java_body_messages
    from minecraft_mod_ai.model_adapters.base import AdapterConfig

    monkeypatch.setattr("minecraft_mod_ai.model_context_budget.fit_messages_to_context",
                        lambda messages, **kwargs: messages)
    behavior = {"behavior_id": "claim", "intent": "Claim only eligible locations",
                "state_reads": ["balance"], "state_writes": ["balance"]}
    context = {**_large_host_context(), "host_java_authoring": True, "assigned_behavior": behavior,
               "current_unit_purpose": "Check eligibility before recording a claim",
               "host_signature": "private boolean check(Object value)",
               "declared_member_api": [{"signature": "public int claimCount()"}]}
    frame = {"kind": "method", "prefix": "private boolean check(Object value) {\n", "suffix": "\n}",
             "host_state_access": context["host_state_access"]}
    messages = java_body_messages([{"role": "user", "content": json.dumps(context)}], frame)
    calls = []

    def generate(request):
        calls.append(request)
        value = json.dumps(request.messages, ensure_ascii=False)
        assert len(value.encode()) < 32768
        assert "RETAINED_BODY_NOT_MODEL_INPUT" not in value
        assert "readState_other_" not in value
        payload = next(json.loads(m["content"]) for m in reversed(request.messages) if m["role"] == "user")
        assert payload["assigned_behavior"] == behavior
        assert payload["requirement"] == context["requirement"]
        assert payload["current_unit_purpose"] == context["current_unit_purpose"]
        assert payload["declared_member_api"] == context["declared_member_api"]
        assert payload["retained_typed_behavior"]["ui_owned_records"]
        assert payload["host_state_access"][0]["methods"] == context["host_state_access"][0]["methods"][:2]
        body = ('return value instanceof java.lang.String.valueOf("x");'
                if len(calls) == 1 else "return value instanceof java.lang.String;")
        return json.dumps({"implementation": body})

    adapter = SimpleNamespace(config=AdapterConfig(role="coder", adapter="llama_cpp"), generate=generate)
    source = DataRouter(adapter).generate_java_source(
        "coder", messages, data_name="large_context_parse_retry", source_frame=frame)
    assert len(calls) == 2
    assert "return value instanceof java.lang.String;" in source
    assert len(frame["host_state_access"][0]["methods"]) == 2002


def test_live_capacity_projection_recognizes_direct_signature_without_reexpanding_host_state():
    from minecraft_mod_ai.construction_java_context import scoped_java_member_messages
    from minecraft_mod_ai.java_authoring_prompt import java_body_messages

    payload = {**_large_host_context(), "host_java_authoring": True,
               "assigned_behaviors": [{"state_reads": ["balance"]}],
               "host_signature": "private int balance()", "member_role": "supporting_member"}
    frame = {"kind": "method", "prefix": "private int balance() {\n", "suffix": "\n}",
             "host_state_access": payload["host_state_access"]}
    original = java_body_messages([{"role": "user", "content": json.dumps(payload)}], frame)
    projected = scoped_java_member_messages(original, frame)
    assert projected != tuple(original), "the live capacity path must recognize immutable_signature"
    task = json.loads(projected[-1]["content"])
    assert task["original_feature_goal"] == payload["original_feature_goal"]
    assert task["retained_typed_behavior"]["ui_owned_records"]
    assert task["owned_runtime_records"] == payload["owned_runtime_records"]
    assert len(task["host_state_access"][0]["methods"]) == 2
    assert len(frame["host_state_access"][0]["methods"]) == 2002


def test_plan_scalar_pages_do_not_reintroduce_callback_or_classpath_catalogs():
    from minecraft_mod_ai.java_authoring_prompt import java_data_messages

    context = {**_large_host_context(), "host_java_authoring": True,
               "current_field": "kind", "completed_plan_fields": {}}
    projected = java_data_messages([{"role": "user", "content": json.dumps(context)}],
                                   "submit_java_construction_plan_1_value_kind")
    task = json.loads(projected[-1]["content"])
    assert task["current_field"] == "kind"
    assert task["requirement"] == context["requirement"]
    assert task["retained_typed_behavior"]["ui_owned_records"]
    assert "available_fabric_callbacks" not in task
    assert "inspected_target_api" not in task


def test_callback_catalog_keeps_exact_sam_without_repeating_event_owner_and_field():
    from minecraft_mod_ai.java_authoring_prompt import java_body_messages

    event = {"event_id": "example.RealEvents#ACTION", "event_owner": "example.RealEvents",
             "event_field": "ACTION", "event_interface": "example.RealEvents.Action",
             "method": "onAction", "parameters": ["example.Player", "java.lang.String"],
             "return_type": "boolean", "type_parameters": ["T extends example.Entity"],
             "throws": ["java.io.IOException"]}
    context = {"host_java_authoring": True, "available_fabric_callbacks": [event]}
    frame = {"kind": "method", "prefix": "void start() {\n", "suffix": "\n}"}
    task = json.loads(java_body_messages([{"role": "user", "content": json.dumps(context)}], frame)[-1]["content"])
    assert task["available_fabric_callbacks"] == [{
        key: value for key, value in event.items() if key not in {"event_owner", "event_field"}}]


def test_runtime_bindings_are_scoped_to_current_work_without_dropping_owned_rules():
    from minecraft_mod_ai.java_authoring_prompt import java_body_messages

    context = {"host_java_authoring": True, "assigned_behavior": {
        "state_reads": ["balance"], "runtime_state_reads": ["Nearby enemies"]},
        "runtime_state_bindings": {"Nearby enemies": {"observation": "Nearby enemies"},
                                   "Unrelated stock": {"observation": "Unrelated stock"},
                                   "selected_location": {"binding": "typed_input"}},
        "owned_runtime_records": [{"record": {"condition": "Only combat-ready players"}}]}
    frame = {"kind": "method", "prefix": "void start() {\n", "suffix": "\n}"}
    task = json.loads(java_body_messages([{"role": "user", "content": json.dumps(context)}], frame)[-1]["content"])
    assert task["runtime_state_bindings"] == {
        key: context["runtime_state_bindings"][key] for key in ("Nearby enemies", "selected_location")}
    assert task["owned_runtime_records"] == context["owned_runtime_records"]


def test_live_capacity_and_field_parse_retry_reach_executable_generated_java(tmp_path, monkeypatch):
    from math import ceil
    from types import SimpleNamespace
    from test_first_pass_java_data_transport import DataRouter
    from minecraft_mod_ai.model_adapters.base import AdapterConfig

    monkeypatch.setattr("minecraft_mod_ai.model_context_budget.fit_messages_to_context",
                        lambda messages, **kwargs: tuple(messages))
    behavior, spec, context = _fixture()
    behavior.update({"state_reads": ["balance"], "state_writes": ["balance"]})
    task = {**_large_host_context(), "host_java_authoring": True, "assigned_behaviors": [behavior]}
    task["available_fabric_callbacks"] *= 112
    task["runtime_state_bindings"] = {name: {"observation": name} for name in ("balance", "other")}
    context["host_state_access"] = task["host_state_access"]
    counts, requests = [], []

    def accounting(request):
        # Deterministic constrained-runtime fixture, not a llama tokenizer claim.
        tokens = ceil(len(json.dumps(request.messages, ensure_ascii=False)) / 3)
        counts.append(tokens)
        return SimpleNamespace(input_tokens=tokens, context_tokens=32768, remaining_tokens=32768-tokens)

    class LiveCoder(SourceCoder):
        rejected_field = False

        def generate_java_source(self, role, messages, *, data_name, source_frame):
            def generate(request):
                requests.append(request)
                payload = json.loads(next(m["content"] for m in reversed(request.messages) if m["role"] == "user"))
                assert payload["retained_typed_behavior"]["ui_owned_records"]
                assert payload["owned_runtime_records"] == task["owned_runtime_records"]
                assert payload["runtime_state_bindings"] == {"balance": {"observation": "balance"}}
                assert len(payload["host_state_access"][0]["methods"]) == 2
                if source_frame["kind"] == "field" and not self.rejected_field:
                    self.rejected_field = True
                    return json.dumps({"implementation": "import example.GameEvents; public class Feature {}"})
                source = SourceCoder.generate_java_source(
                    self, role, [{"role": "user", "content": json.dumps(payload)}],
                    data_name=data_name, source_frame=source_frame)
                return json.dumps({"implementation": source[len(source_frame["prefix"]):-len(source_frame["suffix"])]})

            adapter = SimpleNamespace(config=AdapterConfig(role="coder", adapter="llama_cpp",
                max_context=32768, max_new_tokens=8192), input_context_accounting=accounting, generate=generate)
            return DataRouter(adapter).generate_java_source(
                role, messages, data_name=data_name, source_frame=source_frame)

    coder = LiveCoder()
    source = compose_first_pass_java(coder, spec, [{"role": "user", "content": json.dumps(task)}],
                                     construction_context=context, behaviors=[behavior])
    _run_java(tmp_path, source)
    assert coder.rejected_field and len(requests) == 5
    correction = json.loads(requests[1].messages[-1]["content"])
    assert correction["diagnostic"]
    assert correction["immutable_signature"]["kind"] == "field"
    assert "public class Feature" in correction["previous_implementation"]
    assert max(counts) < 32768 - 2048
    assert len(context["host_state_access"][0]["methods"]) == 2002
