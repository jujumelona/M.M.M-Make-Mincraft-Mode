"""One-operation context and exhaustive on-demand declaration discovery."""
from __future__ import annotations

import json


class HostSymbols(dict):
    """Make a request-time walk of the full host inventory observable."""

    frozen = False

    def items(self):
        assert not self.frozen, "request scanned the full host symbol inventory"
        return super().items()

    def values(self):
        assert not self.frozen, "request scanned the full host symbol inventory"
        return super().values()


def _class(symbols, owner, **extra):
    symbols[owner] = {"kind": "CLASS", "owner": owner,
                      "name": owner.rsplit(".", 1)[-1], "access": 1, **extra}


def _method(symbols, owner, name, parameters=(), **extra):
    key = owner + "#" + name + "(" + ",".join(parameters) + ")"
    symbols[key] = {"kind": "METHOD", "owner": owner, "name": name,
                    "parameters": list(parameters), "return_type": "void", "access": 1,
                    **extra}
    return symbols[key]


def _fixture(payload=None, symbols=None, **kwargs):
    from minecraft_mod_ai.incremental_java_context import IncrementalJavaContext

    spec = {"requirement": "새 세계로 이동하고 자원을 교환한다", "side": "COMMON",
            "target_path": "src/main/java/demo/Feature.java",
            "bindings": {"package_name": "demo", "class_name": "Feature",
                         "java_implementation_requirements": "Preserve atomic transfers"}}
    return IncrementalJavaContext(
        [{"role": "user", "content": json.dumps(payload or {}, ensure_ascii=False)}],
        spec, {"symbols": symbols if symbols is not None else {},
               "target": {"loader": "fabric", "minecraft_version": "26.2"}, **kwargs}, [])


def _program():
    return {"package": "demo", "class_name": "Feature", "members": [{
        "id": "member1", "owner": "Feature", "kind": "method", "name": "claim",
        "purpose": "Remember a claim", "behavior": {
            "behavior_id": "claim", "intent": "Claim the selected eligible world",
            "state_reads": ["balance"], "runtime_state_writes": ["selectedWorld"],
            "source_quote": "Require confirmation before spending resources"},
        "metadata": {"visibility": "public", "static": False, "return_type": "boolean",
                     "parameters": [{"type": "String", "name": "world"}],
                     "type_parameters": "", "throws": [], "annotations": []},
        "body": [],
    }]}


def test_large_inventory_view_preserves_owned_requirements_without_catalogs_or_bodies():
    symbols = HostSymbols()
    _class(symbols, "example.World")
    selected = _method(symbols, "example.World", "teleport", ["String"])
    for index in range(3000):
        _method(symbols, "example.World", f"unrelated{index}")
    methods = [{"name": "readState_balance", "signature": "public static int readState_balance();"},
               {"name": "writeState_selectedWorld", "signature": "public static void writeState_selectedWorld(String value);"}]
    methods.extend({"name": f"readState_other{index}",
                    "signature": f"public static Object readState_other{index}();"}
                   for index in range(3000))
    retained = {"functions": [{"id": "buy", "covers": ["purchase", "spend"],
                               "name": "buy", "return_type": "boolean",
                               "parameters": [{"type": "int", "name": "amount"}],
                               "body": [{"op": "effect", "data": "RETAINED_BODY" * 10000}]}],
                "ui_owned_records": [{"action": "buy", "condition": "Only after confirmation"}],
                "event_bindings": [{"action": "buy", "event": "purchase_screen"}]}
    payload = {"requirement": "새 세계로 이동하고 자원을 교환한다",
               "owned_runtime_records": [{"record": {"condition": "Only eligible worlds", "atomic": True}}],
               "retained_typed_behavior": retained,
               "available_fabric_callbacks": [{"event_id": f"real.Events#EVENT_{i}",
                                                 "event_owner": "real.Events"} for i in range(3000)],
               "runtime_state_bindings": {"balance": {"binding": "world_state"},
                                           "selectedWorld": {"binding": "world_state"},
                                           "typedPlayer": {"binding": "typed_input", "type": "example.Player"},
                                           "unrelated": {"binding": "world_state"}},
               "host_state_access": [{"owner": "example.State", "methods": methods}],
               "inspected_target_api": {"target_api": [selected], "class_members": [{
                   "verified_class": "example.World", "members": list(symbols.values())}]},
               "source_pages": {"Game.java": "RAW_GAMEPLAY_SOURCE" * 10000}}
    before = json.dumps(payload, ensure_ascii=False)
    context = _fixture(payload, symbols)
    symbols.frozen = True
    program = _program()
    scope = [{"name": "world", "type": "String"}, {"name": "confirmed", "type": "boolean"}]
    view = context.view(program["members"][0], scope, program, position=7)
    text = json.dumps(view, ensure_ascii=False)
    assert len(text.encode()) < 16000
    assert view["requirement"] == "새 세계로 이동하고 자원을 교환한다"
    assert view["assigned_behavior"] == program["members"][0]["behavior"]
    assert view["owned_runtime_records"] == payload["owned_runtime_records"]
    assert view["retained_typed_behavior"]["ui_owned_records"] == retained["ui_owned_records"]
    assert view["retained_typed_behavior"]["event_bindings"] == retained["event_bindings"]
    assert view["retained_typed_behavior"]["functions"][0] == {
        "id": "buy", "covers": ["purchase", "spend"], "name": "buy", "return_type": "boolean",
        "parameters": [{"type": "int", "name": "amount"}]}
    assert view["host_state_access"][0]["methods"] == methods[:2]
    assert set(view["runtime_state_bindings"]) == {"balance", "selectedWorld", "typedPlayer"}
    assert view["scope"] == scope
    assert view["position"] == 7
    assert view["implementation_requirements"] == "Preserve atomic transfers"
    assert view["target"]["minecraft_version"] == "26.2"
    assert "RETAINED_BODY" not in text
    assert "RAW_GAMEPLAY_SOURCE" not in text
    assert "unrelated" not in text
    assert "available_fabric_callbacks" not in view
    assert "class_members" not in view
    assert json.dumps(payload, ensure_ascii=False) == before


def test_queued_helper_declaration_remains_callable_before_it_has_a_body():
    context = _fixture()
    program = _program()
    program["members"].append({
        "id": "member2", "owner": "Feature", "kind": "method", "name": "eligible",
        "purpose": "Check eligibility", "behavior": {}, "body": [],
        "metadata": {"visibility": "private", "static": False, "return_type": "boolean",
                     "parameters": [{"type": "String", "name": "world"}],
                     "type_parameters": "", "throws": [], "annotations": []},
    })
    view = context.view(program["members"][0], {"values": []}, program)
    assert view["generated_declarations"][1]["signature"] == "private boolean eligible(String world)"
    assert view["generated_declarations"][1]["id"] == "member2"
    assert "body" not in view["generated_declarations"][1]
    assert view["current_member"]["signature"] == "public boolean claim(String world)"
    assert context.lookup("eligible")["total"] == 0  # generated declarations have their own namespace


def test_generated_declarations_stay_small_and_earlier_helpers_are_searchable():
    from copy import deepcopy
    context = _fixture()
    program = _program()
    for index in range(500):
        member = deepcopy(program["members"][0])
        member.update(id=f"m{index}", name=f"helper{index}", purpose=f"Store record {index}")
        program["members"].append(member)
    view = context.view(program["members"][0], {}, program)
    assert len(json.dumps(view)) < 12000
    assert len(view["generated_declarations"]) == 10
    assert view["generated_declarations_total"] == 501
    assert context.generated(program, "helper3", 0)["generated_declarations"][0]["name"] == "helper3"
    assert context.generated(program, "", 500)["generated_declarations"][0]["name"] == "helper499"


def test_exact_owner_lookup_paginates_every_overload_and_last_member():
    symbols = HostSymbols()
    owner = "example.Transport"
    _class(symbols, owner)
    symbols[owner + "#<init>()"] = {"kind": "CONSTRUCTOR", "owner": owner, "name": "<init>",
                                   "access": 1, "parameters": [], "return_type": "void"}
    for index in range(37):
        _method(symbols, owner, f"move{index:02d}")
    _method(symbols, owner, "move", ["String"])
    generic = _method(symbols, owner, "move", ["java.util.List", "String[]"],
                      generic_parameters=["java.util.List<? extends T>", "String[]"],
                      generic_return_type="java.util.List<T>", type_parameters=["T extends Object"],
                      return_type="java.util.List", varargs=True, access=0x81,
                      signature="public <T> java.util.List<T> move(java.util.List<? extends T> values, String... names)")
    context = _fixture(symbols=symbols)
    symbols.frozen = True
    rows, offset, pages = [], 0, 0
    while offset is not None:
        page = context.lookup(owner, offset)
        assert page["total"] == 40
        assert 0 < len(page["members"]) <= 12
        rows.extend(page["members"])
        offset = page["next_offset"]
        pages += 1
    assert pages >= 4
    assert len(rows) == 40
    assert {row["name"] for row in rows} >= {"<init>", "move36"}
    assert [row for row in rows if row.get("varargs")][0] == generic
    assert context.lookup(owner, 40)["members"] == []
    assert context.lookup(owner, 40)["next_offset"] is None
    exact = context.lookup(owner + "#move")
    assert exact["total"] == 2
    assert len(exact["members"]) == 2


def test_semantic_method_lookup_uses_preindexed_actual_symbols():
    symbols = HostSymbols()
    for index in range(300):
        owner = f"noise.Package{index}"
        _class(symbols, owner)
        _method(symbols, owner, "readValue")
    _class(symbols, "example.Player")
    actual = _method(symbols, "example.Player", "teleportToWorld", ["example.World"])
    _class(symbols, "example.World")
    context = _fixture(symbols=symbols)
    symbols.frozen = True
    assert context.lookup("teleport world")["members"] == [actual]
    assert context.lookup("teleportToWorld")["members"] == [actual]
    missing = context.lookup("whollyNewHelper")
    assert missing["members"] == []
    assert missing["total"] == 0
    assert "helper" in missing["guidance"].lower()
    assert "SOURCE_REQUIRED" not in json.dumps(missing)


def test_lookup_respects_private_client_and_inherited_access_contracts():
    symbols = HostSymbols()
    _class(symbols, "example.Base")
    _class(symbols, "example.Child", superclass="example.Base")
    _class(symbols, "net.minecraft.client.ClientWorld")
    _method(symbols, "example.Base", "publicRun")
    _method(symbols, "example.Base", "protectedRun", access=4)
    _method(symbols, "example.Base", "privateRun", access=2)
    _method(symbols, "example.Base", "packageRun", access=0)
    _method(symbols, "example.Base", "clientRun", ["net.minecraft.client.ClientWorld"])
    _method(symbols, "example.Base", "clientOnly", side="CLIENT")
    _method(symbols, "example.Base", "staticRun", access=9)
    _method(symbols, "example.Base", "finalRun", access=0x11)
    _method(symbols, "net.minecraft.client.ClientWorld", "render")
    context = _fixture(symbols=symbols)
    symbols.frozen = True
    page = context.lookup("example.Child")
    assert {row["name"] for row in page["members"]} == {"publicRun", "staticRun", "finalRun"}
    assert all(row["owner"] == "example.Base" for row in page["members"])
    assert context.lookup("net.minecraft.client.ClientWorld")["members"] == []
    overrides = context.override("example.Base", "protectedRun")
    assert len(overrides) == 1 and overrides[0]["access"] == 4
    assert context.override("example.Base", "privateRun") == []
    assert {row["name"] for row in context.override("example.Base")} == {"protectedRun", "publicRun"}


def test_materialized_lookup_keeps_exact_constructor_generic_varargs_and_throws_signatures():
    headers = ["public Store(String key);",
               "public <T extends Number> java.util.List<T> map(java.util.function.Function<? super T, ? extends T> fn, T... values) throws java.io.IOException;",
               *[f"public int value{index}();" for index in range(28)]]
    context = _fixture({"materialized_prerequisite_api": [{
        "path": "src/main/java/demo/Store.java", "imports": ["import java.util.List;"],
        "types": [{"owner": "demo.Store", "declaration": "public final class Store", "members": headers}]}]})
    rows, offset = [], 0
    while offset is not None:
        page = context.lookup("demo.Store", offset)
        rows.extend(page["members"])
        offset = page["next_offset"]
    assert len(rows) == 30
    assert {row["signature"] for row in rows} == set(headers)
    constructor = next(row for row in rows if row["kind"] == "CONSTRUCTOR")
    assert constructor["parameters"] == ["java.lang.String"]
    generic = next(row for row in rows if row["name"] == "map")
    assert generic["varargs"] is True
    assert generic["generic_parameters"] == ["java.util.function.Function<? super T, ? extends T>", "T[]"]
    assert generic["generic_return_type"] == "java.util.List<T>"
    assert generic["type_parameters"] == ["T extends java.lang.Number"]
    assert generic["throws"] == ["java.io.IOException"]


def test_callbacks_returns_exact_original_records_without_a_selection_step():
    original = [{"event_id": "fabric.Events#USE", "event_owner": "fabric.Events",
                 "event_field": "USE", "event_interface": "fabric.Events.Use",
                 "method": "use", "parameters": ["example.Player"], "return_type": "boolean",
                 "provenance": "actual_callback_record", "extra_contract": {"exact": True}}]
    context = _fixture({"available_fabric_callbacks": original})
    assert context.callbacks() == original
    context.callbacks()[0]["event_field"] = "CHANGED"
    assert context.callbacks() == original


def test_generic_functional_abi_substitution_preserves_parameter_and_result_types():
    symbols = HostSymbols()
    _class(symbols, "java.util.function.Function", access=0x601,
           type_parameters=["T extends java.lang.Object", "R extends java.lang.Object"],
           functional_parameters=["java.lang.Object"], functional_return_type="java.lang.Object",
           functional_contract_sources=["java.util.function.Function#apply(java.lang.Object)"])
    _method(symbols, "java.util.function.Function", "apply", ["java.lang.Object"], access=0x401,
            return_type="java.lang.Object", generic_parameters=["T"], generic_return_type="R")
    _class(symbols, "java.util.function.Consumer", access=0x601,
           type_parameters=["T extends java.lang.Object"], functional_parameters=["java.lang.Object"],
           functional_return_type="void",
           functional_contract_sources=["java.util.function.Consumer#accept(java.lang.Object)"])
    _method(symbols, "java.util.function.Consumer", "accept", ["java.lang.Object"], access=0x401,
            generic_parameters=["T"])
    _class(symbols, "example.NotFunctional")
    context = _fixture(symbols=symbols)
    symbols.frozen = True
    assert context.sam("java.util.function.Function<String, Integer>") == {
        "parameters": ["String"], "return_type": "Integer", "throws": []}
    assert context.sam("java.util.function.Consumer<java.util.List<String>>") == {
        "parameters": ["java.util.List<String>"], "return_type": "void", "throws": []}
    assert context.sam("java.util.function.Consumer<? super String>") == {
        "parameters": ["String"], "return_type": "void", "throws": []}
    assert context.sam("java.util.function.Function<? extends Number, ? super Integer>") == {
        "parameters": ["Number"], "return_type": "Integer", "throws": []}
    assert context.sam("java.util.function.Consumer<?>") == {
        "parameters": ["java.lang.Object"], "return_type": "void", "throws": []}
    assert context.sam("example.NotFunctional") == {}
    assert context.sam("UnknownFunctional") == {}


def test_ambiguous_class_name_search_pages_declarations_without_a_catalog_dump():
    symbols = HostSymbols()
    for index in range(33):
        _class(symbols, f"vendor{index:02d}.Builder")
    context = _fixture(symbols=symbols)
    symbols.frozen = True
    rows, offset = [], 0
    while offset is not None:
        page = context.lookup("Builder", offset)
        assert page["total"] == 33
        assert 0 < len(page["declarations"]) <= 12
        rows.extend(page["declarations"])
        offset = page["next_offset"]
    assert len(rows) == 33
    assert rows[-1]["owner"] == "vendor32.Builder"


def test_callback_fallback_joins_the_full_index_once_and_keeps_actual_registration_abi():
    symbols = HostSymbols()
    event = "net.fabricmc.fabric.api.event.Event"
    owner = "net.fabricmc.fabric.api.event.lifecycle.v1.WorldEvents"
    listener = owner + ".AfterChange"
    _class(symbols, event)
    _method(symbols, event, "register", ["T"], access=1)
    _class(symbols, owner)
    _class(symbols, listener, access=0x601, enclosing_owner=owner)
    _method(symbols, listener, "afterChange", ["demo.World"], access=0x401)
    symbols[owner + "#AFTER_CHANGE"] = {
        "kind": "FIELD", "owner": owner, "name": "AFTER_CHANGE", "access": 9,
        "return_type": event, "generic_return_type": event + "<" + listener + ">"}
    context = _fixture(symbols=symbols)
    symbols.frozen = True
    callbacks = context.callbacks()
    assert len(callbacks) == 1
    assert callbacks[0]["event_id"] == owner + "#AFTER_CHANGE"
    assert callbacks[0]["event_field"] == "AFTER_CHANGE"
    assert callbacks[0]["parameters"] == ["demo.World"]
    assert callbacks[0]["method"] == "afterChange"


def test_bindings_json_is_decoded_before_repeated_operation_views(monkeypatch):
    from minecraft_mod_ai.incremental_java_context import IncrementalJavaContext

    behavior = _program()["members"][0]["behavior"]
    spec = {"requirement": "Original feature", "target_path": "src/main/java/demo/Feature.java",
            "bindings": {"owned_runtime_records_json": '[{"condition":"all worlds"}]',
                         "retained_typed_behavior_json": '{"functions":[{"id":"retained","covers":["spend"]}]}',
                         "runtime_state_bindings_json": '{"balance":{"binding":"world_state"},"input":{"binding":"typed_input"}}'}}
    context = IncrementalJavaContext([{"role": "user", "content": "unstructured text"}], spec, {}, [behavior])

    def reject_repeat_decode(*args, **kwargs):
        raise AssertionError("operation reparsed host JSON")

    monkeypatch.setattr("minecraft_mod_ai.incremental_java_context.json.loads", reject_repeat_decode)
    program = _program()
    for position in range(4):
        view = context.view(program["members"][0], {}, program, position=position)
        assert view["owned_runtime_records"] == [{"condition": "all worlds"}]
        assert view["retained_typed_behavior"]["functions"] == [{"id": "retained", "covers": ["spend"]}]
        assert set(view["runtime_state_bindings"]) == {"balance", "input"}
    assert context.lookup("NoExistingOwner")["total"] == 0


def test_materialized_class_retains_inheritance_and_functional_interface_abi():
    symbols = HostSymbols()
    _class(symbols, "demo.Parent")
    _method(symbols, "demo.Parent", "inherited")
    context = _fixture({"materialized_prerequisite_api": [{"types": [
        {"owner": "demo.Child", "declaration": "public final class Child extends demo.Parent",
         "members": ["public void own();", "public final void fixed();"]},
        {"owner": "demo.Callback", "declaration": "public interface Callback<T>",
         "members": ["boolean accept(T value);", "private void internal();", "default void log(T value);"]},
    ]}]}, symbols)
    page = context.lookup("demo.Child")
    assert {row["name"] for row in page["members"]} == {"inherited", "own", "fixed"}
    assert page["declarations"][0]["access"] & 0x10
    assert context.override("demo.Child", "fixed") == []
    assert context.sam("demo.Callback<String>") == {
        "parameters": ["String"], "return_type": "boolean", "throws": []}
    assert {row["name"] for row in context.lookup("demo.Callback")["members"]} == {"accept", "log"}


def test_initial_selected_calls_follow_the_same_access_and_side_rules_as_lookup():
    symbols = HostSymbols()
    _class(symbols, "example.World")
    selected = [_method(symbols, "example.World", "allowed"),
                _method(symbols, "example.World", "hidden", access=2),
                _method(symbols, "example.World", "render", side="CLIENT")]
    context = _fixture({"inspected_target_api": {"target_api": selected}}, symbols)
    program = _program()
    view = context.view(program["members"][0], {}, program)
    assert [row["name"] for row in view["selected_target_api"]] == ["allowed"]


def test_override_specializes_class_type_arguments_and_preserves_method_type_variables():
    symbols = HostSymbols()
    _class(symbols, "demo.Mapper", access=0x601, type_parameters=["T extends java.lang.Object"])
    _method(symbols, "demo.Mapper", "map", ["java.lang.Object"], access=0x401,
            return_type="java.lang.Object", generic_parameters=["T"], generic_return_type="T")
    _method(symbols, "demo.Mapper", "identity", ["java.lang.Number"], access=0x401,
            return_type="java.lang.Number", generic_parameters=["T"], generic_return_type="T",
            type_parameters=["T extends java.lang.Number"])
    _class(symbols, "demo.GenericChild", access=0x601, type_parameters=["U extends java.lang.Object"],
           interfaces=["demo.Mapper"], generic_interfaces=["demo.Mapper<U>"])
    context = _fixture(symbols=symbols)
    direct = context.override("demo.Mapper<String>", "map")[0]
    assert direct["generic_parameters"] == ["String"]
    assert direct["generic_return_type"] == "String"
    inherited = context.override("demo.GenericChild<Integer>", "map")[0]
    assert inherited["owner"] == "demo.Mapper"
    assert inherited["generic_parameters"] == ["Integer"]
    assert inherited["generic_return_type"] == "Integer"
    shadowed = context.override("demo.Mapper<String>", "identity")[0]
    assert shadowed["type_parameters"] == ["T extends java.lang.Number"]
    assert shadowed["generic_parameters"] == ["T"]
    assert shadowed["generic_return_type"] == "T"
    assert symbols["demo.Mapper#map(java.lang.Object)"]["generic_return_type"] == "T"


def test_current_listener_retains_exact_event_identity_and_typed_ui_parameter_names():
    event = {"event_id": "ui:travel", "event_interface": "demo.Actions.Travel", "method": "travel",
             "parameters": ["demo.Player", "java.lang.String"], "parameter_names": ["player", "destination"],
             "return_type": "boolean", "type_parameters": [], "throws": ["demo.TravelException"],
             "provenance": "host_typed_ui_action"}
    context = _fixture({"available_fabric_callbacks": [event, {**event, "event_id": "ui:trade"}]})
    program = _program()
    member = program["members"][0]
    member["event"] = event
    member["metadata"]["parameters"] = [{"type": "demo.Player", "name": "player"},
                                          {"type": "java.lang.String", "name": "destination"}]
    view = context.view(member, {}, program)
    assert view["current_member"]["event"] == event
    assert view["current_member"]["parameters"] == member["metadata"]["parameters"]
    assert "ui:trade" not in json.dumps(view)
