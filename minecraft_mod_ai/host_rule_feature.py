"""Compile feature entrypoints from observation rules, never model Java source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .host_callback_dispatch import _result_fields, select_callback_results
from .host_fabric_callback_abi import callback_interfaces
from .host_rule_author import RuleAuthor
from .host_rule_observation import bind_rule_observation
from .host_rule_program import RuleProgram
from .java_operation_ir import BOXING, PRIMITIVES, identifier, qualified_name
from .java_region_parser import _parse


def _results(router, event, symbols):
    if event["return_type"] not in {"void", "boolean"} and not _result_fields(event, symbols):
        raise ValueError("RULE_CHECKED_CALLBACK_RESULT_REQUIRED: " + event["return_type"])
    return select_callback_results(router, event, symbols)


def _tracking(author, program, gate, mode):
    if mode == "level":
        return "", "", program.source(gate), {}
    observed = (author.value("Observed value whose change activates this rule")
                if mode == "changed" else gate)
    scope_types = ("net.minecraft.server.MinecraftServer", "net.minecraft.world.level.Level",
                   "net.minecraft.server.level.ServerLevel", "net.minecraft.world.World",
                   "net.minecraft.server.world.ServerWorld", "net.minecraft.world.entity.Entity",
                   "net.minecraft.entity.Entity")
    def tracking_contexts():
        return [row for row in program.menu() if row["type"] not in PRIMITIVES
                and any(program.checker.subtype(row["type"], kind) for kind in scope_types)]
    contexts = tracking_contexts()
    if not contexts:
        author.value("Get the stable world/server/player/entity owning this observation from the callback inputs")
        contexts = tracking_contexts()
    if not contexts:
        raise ValueError("RULE_STABLE_TRACKING_CONTEXT_REQUIRED")
    key = author.ask("rule_tracking_context", "Select the stable world/server/player/entity owning this observation; never a temporary wrapper",
        {"value_ref": {"type": "string", "enum": [row["id"] for row in contexts]}},
        context_values=contexts)["value_ref"]
    policy = author.ask("rule_first_observation", "Choose whether an initial observation fires or only establishes a baseline",
        {"policy": {"type": "string", "enum": ["baseline", "fire"]}})["policy"]
    # containsKey distinguishes no history from every legitimate value,
    # including null, false and a phase index of zero. Values are immutable
    # snapshots; mutable object identity is not a change detector.
    kind = program.kind(observed)
    if kind not in PRIMITIVES and kind not in {*BOXING.values(), "java.lang.String", "java.util.UUID"}:
        raise ValueError("RULE_IMMUTABLE_OBSERVATION_REQUIRED: " + kind)
    initial = "true" if policy == "fire" else "false"
    setup = (
        "Object __mmmContext = " + program.source(key) + ";\n"
        "if (__mmmContext == null) { __MMM_UNHANDLED__ }\n"
        "Object __mmmObserved = " + program.source(observed) + ";\n"
        "boolean __mmmKnown = this.HISTORY.containsKey(__mmmContext);\n"
        "Object __mmmPrevious = this.HISTORY.put(__mmmContext, __mmmObserved);\n")
    changed = ("!java.util.Objects.equals(__mmmPrevious, __mmmObserved)" if mode == "changed"
               else "(java.lang.Boolean.TRUE.equals(__mmmObserved) && !java.lang.Boolean.TRUE.equals(__mmmPrevious))")
    fire = "(__mmmKnown ? " + changed + " : " + initial + ")"
    if mode == "rising":
        fire = "(java.lang.Boolean.TRUE.equals(__mmmObserved) && " + fire + ")"
        gate_source = "true"  # The gate was evaluated exactly once into Observed.
    else:
        gate_source = program.source(gate)
    return ("private final java.util.Map<Object, Object> HISTORY = new java.util.WeakHashMap<>();",
            setup, "(" + gate_source + " && " + fire + ")",
            {"observed": observed, "context": key, "first_observation": policy})


def compose_rule_feature(router, spec, context, *, package, name, behaviors, state_access):
    qualified_name(package)
    identifier(name)
    if not behaviors:
        raise ValueError("RULE_BEHAVIOR_COVERAGE_REQUIRED")
    owner = package + "." + name
    from .construction_operation_context import task_context_bindings
    task_data, _ = task_context_bindings(context)
    data_pages = {**context.get("binding_pages", {}), **task_data}
    implementation = spec.get("bindings", {}).get("java_implementation_requirements")
    if implementation:
        data_pages["implementation_requirements"] = [implementation]
    runtime_bindings = json.loads(spec.get("bindings", {}).get("runtime_state_bindings_json") or "{}")
    candidates = callback_interfaces(context.get("symbols") or {}, side=str(spec.get("side") or "COMMON"))
    from .host_typed_state_access import verified_state_accessor_symbols
    # Generated prerequisites are not yet present in the original target
    # classpath inventory. Integrate *only* their parsed, actual Java methods,
    # never model-invented state names or signatures.
    rule_symbols = dict(context["symbols"])
    for reference, entry in verified_state_accessor_symbols(state_access).items():
        rule_symbols.setdefault(reference, entry)
    if not candidates:
        raise ValueError("RULE_CHECKED_EVENT_ABI_REQUIRED")
    interface, lifecycle = (("net.fabricmc.api.ClientModInitializer", "onInitializeClient")
        if str(spec.get("side") or "").upper() == "CLIENT" else ("net.fabricmc.api.ModInitializer", "onInitialize"))
    members, registrations, records, seen = [], [], [], set()
    for behavior in behaviors:
        behavior_id = str(behavior.get("behavior_id") or "")
        if not behavior_id or behavior_id in seen:
            raise ValueError("RULE_BEHAVIOR_IDENTITY")
        seen.add(behavior_id)
        from .host_rule_provenance import (
            RuleMissingRuntimeProducer, validate_runtime_read_producers)
        input_types = tuple(dict.fromkeys(
            kind for hook in candidates for kind in hook.get("parameters", ())))
        # A single named JVM accessor is a fast path, not the sole way to
        # observe live game state. Complex reads are authored as typed
        # callback-derived expressions later, using the selected hook.
        pending_runtime_reads = []
        try:
            verified_sources = validate_runtime_read_producers(
                behavior, runtime_bindings, rule_symbols, input_types=input_types)
        except RuleMissingRuntimeProducer as error:
            if any(row.get("reason") != "planning_only_world_state"
                   for row in error.missing):
                raise
            pending_runtime_reads = [row["name"] for row in error.missing]
            verified_sources = validate_runtime_read_producers(
                {**behavior, "runtime_state_reads": [
                    name for name in behavior.get("runtime_state_reads", ())
                    if name not in pending_runtime_reads]},
                runtime_bindings, rule_symbols, input_types=input_types)
        if pending_runtime_reads:
            from .host_rule_provenance import unavailable_semantic_runtime_sources
            unavailable = unavailable_semantic_runtime_sources(
                pending_runtime_reads, rule_symbols, input_types=input_types)
            if unavailable:
                raise RuleMissingRuntimeProducer(behavior_id, [{
                    "name": name,
                    "reason": "no_inspected_semantic_source",
                    "required_action": "add an observable producing prerequisite module"} 
                    for name in unavailable])
        # A candidate event must provide the exact typed receiver needed
        # by a directly checked observer. For synthesized observations the
        # host validates the entire typed expression after authoring.
        from .host_rule_provenance import (
            RuleMissingRuntimeProducer, compatible_events_for_runtime_producers)
        options = compatible_events_for_runtime_producers(
            candidates, verified_sources, rule_symbols)
        if not options:
            raise RuleMissingRuntimeProducer(behavior_id, [{
                "name": key,
                "reason": "no_inspected_event_provides_runtime_observer_receiver",
                "verified_symbol_ref": ref}
                for key, ref in verified_sources.items()])
        binding = bind_rule_observation(router, behavior, options)
        event = binding["event"]
        if pending_runtime_reads:
            from .host_rule_provenance import (
                semantic_runtime_candidates, RuleMissingRuntimeProducer)
            # Binding an actual tick hook narrows the set of observable
            # receivers. A method on a completely unrelated callback type
            # cannot certify a runtime state.
            concrete = semantic_runtime_candidates(
                pending_runtime_reads, rule_symbols, event.get("parameters", ()))
            unavailable = [name for name in pending_runtime_reads
                           if not concrete[name]]
            if unavailable:
                raise RuleMissingRuntimeProducer(behavior_id, [{
                    "name": name, "reason": "selected_event_cannot_observe_state",
                    "event_id": event["event_id"],
                    "required_action": "bind the producing event or materialize a prerequisite observer"}
                    for name in unavailable])
        if event.get("type_parameters"):
            raise ValueError("RULE_GENERIC_CALLBACK_REQUIRES_ADAPTER")
        suffix = hashlib.sha256(behavior_id.encode()).hexdigest()[:20]
        handler = "handleRule" + suffix
        registrar = "registerBehavior" + suffix
        params = [{"type": kind, "name": "arg" + str(i)} for i, kind in enumerate(event["parameters"])]
        signature = ", ".join(row["type"] + " " + row["name"] for row in params)
        rule_behavior = {**behavior}
        if verified_sources:
            rule_behavior["verified_runtime_observers"] = verified_sources
        relevant_runtime = {key: runtime_bindings[key] for key in (
            *behavior.get("runtime_state_reads", ()), *behavior.get("runtime_state_writes", ()))
            if key in runtime_bindings}
        if relevant_runtime:
            rule_behavior["runtime_state_observations"] = relevant_runtime
        program = RuleProgram(rule_symbols, params, rule_behavior, owner=owner,
            target_path=spec["target_path"], state_owners=[row["owner"] for row in state_access])
        if verified_sources:
            from .host_rule_provenance import RuleMissingRuntimeProducer
            unreachable = []
            for name, reference in verified_sources.items():
                symbol = rule_symbols[reference]
                if (not symbol.get("static") and not int(symbol.get("access", 0)) & 8
                        and not any(program.checker.assignable(row["type"], symbol["owner"])
                                    for row in params)):
                    unreachable.append({
                        "name": name, "reason": "producer_not_reachable_from_selected_event",
                        "verified_symbol_ref": reference,
                        "event_id": event["event_id"]})
            if unreachable:
                raise RuleMissingRuntimeProducer(behavior_id, unreachable)
        author = RuleAuthor(router, program, data_pages=data_pages)
        from .host_rule_provenance import (
            expression_dependencies, runtime_source_expression_proof)
        runtime_sources = {}
        # Materialize each bound observer as an actual typed IR value, rather
        # than merely trusting the planning label or a matching method name.
        for state_name, reference in verified_sources.items():
            symbol = rule_symbols[reference]
            if symbol.get("static") or int(symbol.get("access", 0)) & 8:
                operands = []
            else:
                matching = [key for key in program.inputs
                            if program.checker.assignable(
                                program.kind(key), symbol["owner"])]
                if not matching:
                    raise RuleMissingRuntimeProducer(behavior_id, [{
                        "name": state_name, "reason": "observer_receiver_not_in_event",
                        "verified_symbol_ref": reference}])
                operands = [matching[0]]
            runtime_sources[state_name] = program.value(
                "call" if symbol["kind"] == "METHOD" else "field",
                operands, symbol_ref=reference, purpose="Runtime read: " + state_name)
        for state_name in pending_runtime_reads:
            # A complex observer is a bounded typed expression built with
            # inspected target symbols, not an invented Java getter. Do not
            # allow another concept's exact getter to certify this read.
            required = program.behavior.get("verified_runtime_observers", {})
            program.behavior["verified_runtime_observers"] = {}
            program.behavior["required_runtime_observation_name"] = state_name
            try:
                try:
                    result_ref = author.value(
                        "Implement a LIVE runtime producer for '" + state_name
                        + "' using the actual selected callback and inspected APIs. "
                        "The value must represent this exact game concept, not "
                        "an unrelated world flag or an arbitrary constant.",
                        require_grounded=True)
                except ValueError as error:
                    raise RuleMissingRuntimeProducer(behavior_id, [{
                        "name": state_name, "reason": "runtime_producer_authoring_failed",
                        "diagnostic": str(error)[:400]}]) from error
            finally:
                program.behavior["verified_runtime_observers"] = required
                program.behavior.pop("required_runtime_observation_name", None)
            if not runtime_source_expression_proof(program, result_ref, state_name):
                raise RuleMissingRuntimeProducer(behavior_id, [{
                    "name": state_name,
                    "reason": "producer_has_no_callback_derived_semantic_api",
                    "typed_value": result_ref}])
            runtime_sources[state_name] = result_ref
        mode = author.ask("rule_activation", "Choose execution semantics: level=every matching observation, rising=false-to-true condition, changed=changed observed value",
            {"mode": {"type": "string", "enum": ["level", "rising", "changed"]}})["mode"]
        gate = author.value("Original trigger and preconditions, including every branch. "
                            "Use EVERY required runtime observation value already "
                            "materialized in the available value IDs.",
                            expected="boolean", require_grounded=True)
        dependencies = expression_dependencies(program, gate)
        omitted = [state_name for state_name, ref in runtime_sources.items()
                   if ref not in dependencies]
        if omitted:
            raise RuleMissingRuntimeProducer(behavior_id, [
                {"name": state_name, "reason": "runtime_read_not_consumed_by_trigger"}
                for state_name in omitted])
        history, setup, condition, tracking = _tracking(author, program, gate, mode)
        history = history.replace("HISTORY", "history" + suffix)
        setup = setup.replace("HISTORY", "history" + suffix)
        # The condition and tracking are immutable once effects begin. Effects
        # can reuse typed observations without changing callback declarations.
        author.effects()
        body = program.render(include_root=False, with_return=False)
        results = _results(router, event, context["symbols"])
        unhandled = "return " + results["unhandled"] + ";" if results else "return;"
        setup = setup.replace("__MMM_UNHANDLED__", unhandled)
        dispatch = (program.declarations() + "\n" + setup
                    + "if (!(" + condition + ")) { " + unhandled + " }\n" + body + "\n")
        if results:
            dispatch += "return __mmmRuleApplied ? " + results["handled"] + " : " + results["unhandled"] + ";"
        if history:
            members.append(history)
        handler_member = ("private " + ("synchronized " if mode != "level" else "")
                          + event["return_type"] + " " + handler + "(" + signature + ") {\n" + dispatch + "\n}")
        from .host_typed_state_access import validate_behavior_state_provenance, validate_host_state_access
        behavior_source = "package " + package + "; class " + name + " {\n" + handler_member + "\n}"
        validate_host_state_access(behavior_source, state_access)
        validate_behavior_state_provenance(behavior_source, state_access, behavior)
        members.append(handler_member)
        members.append("private void " + registrar + "() {\n"
                       + event["event_owner"] + "." + event["event_field"]
                       + ".register(this::" + handler + ");\n}")
        registrations.append("this." + registrar + "();")
        records.append({"behavior": behavior, "observation": binding, "activation": mode,
                        "gate": gate, "tracking": tracking, "operations": program.ir,
                        "runtime_source_expressions": runtime_sources,
                        "semantic_status": "model_authored_unverified"})
    source = ("package " + package + ";\n\npublic final class " + name + " implements " + interface + " {\n"
              + "public " + name + "() {}\n@Override public void " + lifecycle + "() {\n"
              + "\n".join(registrations) + "\n}\n" + "\n\n".join(members) + "\n}\n")
    _parse(source)
    root = Path(context["project_root"]) / ".mmm" / "construction_rule_ir"
    root.mkdir(parents=True, exist_ok=True)
    from .construction_direct_writer import _save_checkpoint
    identity = hashlib.sha256((owner + "\0" + source).encode()).hexdigest()
    _save_checkpoint(root / (identity + ".json"), identity, {
        "schema": "mmm/host-observation-rules-v1", "owner": owner, "rules": records,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "target_compilation_verified": False, "runtime_verified": False,
    })
    return source
