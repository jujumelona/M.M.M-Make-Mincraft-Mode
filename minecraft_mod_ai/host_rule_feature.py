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


def compose_rule_feature(router, spec, context, *, package, name, behaviors, state_access,
                         prerequisite_apis=()):
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
    from .host_prerequisite_api import inspected_prerequisite_symbols
    for reference, entry in inspected_prerequisite_symbols(
            list(prerequisite_apis)).items():
        # Canonical, hash-checked source symbols already include every member.
        # The declaration fallback must not introduce a second alias/signature
        # inventory for the same production owner.
        if rule_symbols.get(entry.get("owner"), {}).get("provenance") != "materialized_source":
            rule_symbols.setdefault(reference, entry)
    for reference, entry in verified_state_accessor_symbols(state_access).items():
        rule_symbols.setdefault(reference, entry)
    if not candidates:
        raise ValueError("RULE_CHECKED_EVENT_ABI_REQUIRED")
    interface, lifecycle = (("net.fabricmc.api.ClientModInitializer", "onInitializeClient")
        if str(spec.get("side") or "").upper() == "CLIENT" else ("net.fabricmc.api.ModInitializer", "onInitialize"))
    from .host_rule_source_plan import (
        RuntimeSourcePlanner, bind_runtime_sources, materialize_runtime_sources)
    source_planner = RuntimeSourcePlanner(rule_symbols)
    members, registrations, records, seen = [], [], [], set()
    for behavior in behaviors:
        behavior_id = str(behavior.get("behavior_id") or "")
        if not behavior_id or behavior_id in seen:
            raise ValueError("RULE_BEHAVIOR_IDENTITY")
        seen.add(behavior_id)
        options, event_source_plans = bind_runtime_sources(
            behavior, runtime_bindings, source_planner, candidates)
        binding = bind_rule_observation(router, behavior, options)
        event = binding["event"]
        if event.get("type_parameters"):
            raise ValueError("RULE_GENERIC_CALLBACK_REQUIRES_ADAPTER")
        suffix = hashlib.sha256(behavior_id.encode()).hexdigest()[:20]
        handler = "handleRule" + suffix
        registrar = "registerBehavior" + suffix
        params = [{"type": kind, "name": "arg" + str(i)} for i, kind in enumerate(event["parameters"])]
        signature = ", ".join(row["type"] + " " + row["name"] for row in params)
        rule_behavior = {**behavior}
        relevant_runtime = {key: runtime_bindings[key] for key in (
            *behavior.get("runtime_state_reads", ()), *behavior.get("runtime_state_writes", ()))
            if key in runtime_bindings}
        if relevant_runtime:
            rule_behavior["runtime_state_observations"] = relevant_runtime
        program = RuleProgram(rule_symbols, params, rule_behavior, owner=owner,
            target_path=spec["target_path"], state_owners=[row["owner"] for row in state_access])
        runtime_sources, selected_sources = materialize_runtime_sources(
            router, program, event_source_plans[event["event_id"]])
        program.behavior["runtime_source_symbols"] = sorted({
            ref for source in selected_sources.values() for ref in source["symbol_refs"]})
        author = RuleAuthor(router, program, data_pages=data_pages)
        from .host_rule_provenance import expression_dependencies, RuleMissingRuntimeProducer
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
        guard_values = program.behavior.get("runtime_source_guards", ())
        guards = ("if (!(" + " && ".join(program.source(ref) for ref in guard_values)
                  + ")) { " + unhandled + " }\n" if guard_values else "")
        dispatch = (program.declarations() + "\n" + guards + setup
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
                        "runtime_source_plans": selected_sources,
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
