"""Plan complete callback-derived observations before authoring rule predicates.

The small model chooses a complete inspected path, never searches the JVM
classpath or constructs arbitrary arguments to discover a runtime producer.
These are structural source plans, not certification of gameplay semantics.
"""
from __future__ import annotations

from collections import defaultdict, deque

from .host_rule_observation import PAGE_SIZE, choose_rule_data
from .host_rule_provenance import (
    RuleMissingRuntimeProducer,
    observation_words,
    resolve_materialized_entity_presence,
)
from .java_operation_ir import PRIMITIVES, parameter_types


MAX_OBSERVER_HOPS = 3


class RuntimeSourcePlanner:
    def __init__(self, symbols):
        self.symbols = symbols
        self.members = defaultdict(list)
        self._ancestors = {}
        self._plans = {}
        self._entity_symbols = {}
        for ref, row in symbols.items():
            kind = row.get("kind")
            if kind == "CLASS" or (kind == "METHOD" and row.get("name") == "getAllEntities"):
                self._entity_symbols[ref] = row
            if kind not in {"METHOD", "FIELD"} or not int(row.get("access", 0)) & 1:
                continue
            if row.get("static") or int(row.get("access", 0)) & 8 or row.get("throws"):
                continue
            if row.get("return_type") in {None, "void"}:
                continue
            if kind == "METHOD" and (parameter_types(row) or not (
                    str(row.get("name") or "").startswith(("get", "is", "has", "can", "read", "peek"))
                    or row.get("name") in {"size", "length", "count"})):
                continue
            self.members[row["owner"]].append((ref, row))

    def ancestors(self, kind):
        if kind not in self._ancestors:
            seen, pending = set(), [kind]
            while pending:
                current = pending.pop()
                if current in seen:
                    continue
                seen.add(current)
                row = self.symbols.get(current, {})
                pending.extend(parent for parent in (
                    row.get("superclass"), *(row.get("interfaces") or ()))
                    if isinstance(parent, str))
            self._ancestors[kind] = seen
        return self._ancestors[kind]

    def plans(self, name, input_types):
        identity = (name, tuple(input_types))
        if identity in self._plans:
            return self._plans[identity]
        wanted = observation_words(name)
        if not wanted:
            return []
        found = []
        for index, kind in enumerate(input_types):
            queue = deque([(kind, (), observation_words(kind.rsplit(".", 1)[-1]), {kind})])
            while queue:
                current, steps, words, visited = queue.popleft()
                if wanted <= words:
                    found.append({"input_index": index, "symbol_refs": list(steps),
                                  "return_type": current})
                    # An exact producer is complete. Do not turn its unrelated
                    # descendants (getClass, reflection flags, etc.) into choices.
                    continue
                if len(steps) >= MAX_OBSERVER_HOPS:
                    continue
                for owner in sorted(self.ancestors(current)):
                    for ref, row in self.members.get(owner, ()):
                        output = row["return_type"]
                        semantic = words | observation_words(row.get("name"))
                        semantic |= observation_words(output.rsplit(".", 1)[-1])
                        next_steps = (*steps, ref)
                        if wanted <= semantic:
                            found.append({"input_index": index, "symbol_refs": list(next_steps),
                                          "return_type": output})
                        elif output not in PRIMITIVES and output not in visited and output in self.symbols:
                            # Object introspection is not a game-state receiver.
                            if output.startswith(("java.lang.reflect.", "java.lang.invoke.")) or output == "java.lang.Class":
                                continue
                            queue.append((output, next_steps, semantic, visited | {output}))
        # A source content type + inspected world enumeration is a complete
        # executable producer even though no getter has the planning label.
        for index, kind in enumerate(input_types):
            entity = resolve_materialized_entity_presence(name, self._entity_symbols, [kind])
            if entity:
                observer = self.symbols[entity["observer_ref"]]
                if observer["owner"] in self.ancestors(kind):
                    found.append({"input_index": index,
                                  "symbol_refs": [entity["observer_ref"]],
                                  "entity_owner": entity["entity_owner"], "return_type": "boolean"})
        unique = {}
        for path in found:
            key = (path["input_index"], tuple(path["symbol_refs"]), path.get("entity_owner"))
            unique[key] = path
        result = sorted(unique.values(), key=lambda p: (
            len(p["symbol_refs"]), p["input_index"], p["symbol_refs"], p.get("entity_owner", "")))
        self._plans[identity] = result
        return result


def bind_runtime_sources(behavior, bindings, planner, events):
    """Require the intersection of sources on one actual callback ABI."""
    names = tuple(dict.fromkeys(behavior.get("runtime_state_reads", ())))
    invalid = [name for name in names if not isinstance(bindings.get(name), dict)
               or bindings[name].get("binding") != "world_state"]
    if invalid:
        raise RuleMissingRuntimeProducer(behavior.get("behavior_id", ""), [
            {"name": name, "reason": "missing_world_observer_binding"} for name in invalid])
    per_event, eligible, available = {}, [], set()
    for event in events:
        sources = {}
        for name in names:
            paths = planner.plans(name, event.get("parameters", ()))
            declared = bindings[name].get("verified_symbol_ref")
            if declared:
                paths = [path for path in paths if path["symbol_refs"] == [declared]]
            sources[name] = paths
            if paths:
                available.add(name)
        if all(sources.values()):
            eligible.append(event)
            per_event[event["event_id"]] = sources
    if names and not eligible:
        missing = [name for name in names if name not in available]
        reason = ("no_inspected_semantic_source" if missing
                  else "no_single_callback_supplies_required_sources")
        raise RuleMissingRuntimeProducer(behavior.get("behavior_id", ""), [
            {"name": name, "reason": reason,
             "required_action": "implement an observer on the producing runtime owner"}
            for name in (missing or names)])
    return eligible, per_event


def materialize_runtime_sources(router, program, source_plans):
    """Emit whole typed paths; only ambiguous complete paths need a model choice."""
    values, selected, shared, guarded = {}, {}, {}, set()
    guards = []

    def require_receiver(ref):
        if ref in guarded:
            return
        guarded.add(ref)
        null = program.value("literal", type=program.kind(ref), value=None)
        guards.append(program.value("binary", [ref, null], operator="!=",
                                    purpose="Runtime source receiver exists"))

    for name, paths in source_plans.items():
        choice = paths[0] if len(paths) == 1 else None
        for offset in range(0, len(paths), PAGE_SIZE) if choice is None else ():
            page = {"p" + str(i): path for i, path in enumerate(paths[offset:offset + PAGE_SIZE])}
            key = choose_rule_data(router, "rule_runtime_source", {
                "behavior": program.behavior, "source_name": name,
                "paths": [{"id": key, **path} for key, path in page.items()],
                "later_paths": len(paths) - offset - len(page)},
                {"source_id": {"type": "string", "enum": [
                    *page, "unavailable", *(["next_page"] if offset + PAGE_SIZE < len(paths) else [])]}},
                "Choose one COMPLETE inspected observation path for this exact concept. "
                "The host emits every receiver and call. Select unavailable if none "
                "represents the original behavior. No Java or API search.")["source_id"]
            if key == "unavailable":
                break
            if key != "next_page":
                choice = page[key]
                break
        if choice is None:
            raise RuleMissingRuntimeProducer(program.behavior.get("behavior_id", ""), [
                {"name": name, "reason": "no_selected_complete_runtime_source"}])
        value = program.inputs[choice["input_index"]]
        prefix = ()
        for ref in choice["symbol_refs"]:
            if prefix:
                require_receiver(value)
            prefix = (*prefix, ref)
            identity = (choice["input_index"], prefix)
            if identity not in shared:
                symbol = program.symbols[ref]
                shared[identity] = program.value(
                    "call" if symbol["kind"] == "METHOD" else "field", [value],
                    symbol_ref=ref, purpose="Runtime read: " + name)
            value = shared[identity]
        if choice.get("entity_owner"):
            require_receiver(value)
            value = program.value("entity_presence", [value], entity_owner=choice["entity_owner"],
                                  purpose="Runtime read: " + name)
        values[name] = value
        selected[name] = choice
    program.behavior["runtime_source_values"] = values
    program.behavior["runtime_source_guards"] = guards
    return values, selected
