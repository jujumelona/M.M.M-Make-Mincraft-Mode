"""Require evidence for runtime-state reads before invoking the small Java coder.

A resolved state-owner *label* is not a value producer. In particular,
world_state + host_classified_unbound_read means planning retained a real
obligation without finding a Java observer or materialized content producer.
Neither a valid Fabric event nor an unrelated world boolean discharges it.
"""
from __future__ import annotations

import json
import re


class RuleMissingRuntimeProducer(ValueError):
    def __init__(self, behavior_id, missing):
        self.behavior_id = behavior_id
        self.missing = tuple(missing)
        details = {"behavior_id": behavior_id, "missing_runtime_producers": missing,
                   "required_action": "materialize a producing content/source module or bind an inspected runtime observer before compiling this behavior"}
        super().__init__("RULE_RUNTIME_PRODUCER_REQUIRED: " + json.dumps(
            details, ensure_ascii=False, sort_keys=True))


def _canonical_words(value):
    # Only exact vocabulary correspondence is deterministic. Similar-looking
    # words, substrings and AI-inferred synonyms are not evidence of a getter.
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(value))
    return tuple(re.findall(r"[^\W_]+", text.casefold()))


def _actual_observer_candidates(name, symbols, input_types):
    expected = _canonical_words(name)
    if not expected:
        return []
    reachable = set()
    pending = list(dict.fromkeys(input_types))
    while pending:
        owner = pending.pop()
        if owner in reachable:
            continue
        reachable.add(owner)
        row = symbols.get(owner, {})
        pending.extend(kind for kind in [
            row.get("superclass"), *(row.get("interfaces") or ())
        ] if isinstance(kind, str))
    options = []
    for key, row in symbols.items():
        if not isinstance(row, dict) or row.get("kind") != "METHOD":
            continue
        if row.get("owner") not in reachable:
            continue
        if not int(row.get("access", 0)) & 1:
            continue
        if row.get("static") or int(row.get("access", 0)) & 8:
            continue
        if row.get("parameters") or row.get("throws"):
            continue
        if row.get("return_type") in {None, "void"}:
            continue
        method = str(row.get("name") or "")
        prefix = next((p for p in ("get", "is", "has", "can", "read")
                       if method.startswith(p) and len(method) > len(p)
                       and method[len(p)].isupper()), None)
        if not prefix or _canonical_words(method[len(prefix):]) != expected:
            continue
        options.append(str(key))
    return sorted(set(options))


def compatible_events_for_runtime_producers(events, sources, symbols):
    """Limit callback selection to inspected hooks that supply every receiver.

    Selecting a world tick first and later discovering that the source getter
    belongs only to a Player creates a second avoidable model failure.
    """
    if not sources:
        return list(events)

    def bases(kind):
        reached, pending = set(), [kind]
        while pending:
            current = pending.pop()
            if current in reached:
                continue
            reached.add(current)
            record = symbols.get(current, {})
            pending.extend(x for x in [
                record.get("superclass"), *(record.get("interfaces") or ())
            ] if isinstance(x, str))
        return reached

    result = []
    for event in events:
        reachable = set().union(*(bases(kind) for kind in event.get("parameters", ())))
        if all(symbols[ref]["owner"] in reachable for ref in sources.values()):
            result.append(event)
    return result


def validate_runtime_read_producers(behavior, runtime_bindings, symbols, *, input_types=()):
    """Reject planning-only observations; accept only checked exact source IDs.

    This is a *necessary*, not sufficient, condition for semantic coverage.
    Verified symbol existence does not prove the implementation's game effect.
    """
    missing = []
    resolved = {}
    for name in dict.fromkeys(behavior.get("runtime_state_reads", ())):
        binding = runtime_bindings.get(name)
        if not isinstance(binding, dict) or binding.get("binding") != "world_state":
            missing.append({"name": name, "reason": "missing_world_observer_binding"})
            continue
        # A model-authored 'host_verified_observer' label is not proof.
        # Reconstruct the exact reachable public accessor inventory from
        # authoritative JVM facts even for an explicitly supplied symbol ID.
        declared = binding.get("verified_symbol_ref")
        derived = _actual_observer_candidates(name, symbols, input_types)
        ref = declared if isinstance(declared, str) else (
            derived[0] if len(derived) == 1 else None)
        evidence = symbols.get(ref) if isinstance(ref, str) else None
        if (not isinstance(evidence, dict)
                or evidence.get("kind") not in {"METHOD", "FIELD"}
                or evidence.get("return_type") in {None, "void"}
                or (evidence.get("kind") == "METHOD" and evidence.get("parameters"))
                or len(derived) != 1 or derived[0] != ref):
            missing.append({
                "name": name,
                "reason": ("ambiguous_runtime_observer" if len(derived) > 1
                           else "planning_only_world_state" if not ref
                           else "unverified_or_unusable_java_observer"),
                "binding_status": binding.get("semantic_status"),
                "matching_candidate_count": len(derived),
            })
            continue
        resolved[name] = ref
    if missing:
        raise RuleMissingRuntimeProducer(str(behavior.get("behavior_id") or ""), missing)
    return resolved


def expression_dependencies(program, root):
    """Return all typed IR nodes actually evaluated by the selected expression."""
    nodes = program.checker.nodes
    if root not in nodes:
        raise ValueError("RULE_OBSERVATION_VALUE_MISSING: " + str(root))
    seen, pending = set(), [root]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        node = nodes.get(current)
        if not isinstance(node, dict):
            raise ValueError("RULE_OBSERVATION_DANGLING_REFERENCE: " + str(current))
        seen.add(current)
        pending.extend(node.get("input_refs", ()))
    return seen


def runtime_source_expression_proof(program, root, semantic_name):
    """Check a live, typed, topic-relevant producer DAG, not a planner label.

    Complex observations can be calculated from several JVM calls, fields and
    comparisons. They need not have a zero-argument getter named after the
    planning concept. The host verifies that the DAG references a real
    callback input and includes an inspected API member relevant to the
    requested observation. This is a structural proof, not a substitute for
    gameplay tests of semantic correctness.
    """
    tokens = set(_canonical_words(semantic_name)) - {
        "state", "runtime", "value", "current", "actual", "has", "get",
        "is", "the", "of", "a", "an", "and",
    }
    if not tokens:
        return False
    graph = expression_dependencies(program, root)
    nodes = program.checker.nodes

    def derived_from_event(ref, visiting=None):
        if visiting is None:
            visiting = set()
        if ref in visiting:
            return False
        node = nodes[ref]
        if node["op"] == "parameter":
            return True
        visiting = visiting | {ref}
        return any(derived_from_event(child, visiting)
                   for child in node.get("input_refs", ()))

    if not derived_from_event(root):
        return False
    for ref in graph:
        node = nodes[ref]
        if node.get("op") not in {"call", "field"}:
            continue
        symbol_ref = node.get("attributes", {}).get("symbol_ref")
        symbol = program.symbols.get(symbol_ref)
        if not isinstance(symbol, dict) or symbol.get("kind") not in {"METHOD", "FIELD"}:
            continue
        # Only a live receiver or a callback-derived API argument grounds this
        # observation. A newly constructed helper and a constant do not.
        if not any(derived_from_event(child) for child in node.get("input_refs", ())):
            continue
        member_name = str(symbol.get("name") or "")
        # Names are evidence hints, not proofs of game effects. Require a
        # relevant inspected member in the *actual expression DAG*.
        if tokens & set(_canonical_words(member_name)):
            return True
    return False
