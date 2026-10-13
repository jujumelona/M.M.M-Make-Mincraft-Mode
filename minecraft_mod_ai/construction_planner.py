"""Turn preserved requirements into small semantic work, not whole-class requests."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from .construction_task import BehaviorRequirement, ConstructionTask, validate_task_graph
from .fixed_template_generation import _validate_native_arguments


_BEHAVIOR_SCHEMA = {"type": "object", "properties": {
    "status": {"type": "string", "enum": ["behavior", "complete"]},
    "intent": {"type": "string", "maxLength": 240},
    "source_quote": {"type": "string", "maxLength": 240},
    "trigger": {"type": "string", "maxLength": 160},
    "effects": {"type": "string", "maxLength": 240},
    "state_reads": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 96}},
    "state_writes": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 96}},
}, "required": ["status", "intent", "source_quote", "trigger", "effects", "state_reads", "state_writes"],
    "additionalProperties": False}


def _id(parent, intent):
    return "behavior_" + hashlib.sha256((parent + "\0" + intent).encode()).hexdigest()[:20]


def author_behavior_requirements(scope, router):
    result = []
    saved = scope.get("behavior_requirements")
    if saved is not None:
        for row in saved:
            row = dict(row)
            for name in ("state_reads", "state_writes"):
                row[name] = tuple(row.get(name, ()))
            result.append(BehaviorRequirement(**row))
        if {row.requirement_id for row in result} != {r["requirement_id"] for r in scope.get("requirements", ())}:
            raise ValueError("CONSTRUCTION_REQUIREMENT_UNRESOLVED: saved behavior coverage")
        return tuple(result)
    for requirement in scope.get("requirements", ()):
        parent, statement = requirement["requirement_id"], requirement["statement"]
        source_text = str(requirement.get("source_span", {}).get("text") or statement)
        prior = []
        while True:
            if router is None:
                raise ValueError("CONSTRUCTION_PLANNER_REQUIRED")
            messages = [{"role": "system", "content": (
                "Decompose the current requirement into independently observable behaviors. "
                "Return one independently triggered behavior at a time in source order, "
                "preserving every requested mechanic rather than grouping unrelated player actions. "
                "Use only the supplied original source_text as authority; quote a matching source span "
                "and do not invent new effects or interactions. Keep coupled atomic transaction "
                "effects together. State reads/writes are only this behavior's dependencies. Return complete only "
                "after every behavior is represented; never invent an interaction or delete a feature. "
                "Do not write Java or choose a fixed gameplay catalogue.")},
                {"role": "user", "content": json.dumps({
                    "requirement_id": parent, "statement": statement,
                    "source_text": source_text,
                    "source_recovery": bool(requirement.get("source_recovery")),
                    "previous_intents": prior}, ensure_ascii=False)}]
            row = _validate_native_arguments(router.generate_structured_data(
                "planner", messages, data_name="construction_behavior_next", parameters=_BEHAVIOR_SCHEMA), _BEHAVIOR_SCHEMA)
            if row["status"] == "complete":
                if not prior:
                    raise ValueError("CONSTRUCTION_REQUIREMENT_UNRESOLVED: " + parent)
                break
            intent, quote = row["intent"].strip(), row["source_quote"]
            from .request_scope import _source_quote_bounds
            bounds = _source_quote_bounds(source_text, quote)
            if not intent or not row["effects"].strip() or bounds is None:
                raise ValueError("CONSTRUCTION_BEHAVIOR_UNGROUNDED: " + parent)
            quote = source_text[bounds[0]:bounds[1]]
            if intent.casefold() in {value.casefold() for value in prior}:
                raise ValueError("CONSTRUCTION_BEHAVIOR_NO_PROGRESS: " + parent)
            prior.append(intent)
            result.append(BehaviorRequirement(_id(parent, intent), parent, intent, quote,
                          row["trigger"], row["effects"], tuple(row["state_reads"]), tuple(row["state_writes"])))
    return tuple(result)



def bind_behavior_scope(scope, router):
    """Bind the execution-only behavior ledger to the original source scope.

    Reuse saved exact child contracts during production retries. Never
    substitute a fake one-behavior result when model authoring fails.
    """
    from copy import deepcopy
    bounded = deepcopy(dict(scope))
    if not bounded.get("requirements"):
        bounded["behavior_requirements"] = []
        return bounded
    behaviors = author_behavior_requirements(bounded, router)
    bounded["behavior_requirements"] = [row.to_dict() for row in behaviors]
    return bounded


def plan_construction_tasks(scope, source_evidence, target_symbols, router):
    behaviors = author_behavior_requirements(scope, router)
    tasks, installed = [], set()
    for behavior in behaviors:
        parent = behavior.requirement_id
        references, dependencies = [], []
        for evidence in source_evidence:
            row = evidence.to_dict() if hasattr(evidence, "to_dict") else dict(evidence)
            if parent not in row.get("requirement_ids", ()):
                continue
            if row.get("behavior_ids") and behavior.behavior_id not in row["behavior_ids"]:
                continue
            # Search hits and unported references never confer install authority.
            if row.get("status") not in {"verified", "installed", "port_required"}:
                continue
            if row.get("mode") not in {"source_transplant", "port"}:
                continue
            source_id = row.get("evidence_id") or row.get("source_id", "")
            if not source_id:
                continue
            key = "source_" + hashlib.sha256(source_id.encode()).hexdigest()[:20]
            if key not in installed:
                tasks.append(ConstructionTask(key, tuple(row.get("requirement_ids", (parent,))),
                    "reuse" if row.get("status") in {"verified", "installed"} and row.get("mode") == "source_transplant" else "port",
                    "Install or adapt pinned source " + source_id, source_refs=(row,),
                    output_symbols=tuple(row.get("symbols", ())), verification_ids=(key + ".compile",)))
                installed.add(key)
            dependencies.append(key)
            references.append(row)
        key = behavior.behavior_id + ".logic"
        tasks.append(ConstructionTask(key, (parent,), "logic", behavior.intent,
            depends_on=tuple(dependencies), source_refs=tuple(references),
            input_symbols=tuple(dict.fromkeys(s for ref in references for s in ref.get("symbols", ()) if s in target_symbols)),
            state_reads=behavior.state_reads, state_writes=behavior.state_writes,
            trigger=behavior.trigger, effects=behavior.effects, behavior_id=behavior.behavior_id,
            verification_ids=(key + ".compile", key + ".behavior")))
        if references:
            glue = behavior.behavior_id + ".glue"
            tasks.append(replace(tasks[-1], task_id=glue, kind="glue",
                intent="Connect " + behavior.intent + " to its installed source and trigger",
                depends_on=(key,), verification_ids=(glue + ".behavior",)))
        tasks.append(ConstructionTask(behavior.behavior_id + ".verify", (parent,), "verify",
            "Verify " + behavior.intent, depends_on=(tasks[-1].task_id,),
            verification_ids=(key + ".compile", key + ".behavior"), behavior_id=behavior.behavior_id))
    validate_task_graph(tasks)
    return tuple(tasks)
