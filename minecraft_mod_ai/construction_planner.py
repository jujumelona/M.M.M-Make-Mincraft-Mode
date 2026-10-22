"""Turn preserved requirements into small semantic work, not whole-class requests."""
from __future__ import annotations

import hashlib
import json

from .construction_task import BehaviorRequirement, ConstructionTask, validate_task_graph
from .fixed_template_generation import _validate_native_arguments
from .construction_trigger_provenance import preserve_grounded_trigger


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


def _id(parent, intent, start, end):
    """Distinct grounded occurrences must not collapse on English paraphrases."""
    signature = chr(0).join((parent, f"{start}:{end}", intent.casefold()))
    return "behavior_" + hashlib.sha256(signature.encode()).hexdigest()[:20]


def _unclaimed_source(source_text, claimed):
    """Preserve *all* remaining user text without inventing new mechanics.

    Only exact anchored character intervals count as covered. Recovery spans
    retain the original source verbatim; none is certified as an interpreted
    behavior merely because the model stopped making progress.
    """
    covered = bytearray(len(source_text))
    for left, right in claimed:
        covered[left:right] = bytes([1]) * (right - left)
    offset = 0
    while offset < len(source_text):
        if covered[offset]:
            offset += 1
            continue
        right = offset + 1
        while right < len(source_text) and not covered[right]:
            right += 1
        # Whitespace and punctuation alone do not represent an omitted action.
        if any(c.isalnum() for c in source_text[offset:right]):
            from .request_scope import _request_chunks
            for base, part in _request_chunks(source_text[offset:right], limit=180):
                if not any(c.isalnum() for c in part):
                    continue
                yield offset + base, offset + base + len(part), part
        offset = right


def _next_source_quote_bounds(source, quote, claimed):
    """Prefer a not-yet-owned occurrence of a repeated verbatim phrase."""
    from .request_scope import _source_quote_bounds
    examined = 0
    fallback = None
    while examined < len(source):
        found = _source_quote_bounds(source[examined:], quote)
        if found is None:
            break
        bounds = (examined + found[0], examined + found[1])
        if fallback is None:
            fallback = bounds
        if bounds not in claimed:
            return bounds
        examined = bounds[0] + 1
    return fallback


def author_behavior_requirements(scope, router, *, checkpoint=None):
    result = []
    saved = scope.get("behavior_requirements")
    if saved is not None:
        requirement_text = {
            item["requirement_id"]: str(
                item.get("source_span", {}).get("text") or item["statement"])
            for item in scope.get("requirements", ())
        }
        for row in saved:
            row = dict(row)
            for name in ("state_reads", "state_writes", "runtime_state_reads", "runtime_state_writes"):
                row[name] = tuple(row.get(name, ()))
            row["trigger"], diagnostic = preserve_grounded_trigger(
                row.get("trigger", ""), source_quote=row["source_quote"],
                source_text=requirement_text.get(row["requirement_id"], row["source_quote"]))
            if diagnostic:
                print("CONSTRUCTION_TRIGGER_PROVENANCE: " +
                      json.dumps({"behavior_id": row["behavior_id"], **diagnostic},
                                 ensure_ascii=False), flush=True)
            result.append(BehaviorRequirement(**row))
        if {row.requirement_id for row in result} != {
            r["requirement_id"] for r in scope.get("requirements", ())
        }:
            raise ValueError("CONSTRUCTION_REQUIREMENT_UNRESOLVED: saved behavior coverage")
        return tuple(result)

    for requirement in scope.get("requirements", ()):
        if checkpoint is not None:
            cached = checkpoint.load(requirement)
            if cached is not None:
                result.extend(cached)
                continue
        beginning = len(result)
        parent, statement = requirement["requirement_id"], requirement["statement"]
        source_text = str(requirement.get("source_span", {}).get("text") or statement)
        if not source_text.strip():
            raise ValueError("CONSTRUCTION_SOURCE_EMPTY: " + parent)
        if router is None:
            raise ValueError("CONSTRUCTION_PLANNER_REQUIRED")
        anchored = []
        seen = set()
        prior = []
        stop_reason = "source_exhausted"
        # One short original feature may contain several mechanics. This finite
        # upper bound is for diagnostic protection, never grounds any invented
        # action or arbitrarily marks unreviewed source as complete.
        page_limit = max(3, min(32, len(source_text) // 24 + 3))
        for _page in range(page_limit):
            remaining = list(_unclaimed_source(source_text, anchored))
            messages = [{"role": "system", "content": (
                "Break the provided ORIGINAL source text into requested, independently "
                "observable Minecraft behaviors in source order. Output one behavior "
                "at a time, or status=complete when no further independent behavior "
                "exists. Never paraphrase previously output behavior as new work. "
                "Keep atomic transactions together, and do not invent mechanics. "
                "Use a contiguous quote from source_text, allowing different "
                "whitespace. Shared quotes are permitted for distinct actions. "
                "Use remaining_source_text and previous behaviors for coverage. "
                "If there is no new action, return status=complete, not a duplicate. "
                "Do not write code.")},
                {"role": "user", "content": json.dumps({
                    "requirement_id": parent,
                    "statement": statement,
                    "source_text": source_text,
                    "source_recovery": bool(requirement.get("source_recovery")),
                    "remaining_source_text": [item[2] for item in remaining],
                    # The host keeps the entire accepted requirement ledger.
                    # Dropping entries after 16 made later decisions forget
                    # already claimed source spans and rediscover old actions.
                    "previous_behaviors": prior,
                }, ensure_ascii=False)}]
            row = _validate_native_arguments(router.generate_structured_data(
                "planner", messages, data_name="construction_behavior_next",
                parameters=_BEHAVIOR_SCHEMA), _BEHAVIOR_SCHEMA)
            if row["status"] == "complete":
                stop_reason = "model_complete"
                break
            intent, quote = row["intent"].strip(), row["source_quote"]
            bounds = _next_source_quote_bounds(source_text, quote, anchored)
            if not intent or not row["effects"].strip() or bounds is None:
                stop_reason = "ungrounded_model_response"
                print("CONSTRUCTION_BEHAVIOR_SOURCE_RECOVERY: " +
                      json.dumps({"requirement_id": parent,
                                  "reason": stop_reason,
                                  "source_text": source_text[:300],
                                  "model_intent": intent[:160],
                                  "model_quote": quote[:160]}, ensure_ascii=False),
                      flush=True)
                break
            left, right = bounds
            quote = source_text[left:right]
            identity = (left, right, intent.casefold())
            if identity in seen:
                stop_reason = "duplicate_source_behavior"
                print("CONSTRUCTION_BEHAVIOR_DUPLICATE: " +
                      json.dumps({"requirement_id": parent,
                                  "intent": intent, "source_quote": quote,
                                  "source_start": left, "source_end": right},
                                 ensure_ascii=False), flush=True)
                break
            # Two actions may legitimately share the same English paraphrase,
            # provided their original source spans are different.
            seen.add(identity)
            anchored.append((left, right))
            prior.append({"intent": intent, "source_quote": quote,
                          "source_start": left, "source_end": right})
            grounded_trigger, diagnostic = preserve_grounded_trigger(
                row["trigger"], source_quote=quote, source_text=source_text)
            if diagnostic:
                print("CONSTRUCTION_TRIGGER_PROVENANCE: " +
                      json.dumps({"requirement_id": parent, **diagnostic},
                                 ensure_ascii=False), flush=True)
            result.append(BehaviorRequirement(
                _id(parent, intent, left, right), parent, intent, quote,
                grounded_trigger, row["effects"], tuple(row["state_reads"]),
                tuple(row["state_writes"]),
            ))
        else:
            stop_reason = "page_limit"

        missing = list(_unclaimed_source(source_text, anchored))
        # If the model cannot interpret a remaining source excerpt, preserve
        # its *exact* text as an unresolved work obligation. No inferred event,
        # trigger, or gameplay effect is silently substituted.
        for left, right, original in missing:
            raw = original.strip()
            if not raw:
                continue
            result.append(BehaviorRequirement(
                _id(parent, raw, left, right), parent, raw, original, "",
                original, (), (), "source_preserved_unresolved",
            ))
        print("CONSTRUCTION_BEHAVIOR_COVERAGE: " + json.dumps({
            "requirement_id": parent,
            "source_excerpt": source_text[:240],
            "source_characters": len(source_text),
            "grounded_behaviors": len(prior),
            "unresolved_source_spans": len(missing),
            "stop_reason": stop_reason,
        }, ensure_ascii=False), flush=True)
        if checkpoint is not None:
            checkpoint.save(requirement, result[beginning:])
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
    from .construction_behavior_checkpoint import BehaviorPlanningCheckpoint
    checkpoint = BehaviorPlanningCheckpoint(bounded, router)
    # Read -> model -> accept -> publish is one scope-owned transaction.
    # A second notebook waits and then restores accepted per-requirement
    # records instead of independently inferring and racing to overwrite.
    with checkpoint.claim():
        behaviors = author_behavior_requirements(bounded, router, checkpoint=checkpoint)
    bounded["behavior_requirements"] = [row.to_dict() for row in behaviors]
    return bounded


def plan_construction_tasks(scope, source_evidence, target_symbols, router):
    behaviors = author_behavior_requirements(scope, router)
    tasks = []
    for behavior in behaviors:
        parent = behavior.requirement_id
        key = behavior.behavior_id + ".logic"
        tasks.append(ConstructionTask(
            key, (parent,), "logic", behavior.intent,
            state_reads=behavior.state_reads, state_writes=behavior.state_writes,
            trigger=behavior.trigger, effects=behavior.effects,
            behavior_id=behavior.behavior_id,
            verification_ids=(key + ".compile", key + ".behavior"),
        ))
        tasks.append(ConstructionTask(
            behavior.behavior_id + ".verify", (parent,), "verify",
            "Verify " + behavior.intent, depends_on=(key,),
            verification_ids=(key + ".compile", key + ".behavior"),
            behavior_id=behavior.behavior_id,
        ))
    validate_task_graph(tasks)
    return tuple(tasks)
