"""Persist the handoff from worksheet records to concrete feature owners."""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from .authored_structured_design import active_concern_records

TYPED_UI_OWNER = "__typed_ui__"


def feature_module_id(feature_id: str) -> str:
    return "feature_" + re.sub(r"[^a-z0-9_]", "_", feature_id.casefold())


def _runtime_records(sections):
    for section in ("behavior_contract", "algorithm", "failure_and_limits", "state_model", "integration"):
        for concern, rows in active_concern_records(sections, section).items():
            if section == "state_model" and concern not in {"transitions", "updates"}:
                continue
            if section == "integration" and concern != "entry_points":
                continue
            for index, row in enumerate(rows):
                yield f"{section}.{concern}[{index}]", row


def author_runtime_ownership(router, scope, sections, *, budget=None):
    """Choose one owner per runtime record using bounded feature identities.

    Infrastructure remains in the Typed host. Existing mutating UI transactions
    may retain their dedicated host dispatcher, never a second generic handler.
    """
    from .authored_ui_action_bindings import _mutating_ui_actions
    from .fixed_template_generation import generate_fixed_template_value

    features = [row for row in scope.get("requirements", ())
                if row.get("implementation_role", "gameplay") != "interface"]
    if not features:
        return []
    choices = [row["requirement_id"] for row in features]
    actions = _mutating_ui_actions(sections)
    groups = {}
    for ref, record in _runtime_records(sections):
        available = [*choices]
        if actions and not ref.startswith("integration."):
            available.extend(TYPED_UI_OWNER + ":" + action for action in actions)
        if len(available) == 1:
            owner = available[0]
        else:
            if budget is not None:
                budget.consume("request_scope.runtime_owner")
            owner = generate_fixed_template_value(router, "planner", (
                {"role": "system", "content": (
                    "Assign this existing worksheet record to exactly one original requested feature. "
                    "Do not add mechanics or change the record. Choose __typed_ui__:ACTION only for a "
                    "transaction/state rule already implemented by one of the listed UI actions. "
                    "Custom gameplay triggers belong to their actual feature, not player join or an invented command."
                )},
                {"role": "user", "content": json.dumps({
                    "record_ref": ref, "record": record,
                    "features": [{"id": row["requirement_id"], "statement": row["statement"]} for row in features],
                    "existing_ui_actions": actions,
                }, ensure_ascii=False, separators=(",", ":"))},
            ), response_schema={"type": "string", "enum": available}, enable_tools=False)
        action = ""
        if owner.startswith(TYPED_UI_OWNER + ":"):
            owner, action = TYPED_UI_OWNER, owner[len(TYPED_UI_OWNER) + 1:]
        group = groups.setdefault((owner, action), {
            "feature_id": owner,
            "owner_module": "authored_typed_plan" if owner == TYPED_UI_OWNER else feature_module_id(owner),
            "worksheet_record_refs": [],
            **({"ui_action": action} if action else {}),
        })
        group["worksheet_record_refs"].append(ref)
    return list(groups.values())


def owned_record_refs(sections, owners, *, include_typed_ui=False):
    known = dict(_runtime_records(sections))
    result, seen = set(), set()
    for owner in owners:
        for ref in owner.get("worksheet_record_refs", ()):
            if ref not in known:
                raise ValueError("RUNTIME_OWNER_RECORD_UNKNOWN: " + ref)
            if ref in seen:
                raise ValueError("RUNTIME_OWNER_RECORD_DUPLICATE: " + ref)
            seen.add(ref)
            if include_typed_ui or owner["feature_id"] != TYPED_UI_OWNER:
                result.add(ref)
    return result


def runtime_sections(sections, owners, *, for_ui=False):
    """Project only the records implemented by this Typed dispatcher."""
    moved = owned_record_refs(sections, owners, include_typed_ui=True)
    ui = {ref for owner in owners if owner["feature_id"] == TYPED_UI_OWNER
          for ref in owner["worksheet_record_refs"]}
    removed = (set(dict(_runtime_records(sections))) - ui
               if for_ui and owners else moved)
    result = deepcopy(dict(sections or {}))
    for section, value in result.items():
        spec = value.get("specification", {})
        for concern, rows in list(spec.items()):
            if concern == "inapplicable_concerns" or not isinstance(rows, list):
                continue
            spec[concern] = [row for index, row in enumerate(rows)
                             if f"{section}.{concern}[{index}]" not in removed]
    return result


def owner_records(sections, owners, feature_id):
    records = dict(_runtime_records(sections))
    return [{"record_ref": ref, "record": records[ref],
             **({"ui_action": owner["ui_action"]} if owner.get("ui_action") else {})}
            for owner in owners if owner["feature_id"] == feature_id
            for ref in owner["worksheet_record_refs"]]


def ui_record_action_guards(sections, owners):
    """Retain exact action identity after callback-owned rows are projected out."""
    assigned = {ref: owner["ui_action"] for owner in owners
                if owner["feature_id"] == TYPED_UI_OWNER
                for ref in owner["worksheet_record_refs"]}
    projected = {}
    indices = {}
    for ref, _record in _runtime_records(sections):
        if ref not in assigned:
            continue
        concern = ref.split("[", 1)[0]
        index = indices.get(concern, 0)
        projected[f"{concern}[{index}]"] = assigned[ref]
        indices[concern] = index + 1
    return projected


def generated_owner_source_context(project_root, spec):
    """Read materialized prerequisite APIs without changing approved contracts."""
    root = Path(project_root).resolve()
    paths = json.loads(spec.get("bindings", {}).get("content_dependency_paths_json", "[]"))
    readable = []
    for relative in dict.fromkeys(paths):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not str(relative).startswith(("src/main/java/", "src/client/java/")):
            raise ValueError("RUNTIME_OWNER_SOURCE_PATH_INVALID")
        if not path.is_file() or path.stat().st_size > 512 * 1024:
            continue  # Optional state files exist only when authored state requires them.
        readable.append((relative, path))
    result = []
    per_file = max(1, 9000 // max(1, len(readable)))
    for relative, path in readable:
        raw = path.read_bytes()
        source = raw.decode("utf-8")
        # Exact declarations include methods below a long implementation body;
        # an arbitrary prefix alone would still force the coder to guess APIs.
        declarations = "\n".join(match.group(0) for match in re.finditer(
            r"\b(?:public|protected)\s+[^{};]+[;{]", source))
        declarations = declarations[:min(1800, per_file // 2)]
        excerpt = source[:min(2500, per_file - len(declarations))]
        result.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                       "source": excerpt, "public_declarations": declarations,
                       "truncated": len(excerpt) < len(source),
                       "authority": "materialized_prerequisite_source"})
    return result
