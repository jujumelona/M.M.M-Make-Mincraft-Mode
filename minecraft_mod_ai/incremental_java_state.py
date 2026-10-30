"""Keep the existing typed-state contracts with mapped operation diagnostics."""
from __future__ import annotations

import re

from .incremental_java_render import render_program, render_member_fragment
from .java_region_parser import _parse, _text


def _repair_entrypoint(program, diagnostic):
    no_effect = re.search(
        r"INCREMENTAL_JAVA_NO_OBSERVABLE_ACTION: entrypoint=([A-Za-z0-9_]+)",
        diagnostic,
    )
    if no_effect:
        member = next(
            (row for row in program["members"] if row["id"] == no_effect[1]), None,
        )
        if member is None:
            return None, diagnostic
        behavior_id = (member.get("behavior") or {}).get("behavior_id") or member["id"]
        return member, "CONSTRUCTION_FILE_BEHAVIOR_NOOP: " + str(behavior_id)
    direct_missing = re.search(
        r"INCREMENTAL_JAVA_(?:INSPECTED_OCCURRENCE_MISSING|NATIVE_MUTATION_UNPROVEN): ([^ ;]+)",
        diagnostic,
    )
    if direct_missing:
        identity = direct_missing[1]
        candidates = [row for row in program["members"]
                      if (row.get("behavior") or {}).get("behavior_id") == identity]
        native = "NATIVE_MUTATION_UNPROVEN" in diagnostic
        member = next((row for row in candidates if row.get("event") and native), None)
        if member is None:
            member = next((row for row in candidates if row.get("registrar")), None)
        if member is not None:
            return member, "CONSTRUCTION_FILE_BEHAVIOR_NOOP: " + diagnostic
    match = re.search(
        r"CONSTRUCTION_FILE_BEHAVIOR_(?:METHOD_MISSING|NOOP): (.+)", diagnostic,
    )
    if match:
        member = next(
            (row for row in program["members"] if row.get("registrar")
             and str((row.get("behavior") or {}).get("behavior_id")) == match[1].strip()),
            None,
        )
        return member, diagnostic
    return None, diagnostic


def map_behavior_diagnostic(source, spec, context, diagnostic):
    program = (context.get("incremental_state") or {}).get("program")
    if not program:
        return diagnostic
    member, diagnostic = _repair_entrypoint(program, diagnostic)
    if member is None:
        return diagnostic
    raw, mapping = render_program(program)
    offset = len(source.splitlines()) - len(raw.splitlines())
    line = next(line for line, location in mapping.items() if location["member_id"] == member["id"]
                and location["slot"] == "declaration")
    filename = spec["target_path"].rsplit("/", 1)[-1]
    return f"{filename}:{line + offset}: error: {diagnostic}"


def validate_program_state(source, spec, context, *, validators):
    validate_host_state_access, validate_behavior_state_provenance = validators
    state = context.get("incremental_state") or {}
    program = state.get("program")
    contracts = context.get("host_state_access") or context.get("state_access") or ()
    if not program or not contracts:
        return
    raw, line_map = render_program(program)
    offset = len(source.splitlines()) - len(raw.splitlines())
    for member in program["members"]:
        if member["kind"] not in {"field", "method", "constructor"}:
            continue
        fragment = render_member_fragment(program, member["id"])
        try:
            validate_host_state_access(fragment, contracts)
            validate_behavior_state_provenance(fragment, contracts, member.get("behavior"))
        except ValueError as exc:
            diagnostic = str(exc).split(". Use only declared typed state accessors:", 1)[0]
            line = _state_error_line(raw, line_map, member["id"], diagnostic, contracts)
            filename = spec["target_path"].rsplit("/", 1)[-1]
            raise ValueError(f"{filename}:{line + offset}: error: {diagnostic}") from exc


def _state_error_line(source, mapping, member_id, diagnostic, contracts):
    primary = diagnostic.split(". Use only declared typed state accessors:", 1)[0]
    pattern = (r"\b(?:getState|setState|exportState|importState|clearScopedState|defaultStateSnapshot)\b"
               if "JAVA_HOST_RAW_STATE_ACCESS:" in primary else r"(?:readState_|writeState_)[A-Za-z0-9_$]+")
    names = re.findall(pattern, primary)[:1]
    matches = [(line, location) for line, location in mapping.items() if location["member_id"] == member_id
               and location["slot"] != "registrar"]
    accepted_lines = {line for line, _ in matches}
    owners = {row["owner"] for row in contracts}
    owners.update(owner.rsplit(".", 1)[-1] for owner in tuple(owners))
    for line in _state_call_lines(source, names, owners):
        if line in accepted_lines:
            return line
    return matches[0][0]


def _state_call_lines(source, names, owners):
    raw, root = _parse(source)
    pending = [root]
    while pending:
        node = pending.pop()
        if node.type in {"method_invocation", "method_reference"}:
            reference = node.type == "method_reference"
            name = node.named_children[-1] if reference else node.child_by_field_name("name")
            receiver = node.named_children[0] if reference else node.child_by_field_name("object")
            if name is not None and receiver is not None and _text(raw, name) in names:
                owner = re.sub(r"\s+", "", _text(raw, receiver))
                if owner in owners:
                    yield name.start_point.row + 1
        pending.extend(reversed(node.named_children))
