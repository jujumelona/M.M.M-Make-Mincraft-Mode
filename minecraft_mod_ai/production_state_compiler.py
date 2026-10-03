from __future__ import annotations

"""Production-only semantic compiler for authored state models.

Planning remains free Markdown. At production entry this module lowers only the
approved state_model section into bounded records consumed by the deterministic
host state compiler. The model never emits Java, Java symbols, imports, Fabric
APIs, or host-private AST objects on this path.
"""

from collections.abc import Mapping, Sequence
import json
import re
from typing import Any

from .authored_ir_parser import authored_section_id, parse_markdown_heading
from .authored_plan import AuthoredPlan
from .planning_detail_slots import DETAIL_RECORDS
from .structured_state_runtime import (
    validate_state_expression,
    validate_structured_state_section,
)

_STATE_CONCERNS = tuple(DETAIL_RECORDS["state_model"])
_MAX_CONCERN_RECORDS = 12
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ASSIGNMENT = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\+=|-=|\*=|/=|=)\s*(.*?)\s*$"
)


def _section_text(text: str, section: str) -> str:
    lines = str(text or "").splitlines()
    start = -1
    depth = 0
    for index, line in enumerate(lines):
        heading = parse_markdown_heading(line)
        if heading is None:
            continue
        heading_depth, title = heading
        if authored_section_id(title) == section:
            start = index
            depth = heading_depth
            break
    if start < 0:
        return ""

    end = len(lines)
    for index in range(start + 1, len(lines)):
        heading = parse_markdown_heading(lines[index])
        if heading is None:
            continue
        heading_depth, title = heading
        if heading_depth <= depth and authored_section_id(title):
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def _concern_source(source: str, concern: str) -> str:
    """Return only one authored concern block from a canonical section.

    The free-Markdown planner emits one top-level bullet per concern. Production
    extraction must not repeatedly send the entire state_model to a small model:
    unrelated concern rows materially increase false EMPTY classifications.
    """

    text = str(source or "")
    lines = text.splitlines()
    start = -1
    bullet = re.compile(
        r"^\s*-\s*(?:\*\*|__)?([A-Za-z_][A-Za-z0-9_]*)(?:\*\*|__)?\s*:\s*(.*)$"
    )
    for index, line in enumerate(lines):
        match = bullet.match(line)
        if match is not None and match.group(1) == concern:
            start = index
            break
    if start < 0:
        return ""

    end = len(lines)
    for index in range(start + 1, len(lines)):
        match = bullet.match(lines[index])
        if match is not None and match.group(1) in _STATE_CONCERNS:
            end = index
            break
        if parse_markdown_heading(lines[index]) is not None:
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def _concern_has_explicit_payload(source: str, concern: str) -> bool:
    """Detect authored state values across canonical and free-Markdown shapes.

    The free planner writes readable Markdown, so a concern can carry its values in
    nested bullets instead of on the concern header or as field=value records. Treat
    those child bullets as authored payload; otherwise production can silently classify
    real state requirements as EMPTY and the host-only state compiler rejects the job.
    """

    block = _concern_source(source, concern)
    if not block:
        return False
    lines = block.splitlines()
    first = lines[0]
    match = re.match(
        rf"^\s*-\s*(?:\*\*|__)?{re.escape(concern)}(?:\*\*|__)?\s*:\s*(.*)$",
        first,
    )
    if match is None:
        return False
    body = match.group(1).strip()
    fields = tuple(DETAIL_RECORDS["state_model"][concern].split())
    if not fields:
        return bool(body or any(line.strip() for line in lines[1:]))

    field_pattern = r"\s+".join(re.escape(field) for field in fields)
    normalized_body = " ".join(body.split())
    normalized_descriptor = " ".join(fields)

    # A bare field descriptor is only the planner template. Inline canonical rows
    # count as explicit only when they carry the declared field layout; arbitrary
    # prose shorthand still goes through the bounded extractor below.
    if normalized_body and normalized_body != normalized_descriptor:
        descriptor = re.match(rf"^{field_pattern}\s*:\s*(.+)$", body)
        if descriptor is not None:
            return bool(descriptor.group(1).strip())

    # Structured projections use field=value rows. Free Markdown plans instead use
    # nested list items such as "- **variables**:" followed by "- `credits`: ...".
    if any(
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(field)}\s*=", block)
        for field in fields
    ):
        return True

    return any(
        re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)\S", line.strip())
        for line in lines[1:]
        if line.strip()
    )

def _nested_concern_entries(source: str, concern: str) -> list[tuple[str, str]]:
    """Read nested Markdown key/value bullets owned by one state concern."""

    block = _concern_source(source, concern)
    if not block:
        return []
    entries: list[tuple[str, str]] = []
    pattern = re.compile(
        r"^\s*[-*+]\s+(?:`([^`]+)`|\*\*([^*]+)\*\*|__([^_]+)__|"
        r"([A-Za-z_][A-Za-z0-9_]*))\s*:\s*(.*?)\s*$"
    )
    for line in block.splitlines()[1:]:
        match = pattern.match(line)
        if match is None:
            continue
        key = next((part for part in match.groups()[:4] if part), "").strip()
        value = str(match.group(5) or "").strip()
        if key and value:
            entries.append((key, value))
    return entries


def _markdown_code_values(value: str) -> list[str]:
    return [
        item.strip()
        for item in re.findall(r"`([^`]*)`", str(value or ""))
        if item.strip()
    ]


def _plain_markdown_value(value: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", str(value or ""))
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    return " ".join(text.split()).strip()


def _labeled_code_value(value: str, *labels: str) -> str:
    label_pattern = "|".join(re.escape(label) for label in labels)
    match = re.search(
        rf"(?i)(?:{label_pattern})\s*[:=]?\s*`([^`]*)`",
        str(value or ""),
    )
    return str(match.group(1) or "").strip() if match is not None else ""


def _assignment_fragments(value: str) -> list[str]:
    result: list[str] = []
    for item in _markdown_code_values(value):
        if re.search(r"(?<![=!<>])(?:\+=|-=|\*=|/=|=(?!=))", item):
            result.append(item)
    return result


def _expression_fragments(value: str) -> list[str]:
    result: list[str] = []
    for item in _markdown_code_values(value):
        if not re.search(r"(?:==|!=|>=|<=|>|<|&&|\|\|)", item):
            continue
        # Human prose frequently writes percentages as 100%, while the state DSL
        # interprets '%' as modulo. Preserve the numeric threshold, not that typo.
        item = re.sub(r"(?<=\d)%\b|(?<=\d)%$", "", item)
        result.append(item)
    return result


def _deterministic_variable_records(source: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for raw_name, body in _nested_concern_entries(source, "variables"):
        name = _stable_identifier(raw_name, fallback=f"state_{len(records) + 1}")
        codes = _markdown_code_values(body)

        owner_match = re.search(r"(?i)^(.*?)(?:\s+소유|\s+owner\b)", body)
        owner_codes = (
            _markdown_code_values(owner_match.group(1))
            if owner_match is not None
            else []
        )
        owner = "|".join(owner_codes) if owner_codes else (codes[0] if codes else "system")

        after_owner = body[owner_match.end():] if owner_match is not None else body
        type_codes = _markdown_code_values(after_owner)
        type_name = type_codes[0] if type_codes else (
            codes[1] if len(codes) > 1 else "object"
        )

        default = _labeled_code_value(body, "기본값", "default") or "null"
        if type_name.casefold() in {"byte", "short", "int", "integer", "long"}:
            default = re.sub(r"(?i)^([-+]?\d+)l$", r"\1", default)

        domain = _labeled_code_value(body, "범위", "domain") or "any"
        unit = _labeled_code_value(body, "단위", "unit") or "value"
        records.append({
            "name": name,
            "owner": owner or "system",
            "type": type_name or "object",
            "unit": unit,
            "default": default,
            "domain": domain,
        })
    return records


def _deterministic_concern_records(
    source: str,
    concern: str,
    *,
    declared_names: Sequence[str],
) -> list[dict[str, str]]:
    """Lower planner Markdown without depending on a model response.

    Only shapes that can be mapped without inventing gameplay semantics are emitted.
    Concerns that cannot be represented safely remain available to the bounded model
    extractor below.
    """

    del declared_names
    entries = _nested_concern_entries(source, concern)
    if not entries:
        return []
    if concern == "variables":
        return _deterministic_variable_records(source)

    values = {key: value for key, value in entries}
    if concern == "transitions":
        required = ("from_state", "trigger", "guard", "mutation", "to_state")
        if not all(name in values for name in required):
            return []
        guards = _expression_fragments(values["guard"])
        mutations = _assignment_fragments(values["mutation"])
        if not guards:
            return []
        return [{
            "from_state": _plain_markdown_value(values["from_state"]),
            "trigger": _plain_markdown_value(values["trigger"]),
            "guard": " && ".join(f"({item})" for item in guards),
            "mutation": "; ".join(mutations),
            "to_state": _plain_markdown_value(values["to_state"]),
        }]

    if concern == "invariants":
        if "condition" in values:
            conditions = _expression_fragments(values["condition"])
            enforcement = values.get("enforcement", values["condition"])
        elif "condition_enforcement" in values:
            conditions = _expression_fragments(values["condition_enforcement"])
            enforcement = values["condition_enforcement"]
        else:
            return []
        if not conditions:
            return []
        return [{
            "condition": " && ".join(f"({item})" for item in conditions),
            "enforcement": _plain_markdown_value(enforcement),
        }]

    if concern == "initialization":
        if not all(name in values for name in ("owner", "trigger", "initial_state")):
            return []
        mutations = _assignment_fragments(values["initial_state"])
        if not mutations:
            # Variable defaults already own pure initial-value initialization. Do not
            # manufacture assignments from prose that does not name state variables.
            return []
        return [{
            "owner": _plain_markdown_value(values["owner"]),
            "trigger": _plain_markdown_value(values["trigger"]),
            "initial_state": "; ".join(mutations),
        }]

    if concern == "updates":
        if not all(name in values for name in ("trigger", "mutation", "owner")):
            return []
        mutations = _assignment_fragments(values["mutation"])
        if not mutations:
            return []
        return [{
            "trigger": _plain_markdown_value(values["trigger"]),
            "mutation": "; ".join(mutations),
            "owner": _plain_markdown_value(values["owner"]),
        }]

    if concern == "cleanup":
        if not all(name in values for name in ("event", "action", "retained_state")):
            return []
        mutations = _assignment_fragments(values["action"])
        if not mutations:
            return []
        return [{
            "event": _plain_markdown_value(values["event"]),
            "action": "; ".join(mutations),
            "retained_state": _plain_markdown_value(values["retained_state"]),
        }]

    if concern == "concurrency":
        required = ("entry_path", "ownership", "reentrancy_rule")
        if not all(name in values for name in required):
            return []
        return [{
            name: _plain_markdown_value(values[name])
            for name in required
        }]

    return []


def _json_key(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decode_scalar_text(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        try:
            return str(json.loads(text))
        except Exception:
            return text[1:-1]
    if len(text) >= 2 and text[0] == text[-1] == "'":
        return text[1:-1]
    return text


def _extract_known_field(record_text: str, field: str) -> str:
    pattern = (
        r'(?is)(?:^|[,;\n{])\s*"?'
        + re.escape(field)
        + r'"?\s*[:=]\s*'
        + r'("(?:\\.|[^"\\])*"|[^,;\n}]+)'
    )
    match = re.search(pattern, record_text)
    return _decode_scalar_text(match.group(1)) if match is not None else ""


def _parse_semantic_page(raw: str, *, fields: Sequence[str]) -> tuple[list[dict[str, str]], bool]:
    """Parse meaning without requiring the small model to produce valid JSON."""
    text = str(raw or "").strip()
    text = re.sub(r'^```(?:text|json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    status = re.search(
        r'(?im)(?:complete"?\s*[:=]\s*(true|false|yes|no|1|0)|^\s*STATUS\s*[:=]?\s*(COMPLETE|MORE)\s*$)',
        text,
    )
    token = next((part for part in status.groups() if part), '') if status else ''
    complete = token.casefold() in {'true', 'yes', '1', 'complete'}

    blocks = re.findall(
        r'(?is)(?:^|\n)\s*RECORD\s*(.*?)(?=(?:\n\s*(?:END|RECORD|STATUS)\b)|\Z)',
        text,
    )
    if not blocks:
        blocks = [
            match.group(1)
            for match in re.finditer(r'\{([^{}]*)\}', text, flags=re.DOTALL)
            if any(re.search(r'"?' + re.escape(field) + r'"?\s*[:=]', match.group(1), re.I) for field in fields)
        ]
    if not blocks:
        blocks = [text]

    records: list[dict[str, str]] = []
    for block in blocks:
        record = {field: _extract_known_field(block, field) for field in fields}
        if not any(record.values()):
            for line in block.splitlines():
                match = re.match(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\t|[:=])\s*(.*?)\s*$', line)
                if match is not None and match.group(1) in fields:
                    record[match.group(1)] = _decode_scalar_text(match.group(2))
        if any(record.values()):
            records.append(record)
    return records, complete

def _generate_concern_records(
    router: Any,
    *,
    source: str,
    concern: str,
    declared_names: Sequence[str],
) -> list[dict[str, str]]:
    """Extract one bounded concern snapshot with exactly one model call.

    Production must never depend on the model deciding when pagination is complete.
    The approved plan is finite and each state concern is lowered once; the host owns
    the record cap, de-duplication, parsing, normalization, and downstream compilation.
    """

    deterministic = _deterministic_concern_records(
        source,
        concern,
        declared_names=declared_names,
    )
    if deterministic:
        return deterministic

    fields = tuple(DETAIL_RECORDS["state_model"][concern].split())
    payload = {
        "stage": "production_semantic_lowering",
        "section": "state_model",
        "concern": concern,
        "fields": fields,
        "approved_state_model": _concern_source(source, concern) or source,
        "contains_authored_values": _concern_has_explicit_payload(source, concern),
        "declared_state_variables": list(declared_names),
        "record_limit": _MAX_CONCERN_RECORDS,
    }
    system = (
        "You are a bounded production semantic extractor. Read only the approved "
        "state_model and extract every concrete record for the selected concern in ONE "
        "response. Never redesign or extend the approved plan. Never invent extra "
        "invariants, transitions, variables, lifecycle rules, or repeated variants. "
        "Do not emit Java, imports, classes, methods, Fabric/Minecraft APIs, JSON, "
        "JSON Schema, or prose. For guard/condition expressions use identifiers, "
        "literals, parentheses and operators ! + - * / % == != >= <= > < && ||. "
        "For mutations/actions/initial_state use semicolon-separated assignments with "
        "=, +=, -=, *=, /=. Mutation targets should use declared_state_variables. "
        "String/state literals must be quoted. Never use Java enum/member syntax such "
        "as ShipStatus.COMPLETE; write a quoted literal such as \"COMPLETE\". "
        "The host supplies only the selected concern when it can isolate it. "
        "Approved authored values may be written as nested Markdown bullets rather than "
        "field=value rows. Map those bullets semantically into the supplied fields; a "
        "noncanonical Markdown shape is never a reason to return EMPTY. "
        "The user payload contains contains_authored_values, computed deterministically by "
        "the host from the authored concern block. When it is true, STATUS=EMPTY is invalid "
        "and at least one RECORD must be emitted. "
        "If this concern truly has no concrete requirement in the approved plan, output only "
        "STATUS=EMPTY. Otherwise output at most the host record_limit records. "
        "Never paginate and never output STATUS=MORE."
    )
    protocol = (
        system
        + " Output format:\n"
        + "STATUS=DONE\nRECORD\n"
        + "\n".join(field + "=<value>" for field in fields)
        + "\nEND\n"
        + "Repeat RECORD/END only for additional explicit records."
    )
    raw = router.generate_text(
        "planner",
        (
            {"role": "system", "content": protocol},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ),
        response_format="text",
        response_schema=None,
        enable_tools=False,
        output_token_ceiling=2048,
        force_non_thinking=True,
    )
    rows, terminal = _parse_semantic_page(raw, fields=fields)
    accepted: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows[:_MAX_CONCERN_RECORDS]:
        if not isinstance(row, Mapping):
            continue
        normalized = {
            field: str(row.get(field) or "").strip()
            for field in fields
        }
        if not any(normalized.values()):
            continue
        key = _json_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        accepted.append(normalized)

    if accepted:
        return accepted

    explicit_payload = _concern_has_explicit_payload(source, concern)
    if explicit_payload:
        if _nested_concern_entries(source, concern):
            return []
        raise ValueError(
            "PRODUCTION_STATE_LOWERING_FALSE_EMPTY: "
            f"state_model.{concern} contains authored values but extraction returned no records"
        )
    if terminal:
        return []
    if re.search(r"(?im)^\s*STATUS\s*[:=]?\s*(?:EMPTY|DONE|COMPLETE)\s*$", raw):
        return []
    if not str(raw or "").strip():
        return []
    raise ValueError(
        f"PRODUCTION_STATE_LOWERING_UNREADABLE: state_model.{concern}"
    )


def _stable_identifier(value: str, *, fallback: str) -> str:
    raw = str(value or "").strip()
    if _IDENTIFIER.fullmatch(raw):
        return raw
    text = re.sub(r"[^0-9A-Za-z_]+", "_", raw)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = "_" + text
    return text


def _transform_unquoted(text: str, transform) -> str:
    """Apply lexical normalization only outside quoted string literals."""

    source = str(text or "")
    output: list[str] = []
    buffer: list[str] = []
    quote = ""

    def flush_unquoted() -> None:
        if buffer:
            output.append(transform("".join(buffer)))
            buffer.clear()

    index = 0
    while index < len(source):
        char = source[index]
        if quote:
            output.append(char)
            if char == "\\" and index + 1 < len(source):
                index += 1
                output.append(source[index])
            elif char == quote:
                quote = ""
            index += 1
            continue
        if char in {'"', "'"}:
            flush_unquoted()
            quote = char
            output.append(char)
            index += 1
            continue
        buffer.append(char)
        index += 1
    flush_unquoted()
    return "".join(output)


def _replace_aliases(text: str, aliases: Mapping[str, str]) -> str:
    def replace_piece(piece: str) -> str:
        result = piece
        for old, new in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
            if not old or old == new:
                continue
            result = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])",
                new,
                result,
                flags=re.IGNORECASE,
            )
        return result

    return _transform_unquoted(text, replace_piece)


def _normalize_logic_tokens(text: str) -> str:
    def normalize_piece(piece: str) -> str:
        result = piece
        result = result.replace("&&&", "&&").replace("|||", "||")
        result = re.sub(r"\bAND\b", "&&", result, flags=re.IGNORECASE)
        result = re.sub(r"\bOR\b", "||", result, flags=re.IGNORECASE)
        result = re.sub(r"\bNOT\b\s+", "!", result, flags=re.IGNORECASE)
        result = re.sub(
            r"\b(?:greater\s+than\s+or\s+equal\s+to|at\s+least)\b",
            ">=",
            result,
            flags=re.IGNORECASE,
        )
        result = re.sub(
            r"\b(?:less\s+than\s+or\s+equal\s+to|at\s+most)\b",
            "<=",
            result,
            flags=re.IGNORECASE,
        )
        result = re.sub(r"\bgreater\s+than\b", ">", result, flags=re.IGNORECASE)
        result = re.sub(r"\bless\s+than\b", "<", result, flags=re.IGNORECASE)
        result = re.sub(r"\bnot\s+equal\s+to\b", "!=", result, flags=re.IGNORECASE)
        result = re.sub(r"\bequal\s+to\b", "==", result, flags=re.IGNORECASE)
        result = re.sub(r"\bis\s+not\b", "!=", result, flags=re.IGNORECASE)
        result = re.sub(r"\bis\b", "==", result, flags=re.IGNORECASE)
        result = result.replace("<>", "!=")
        result = re.sub(r"(?<![!<>=])=(?!=)", "==", result)
        result = re.sub(
            r"\b[A-Za-z_][A-Za-z0-9_]*\.([A-Z][A-Z0-9_]*)\b",
            lambda match: json.dumps(match.group(1)),
            result,
        )
        result = re.sub(r"\b(?:this|context)\.([A-Za-z_][A-Za-z0-9_]*)\b", r"\1", result)
        result = re.sub(r"\bTRUE\b", "true", result, flags=re.IGNORECASE)
        result = re.sub(r"\bFALSE\b", "false", result, flags=re.IGNORECASE)
        result = re.sub(r"\bNULL\b", "null", result, flags=re.IGNORECASE)
        return result

    result = _transform_unquoted(str(text or "").strip(), normalize_piece)
    return result.rstrip(";").strip()


def _string_like(record: Mapping[str, str]) -> bool:
    type_name = str(record.get("type") or "").casefold()
    domain = str(record.get("domain") or "").casefold()
    return (
        any(token in type_name for token in ("string", "text", "enum", "status", "mode"))
        or "enum" in domain
        or "|" in domain
    )


def _split_expression_segments(text: str) -> list[str]:
    operators = ("->", "||", "&&", "==", "!=", ">=", "<=", "(", ")", "!", "+", "-", "*", "/", "%", ">", "<")
    source = str(text or "")
    segments: list[str] = []
    buffer: list[str] = []
    quote = ""

    def flush() -> None:
        if buffer:
            segments.append("".join(buffer))
            buffer.clear()

    index = 0
    while index < len(source):
        char = source[index]
        if quote:
            buffer.append(char)
            if char == "\\" and index + 1 < len(source):
                index += 1
                buffer.append(source[index])
            elif char == quote:
                quote = ""
            index += 1
            continue
        if char in {'"', "'"}:
            quote = char
            buffer.append(char)
            index += 1
            continue

        matched = next(
            (operator for operator in operators if source.startswith(operator, index)),
            "",
        )
        if matched:
            flush()
            segments.append(matched)
            index += len(matched)
            continue
        buffer.append(char)
        index += 1
    flush()
    return segments


def _simple_expression_atom(value: str) -> bool:
    atom = str(value or "").strip()
    if not atom:
        return True
    if atom[0:1] in {'"', "'"} and atom[-1:] == atom[0]:
        return True
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", atom):
        return True
    return atom.casefold() in {"true", "false", "null"}


def _identifier_signature(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value or "").casefold())


def _canonicalize_expression_operands(
    expression: str,
    *,
    variables: Mapping[str, Mapping[str, str]],
) -> str:
    text = re.sub(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\)",
        r"\1",
        str(expression or ""),
    )
    segments = _split_expression_segments(text)
    comparison_ops = {"==", "!=", ">=", "<=", ">", "<"}
    variable_by_signature = {
        _identifier_signature(name): name
        for name in variables
        if _identifier_signature(name)
    }

    def neighbor_atom(index: int, direction: int) -> str:
        pos = index + direction
        while 0 <= pos < len(segments):
            value = segments[pos].strip()
            if value:
                return value
            pos += direction
        return ""

    for index, raw in enumerate(list(segments)):
        atom = raw.strip()
        if not atom or atom in {"->", "||", "&&", "==", "!=", ">=", "<=", "(", ")", "!", "+", "-", "*", "/", "%", ">", "<"}:
            continue
        if _simple_expression_atom(atom):
            segments[index] = atom
            continue

        declared_name = variable_by_signature.get(_identifier_signature(atom))
        if declared_name:
            segments[index] = declared_name
            continue

        previous = neighbor_atom(index, -1)
        following = neighbor_atom(index, 1)
        previous_op = previous if previous in comparison_ops else ""
        following_op = following if following in comparison_ops else ""

        if previous_op in {"==", "!="}:
            other = neighbor_atom(index - 1, -1)
            if other in variables and _string_like(variables[other]) and atom not in variables:
                segments[index] = json.dumps(atom)
                continue
        if following_op in {"==", "!="}:
            other = neighbor_atom(index + 1, 1)
            if other in variables and _string_like(variables[other]) and atom not in variables:
                segments[index] = json.dumps(atom)
                continue

        if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", atom):
            segments[index] = atom
            continue
        segments[index] = _stable_identifier(atom, fallback="state_value")

    return " ".join(part.strip() for part in segments if part.strip())


def _validated_expression_or_fallback(
    text: str,
    *,
    fallback: str,
    original: str,
) -> str:
    try:
        validate_state_expression(text)
        return text
    except ValueError:
        from .root_cause_trace import emit_root_cause

        emit_root_cause(
            "production_state_expression_degraded",
            stage="production",
            operation="state_expression_canonicalization",
            gate="host_expression_parser",
            result="PASS",
            details={
                "original": str(original or "")[:512],
                "canonical": str(text or "")[:512],
                "fallback": fallback,
            },
        )
        return fallback


def _normalize_expression(
    value: str,
    *,
    aliases: Mapping[str, str],
    variables: Mapping[str, Mapping[str, str]],
    fallback: str = "true",
) -> str:
    original = str(value or "")
    text = _replace_aliases(original, aliases)
    text = _normalize_logic_tokens(text)
    text = _canonicalize_expression_operands(text, variables=variables)
    lowered = text.casefold()
    if not text or lowered in {"none", "n/a", "na", "always", "no guard", "no condition"}:
        return fallback
    return _validated_expression_or_fallback(
        text,
        fallback=fallback,
        original=original,
    )


def _split_unquoted_statements(text: str) -> list[str]:
    source = str(text or "")
    rows: list[str] = []
    buffer: list[str] = []
    quote = ""
    index = 0
    while index < len(source):
        char = source[index]
        if quote:
            buffer.append(char)
            if char == "\\" and index + 1 < len(source):
                index += 1
                buffer.append(source[index])
            elif char == quote:
                quote = ""
            index += 1
            continue
        if char in {'"', "'"}:
            quote = char
            buffer.append(char)
        elif char == ";":
            rows.append("".join(buffer))
            buffer.clear()
        else:
            buffer.append(char)
        index += 1
    rows.append("".join(buffer))
    return rows


def _resolve_declared_state_name(
    name: str,
    *,
    aliases: Mapping[str, str],
    variables: Mapping[str, Mapping[str, str]],
) -> str | None:
    raw = str(name or "").strip()
    direct = aliases.get(raw)
    if direct in variables:
        return direct
    signature = _identifier_signature(raw)
    for declared in variables:
        if _identifier_signature(declared) == signature:
            return declared
    return None


def _normalize_mutation_syntax(text: str) -> str:
    def normalize_piece(piece: str) -> str:
        result = re.sub(
            r"\b(?:increase|increment)\s+([A-Za-z_][A-Za-z0-9_]*)\s+by\s+(.+)$",
            r"\1 += \2",
            piece,
            flags=re.IGNORECASE,
        )
        result = re.sub(
            r"\b(?:decrease|decrement)\s+([A-Za-z_][A-Za-z0-9_]*)\s+by\s+(.+)$",
            r"\1 -= \2",
            result,
            flags=re.IGNORECASE,
        )
        result = re.sub(
            r"\bset\s+([A-Za-z_][A-Za-z0-9_]*)\s+to\s+(.+)$",
            r"\1 = \2",
            result,
            flags=re.IGNORECASE,
        )
        result = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\+\+\b", r"\1 += 1", result)
        result = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)--\b", r"\1 -= 1", result)
        return result

    return _transform_unquoted(text, normalize_piece)


def _normalize_mutation(
    value: str,
    *,
    aliases: Mapping[str, str],
    variables: Mapping[str, Mapping[str, str]],
) -> str | None:
    """Return canonical state assignments, empty program, or None for out-of-scope action.

    Only assignments whose targets already exist in the authoritative variable table are
    state mutations. Opaque lifecycle/persistence/network/UI actions are not translated
    into fake state code and are excluded from state_model records.
    """

    text = _replace_aliases(value, aliases).strip()
    lowered = text.casefold()
    if not text or lowered in {"none", "n/a", "na", "no change"}:
        return ""

    text = _normalize_mutation_syntax(text)

    normalized_rows: list[str] = []
    saw_out_of_scope = False
    for raw in _split_unquoted_statements(text):
        row = raw.strip()
        if not row:
            continue
        match = _ASSIGNMENT.fullmatch(row)
        if match is None:
            saw_out_of_scope = True
            continue
        name, operator, rhs = match.groups()
        declared = _resolve_declared_state_name(
            name,
            aliases=aliases,
            variables=variables,
        )
        if declared is None:
            saw_out_of_scope = True
            continue
        rhs = _normalize_expression(
            rhs,
            aliases=aliases,
            variables=variables,
            fallback="",
        )
        if not rhs:
            saw_out_of_scope = True
            continue
        normalized_rows.append(f"{declared} {operator} {rhs}")

    if normalized_rows:
        return "; ".join(normalized_rows)
    if saw_out_of_scope:
        return None
    return ""
def _normalize_records(
    records: Mapping[str, Sequence[Mapping[str, str]]],
) -> dict[str, list[dict[str, str]]]:
    variables: dict[str, dict[str, str]] = {}
    aliases: dict[str, str] = {}

    for index, raw in enumerate(records.get("variables", ())):
        name_raw = str(raw.get("name") or raw.get("unit") or "").strip()
        name = _stable_identifier(name_raw, fallback=f"state_{index + 1}")
        aliases[name_raw] = name
        aliases[name_raw.replace(" ", "_")] = name
        if name in variables:
            continue
        variables[name] = {
            "name": name,
            "owner": str(raw.get("owner") or "system").strip() or "system",
            "type": str(raw.get("type") or "object").strip() or "object",
            "unit": str(raw.get("unit") or "value").strip() or "value",
            "default": str(raw.get("default") or "null").strip() or "null",
            "domain": str(raw.get("domain") or "any").strip() or "any",
        }

    result: dict[str, list[dict[str, str]]] = {
        concern: [] for concern in _STATE_CONCERNS
    }

    for raw in records.get("transitions", ()):
        guard = _normalize_expression(
            str(raw.get("guard") or ""),
            aliases=aliases,
            variables=variables,
            fallback="false",
        )
        mutation = _normalize_mutation(
            str(raw.get("mutation") or ""),
            aliases=aliases,
            variables=variables,
        )
        if mutation is None:
            mutation = ""
        result["transitions"].append({
            "from_state": str(raw.get("from_state") or "any").strip() or "any",
            "trigger": str(raw.get("trigger") or "event").strip() or "event",
            "guard": guard,
            "mutation": mutation,
            "to_state": str(raw.get("to_state") or raw.get("from_state") or "any").strip() or "any",
        })

    for raw in records.get("invariants", ()):
        result["invariants"].append({
            "condition": _normalize_expression(
                str(raw.get("condition") or ""),
                aliases=aliases,
                variables=variables,
                fallback="false",
            ),
            "enforcement": str(raw.get("enforcement") or "host runtime check").strip()
            or "host runtime check",
        })

    for concern, field, defaults in (
        ("initialization", "initial_state", ("system", "initialize")),
        ("updates", "mutation", ("update", "server")),
        ("cleanup", "action", ("shutdown", "declared state")),
    ):
        for raw in records.get(concern, ()):
            mutation = _normalize_mutation(
                str(raw.get(field) or ""),
                aliases=aliases,
                variables=variables,
            )
            if mutation is None or mutation == "":
                continue
            if concern == "initialization":
                result[concern].append({
                    "owner": str(raw.get("owner") or defaults[0]).strip() or defaults[0],
                    "trigger": str(raw.get("trigger") or defaults[1]).strip() or defaults[1],
                    "initial_state": mutation,
                })
            elif concern == "updates":
                result[concern].append({
                    "trigger": str(raw.get("trigger") or defaults[0]).strip() or defaults[0],
                    "mutation": mutation,
                    "owner": str(raw.get("owner") or defaults[1]).strip() or defaults[1],
                })
            else:
                result[concern].append({
                    "event": str(raw.get("event") or defaults[0]).strip() or defaults[0],
                    "action": mutation,
                    "retained_state": str(raw.get("retained_state") or defaults[1]).strip()
                    or defaults[1],
                })

    for raw in records.get("concurrency", ()):
        result["concurrency"].append({
            "entry_path": str(raw.get("entry_path") or "state transition entry").strip()
            or "state transition entry",
            "ownership": str(raw.get("ownership") or "server").strip() or "server",
            "reentrancy_rule": str(raw.get("reentrancy_rule") or "serialized").strip()
            or "serialized",
        })

    result["variables"] = list(variables.values())
    return result


def normalize_structured_state_section(
    section: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize any structured state authority through the production SSOT.

    This is the only production boundary for state-model semantics, regardless of
    whether records came from the dedicated state compiler or from a structured
    authored design. External subsystem actions are excluded, expressions are
    canonicalized, and the result is validated by the host DSL parser before any
    Java lowering occurs.
    """
    raw_specification = section.get("specification")
    specification = (
        raw_specification
        if isinstance(raw_specification, Mapping)
        else section
    )
    raw: dict[str, list[dict[str, str]]] = {
        concern: [] for concern in _STATE_CONCERNS
    }
    for concern in _STATE_CONCERNS:
        rows = specification.get(concern)
        if not isinstance(rows, Sequence) or isinstance(
            rows, (str, bytes, bytearray)
        ):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            raw[concern].append(
                {str(key): str(value) for key, value in row.items()}
            )

    normalized = _normalize_records(raw)
    result_specification: dict[str, Any] = {
        concern: normalized.get(concern, [])
        for concern in _STATE_CONCERNS
    }
    explicit_inapplicable = specification.get("inapplicable_concerns")
    inapplicable_by_name: dict[str, dict[str, str]] = {}
    if isinstance(explicit_inapplicable, Sequence) and not isinstance(
        explicit_inapplicable, (str, bytes, bytearray)
    ):
        for row in explicit_inapplicable:
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("concern") or "").strip()
            if name:
                inapplicable_by_name[name] = {
                    "concern": name,
                    "reason": str(row.get("reason") or "Not required.").strip()
                    or "Not required.",
                }
    for concern in _STATE_CONCERNS:
        if not result_specification[concern]:
            inapplicable_by_name.setdefault(
                concern,
                {
                    "concern": concern,
                    "reason": (
                        "No canonical state mutation/condition remains after "
                        "production semantic normalization."
                    ),
                },
            )
    result_specification["inapplicable_concerns"] = list(
        inapplicable_by_name.values()
    )
    normalized_section = {
        "specification": result_specification,
        "constraint_evidence_refs": list(
            section.get("constraint_evidence_refs") or []
        ),
    }
    validate_structured_state_section(normalized_section)
    return normalized_section


def compile_production_state_section(router: Any, plan: AuthoredPlan) -> dict[str, Any]:
    """Return canonical structured state without invoking a production model.

    The small model may translate authored intent into the bounded state worksheet
    during planning. Production accepts only that canonical worksheet and performs
    deterministic normalization/validation.
    """

    _ = router  # Compatibility only; production state compilation is model-free.
    structured = plan.structured_sections
    if not isinstance(structured, Mapping):
        structured = {}
    raw_state = structured.get("state_model")
    if isinstance(raw_state, Mapping):
        return normalize_structured_state_section(raw_state)

    source = _section_text(plan.text, "state_model")
    if source:
        raise ValueError(
            "PRODUCTION_STATE_STRUCTURED_AUTHORITY_REQUIRED: state_model exists in "
            "the authored design but no canonical structured state worksheet was supplied"
        )
    return {}


__all__ = [
    "compile_production_state_section",
    "normalize_structured_state_section",
]
