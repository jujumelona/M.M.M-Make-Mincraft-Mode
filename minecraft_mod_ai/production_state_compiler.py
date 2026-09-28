from __future__ import annotations

"""Production-only semantic compiler for authored state models.

Planning remains free Markdown. At production entry this module lowers only the
approved state_model section into bounded records consumed by the deterministic
host state compiler. The model never emits Java, Java symbols, imports, Fabric
APIs, or host-private AST objects on this path.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
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

    fields = tuple(DETAIL_RECORDS["state_model"][concern].split())
    payload = {
        "stage": "production_semantic_lowering",
        "section": "state_model",
        "concern": concern,
        "fields": fields,
        "approved_state_model": source,
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
        "If this concern has no concrete requirement in the approved plan, output only "
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


def _replace_aliases(text: str, aliases: Mapping[str, str]) -> str:
    result = str(text or "")
    for old, new in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if not old or old == new:
            continue
        result = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])",
            new,
            result,
        )
    return result


def _normalize_logic_tokens(text: str) -> str:
    result = str(text or "").strip().rstrip(";")
    result = result.replace("&&&", "&&").replace("|||", "||")
    result = re.sub(r"\bAND\b", "&&", result, flags=re.IGNORECASE)
    result = re.sub(r"\bOR\b", "||", result, flags=re.IGNORECASE)
    result = re.sub(r"\bNOT\b\s+", "!", result, flags=re.IGNORECASE)
    result = re.sub(r"\b(?:greater\s+than\s+or\s+equal\s+to|at\s+least)\b", ">=", result, flags=re.IGNORECASE)
    result = re.sub(r"\b(?:less\s+than\s+or\s+equal\s+to|at\s+most)\b", "<=", result, flags=re.IGNORECASE)
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
    return " ".join(result.split())


def _string_like(record: Mapping[str, str]) -> bool:
    type_name = str(record.get("type") or "").casefold()
    domain = str(record.get("domain") or "").casefold()
    return (
        any(token in type_name for token in ("string", "text", "enum", "status", "mode"))
        or "enum" in domain
        or "|" in domain
    )


def _split_expression_segments(text: str) -> list[str]:
    operators = ("||", "&&", "==", "!=", ">=", "<=", "(", ")", "!", "+", "-", "*", "/", "%", ">", "<")
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
        if not atom or atom in {"||", "&&", "==", "!=", ">=", "<=", "(", ")", "!", "+", "-", "*", "/", "%", ">", "<"}:
            continue
        if _simple_expression_atom(atom):
            segments[index] = atom
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

    rebuilt = " ".join(part.strip() for part in segments if part.strip())

    balanced: list[str] = []
    depth = 0
    for part in _split_expression_segments(rebuilt):
        token = part.strip()
        if token == "(":
            depth += 1
            balanced.append(token)
        elif token == ")":
            if depth:
                depth -= 1
                balanced.append(token)
        elif token:
            balanced.append(token)
    if depth:
        balanced.extend(")" for _ in range(depth))
    return " ".join(balanced)


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


def _infer_variable(name: str, rhs: str) -> dict[str, str]:
    value = str(rhs or "").strip()
    lowered = value.casefold()
    if value.startswith(('"', "'")):
        type_name, default, domain = "string", value.strip("'\"") or "unset", "text"
    elif lowered in {"true", "false"}:
        type_name, default, domain = "boolean", "false", "boolean"
    elif re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value):
        type_name, default, domain = "double", "0", "number"
    else:
        type_name, default, domain = "object", "null", "any"
    return {
        "name": name,
        "owner": "system",
        "type": type_name,
        "unit": "value",
        "default": default,
        "domain": domain,
    }


def _normalize_mutation(
    value: str,
    *,
    aliases: dict[str, str],
    variables: dict[str, dict[str, str]],
    allow_opaque_noop: bool = False,
) -> str:
    text = _replace_aliases(value, aliases).strip()
    lowered = text.casefold()
    if not text or lowered in {"none", "n/a", "na", "noop", "no-op", "no change"}:
        return "noop"

    text = re.sub(
        r"\b(?:increase|increment)\s+([A-Za-z_][A-Za-z0-9_]*)\s+by\s+(.+)$",
        r"\1 += \2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:decrease|decrement)\s+([A-Za-z_][A-Za-z0-9_]*)\s+by\s+(.+)$",
        r"\1 -= \2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bset\s+([A-Za-z_][A-Za-z0-9_]*)\s+to\s+(.+)$",
        r"\1 = \2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\+\+\b", r"\1 += 1", text)
    text = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)--\b", r"\1 -= 1", text)

    normalized_rows: list[str] = []
    opaque_rows: list[str] = []
    for raw in text.split(";"):
        row = raw.strip()
        if not row:
            continue
        match = _ASSIGNMENT.fullmatch(row)
        if match is None:
            if allow_opaque_noop:
                opaque_rows.append(row)
                continue
            raise ValueError(
                "PRODUCTION_STATE_MUTATION_UNSUPPORTED: " + repr(row)
            )
        name, operator, rhs = match.groups()
        stable = aliases.get(name, _stable_identifier(name, fallback="state_value"))
        aliases.setdefault(name, stable)
        if stable not in variables:
            variables[stable] = _infer_variable(stable, rhs)
        rhs = _normalize_expression(
            rhs,
            aliases=aliases,
            variables=variables,
            fallback=stable,
        )
        normalized_rows.append(f"{stable} {operator} {rhs}")
    if normalized_rows:
        return "; ".join(normalized_rows)
    if opaque_rows or allow_opaque_noop:
        return "noop"
    return "noop"


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
            allow_opaque_noop=True,
        )
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
                allow_opaque_noop=True,
            )
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


def compile_production_state_section(router: Any, plan: AuthoredPlan) -> dict[str, Any]:
    source = _section_text(plan.text, "state_model")
    if not source:
        return {}

    raw: dict[str, list[dict[str, str]]] = {}
    declared_names: list[str] = []
    for concern in _STATE_CONCERNS:
        rows = _generate_concern_records(
            router,
            source=source,
            concern=concern,
            declared_names=declared_names,
        )
        raw[concern] = rows
        if concern == "variables":
            declared_names = [
                _stable_identifier(str(row.get("name") or row.get("unit") or ""), fallback=f"state_{index + 1}")
                for index, row in enumerate(rows)
            ]

    normalized = _normalize_records(raw)
    specification: dict[str, Any] = {
        concern: normalized.get(concern, [])
        for concern in _STATE_CONCERNS
    }
    specification["inapplicable_concerns"] = [
        {
            "concern": concern,
            "reason": "No concrete requirement for this concern in the approved state_model.",
        }
        for concern in _STATE_CONCERNS
        if not specification[concern]
    ]
    section = {
        "specification": specification,
        "constraint_evidence_refs": [],
    }
    validate_structured_state_section(section)
    return section


def bind_production_state_contract(router: Any, plan: AuthoredPlan) -> AuthoredPlan:
    """Attach deterministic state records after planning, immediately before production."""

    if plan.structured_sections.get("state_model"):
        return plan
    section = compile_production_state_section(router, plan)
    if not section:
        return plan
    structured = deepcopy(plan.structured_sections)
    structured["state_model"] = section
    return AuthoredPlan(
        requested_prompt=plan.requested_prompt,
        text=plan.text,
        existing_input_sha256=plan.existing_input_sha256,
        media_paths=plan.media_paths,
        schema_version=plan.schema_version,
        structured_sections=structured,
    )


__all__ = [
    "bind_production_state_contract",
    "compile_production_state_section",
]
