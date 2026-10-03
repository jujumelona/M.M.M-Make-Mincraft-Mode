from __future__ import annotations

"""Deterministic production compiler for canonical structured state models.

Semantic interpretation belongs to the bounded planning worksheet. Production
never asks a model to reinterpret Markdown or emit state Java; it only normalizes,
validates, and host-compiles the canonical structured state authority.
"""

from collections.abc import Mapping, Sequence
import json
import re
from typing import Any

from .authored_ir_parser import authored_section_id, parse_markdown_heading
from .authored_plan import AuthoredPlan
from .planning_detail_slots import DETAIL_RECORDS
from .structured_state_runtime import (
    render_state_model_concern,
    validate_state_expression,
    validate_structured_state_section,
)

_STATE_CONCERNS = tuple(DETAIL_RECORDS["state_model"])
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


def render_production_state_java(
    section: Mapping[str, Any],
    *,
    package_name: str,
    symbol: str = "AuthoredStateModel",
) -> str:
    """Render one complete state owner class from canonical structured authority."""

    if not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", symbol):
        raise ValueError(f"PRODUCTION_STATE_SYMBOL_INVALID: {symbol!r}")
    package = str(package_name or "").strip()
    if package and re.fullmatch(
        r"[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*",
        package,
    ) is None:
        raise ValueError(f"PRODUCTION_STATE_PACKAGE_INVALID: {package!r}")

    normalized = normalize_structured_state_section(section)
    specification = normalized["specification"]
    obligations: list[str] = []
    active: list[str] = []
    for concern in _STATE_CONCERNS:
        rows = specification.get(concern)
        if not isinstance(rows, list) or not rows:
            continue
        active.append(concern)
        obligations.append(
            json.dumps(
                {
                    "instruction": json.dumps(
                        {"concern": concern},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "structured_records": rows,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    if not active:
        raise ValueError(
            "PRODUCTION_STATE_STRUCTURED_AUTHORITY_REQUIRED: "
            "state operations require at least one canonical state concern record"
        )

    task = {"implementation_obligations": obligations}
    members: list[str] = []
    include_runtime = True
    for concern in active:
        rendered = render_state_model_concern(
            task,
            concern,
            include_runtime=include_runtime,
        )
        if not rendered:
            continue
        members.append(rendered)
        include_runtime = False

    if not members:
        raise ValueError("PRODUCTION_STATE_HOST_COMPILER_EMPTY")

    body = "\n\n".join(members)
    indented = "\n".join(
        ("    " + line if line else "")
        for line in body.splitlines()
    )
    prefix = f"package {package};\n\n" if package else ""
    return (
        prefix
        + "// MMM:TYPED_PLAN_STATE_OWNER\n"
        + f"public final class {symbol} {{\n"
        + f"    private {symbol}() {{}}\n\n"
        + indented
        + "\n}\n"
    )



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
    "render_production_state_java",
]
