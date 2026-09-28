"""Build Java structures from native scalar decisions, never nested model JSON.

Each cursor retains the selected concern, enclosing declaration and accepted parts.
The host owns arrays, object structure and completion bounds. Invalid decisions stop
at this boundary; they cannot discard accepted work through whole-region retries.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from .custom_module_errors import AtomicJavaDecisionError
from .implementation_ir import OutputBudgetExhausted
from .llama_finish_reason_contract import (
    CONTEXT_PRESSURE,
    OUTPUT_EXHAUSTED,
    completion_boundary_error,
)
from .model_adapters.base import NativeToolDecisionRejected
from .model_output_atomicity_contract import (
    MAX_MODEL_FIELDS,
    MAX_MODEL_STRING_CHARS,
    assert_atomic_model_schema,
)

MAX_ASSEMBLY_CALLS = 128
MAX_PART_ITEMS = 32
_MODIFIERS = {
    "fields": ["public", "protected", "private", "static", "final", "volatile", "transient"],
    "methods": ["public", "protected", "private", "static", "final", "synchronized"],
    "classes": ["public", "protected", "private", "static", "final", "abstract"],
    "records": ["public", "protected", "private", "static"],
    "enums": ["public", "protected", "private", "static"],
}
_TYPE_PATTERN = (
    r"^(?!.*\b(?:public|protected|private|static|final|volatile|transient|"
    r"abstract|synchronized|native)\b)[A-Za-z_$][A-Za-z0-9_$.,<>?\[\] @]*$"
)


def _closed(properties: Mapping[str, Any], required=()) -> dict[str, Any]:
    return {"type": "object", "properties": dict(properties),
            "required": list(required), "additionalProperties": False}


def _scalar_schema(schema: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = deepcopy(dict(schema))
    if result.get("type") == "string":
        result["maxLength"] = min(result.get("maxLength", MAX_MODEL_STRING_CHARS), MAX_MODEL_STRING_CHARS)
        if key in {"type", "return_type"}:
            result["pattern"] = _TYPE_PATTERN
            result["description"] = "Java type only; modifiers belong exclusively in modifiers."
    return result


def _assembly_context(value: Any, selected_path: list, path: tuple = ()) -> Any:
    """Retain declarations everywhere and executable bodies only at the cursor."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if key == "body" and tuple(selected_path[:len(path)]) != path:
                result["accepted_body_sha256"] = hashlib.sha256(
                    json.dumps(item, ensure_ascii=False).encode("utf-8")
                ).hexdigest()
            else:
                result[key] = _assembly_context(item, selected_path, (*path, key))
        return result
    if isinstance(value, list):
        return [_assembly_context(item, selected_path, (*path, index)) for index, item in enumerate(value)]
    return value


class JavaStructureAssembly:
    def __init__(self, callback: Callable, payload: Mapping[str, Any], *, output_token_ceiling: int | None,
                 config: Any = None):
        self.callback = callback
        self.payload = dict(payload)
        self.root: dict[str, Any] = {}
        self.calls = 0
        self.output_token_ceiling = output_token_ceiling
        self.config = config

    def _ask(self, schema: Mapping[str, Any], path: list, purpose: str) -> dict[str, Any]:
        if self.calls >= MAX_ASSEMBLY_CALLS:
            raise OutputBudgetExhausted(
                "OUTPUT_BUDGET_EXHAUSTED: Java assembly call limit requires further decomposition."
            )
        assert_atomic_model_schema(schema, surface="native Java assembly")
        self.calls += 1
        # Replace the old one-shot output instructions, retaining semantic authority.
        payload = {key: value for key, value in self.payload.items()
                   if key not in {"generation_recipe", "scope"}}
        payload["assembly"] = {
            "path": path, "purpose": purpose, "accepted_structure": _assembly_context(self.root, path),
            "remaining_calls": MAX_ASSEMBLY_CALLS - self.calls,
        }
        kwargs = {
            "tool_name": "emit_java_part", "parameters": dict(schema),
            "description": "Fill the selected Java declaration or statement using native scalar arguments.",
        }
        signature = inspect.signature(self.callback)
        if self.output_token_ceiling is not None and (
            "output_token_ceiling" in signature.parameters
            or any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
        ):
            kwargs["output_token_ceiling"] = self.output_token_ceiling
        messages = [
                {"role": "system", "content": (
                    "Implement the host-selected concern through native emit_java_part calls. "
                    "Fill only the current assembly.path using the supplied scalar schema. "
                    "The host constructs objects/arrays; never serialize them into strings. "
                    "Keep the authored requirements, dependency_api and available_sibling_api authoritative. "
                    "Reuse exact sibling declarations; do not redeclare them or change their types/defaults. "
                    "Accepted structure and enclosing declarations remain fixed. "
                    "For part selection choose a needed part or done when this enclosing object is complete. "
                    "A body value is one complete Java statement or balanced control-flow block, "
                    "not a fragment of JSON or a partial brace. Split long logic into named helper methods. "
                    "For a declaration, type/return_type contains only a Java type, never modifiers. "
                    "The host adds static to outer fields/methods. Omit unnecessary optional scalar values. "
                    "Do not invent Minecraft/Fabric APIs or metadata-only gameplay implementations."
                )},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
            ]
        from .model_context_budget import request_message_budget

        tools = [{"type": "function", "function": {
            "name": kwargs["tool_name"], "description": kwargs["description"], "parameters": schema,
        }}]
        budget = request_message_budget(self.config, tools)
        # Preserve complete requirements/cursor instead of truncating authoritative JSON.
        size = len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if size + 2048 > budget:
            raise OutputBudgetExhausted(
                f"OUTPUT_BUDGET_EXHAUSTED: Java assembly context needs decomposition ({size + 2048}>{budget})."
            )
        try:
            value = self.callback("coder", messages, **kwargs)
            Draft202012Validator(schema).validate(value)
        except (NativeToolDecisionRejected, ValidationError) as exc:
            raise AtomicJavaDecisionError(
                f"ATOMIC_JAVA_ASSEMBLY_INVALID: {path}: {exc}",
                response=getattr(exc, "rejections", getattr(exc, "instance", None)),
            ) from exc
        except Exception as exc:
            from .llama_context_safety_contract import ContextPackingError
            from .llama_exact_context import ExactContextOverflow

            boundary = completion_boundary_error(exc)
            current: BaseException | None = exc
            seen: set[int] = set()
            context_overflow = False
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                if isinstance(current, (ContextPackingError, ExactContextOverflow)):
                    context_overflow = True
                    break
                current = getattr(current, "cause", None) or current.__cause__ or current.__context__
            if context_overflow or (boundary is not None and boundary.kind in {OUTPUT_EXHAUSTED, CONTEXT_PRESSURE}):
                raise OutputBudgetExhausted(
                    f"OUTPUT_BUDGET_EXHAUSTED: Java component {path} requires decomposition."
                ) from exc
            raise
        return dict(value)

    def _object(self, schema: Mapping[str, Any], target: dict, path: list) -> None:
        properties = schema["properties"]
        scalars = {key: _scalar_schema(value, key) for key, value in properties.items()
                   if value.get("type") not in {"array", "object"}}
        if len(path) == 2 and "name" in scalars:
            category = path[0]
            kinds = {"fields": "field", "classes": "type", "records": "type", "enums": "type"}
            reserved = {
                row["symbol"] for row in self.payload.get("available_sibling_api", ())
                if isinstance(row, Mapping) and row.get("kind") == kinds.get(category)
                and row.get("symbol")
            }
            reserved.update(item["name"] for item in self.root.get(category, ())
                            if item is not target and isinstance(item, Mapping) and item.get("name")
                            and category != "methods")
            if category in {"classes", "records", "enums"}:
                reserved.add(self.payload.get("host_selected_class", ""))
                for other in {"classes", "records", "enums"} - {category}:
                    reserved.update(item["name"] for item in self.root.get(other, ()) if item.get("name"))
            reserved.discard("")
            if reserved:
                scalars["name"]["not"] = {"enum": sorted(reserved)}
        required = set(schema.get("required", ()))
        names = list(scalars)
        for start in range(0, len(names), MAX_MODEL_FIELDS):
            keys = names[start:start + MAX_MODEL_FIELDS]
            target.update(self._ask(
                _closed({key: scalars[key] for key in keys}, [key for key in keys if key in required]),
                path, "Declare this component's identity, type and initial value.",
            ))
        arrays = {key: value for key, value in properties.items() if value.get("type") == "array"}
        while arrays:
            selected = self._ask(
                _closed({"part": {"type": "string", "enum": [*arrays, "done"]}}, ["part"]),
                path, "Select the next necessary part of this component; done closes it.",
            )["part"]
            if selected == "done":
                break
            values = target.setdefault(selected, [])
            if len(values) >= MAX_PART_ITEMS:
                raise OutputBudgetExhausted(
                    f"OUTPUT_BUDGET_EXHAUSTED: {path + [selected]} needs decomposition."
                )
            item_schema = arrays[selected]["items"]
            item_path = [*path, selected, len(values)]
            if item_schema.get("type") == "object":
                item: dict[str, Any] = {}
                values.append(item)
                self._object(item_schema, item, item_path)
            else:
                scalar = _scalar_schema(item_schema, selected)
                if selected == "modifiers":
                    scalar["enum"] = _MODIFIERS.get(str(path[-2]) if len(path) >= 2 else "", [])
                value = self._ask(_closed({"value": scalar}, ["value"]), item_path,
                                  "Append one complete value to the selected part.")["value"]
                if arrays[selected].get("uniqueItems") and value in values:
                    raise AtomicJavaDecisionError(
                        f"ATOMIC_JAVA_ASSEMBLY_INVALID: duplicate {selected} value.", response=value,
                    )
                values.append(value)
        # These are host-created containers. Validate their required shape too.
        try:
            Draft202012Validator(schema).validate(target)
        except ValidationError as exc:
            raise AtomicJavaDecisionError(
                f"ATOMIC_JAVA_ASSEMBLY_INVALID: incomplete component {path}: {exc.message}", response=target,
            ) from exc
        if path and len(path) >= 2 and path[-2] in {"fields", "methods"}:
            from .atomic_java_admission import _semantic_component_issue

            reason, _ = _semantic_component_issue(str(path[-2]), target)
            # Nested constructors have a separate renderer contract.
            if reason and not (len(path) > 2 and target.get("name") == "<init>"):
                raise AtomicJavaDecisionError(
                    f"ATOMIC_JAVA_ASSEMBLY_INVALID: {path}: {reason}", response=target,
                )

    def run(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        self._object(parameters, self.root, [])
        return self.root
