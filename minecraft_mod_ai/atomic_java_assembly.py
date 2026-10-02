"""Build Java structures from native scalar decisions, never nested model JSON.

Each cursor retains the selected concern, enclosing declaration and accepted parts.
The host owns arrays, object structure and completion bounds. Invalid decisions stop
at this boundary; they cannot discard accepted work through whole-region retries.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from .custom_module_errors import AtomicJavaDecisionError
from .execution_contract_policy import (
    DEFAULT_ATOMIC_SCHEMA_LIMITS,
    JAVA_ATOMIC_ASSEMBLY_CONTEXT_MARGIN_BYTES,
    JAVA_ATOMIC_ASSEMBLY_MAX_CALLS as MAX_ASSEMBLY_CALLS,
    JAVA_ATOMIC_ASSEMBLY_MAX_PART_ITEMS as MAX_PART_ITEMS,
    java_atomic_assembly_system_prompt,
)
from .implementation_ir import OutputBudgetExhausted
from .llama_finish_reason_contract import (
    CONTEXT_PRESSURE,
    OUTPUT_EXHAUSTED,
    completion_boundary_error,
)
from .model_adapters.base import NativeToolDecisionRejected
from .model_output_atomicity_contract import (
    MAX_MODEL_FIELDS,
    assert_atomic_model_schema,
)

_TYPE_PATTERN = (
    r"^(?!.*\b(?:public|protected|private|static|final|volatile|transient|"
    r"abstract|synchronized|native)\b)[A-Za-z_$][A-Za-z0-9_$.,<>?\[\] @]*$"
)

_EXECUTABLE_SCALAR_PATTERN = (
    r"^(?![\s\S]*\b(?:package|import)\s+[A-Za-z_$])"
    r"(?![\s\S]*\b(?:class|interface|enum|record)\s+[A-Za-z_$])"
    r"(?![\s\S]*\b(?:public|protected|private)\s+(?:static\s+)?"
    r"[A-Za-z_$][A-Za-z0-9_$.,<>?\[\] ]*\s+[A-Za-z_$][A-Za-z0-9_$]*\s*\()"
    r"(?!\s*static\s*\{)[\s\S]*$"
)


def _closed(properties: Mapping[str, Any], required=()) -> dict[str, Any]:
    return {"type": "object", "properties": dict(properties),
            "required": list(required), "additionalProperties": False}


def _scalar_schema(schema: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = deepcopy(dict(schema))
    if result.get("type") == "string":
        result["maxLength"] = min(
            result.get("maxLength", DEFAULT_ATOMIC_SCHEMA_LIMITS.max_string_chars),
            DEFAULT_ATOMIC_SCHEMA_LIMITS.max_string_chars,
        )
        if key in {"type", "return_type"}:
            result["pattern"] = _TYPE_PATTERN
            result["description"] = (
                "Java type only. Declaration visibility and modifiers are host-owned; "
                "the model must never emit them."
            )
        elif key in {"body", "statements"}:
            result["pattern"] = _EXECUTABLE_SCALAR_PATTERN
            result["description"] = (
                "One executable Java statement or balanced control-flow block only. "
                "Package/import directives, type declarations, method declarations, and "
                "static initializer blocks are host-owned and cannot appear here."
            )
        elif key == "initializer":
            # Qwen native tool calls naturally encode literal booleans/numbers/null as
            # JSON scalars. Accept those transport forms and canonicalize them to Java
            # source text before the host-owned structure is validated/rendered.
            result["type"] = ["string", "number", "boolean", "null"]
            result["description"] = (
                str(result.get("description") or "").strip()
                + " Native JSON string/number/boolean/null literals are accepted; "
                "the host canonicalizes them to Java initializer source."
            ).strip()
    return result


def _normalize_java_scalar_arguments(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "initializer" not in result:
        return result
    initializer = result["initializer"]
    if isinstance(initializer, str):
        return result
    if initializer is None:
        result["initializer"] = "null"
        return result
    if isinstance(initializer, bool):
        result["initializer"] = "true" if initializer else "false"
        return result
    if isinstance(initializer, int):
        result["initializer"] = str(initializer)
        return result
    if isinstance(initializer, float):
        if not math.isfinite(initializer):
            raise AtomicJavaDecisionError(
                "ATOMIC_JAVA_ASSEMBLY_INVALID: Java initializer must be a finite JSON number.",
                response=initializer,
            )
        result["initializer"] = json.dumps(
            initializer,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
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
    def __init__(
        self,
        callback: Callable,
        payload: Mapping[str, Any],
        *,
        output_token_ceiling: int | None,
        config: Any = None,
        multi_callback: Callable | None = None,
    ):
        self.callback = callback
        self.multi_callback = multi_callback if callable(multi_callback) else None
        self.payload = dict(payload)
        self.root: dict[str, Any] = {}
        self.calls = 0
        self.output_token_ceiling = output_token_ceiling
        self.config = config

    def _request(
        self,
        schema: Mapping[str, Any],
        path: list,
        purpose: str,
        *,
        multiple: bool,
        limit: int = 1,
    ) -> tuple[dict[str, Any], ...]:
        if self.calls >= MAX_ASSEMBLY_CALLS:
            raise OutputBudgetExhausted(
                "OUTPUT_BUDGET_EXHAUSTED: Java assembly call limit requires further decomposition."
            )
        assert_atomic_model_schema(schema, surface="native Java assembly")
        self.calls += 1
        payload = {key: value for key, value in self.payload.items()
                   if key not in {"generation_recipe", "scope"}}
        recipe = self.payload.get("generation_recipe")
        if isinstance(recipe, Mapping):
            payload["compiler_contract"] = {
                "first_pass_goal": recipe.get("first_pass_goal"),
                "rules": list(recipe.get("compiler_first_rules") or ()),
                "jdk_package_anchors": dict(recipe.get("jdk_package_anchors") or {}),
                "sibling_api_is_authoritative": bool(
                    recipe.get("sibling_api_is_authoritative")
                ),
                "never_mutate_final_sibling_fields": bool(
                    recipe.get("never_mutate_final_sibling_fields")
                ),
            }
        payload["assembly"] = {
            "path": path,
            "purpose": purpose,
            "accepted_structure": _assembly_context(self.root, path),
            "remaining_calls": MAX_ASSEMBLY_CALLS - self.calls,
        }
        use_multi = bool(multiple and self.multi_callback is not None)
        callback = self.multi_callback if use_multi else self.callback
        statement_path = bool(path and path[-1] in {"body", "statements"})
        kwargs = {
            "tool_name": "emit_java_statement" if statement_path else "emit_java_part",
            "parameters": dict(schema),
            "description": (
                "Emit exactly one executable Java statement or balanced control-flow block "
                "for an already-declared host-owned method/lifecycle body. Never emit a "
                "method signature, declaration wrapper, import, type declaration, or static initializer."
                if statement_path
                else (
                    "Emit one Java sibling item per native function call."
                    if use_multi
                    else "Fill the selected Java declaration using native scalar arguments."
                )
            ),
        }
        signature = inspect.signature(callback)
        if self.output_token_ceiling is not None and (
            "output_token_ceiling" in signature.parameters
            or any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
        ):
            kwargs["output_token_ceiling"] = self.output_token_ceiling
        messages = [
            {"role": "system", "content": java_atomic_assembly_system_prompt()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]
        from .model_context_budget import request_message_budget

        tools = [{"type": "function", "function": {
            "name": kwargs["tool_name"], "description": kwargs["description"], "parameters": schema,
        }}]
        budget = request_message_budget(self.config, tools)
        size = len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if size + JAVA_ATOMIC_ASSEMBLY_CONTEXT_MARGIN_BYTES > budget:
            raise OutputBudgetExhausted(
                f"OUTPUT_BUDGET_EXHAUSTED: Java assembly context needs decomposition ({size + JAVA_ATOMIC_ASSEMBLY_CONTEXT_MARGIN_BYTES}>{budget})."
            )
        try:
            raw = callback("coder", messages, **kwargs)
            values = tuple(raw) if use_multi else (raw,)
            if not values:
                raise AtomicJavaDecisionError(
                    f"ATOMIC_JAVA_ASSEMBLY_INVALID: {path}: native tool turn returned no items."
                )
            if len(values) > max(1, int(limit)):
                raise AtomicJavaDecisionError(
                    f"ATOMIC_JAVA_ASSEMBLY_INVALID: {path}: native tool turn returned "
                    f"{len(values)} items, limit is {limit}.",
                    response=values,
                )
            validated: list[dict[str, Any]] = []
            for value in values:
                normalized = _normalize_java_scalar_arguments(value)
                Draft202012Validator(schema).validate(normalized)
                validated.append(normalized)
            return tuple(validated)
        except (NativeToolDecisionRejected, ValidationError) as exc:
            raise AtomicJavaDecisionError(
                f"ATOMIC_JAVA_ASSEMBLY_INVALID: {path}: {exc}",
                response=getattr(exc, "rejections", getattr(exc, "instance", None)),
            ) from exc
        except AtomicJavaDecisionError:
            raise
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
            if context_overflow or (
                boundary is not None and boundary.kind in {OUTPUT_EXHAUSTED, CONTEXT_PRESSURE}
            ):
                raise OutputBudgetExhausted(
                    f"OUTPUT_BUDGET_EXHAUSTED: Java component {path} requires decomposition."
                ) from exc
            raise

    def _ask(self, schema: Mapping[str, Any], path: list, purpose: str) -> dict[str, Any]:
        return self._request(schema, path, purpose, multiple=False, limit=1)[0]

    def _ask_many(
        self,
        schema: Mapping[str, Any],
        path: list,
        purpose: str,
        *,
        limit: int,
    ) -> tuple[dict[str, Any], ...]:
        return self._request(schema, path, purpose, multiple=True, limit=limit)

    def _object(
        self,
        schema: Mapping[str, Any],
        target: dict,
        path: list,
        *,
        scalars_seeded: bool = False,
    ) -> None:
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
        if not scalars_seeded:
            for start in range(0, len(names), MAX_MODEL_FIELDS):
                keys = names[start:start + MAX_MODEL_FIELDS]
                target.update(self._ask(
                    _closed({key: scalars[key] for key in keys}, [key for key in keys if key in required]),
                    path, "Declare this component's identity, type and initial value.",
                ))
        # Schema metadata and cursor state must never share the same mutable
        # container. array_specs is immutable for the lifetime of this object
        # assembly; remaining_parts alone tracks which native multi-call batches
        # are still open.
        array_specs = {
            key: value
            for key, value in properties.items()
            if value.get("type") == "array"
        }
        remaining_parts = list(array_specs)
        while remaining_parts:
            selected = self._ask(
                _closed(
                    {"part": {"type": "string", "enum": [*remaining_parts, "done"]}},
                    ["part"],
                ),
                path,
                "Select the next necessary part of this component; done closes it.",
            )["part"]
            if selected == "done":
                break
            values = target.setdefault(selected, [])
            if len(values) >= MAX_PART_ITEMS:
                raise OutputBudgetExhausted(
                    f"OUTPUT_BUDGET_EXHAUSTED: {path + [selected]} needs decomposition."
                )
            selected_spec = array_specs[selected]
            item_schema = selected_spec["items"]
            # Native multi-call turns emit the complete sibling batch for the
            # selected part. Close only the cursor state; keep selected_spec
            # available for validation/rendering below.
            if self.multi_callback is not None:
                remaining_parts.remove(selected)
            item_path = [*path, selected, len(values)]
            remaining = MAX_PART_ITEMS - len(values)
            if item_schema.get("type") == "object":
                item_properties = item_schema.get("properties", {})
                item_scalars = {
                    key: _scalar_schema(value, key)
                    for key, value in item_properties.items()
                    if value.get("type") not in {"array", "object"}
                }
                item_required = set(item_schema.get("required", ()))
                scalar_names = list(item_scalars)
                if (
                    self.multi_callback is not None
                    and item_scalars
                    and len(scalar_names) <= MAX_MODEL_FIELDS
                ):
                    reserved: set[str] = set()
                    if len(item_path) == 2 and "name" in item_scalars:
                        category = item_path[0]
                        kinds = {
                            "fields": "field",
                            "classes": "type",
                            "records": "type",
                            "enums": "type",
                        }
                        reserved.update(
                            row["symbol"]
                            for row in self.payload.get("available_sibling_api", ())
                            if isinstance(row, Mapping)
                            and row.get("kind") == kinds.get(category)
                            and row.get("symbol")
                        )
                        reserved.update(
                            item["name"]
                            for item in self.root.get(category, ())
                            if isinstance(item, Mapping) and item.get("name")
                            and category != "methods"
                        )
                        if category in {"classes", "records", "enums"}:
                            reserved.add(self.payload.get("host_selected_class", ""))
                            for other in {"classes", "records", "enums"} - {category}:
                                reserved.update(
                                    item["name"]
                                    for item in self.root.get(other, ())
                                    if isinstance(item, Mapping) and item.get("name")
                                )
                        reserved.discard("")
                        if reserved:
                            item_scalars["name"]["not"] = {"enum": sorted(reserved)}
                    seed_schema = _closed(
                        item_scalars,
                        [key for key in scalar_names if key in item_required],
                    )
                    seeds = self._ask_many(
                        seed_schema,
                        [*path, selected],
                        "Declare one or more sibling components; one native call per sibling.",
                        limit=remaining,
                    )
                    batch_names: set[str] = set()
                    for seed in seeds:
                        seed_name = str(seed.get("name") or "").strip()
                        if seed_name and (seed_name in reserved or seed_name in batch_names):
                            raise AtomicJavaDecisionError(
                                f"ATOMIC_JAVA_ASSEMBLY_INVALID: duplicate or reserved "
                                f"{selected} name {seed_name!r}.",
                                response=seed,
                            )
                        if seed_name:
                            batch_names.add(seed_name)
                        index = len(values)
                        item = dict(seed)
                        values.append(item)
                        self._object(
                            item_schema,
                            item,
                            [*path, selected, index],
                            scalars_seeded=True,
                        )
                else:
                    item: dict[str, Any] = {}
                    values.append(item)
                    self._object(item_schema, item, item_path)
            else:
                scalar = _scalar_schema(item_schema, selected)
                executable = selected in {"body", "statements"}
                argument = "statement" if executable else "value"
                purpose = "Append one or more complete values; one native call per value."
                if executable:
                    owner_name = str(target.get("name") or "").strip()
                    owner_return = str(target.get("return_type") or "").strip()
                    signature = (
                        f"{owner_return} {owner_name}(...)" if owner_name else "host-owned lifecycle body"
                    )
                    purpose = (
                        f"Emit executable statements only for existing {signature}. "
                        "The declaration/header already exists and must not be repeated."
                    )
                emitted = self._ask_many(
                    _closed({argument: scalar}, [argument]),
                    [*path, selected],
                    purpose,
                    limit=remaining,
                )
                for row in emitted:
                    value = row[argument]
                    if selected_spec.get("uniqueItems") and value in values:
                        raise AtomicJavaDecisionError(
                            f"ATOMIC_JAVA_ASSEMBLY_INVALID: duplicate {selected} value.",
                            response=value,
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
