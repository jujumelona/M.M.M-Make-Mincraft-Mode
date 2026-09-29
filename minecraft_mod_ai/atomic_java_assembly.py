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
        result["maxLength"] = min(
            result.get("maxLength", MAX_MODEL_STRING_CHARS),
            MAX_MODEL_STRING_CHARS,
        )
        if key in {"type", "return_type"}:
            result["pattern"] = _TYPE_PATTERN
            result["description"] = (
                "Java type only. Declaration visibility and modifiers are host-owned; "
                "the model must never emit them."
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
        kwargs = {
            "tool_name": "emit_java_part",
            "parameters": dict(schema),
            "description": (
                "Emit one Java sibling item per native function call."
                if use_multi
                else "Fill the selected Java declaration or statement using native scalar arguments."
            ),
        }
        signature = inspect.signature(callback)
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
                "When the current path is a sibling batch, emit one function call per sibling item. "
                "A batch part is single-use: when a part is selected, emit every needed sibling item "
                "for that part in the same native turn because the host closes that part immediately. "
                "Keep the authored requirements, dependency_api, available_sibling_api, and "
                "compiler_contract authoritative. The target is first-pass compilable Java, not code that "
                "expects a compiler-repair round. Reuse exact sibling declarations; do not redeclare them "
                "or change their types/defaults. Never reassign a final sibling or concern-local final field. "
                "If a field is final, initialize it at declaration time. If an authoritative API returns Object "
                "but this method needs a narrower generic/container type, narrow with an explicit runtime type "
                "check and a type-compatible fallback; never use a raw/unchecked cast as a shortcut. "
                "Use canonical JDK packages; Lock/ReentrantLock live in java.util.concurrent.locks. "
                "Accepted structure and enclosing declarations remain fixed. "
                "For part selection choose a needed part or done when this enclosing object is complete. "
                "A body value is one complete Java statement or balanced control-flow block, "
                "not a fragment of JSON or a partial brace. Split long logic into named helper methods. "
                "For a declaration, type/return_type contains only a Java type. "
                "All Java declaration modifiers and visibility are host-owned; the model never emits them. "
                "The host adds static to outer fields/methods and owns nested-type visibility. "
                "Omit unnecessary optional scalar values. "
                "A field marked final must have a declaration initializer and generated executable code must never "
                "rebind a final field. Preserve generic types exactly. If an authoritative API returns Object, "
                "do not directly return it from a narrower typed method; inspect/narrow the runtime value first. "
                "For JDK locks use java.util.concurrent.locks.Lock/ReentrantLock (or simple Lock/ReentrantLock, "
                "which the host canonicalizes), never java.util.concurrent.Lock/ReentrantLock. "
                "Do not invent Minecraft/Fabric APIs or metadata-only gameplay implementations."
            )},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]
        from .model_context_budget import request_message_budget

        tools = [{"type": "function", "function": {
            "name": kwargs["tool_name"], "description": kwargs["description"], "parameters": schema,
        }}]
        budget = request_message_budget(self.config, tools)
        size = len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if size + 2048 > budget:
            raise OutputBudgetExhausted(
                f"OUTPUT_BUDGET_EXHAUSTED: Java assembly context needs decomposition ({size + 2048}>{budget})."
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
            # Production uses native multi-call turns. A selected array part is one
            # complete batch, not a resumable cursor. Closing it immediately prevents
            # the model from reopening fields/methods/body and redeclaring items that
            # were already accepted in the preceding native turn.
            if self.multi_callback is not None:
                arrays.pop(selected)
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
                emitted = self._ask_many(
                    _closed({"value": scalar}, ["value"]),
                    [*path, selected],
                    "Append one or more complete values; one native call per value.",
                    limit=remaining,
                )
                for row in emitted:
                    value = row["value"]
                    if arrays[selected].get("uniqueItems") and value in values:
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
