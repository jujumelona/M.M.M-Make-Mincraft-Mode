"""Construct Java source from small, independently completed model actions.

This is a first-pass production protocol, not an output-exhaustion retry.  The
host owns the package boundary, final closing brace, and source assembly.  The
model supplies typed metadata and completes fixed-prefix Java declarations. Required
Fabric entrypoint declarations are immutable host data, not model suggestions.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any

from .first_pass_java_declarations import (
    IDENTIFIER,
    MEMBER_KINDS,
    canonicalize_field_type_metadata,
    declaration_frame,
    declaration_header,
    metadata_schema,
    render_declaration,
    validate_declaration_metadata,
)
from .fixed_template_generation import (
    _validate_native_arguments,
)
from .model_output_atomicity_contract import assert_atomic_model_schema

_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "imports": {"type": "array", "maxItems": 24, "items": {"type": "string", "maxLength": 240}},
        "superclass": {"type": "string", "maxLength": 240},
        "interfaces": {"type": "string", "maxLength": 480},
        "units": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(MEMBER_KINDS)},
                    "name": IDENTIFIER,
                    "purpose": {"type": "string", "maxLength": 160},
                },
                "required": ["kind", "name", "purpose"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["imports", "superclass", "interfaces", "units"],
    "additionalProperties": False,
}

_QUALIFIED_JAVA_IMPORT = re.compile(
    r"(?:[A-Za-z_$][A-Za-z0-9_$]*\.)+[A-Za-z_$][A-Za-z0-9_$]*|"
    r"(?:[A-Za-z_$][A-Za-z0-9_$]*\.)+\*"
)


def _assemble_java_imports(entries: Sequence[str]) -> str:
    """Create Java import statements from model-supplied qualified type names.

    The model supplies dependency references, not Java declaration syntax.
    Legacy import statements are accepted as suggestions, but arbitrary prose
    and executable fragments never enter the generated source header.
    """
    statements: list[str] = []
    for suggestion in entries:
        if not isinstance(suggestion, str):
            continue
        for fragment in re.split(r"[;\r\n,]", suggestion):
            reference = fragment.strip()
            reference = re.sub(r"^import\s+", "", reference).strip()
            is_static = reference.startswith("static ")
            if is_static:
                reference = reference[len("static "):].strip()
            if not _QUALIFIED_JAVA_IMPORT.fullmatch(reference):
                continue
            statements.append(
                "import " + ("static " if is_static else "") + reference + ";"
            )
    return "\n".join(dict.fromkeys(statements))


def _one_structured_action(
    router: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    name: str,
    schema: Mapping[str, Any],
    page_structure: bool = True,
) -> Mapping[str, Any]:
    """One attempt for one small action. No schema-repair or source regeneration."""
    assert_atomic_model_schema(schema, surface=f"first-pass Java {name}")
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or not properties:
        raise ValueError("JAVA_DATA_PAGE_REQUIRES_DECLARED_FIELDS")
    # Select page boundaries from the data structure, not an invented token cap.
    # Arrays are planned once by count and authored one entry at a time. Every
    # page is an independent first-pass request, never a failed-output replay.
    if page_structure and len(properties) > 1:
        merged: dict[str, Any] = {}
        for field, child in properties.items():
            page = {"type": "object", "properties": {field: child},
                    "required": [field], "additionalProperties": False}
            merged.update(_one_structured_action(
                router, [*messages, {"role": "user", "content": json.dumps(
                    {"current_field": field, "completed_plan_fields": merged},
                    ensure_ascii=False)}], name=f"{name}_{field}", schema=page,
            ))
        return _validate_native_arguments(merged, schema)

    field, child = next(iter(properties.items()))
    if page_structure and child.get("type") == "object":
        value = _one_structured_action(
            router, messages, name=f"{name}_{field}", schema=child,
        )
        return _validate_native_arguments({field: value}, schema)
    if page_structure and child.get("type") == "array":
        maximum = int(child["maxItems"])
        minimum = int(child.get("minItems", 0))
        count_schema = {"type": "object", "properties": {"count": {
            "type": "integer", "enum": list(range(minimum, maximum + 1))}},
            "required": ["count"], "additionalProperties": False}
        count = _one_structured_action(
            router, [*messages, {"role": "user", "content":
                f"Choose the number of necessary {field} entries; include all requirements."}],
            name=f"{name}_count", schema=count_schema,
        )["count"]
        values: list[Any] = []
        for index in range(count):
            page_schema = {"type": "object", "properties": {"value": child["items"]},
                           "required": ["value"], "additionalProperties": False}
            item = _one_structured_action(
                router, [*messages, {"role": "user", "content": json.dumps({
                    "field": field, "index": index, "count": count,
                    "previous_entries": values,
                    "instruction": "Return the next necessary entry without duplicates."},
                    ensure_ascii=False)}], name=f"{name}_{index + 1}", schema=page_schema,
            )
            values.append(item["value"])
        return _validate_native_arguments({field: values}, schema)

    response = router.generate_structured_data(
        "coder", messages, data_name=name,
        parameters=dict(schema),
    )
    if not isinstance(response, Mapping):
        raise ValueError(f"JAVA_FIRST_PASS_ACTION_MISSING: {name}")
    return _validate_native_arguments(response, schema)


def _java_identity(spec: Mapping[str, Any]) -> tuple[str, str] | None:
    """Infer only the declared compilation-unit identity, never a guessed API."""
    if str(spec.get("language", "")).casefold() != "java":
        return None
    if spec.get("java_prefix") or spec.get("java_suffix"):
        # Fragment/anchor contracts do not own their surrounding Java class.
        return None
    path = str(spec.get("target_path") or "").replace("\\", "/")
    source_root = next(
        (prefix for prefix in ("src/main/java/", "src/client/java/")
         if path.startswith(prefix)), None,
    )
    if source_root is None or not path.endswith(".java"):
        return None
    relative = PurePosixPath(path[len(source_root):])
    inferred_class = relative.stem
    inferred_package = ".".join(relative.parts[:-1])
    bindings = spec.get("bindings") or {}
    if not isinstance(bindings, Mapping):
        return None
    class_name = str(bindings.get("class_name") or inferred_class)
    package_name = str(bindings.get("package_name") or inferred_package)
    identifier = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
    if not identifier.fullmatch(class_name) or not package_name or any(
        not identifier.fullmatch(part) for part in package_name.split(".")
    ):
        return None
    if class_name != inferred_class or package_name != inferred_package:
        # Respect the actual target path rather than manufacturing a mismatched class.
        return None
    return package_name, class_name


def can_compose_first_pass_java(spec: Mapping[str, Any]) -> bool:
    return _java_identity(spec) is not None


def _owned_behavior_id(unit):
    return next((unit.get(key) for key in (
        "host_required_behavior", "host_handler_behavior", "host_condition_behavior",
        "host_effect_behavior") if unit.get(key)), None)


def _scope_host_implementation_messages(
    messages: Sequence[Mapping[str, str]],
    *,
    behavior: Mapping[str, Any] | None,
    purpose: str,
    for_body: bool = False,
) -> list[dict[str, str]]:
    """Keep full checked APIs, but only the behavior this member owns.

    The original whole-feature context remains authoritative for PLANNING and
    integration. Sending it unchanged to EACH per-member body generator made
    a 9B model author the entire mod inside one method, even after the host
    had chosen a single behavior. This view never mutates original messages,
    verified API evidence or workspace source.
    """
    if not behavior:
        return [dict(message) for message in messages]
    identifier = str(behavior.get("behavior_id") or "")
    scoped = []
    for message in messages:
        item = dict(message)
        if item.get("role") != "user":
            scoped.append(item)
            continue
        try:
            payload = json.loads(item.get("content", ""))
        except (ValueError, TypeError):
            scoped.append(item)
            continue
        if not isinstance(payload, dict) or not isinstance(
            payload.get("assigned_behaviors"), list
        ) or "inspected_target_api" not in payload:
            scoped.append(item)
            continue
        payload["requirement"] = purpose
        payload["original_feature_goal"] = purpose
        payload["assigned_behaviors"] = [dict(behavior)]
        # Hook selection is already finished by the host. Sending alternative
        # event owners again tells the small coder to invent another registration.
        payload.pop("source_owned_inspected_hooks", None)
        if isinstance(payload.get("source_owned_producer_obligations"), list):
            payload["source_owned_producer_obligations"] = [
                dict(item) for item in payload["source_owned_producer_obligations"]
                if item.get("behavior_id") == identifier
            ]
        if for_body:
            # Retrieval emits complete inventories for each candidate owner.
            # The 17th Colab run repeated 99K characters on every tiny body
            # request, mostly constructors of unrelated JDK Location APIs.
            # Body inference needs callable evidence selected by retrieval,
            # not every other member of those owners. Full symbol records stay
            # in the authoritative host index and remain checked by javac.
            evidence = payload.get("inspected_target_api")
            if isinstance(evidence, dict):
                evidence = dict(evidence)
                from .construction_file_api_evidence import scoped_body_api_records
                evidence["class_members"] = scoped_body_api_records(evidence)
                payload["inspected_target_api"] = evidence
        order = payload.get("feature_work_order")
        if isinstance(order, dict):
            order = dict(order)
            order["behavior_ids"] = [identifier]
            order["intents"] = [str(behavior.get("intent") or "")]
            order["effects"] = [str(behavior.get("effects") or "")]
            order["state_reads"] = list(behavior.get("state_reads") or [])
            order["state_writes"] = list(behavior.get("state_writes") or [])
            payload["feature_work_order"] = order
        item["content"] = json.dumps(payload, ensure_ascii=False, default=str)
        scoped.append(item)
    return scoped


def compose_first_pass_java(
    router: Any,
    spec: Mapping[str, Any],
    messages: Sequence[Mapping[str, str]],
    *,
    source_context: Sequence[Any] = (),
    construction_context: Mapping[str, Any] | None = None,
    behaviors: Sequence[Mapping[str, Any]] = (),
) -> str:
    identity = _java_identity(spec)
    if identity is None:
        raise ValueError("JAVA_FIRST_PASS_REQUIRES_COMPLETE_COMPILATION_UNIT")
    package_name, class_name = identity

    host_mode = construction_context is not None
    callback_api_index = None
    task = {
        "package": package_name,
        "class": class_name,
        "requirement": spec.get("requirement", ""),
        "bindings": spec.get("bindings", {}),
        "prerequisite_source": source_context,
        "side": spec.get("side"),
        "target_path": spec.get("target_path"),
        "implementation_requirements": spec.get("bindings", {}).get("java_implementation_requirements", ""),
    }
    bindings = task["bindings"]
    if host_mode:
        if bindings.get("request_feature_entrypoint") and not behaviors:
            raise ValueError("JAVA_HOST_BEHAVIOR_COVERAGE_MISSING")
        contract = (construction_context or {}).get("authoring_contract") or {}
        if contract.get("canonical_leaf") and contract.get("responsibility") and not bindings.get("request_feature_entrypoint"):
            task["requirement"] = str(contract["responsibility"])
        task["assigned_behaviors"] = [dict(item) for item in behaviors]
        task["feature_work_order"] = (construction_context or {}).get("feature_work_order")
    # Use the persisted owner roles which already select Fabric entrypoints.
    # COMMON request-feature jobs and CLIENT canonical jobs have fixed loader APIs.
    if bindings.get("request_feature_entrypoint"):
        if str(spec.get("side") or "").upper() == "CLIENT":
            entrypoint_interface = "net.fabricmc.api.ClientModInitializer"
            entrypoint_method = "onInitializeClient"
        else:
            entrypoint_interface = "net.fabricmc.api.ModInitializer"
            entrypoint_method = "onInitialize"
    elif str(spec.get("side") or "").upper() == "CLIENT":
        entrypoint_interface = "net.fabricmc.api.ClientModInitializer"
        entrypoint_method = "onInitializeClient"
    else:
        entrypoint_interface = entrypoint_method = ""
    plan_schema = deepcopy(_PLAN_SCHEMA)
    if host_mode:
        # The decoder never receives an unbounded import-generation task.
        plan_schema["properties"].pop("imports")
        plan_schema["required"] = list(plan_schema["properties"])
    plain_host_header = bool(
        host_mode and (construction_context or {}).get("authoring_contract", {})
        .get("host_requirements", {}).get("java_class_header") == "plain"
    )
    if plain_host_header:
        # A registry holder is a plain Java owner, not an EntityType or
        # Registry implementation. Never ask a small coder to invent
        # inheritance when the canonical leaf has already fixed it.
        for field in ("superclass", "interfaces"):
            plan_schema["properties"].pop(field, None)
        plan_schema["required"] = list(plan_schema["properties"])
    if entrypoint_interface:
        # Initializer inheritance, constructor and lifecycle are already selected
        # by the source contract. Do not ask the model to select them again.
        for field in ("superclass", "interfaces"):
            plan_schema["properties"].pop(field, None)
        unit_schema = plan_schema["properties"]["units"]["items"]
        unit_schema["properties"].pop("name")
        unit_schema["properties"]["kind"]["enum"] = [kind for kind in MEMBER_KINDS if kind != "constructor"]
        unit_schema["required"] = list(unit_schema["properties"])
        plan_schema["properties"]["units"]["minItems"] = 0
        plan_schema["required"] = list(plan_schema["properties"])
        task["host_entrypoint"] = {"interface": entrypoint_interface, "method": entrypoint_method}
    plan_messages = [
        *messages,
        {
            "role": "system",
            "content": (
                "Plan typed Java field/method/constructor/nested-type responsibilities. "
                "The host fixes every declaration from typed metadata; later complete "
                "one Java declaration from its immutable signature per request. "
                "Account for every required "
                "behavior and integration, including constructors and registration "
                "when required; never simplify the gameplay specification. "
                + ("The host resolves type imports from inspected source and JDK APIs. "
                   "Never supply an imports field, import statements or package statements. "
                   if host_mode else
                   "List Java imports as an array of fully-qualified type references, "
                   "for example ['java.util.List', 'java.util.Map']. "
                   "Do not include import keywords, semicolons or package lines. "
                   "Use 'static package.Type.member' only for a static import. ")
                + "The HOST writes the package and class header using the exact "
                f"class name {class_name}; do not generate those declarations. "
                "Provide superclass as just the optional Java type name (no "
                "'extends' keyword), and interfaces as optional comma-separated "
                "Java type names (no 'implements' keyword). Empty strings mean "
                "none. Preserve actual API inheritance/interface requirements. "
                "When host_entrypoint is present, its public constructor, interface and "
                "initialization method are already owned by the host. Plan only needed "
                "supporting members; do not plan a second lifecycle method. The host "
                "assigns internal helper names and later authors the required lifecycle body. "
                "Plan only real necessary supporting members, at most ten; "
                "each planned unit is one complete declaration with one coherent "
                "responsibility. No code in the plan units."
            ),
        },
        {"role": "user", "content": json.dumps(task, ensure_ascii=False, default=str)},
    ]
    if host_mode and bindings.get("request_feature_entrypoint"):
        # The host already owns the exact behavior set and Fabric entrypoint.
        # Asking the model to also invent "optional" members caused it to
        # choose a random int field, then emit registrar calls as that FIELD'S
        # initializer (production-logs 14). These are not optional features:
        # every assigned behavior receives its own host-declared handler and
        # registration method immediately below. No pre-planning model turn.
        plan = {"units": []}
    else:
        plan = _one_structured_action(
            router,
            plan_messages,
            name="submit_java_construction_plan",
            schema=plan_schema,
        )

    imports = "" if host_mode else _assemble_java_imports(plan["imports"])
    superclass = "" if plain_host_header else str(plan.get("superclass") or "").strip()
    interfaces = (entrypoint_interface or ("" if plain_host_header
                                          else str(plan.get("interfaces") or "").strip()))
    # The model never authors identity, scope, or outer braces. It supplies only
    # optional Java type references used in a deterministic host-owned header.
    # Reject executable fragments, not legitimate inheritance relationships.
    for type_clause in (superclass, interfaces):
        if any(marker in type_clause for marker in ("{", "}", ";", chr(10), chr(13), "/*", "//")):
            raise ValueError("JAVA_FIRST_PASS_TYPE_REFERENCE_INVALID")
    declaration = (
        f"public {'final ' if entrypoint_interface else ''}class {class_name}"
        + (f" extends {superclass}" if superclass else "")
        + (f" implements {interfaces}" if interfaces else "")
        + " {"
    )
    # Real Fabric listener interfaces supply the callback ABI. The model
    # selects only the matching verified trigger; Java parameters and the
    # method-reference registration are host-authored, not guessed locals.
    bound_events = {}
    event_candidates = []
    pending_custom_source_sites = []
    pending_ui_source_plans = []
    # Source-owned features must supply their own real execution producer.
    # The restricted rule path above is allowed to bind only inspected hooks
    # whose observation contract is already proven. Reusing the same hook
    # picker here produced JAVA_HOST_BEHAVIOR_EVENT_UNBOUND even after the
    # feature was deliberately routed away from it.
    source_owned = bool(host_mode and (construction_context or {}).get("host_source_owned"))
    source_work_graph = ((construction_context or {}).get("source_owned_work_graph")
                         if source_owned else None)
    if host_mode and bindings.get("request_feature_entrypoint"):
        from .host_fabric_callback_abi import callback_interfaces
        symbol_index = (construction_context or {}).get("symbols") or {}
        event_candidates = callback_interfaces(symbol_index, side=str(spec.get("side") or "COMMON"))
        site_bindings = (construction_context or {}).get("source_site_bindings") or {}
        if not isinstance(site_bindings, Mapping):
            raise ValueError("JAVA_SOURCE_SITE_BINDINGS_INVALID")
        if symbol_index and not event_candidates and not source_owned:
            # The rule-only route cannot invent a custom owner. Source-owned
            # mode below searches exact inspected source sites instead.
            raise ValueError("JAVA_HOST_FABRIC_EVENT_ABI_UNAVAILABLE")
        if event_candidates or source_owned:
            from .host_behavior_event_binding import select_behavior_callback
            for behavior in behaviors:
                behavior_id = str(behavior["behavior_id"])
                if not source_owned:
                    bound_events[behavior_id] = select_behavior_callback(
                        router, behavior, event_candidates)
                    continue
                from .source_owned_hook import choose_source_hook, SourceHookUnavailable
                # A verified Fabric callback alone cannot provide a missing
                # world/UI observation. When the immutable work graph requires
                # a producer BEFORE the callback, first bind the real materialized
                # source method. Otherwise the picker selects an unrelated
                # tick/player event and the coder invents an API to compensate.
                from .feature_source_work_graph import requires_original_occurrence_site
                requires_original_producer = bool(
                    source_work_graph is not None
                    and requires_original_occurrence_site(
                        source_work_graph, behavior_id)
                )
                if event_candidates and not requires_original_producer:
                    try:
                        bound_events[behavior_id] = choose_source_hook(
                            router, behavior, event_candidates)
                        continue
                    except SourceHookUnavailable:
                        # SOURCE_REQUIRED is not a second opportunity to
                        # substitute a player_join or server_tick callback.
                        pass
                source_ref = (behavior.get("source_site_ref")
                              or site_bindings.get(behavior_id))
                if not source_ref:
                    from .source_owned_site_choice import choose_verified_source_site
                    try:
                        source_ref = choose_verified_source_site(
                            router, behavior, symbol_index,
                            side=str(spec.get("side") or "COMMON").upper(),
                        )
                    except SourceHookUnavailable:
                        # New player-initiated gameplay need not already have
                        # its own Java method. Build its original input source
                        # through the existing Typed Action host, before asking
                        # the coder for a listener body. Passive occurrences
                        # remain unsupported here; no unrelated hook is used.
                        root = (construction_context or {}).get("project_root")
                        if not root or str(spec.get("side") or "COMMON").upper() == "CLIENT":
                            raise
                        from pathlib import Path
                        from .source_owned_ui_event import author_typed_ui_occurrence
                        manifest = Path(root) / "src/main/resources/fabric.mod.json"
                        if not manifest.is_file():
                            raise ValueError("SOURCE_SITE_FABRIC_MANIFEST_REQUIRED")
                        metadata = json.loads(manifest.read_text(encoding="utf-8"))
                        target = (construction_context or {}).get("target") or {}
                        produced = author_typed_ui_occurrence(
                            router, behavior, symbols=symbol_index, package=package_name,
                            mod_id=metadata["id"], minecraft_version=target.get(
                                "minecraft_version") or bindings.get("minecraft_version"),
                        )
                        pending_ui_source_plans.append(produced)
                        bound_events[behavior_id] = produced["callback"]
                        if source_work_graph is not None:
                            from .feature_source_work_graph import bind_planned_ui_occurrence
                            source_work_graph = bind_planned_ui_occurrence(
                                source_work_graph, behavior_id, produced)
                        continue
                root = (construction_context or {}).get("project_root")
                if not root:
                    raise ValueError("JAVA_CUSTOM_SOURCE_PROJECT_ROOT_REQUIRED")
                # Plan the exact Java/Mixin ABI *without writing files*.
                # An unfinished coder response must not leave a live Mixin
                # registered in fabric.mod.json or poison the next run.
                from pathlib import Path
                from .source_owned_custom_event import plan_verified_custom_event
                manifest = Path(root) / "src/main/resources/fabric.mod.json"
                if not manifest.is_file():
                    raise ValueError("SOURCE_SITE_FABRIC_MANIFEST_REQUIRED")
                metadata = json.loads(manifest.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict) or not isinstance(metadata.get("id"), str):
                    raise ValueError("SOURCE_SITE_FABRIC_MANIFEST_INVALID")
                production_args = {
                    "project_root": root, "symbols": symbol_index,
                    "source_site_ref": str(source_ref), "behavior_id": behavior_id,
                    "package": package_name,
                    "at": str(behavior.get("source_injection_point") or "HEAD"),
                    "side": str(spec.get("side") or "COMMON").upper(),
                    "java_version": (construction_context or {}).get(
                        "target", {}).get("java_version", 25),
                }
                produced = plan_verified_custom_event(
                    symbols=symbol_index, source_site_ref=str(source_ref),
                    behavior_id=behavior_id, package=package_name,
                    mod_id=metadata["id"], at=production_args["at"],
                    side=production_args["side"],
                    java_version=production_args["java_version"],
                )
                pending_custom_source_sites.append(production_args)
                bound_events[behavior_id] = produced["callback"]
                if source_work_graph is not None:
                    # Keep newly materialized events separate from the
                    # pre-existing Fabric callback inventory.
                    source_work_graph = deepcopy(source_work_graph)
                    for row in source_work_graph["tasks"]:
                        if row.get("behavior_id") != behavior_id:
                            continue
                        if row.get("kind") == "custom_event_source":
                            row["status"] = "materialized"
                            row["evidence"]["source_site_ref"] = str(source_ref)
                            row["evidence"]["source_descriptor"] = produced["source_descriptor"]
                            row["evidence"]["materialized_paths"] = sorted(produced["files"])
                            row["evidence"]["runtime_verified"] = False
                        elif row.get("kind") == "occurrence_hook":
                            row["evidence"].setdefault(
                                "materialized_custom_event_ids", []
                            ).append(produced["callback"]["event_id"])
                            row["evidence"]["source_site_ref"] = str(source_ref)
    if source_work_graph is not None:
        from .feature_source_work_graph import bind_selected_hooks
        source_work_graph = bind_selected_hooks(source_work_graph, bound_events)
    units = []
    for index, item in enumerate(plan["units"], 1):
        unit = dict(item)
        if entrypoint_interface:
            # These helpers have no preexisting public API. Select their names
            # before inference, outside the reserved entrypoint namespace.
            unit["name"] = ("MmmMember" if unit["kind"] in {"class", "record", "enum"} else "mmmMember") + str(index)
        elif unit["kind"] == "constructor":
            unit["name"] = class_name
        units.append(unit)
    if host_mode and bindings.get("request_feature_entrypoint"):
        if not behaviors:
            raise ValueError("JAVA_HOST_BEHAVIOR_COVERAGE_MISSING")
        seen_behaviors: set[str] = set()
        for row in behaviors:
            behavior_id = str(row.get("behavior_id") or "").strip()
            if not behavior_id or behavior_id in seen_behaviors or not row.get("intent"):
                raise ValueError("JAVA_HOST_BEHAVIOR_INVALID")
            seen_behaviors.add(behavior_id)
            registrar = "registerBehavior" + hashlib.sha256(
                behavior_id.encode("utf-8")).hexdigest()[:20]
            # The registrar installs the verified trigger and is NOT the owner
            # of all gameplay state, UI, economy and effects. Previously a
            # single no-argument registrar was asked to implement the entire
            # feature, causing the model to emit class-level fields as method
            # statements until it exhausted its output context.
            handler = "handleBehavior" + hashlib.sha256(
                behavior_id.encode("utf-8")).hexdigest()[:20]
            event = bound_events.get(behavior_id)
            dispatch = {}
            if event:
                condition = "matchesBehavior" + hashlib.sha256(behavior_id.encode()).hexdigest()[:20]
                effect = "applyBehavior" + hashlib.sha256(behavior_id.encode()).hexdigest()[:20]
                observer_units = []
                if source_work_graph is not None:
                    from .feature_source_work_graph import runtime_observer_methods
                    for observer in runtime_observer_methods(source_work_graph, behavior_id):
                        observer_units.append({
                            "kind": "method", "name": observer["name"],
                            "purpose": (
                                "Resolve required runtime observation '"
                                + observer["source_name"] + "' from the verified "
                                "event receiver/API. Return true only if the "
                                "actual observation and prerequisites are satisfied. "
                                "Do not fabricate state or treat a placeholder "
                                "null-check as evidence. Required work: "
                                + observer["required_action"]
                            ),
                            "host_condition_behavior": behavior_id,
                            "host_observer_source_task_id": observer["task_id"],
                            "host_observer_source_name": observer["source_name"],
                            "verified_event_callback": event,
                        })
                units.extend([
                    {"kind": "method", "name": condition,
                     "purpose": "Test only the original trigger and preconditions: "
                                + str(row.get("trigger") or row["intent"])
                                + ". Return boolean; perform no gameplay effects or state mutations.",
                     "host_condition_behavior": behavior_id, "verified_event_callback": event},
                    *observer_units,
                    {"kind": "method", "name": effect,
                     "purpose": "Apply only these effects after all source observations "
                                "and the host's trigger predicate passed: "
                                + str(row.get("effects") or row["intent"]),
                     "host_effect_behavior": behavior_id, "verified_event_callback": event},
                ])
                dispatch = {
                    "host_condition_method": condition,
                    "host_effect_method": effect,
                    "host_observer_methods": [unit["name"] for unit in observer_units],
                }
            units.append({
                **dispatch,
                "kind": "method", "name": handler,
                "purpose": (
                    "Implement the runtime effects for behavior "
                    + behavior_id + ": " + str(row["intent"])
                    + "; trigger=" + str(row.get("trigger") or "")
                    + "; effects=" + str(row.get("effects") or "")
                ),
                "host_handler_behavior": behavior_id,
                "verified_event_callback": bound_events.get(behavior_id),
            })
            units.append({
                "kind": "method", "name": registrar,
                "purpose": (
                    "Register the real Fabric event/callback for behavior "
                    + behavior_id + " and dispatch to the host-declared helper "
                    + handler + "; do not implement gameplay state or declare "
                    "new members in this registration method."
                ),
                "host_required_behavior": behavior_id,
                "host_behavior_handler": handler,
                "verified_event_callback": bound_events.get(behavior_id),
            })
    declarations: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, unit in enumerate(units, 1):
        if host_mode:
            # The inspected classpath and materialized sources are already in
            # the original host message. Duplicating **task here previously
            # made a *single scalar metadata* request exceed 69K input chars,
            # distracting the 9B coder from its own kind/name contract.
            # Retain the full evidence once; only the task-specific metadata
            # is added here. No feature requirements or verified APIs are cut.
            unit_task = {
                "target_path": task["target_path"],
                "side": task["side"],
                "class_declaration": declaration,
                "all_units": units,
                "current_unit": unit,
                "current_unit_purpose": unit["purpose"],
                "previous_declaration_metadata": declarations,
            }
            behavior_id = _owned_behavior_id(unit)
            selected_behavior = next((
                dict(item) for item in behaviors
                if item.get("behavior_id") == behavior_id
            ), None)
            metadata_context = _scope_host_implementation_messages(
                messages, behavior=selected_behavior, purpose=unit["purpose"])
            if source_work_graph is not None and behavior_id:
                from .feature_source_work_graph import member_source_work
                unit_task["source_owned_work"] = member_source_work(
                    source_work_graph, behavior_id, member_role=(
                        "register_callback" if unit.get("host_required_behavior")
                        else "implement_callback_effects"))
        else:
            unit_task = {
                **task, "imports": imports,
                "class_declaration": declaration,
                "all_units": units,
                "current_unit": unit,
                "previous_declaration_metadata": declarations,
            }
            metadata_context = list(messages)
        metadata_messages = [
            *metadata_context,
            {
                "role": "system",
                "content": (
                    "Supply only typed declaration metadata for current_unit. The host "
                    "owns its kind and exact name. Types are Java types, parameters are "
                    "type/name records, and annotations are annotation names with optional "
                    "arguments. Do not place a complete declaration or class in a scalar. "
                    "Use implementation_requirements, side and verified prerequisite APIs. "
                    "Preserve exact override names for nested API implementations. "
                    "For a nested type supply typed field/constructor/method metadata. "
                    "Implementation is requested only after every signature is available."
                ),
            },
            {"role": "user", "content": json.dumps(unit_task, ensure_ascii=False, default=str)},
        ]
        kind, member_name = unit["kind"], unit["name"]
        nested_type = kind in {"class", "record", "enum"}
        event_callback = unit.get("verified_event_callback")
        if host_mode and unit.get("host_required_behavior"):
            # Lifecycle registration has no parameters, but the actual Event
            # registration will be composed from a verified SAM below.
            parts = {"visibility": "private", "static": False,
                     "return_type": "void", "type_parameters": "",
                     "parameters": [], "throws": [], "annotations": []}
        elif host_mode and _owned_behavior_id(unit) and event_callback:
            # A callback's inputs/return value belong to Fabric's actual SAM.
            # Never ask a 9B model to invent 'player' or other locals in a
            # no-argument lifecycle method.
            from .java_operation_ir import KEYWORDS
            names = []
            used = set()
            for index_arg, type_name in enumerate(event_callback["parameters"]):
                simple = type_name.rsplit(".", 1)[-1].replace("[]", "Array")
                name_arg = (simple[:1].lower() + simple[1:]
                            if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", simple)
                            and simple[:1].lower() + simple[1:] not in KEYWORDS
                            else f"eventArg{index_arg}")
                if event_callback.get("provenance") == "host_typed_ui_action":
                    # The host's form ABI owns input names as well as types.
                    # Two strings with different meanings cannot collapse to
                    # anonymous callback-local names.
                    name_arg = event_callback["parameter_names"][index_arg]
                while name_arg in used:
                    name_arg += str(index_arg)
                used.add(name_arg)
                names.append({"type": type_name, "name": name_arg})
            parts = {"visibility": "private", "static": False,
                     "return_type": ("boolean" if unit.get("host_condition_behavior")
                                     else "boolean" if unit.get("host_effect_behavior")
                                     else event_callback["return_type"]),
                     "type_parameters": ", ".join(event_callback.get("type_parameters") or []),
                     "parameters": names, "throws": list(event_callback.get("throws") or []),
                     "annotations": []}
        else:
            member_schema = metadata_schema(kind, member_name)
            if host_mode and unit.get("host_handler_behavior"):
                # Runtime effect handlers are commands. The host, not the 9B
                # model, decides their result type. Parameter names/types remain
                # model-authored because the verified event API determines them.
                member_schema["properties"]["return_type"] = {"const": "void"}
            parts = dict(_one_structured_action(
                router,
                metadata_messages,
                name=f"submit_java_declaration_{index}",
                schema=member_schema,
                page_structure=nested_type,
            ))
        parts = canonicalize_field_type_metadata(kind, member_name, parts)
        validate_declaration_metadata(kind, member_name, parts, owner=class_name)
        declarations.append((unit, parts))

    # All method/field signatures are fixed before any body is authored. Forward
    # calls therefore use actual parameter/return types rather than guessed APIs.
    signature_index: list[dict[str, str]] = []
    owner_types: dict[str, dict[str, Any]] = {
        class_name: {"kind": "class", "name": class_name, "declaration": declaration},
    }
    def index_declaration(kind, name, parts, owner):
        header = declaration_header(kind, name, parts)
        signature_index.append({"owner": owner, "kind": kind, "name": name,
                                "signature": header})
        if kind in {"class", "record", "enum"}:
            owner_types[name] = {"kind": kind, "name": name, "declaration": header}
            if kind == "enum":
                owner_types[name]["constants"] = deepcopy(parts["constants"])
                for constant in parts["constants"]:
                    signature_index.append({"owner": name, "kind": "enum_constant",
                                            "name": constant["name"], "signature": constant["name"]})
            elif kind == "record":
                owner_types[name]["components"] = deepcopy(parts["components"])
            for child in parts["fields"]:
                index_declaration("field", child["name"], child, name)
            for child in parts["constructors"]:
                index_declaration("constructor", name, child, name)
            for child in parts["methods"]:
                index_declaration("method", child["name"], child, name)
    for unit, parts in declarations:
        index_declaration(unit["kind"], unit["name"], parts, class_name)

    finished: list[str] = []
    def implement(kind, name, parts, *, owner, action, purpose):
        nonlocal callback_api_index
        if kind in {"class", "record", "enum"}:
            for index, child in enumerate(parts["fields"], 1):
                implement("field", child["name"], child, owner=name, action=f"{action}_field_{index}", purpose=purpose)
            for index, child in enumerate(parts["constructors"], 1):
                implement("constructor", name, child, owner=name, action=f"{action}_constructor_{index}", purpose=purpose)
            for index, child in enumerate(parts["methods"], 1):
                implement("method", child["name"], child, owner=name, action=f"{action}_method_{index}", purpose=purpose)
            return
        # No static initializer blocks are authored by this protocol. A final
        # field needs an expression unless an instance constructor can assign it.
        allow_uninitialized = not parts.get("final") or (
            not parts.get("static") and any(
                member["kind"] == "constructor" and member["owner"] == owner
                for member in signature_index
            )
        )
        frame = declaration_frame(kind, name, parts, allow_uninitialized=allow_uninitialized)
        if host_mode and (construction_context or {}).get("host_state_access"):
            frame["host_state_access"] = construction_context["host_state_access"]
        host_unit = next((entry for entry in units if entry["name"] == name), {})
        unit_event = next((entry.get("verified_event_callback")
                           for entry in units if entry["name"] == name), None)
        if host_mode and unit_event and name.startswith("registerBehavior"):
            handler = next(entry["host_behavior_handler"] for entry in units
                           if entry["name"] == name)
            statement = (unit_event["event_owner"] + "."
                         + unit_event["event_field"] + ".register(this::" + handler + ");")
            parts["source"] = frame["prefix"] + statement + frame["suffix"]
            return
        if host_mode and host_unit.get("host_condition_method"):
            from .host_callback_dispatch import (
                render_callback_dispatch, select_callback_results,
            )
            arguments = ", ".join(param["name"] for param in parts["parameters"])
            results = select_callback_results(
                router, unit_event, (construction_context or {}).get("symbols") or {},
            )
            statement = render_callback_dispatch(
                host_unit["host_condition_method"], host_unit["host_effect_method"],
                arguments, results,
                observers=host_unit.get("host_observer_methods", ()),
            )
            parts["source"] = frame["prefix"] + statement + frame["suffix"]
            return
        # The surrounding user message already contains the full inspected API
        # evidence and materialized project source. Re-sending **task duplicates
        # the whole design/context on every body action (observed at 69K chars
        # for a single registrar), exhausting the 9B model before one edit.
        selected_host_unit = next((
            unit for unit in units
            if unit["kind"] == kind and unit["name"] == name
        ), {})
        owned_behavior_id = _owned_behavior_id(selected_host_unit)
        owned_behavior = next((
            dict(item) for item in behaviors
            if item.get("behavior_id") == owned_behavior_id
        ), None)
        if host_mode and owned_behavior is not None:
            # Bind the immutable behavior's own state identifiers to the AST
            # admission gate. Compiler-valid writes to another behavior's
            # storage are never an acceptable syntax "repair".
            frame["host_behavior_contract"] = owned_behavior
        body_task = {
            "target_path": task["target_path"],
            "class": class_name,
            "requirement": purpose if (
                host_mode and owned_behavior_id
            ) else task["requirement"],
            "side": task["side"],
            "implementation_requirements": task["implementation_requirements"],
            "member_role": (
                "register_callback" if selected_host_unit.get("host_required_behavior")
                else "test_callback_condition" if selected_host_unit.get("host_condition_behavior")
                else "implement_callback_effects" if (selected_host_unit.get("host_handler_behavior")
                                                      or selected_host_unit.get("host_effect_behavior"))
                else "supporting_member"
            ),
            "verified_event_callback": selected_host_unit.get("verified_event_callback"),
            "handler_signature": next((
                item["signature"] for item in signature_index
                if item["name"] == selected_host_unit.get("host_behavior_handler")
            ), None),
            "current_unit_purpose": purpose,
            "observed_source_name": selected_host_unit.get("host_observer_source_name"),
            "observed_source_task_id": selected_host_unit.get("host_observer_source_task_id"),
            "assigned_behavior": owned_behavior,
            "runtime_state_observations": {
                key: value for key, value in json.loads(
                    bindings.get("runtime_state_bindings_json") or "{}").items()
                if owned_behavior is not None and key in {
                    *owned_behavior.get("runtime_state_reads", ()),
                    *owned_behavior.get("runtime_state_writes", ()),
                }
            },
            "class_declaration": declaration,
            "owner_type": owner_types[owner],
            "owner": owner,
            "declaration_metadata": parts,
            "host_signature": declaration_header(kind, name, parts),
            "response_source_frame": frame,
            "declared_member_api": signature_index,
            "host_state_work_order_scope": (
                (construction_context or {}).get("host_state_work_order_scope")
                if host_mode else None
            ),
            # Source already generated for a different member is context, not
            # an invitation to generate another complete compilation unit.
            "previous_complete_members": finished[-3:],
        }
        if source_work_graph is not None and owned_behavior_id:
            from .feature_source_work_graph import member_source_work
            body_task["source_owned_work"] = member_source_work(
                source_work_graph, owned_behavior_id,
                member_role=body_task["member_role"])
        if host_mode and unit_event:
            # Free-text gameplay retrieval can have no target matches. The
            # selected real callback still provides exact receiver and return
            # types; their checked callable APIs must reach the body writer.
            if callback_api_index is None:
                from .construction_file_api_evidence import FileAPIEvidence
                callback_api_index = FileAPIEvidence(
                    construction_context.get("symbols") or {},
                    target_path=task["target_path"], owner=package_name + "." + class_name,
                )
            # Include the inspected callback result owner as well as inputs.
            # Result enums/constants are required to compose exact SAM returns
            # even if unrelated to the free-text gameplay trigger.
            callback_types = [*unit_event.get("parameters", ()),
                              unit_event.get("return_type", "")]
            body_task["callback_input_api"] = callback_api_index.callback_member_records(
                callback_types,
                intent=purpose,
                predicate=bool(selected_host_unit.get("host_condition_behavior")),
            )
        if not host_mode:
            # Without the host's initial evidence payload this is the only
            # transport of prerequisite sources.
            body_task["prerequisite_source"] = source_context
        elif not messages:
            body_task["assigned_behaviors"] = [dict(item) for item in behaviors]
        member_messages = (
            _scope_host_implementation_messages(
                messages, behavior=owned_behavior, purpose=purpose,
                for_body=True)
            if host_mode else list(messages)
        )
        body_messages = [
            *member_messages,
            {"role": "system", "content": (
                "Return JSON with one implementation string containing only the field "
                "initializer or method/constructor body. The host fixes the signature "
                "and adds all declaration delimiters. Do not repeat the signature. "
                "An empty string may select an approved uninitialized field. "
                "Never return package/import directives, an outer class or extra members. "
                "Respect implementation_requirements, "
                "side, current_unit_purpose and actual prerequisite/member APIs. "
                "For shared state use only host_state_access typed readState_/writeState_ "
                "methods; the host owns state keys, value casts and context signatures. "
                "An undeclared work-order state is NOT a Java accessor: use only "
                "verified callback/world inputs when that preserves behavior, never "
                "substitute an unrelated state field. Persistent state absent from "
                "the canonical plan must be resolved upstream, not fabricated in Java. "
                "declared_member_api contains the exact signatures of all helpers, "
                "including forward references. Implement the requested behavior "
                "without placeholders or TODOs."
            )},
            {"role": "user", "content": json.dumps(body_task, ensure_ascii=False, default=str)},
        ]
        if host_mode and unit_event and owned_behavior_id and kind == "method":
            # The small coder chooses semantic operations and expression
            # values. The host compiles all statement/control-flow syntax;
            # effects can never emit a callback result or return statement.
            from .host_callback_body import author_callback_body

            parts["source"] = author_callback_body(
                router, body_messages, data_name=action, frame=frame,
                predicate=bool(selected_host_unit.get("host_condition_behavior")),
                effect_receipt=bool(selected_host_unit.get("host_effect_behavior")),
            )
        else:
            parts["source"] = router.generate_java_source("coder", body_messages,
                data_name=action, source_frame=frame)
    for index, (unit, parts) in enumerate(declarations, 1):
        kind, member_name = unit["kind"], unit["name"]
        implement(kind, member_name, parts, owner=class_name,
                  action=f"submit_java_implementation_{index}", purpose=unit["purpose"])
        finished.append(render_declaration(kind, member_name, parts, owner=class_name))

    if entrypoint_interface:
        # The lifecycle body is generated after helpers exist, against the exact
        # assigned COMMON/CLIENT contract. Neither its name nor signature is AI data.
        lifecycle = {"visibility": "public", "static": False, "return_type": "void",
                     "parameters": [], "throws": [], "annotations": ["Override"],
                     "type_parameters": ""}
        frame = declaration_frame("method", entrypoint_method, lifecycle)
        if host_mode and (construction_context or {}).get("host_state_access"):
            frame["host_state_access"] = construction_context["host_state_access"]
        lifecycle_messages = [
            *messages,
            {"role": "system", "content": (
                "Return JSON with one implementation string containing only the "
                "initialization method body. The host supplies its signature and "
                "outer braces. Complete its body without repeating the signature. "
                "Register and implement the exact original feature using the "
                "verified APIs and completed helper declarations. Respect the assigned "
                "side and implementation_requirements; retain existing typed behavior "
                "without duplicating its effects. Return no package/import directives, "
                "class declaration or extra method."
            )},
            {"role": "user", "content": json.dumps({
                "requirement": task["requirement"], "side": task["side"],
                "target_path": task["target_path"], "imports": imports,
                "host_signature": declaration_header("method", entrypoint_method, lifecycle),
                "response_source_frame": frame,
                "complete_members": finished[-6:]}, ensure_ascii=False, default=str)},
        ]
        if host_mode and bindings.get("request_feature_entrypoint"):
            calls = [unit["name"] for unit in units if unit.get("host_required_behavior")]
            lifecycle["source"] = frame["prefix"] + "".join(
                "    this." + name + "();\n" for name in calls
            ).rstrip("\n") + frame["suffix"]
        else:
            lifecycle["source"] = router.generate_java_source("coder", lifecycle_messages,
                data_name="submit_java_entrypoint_source", source_frame=frame)
        finished.append(render_declaration("method", entrypoint_method, lifecycle, owner=class_name))

    source = (
        f"package {package_name};\n\n"
        + (imports + "\n\n" if imports else "")
        + declaration + "\n\n"
        + "\n\n".join(finished)
        + "\n}\n"
    )
    if host_mode:
        from .host_java_imports import host_add_verified_imports
        source = host_add_verified_imports(
            source, symbols=dict(construction_context.get("symbols") or {}),
            owner=package_name + "." + class_name,
            target_path=str(spec["target_path"]))
    if pending_custom_source_sites:
        from .source_owned_custom_event import materialize_verified_custom_event
        for site in pending_custom_source_sites:
            materialize_verified_custom_event(**site)
    if pending_ui_source_plans:
        from .source_owned_ui_event import materialize_typed_ui_occurrence
        for planned in pending_ui_source_plans:
            materialize_typed_ui_occurrence(
                planned, project_root=construction_context["project_root"],
                symbols=construction_context.get("symbols") or {})
    return source
