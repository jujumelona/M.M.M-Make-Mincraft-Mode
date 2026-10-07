"""Lower a saved Typed PlanIR design through deterministic host production."""

from __future__ import annotations

import hashlib
from copy import deepcopy
import json
from collections.abc import Iterable, Mapping
from typing import Any

from .authored_plan import AuthoredPlan
from .complete_spec import (
    AssetRequest,
    CompleteProposal,
    ProductionModule,
    complete_proposal_from_parts,
)
from .host_target_binding import bind_existing_project, bind_platform
from .spec import ModSpec, Proposal, ProposalStatus
from .spec_identity import canonical_spec_id
from .platform_backend_contract import deterministic_backend_capabilities
from .target_contract import target_coordinates_from_mapping
from .typed_host_generation_contract import (
    normalize_typed_network_sync_config,
    normalize_typed_resource_policy_config,
    normalize_typed_state_store_config,
)

_TARGET_KEYS = ("minecraft_version", "loader", "mappings")
_AUTHORED_EXECUTION_SCHEMA = "mmm/authored-execution-manifest-v2"

def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _main_class_name(mod_id: str) -> str:
    """Match the canonical Fabric template provider's host-owned entrypoint name."""

    return "".join(part.capitalize() for part in str(mod_id).split("_")) + "Mod"

_CANDIDATE_ARTIFACT_EXECUTOR_TYPES = frozenset({
    "deterministic_renderer",
    "deterministic",
    "template",
})


def _fact_artifact_route_ready(fact: Any, version_context: Any) -> bool:
    """Return whether a registered canonical leaf can produce a build candidate.

    not_reviewed is deliberately not production admission. It is, however, a
    valid candidate-generation state when the registered implementation is a
    package-owned Python generator (or an evidence-free deterministic renderer).
    The build/JDT/runtime stages remain responsible for proving that candidate.
    """

    from .artifact_expansion import FACT_TO_CANONICAL_LEAVES
    from .prompt_fact_types import FactType
    from .registered_leaf_binding import require_registered_leaf_binding

    # CONTENT_RELATION is runtime/content topology, not a file-generating
    # artifact fact and not production scheduling. Its executable meaning is
    # lowered into module config/executable_relations by authored content planning.
    # Treating it as an item integration leaf manufactures bogus initializer jobs for
    # GUI/entity/etc. modules and cross-wires their typed ports.
    if fact.fact_type == FactType.CONTENT_RELATION:
        return True

    leaves = FACT_TO_CANONICAL_LEAVES.get(fact.fact_type)
    if not leaves:
        return False
    for leaf in leaves:
        binding = require_registered_leaf_binding(version_context, leaf)
        if binding.get("state") == "admitted":
            continue
        implementation = binding.get("implementation")
        if not isinstance(implementation, Mapping):
            return False
        implementation_id = str(implementation.get("implementation_id") or "")
        executor_type = str(implementation.get("executor_type") or "")
        if implementation_id.startswith("python_generator:"):
            continue
        if executor_type in _CANDIDATE_ARTIFACT_EXECUTOR_TYPES:
            continue
        return False
    return True


def _compile_new_authored_modules(
    plan: AuthoredPlan,
    *,
    mod_id: str,
    package_name: str,
    target: Mapping[str, Any],
    production_state_section: Mapping[str, Any] | None = None,
    deterministic_module_kinds: Iterable[str] | None = None,
) -> tuple[tuple[ProductionModule, ...], dict[str, Any]]:
    """Lower the persisted Typed PlanIR through deterministic host backends only."""
    main_symbol = _main_class_name(mod_id)
    main_path = f"src/main/java/{package_name.replace('.', '/')}/{main_symbol}.java"

    if not getattr(plan, "typed_plan_ir", {}):
        raise ValueError(
            "TYPED_PLAN_REQUIRED: untyped authored production has been removed."
        )
    from .typed_plan_ir import (
        typed_plan_capability_ids,
        typed_plan_uses_state,
        validate_typed_plan_ir,
    )
    from .typed_platform_ir import (
        PLATFORM_HOST_KINDS,
        network_sync_requires_state,
    )
    from .typed_plan_support import assert_typed_plan_host_support

    source_sha = "sha256:" + hashlib.sha256(plan.text.encode("utf-8")).hexdigest()
    stored_sha = str(plan.typed_plan_ir.get("source_sha256") or "")
    normalized_stored_sha = (
        stored_sha if stored_sha.startswith("sha256:") else "sha256:" + stored_sha
    )
    if normalized_stored_sha != source_sha:
        raise ValueError("TYPED_PLAN_SOURCE_HASH_MISMATCH")

    from .typed_host_capabilities import (
        typed_host_capability_contracts,
    )

    capability_contracts = typed_host_capability_contracts()
    validated_plan = validate_typed_plan_ir(
        plan.typed_plan_ir,
        capabilities=capability_contracts,
    )
    capability_ids = typed_plan_capability_ids(validated_plan)
    bound_capabilities = {
        capability_id: deepcopy(capability_contracts[capability_id])
        for capability_id in capability_ids
    }
    from .authored_content_contract import content_owned_refs

    assert_typed_plan_host_support(
        plan.structured_sections,
        validated_plan,
        externally_covered_refs=content_owned_refs(
            plan.structured_sections,
            getattr(plan, "content_design", {}),
        ),
    )

    raw_platform_modules = validated_plan.get("platform_modules", [])
    target_deterministic_kinds = (
        frozenset(
            str(kind).strip()
            for kind in deterministic_module_kinds
            if str(kind).strip()
        )
        if deterministic_module_kinds is not None
        else None
    )
    from .platform_backend_contract import missing_production_backend_capabilities

    platform_modules: list[ProductionModule] = []
    state_store_config: dict[str, Any] | None = None
    network_sync_config: dict[str, Any] | None = None
    resource_policy_config: dict[str, Any] | None = None
    for item in raw_platform_modules:
        module_id = str(item["module_id"])
        kind = str(item["kind"])
        if module_id == "authored_typed_plan":
            raise ValueError(
                "TYPED_PLATFORM_MODULE_ID_CONFLICT: authored_typed_plan"
            )
        if kind == "state_store":
            if state_store_config is not None:
                raise ValueError(
                    "TYPED_PLATFORM_STATE_STORE_DUPLICATE"
                )
            state_store_config = normalize_typed_state_store_config(
                item["config"],
                structured_sections=plan.structured_sections,
                state_section=(
                    production_state_section
                    if isinstance(production_state_section, Mapping)
                    else None
                ),
                covers=item["covers"],
            )
            continue
        if kind == "network_sync":
            if network_sync_config is not None:
                raise ValueError("TYPED_PLATFORM_NETWORK_SYNC_DUPLICATE")
            network_sync_config = normalize_typed_network_sync_config(
                item["config"],
                covers=item["covers"],
            )
            continue
        if kind == "resource_policy":
            if resource_policy_config is not None:
                raise ValueError("TYPED_PLATFORM_RESOURCE_POLICY_DUPLICATE")
            resource_policy_config = normalize_typed_resource_policy_config(
                item["config"],
            )
            continue
        if target_deterministic_kinds is not None:
            missing_backend = missing_production_backend_capabilities(
                target_deterministic_kinds,
                kind,
                item["config"],
            )
            if missing_backend:
                raise ValueError(
                    "TYPED_PLATFORM_DETERMINISTIC_BACKEND_REQUIRED: "
                    f"{kind!r} requires missing backend capabilities "
                    f"{sorted(missing_backend)}"
                )
        platform_modules.append(
            ProductionModule(
                module_id=module_id,
                kind=kind,
                config=deepcopy(dict(item["config"])),
                required_gates=("target_compile",),
            )
        )

    from .authored_structured_design import active_concern_records

    state_section = (
        deepcopy(dict(production_state_section))
        if isinstance(production_state_section, Mapping)
        else {}
    )
    active_state = active_concern_records(
        plan.structured_sections,
        "state_model",
    )
    network_sync_covers = set(
        network_sync_config.get("__covers", ())
        if isinstance(network_sync_config, Mapping)
        else ()
    )
    network_sync_needs_state = network_sync_requires_state(
        tuple(network_sync_covers)
    )
    state_required = (
        typed_plan_uses_state(validated_plan)
        or state_store_config is not None
        or network_sync_needs_state
        or any(bool(rows) for rows in active_state.values())
    )
    if state_required and not state_section:
        raise ValueError(
            "TYPED_PLAN_STATE_AUTHORITY_REQUIRED: active state semantics, "
            "state operations, or persistent state require canonical "
            "structured state_model authority."
        )

    program_symbol = "AuthoredProgram"
    program_path = (
        f"src/main/java/{package_name.replace('.', '/')}/{program_symbol}.java"
    )
    task_id = "authored_typed_plan"
    module = ProductionModule(
        module_id=task_id,
        kind="typed_host",
        config={
            "typed_plan_ir": deepcopy(validated_plan),
            "typed_plan_package": package_name,
            "typed_plan_path": program_path,
            "typed_plan_capabilities": deepcopy(bound_capabilities),
            "typed_plan_state_section": state_section,
            "typed_plan_structured_sections": deepcopy(plan.structured_sections),
            "typed_state_store": state_store_config,
            "typed_network_sync": network_sync_config,
            "typed_resource_policy": resource_policy_config,
            **dict(target),
        },
        required_gates=("target_compile",),
    )
    from .typed_plan_production import (
        validate_typed_plan_generation_contract,
    )

    typed_host_contract = validate_typed_plan_generation_contract(
        module,
        package_name=package_name,
        mod_id=mod_id,
    )
    manifest = {
        "schema_version": _AUTHORED_EXECUTION_SCHEMA,
        "policy": "host_typed_plan_ir",
        "source_text_sha256": source_sha,
        "source_bytes": len(plan.text.encode("utf-8")),
        "unit_count": 1,
        "units": [{
            "module_id": task_id,
            "path": program_path,
            "symbol": program_symbol,
            "source_sha256": source_sha,
        }],
        "graph_status": "not_required",
        "typed_program": {
            "path": program_path,
            "symbol": program_symbol,
            "dry_compile": dict(typed_host_contract),
            "state_required": state_required,
            "state_store": state_store_config is not None,
            "network_sync": network_sync_config is not None,
            "resource_policy": resource_policy_config is not None,
            "capabilities": list(capability_ids),
        },
        "platform_modules": [
            {
                "module_id": str(item["module_id"]),
                "kind": str(item["kind"]),
                "covers": list(item["covers"]),
            }
            for item in raw_platform_modules
            if str(item["kind"]) not in PLATFORM_HOST_KINDS
        ],
        "entrypoint": {
            "owner": "host_scaffold",
            "path": main_path,
            "symbol": main_symbol,
            "feature_symbols": [program_symbol],
        },
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    return (module, *platform_modules), manifest


def materialize_authored_execution_scaffold(
    proposal: CompleteProposal,
    project_root: Any,
) -> Any:
    """Materialize the deterministic Typed PlanIR scaffold only."""

    from pathlib import Path

    root = Path(project_root).expanduser().resolve()
    game_design = getattr(proposal, "game_design", None)
    design = game_design if isinstance(game_design, Mapping) else {}
    manifest = design.get("_authored_execution_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("AUTHORED_TYPED_EXECUTION_MANIFEST_REQUIRED")
    if manifest.get("schema_version") != _AUTHORED_EXECUTION_SCHEMA:
        raise ValueError("AUTHORED_SCAFFOLD_SCHEMA_MISMATCH")
    policy = str(manifest.get("policy") or "")
    if policy != "host_typed_plan_ir":
        raise ValueError(
            "AUTHORED_SCAFFOLD_POLICY_MISMATCH: only Typed PlanIR is supported"
        )

    expected_manifest = dict(manifest)
    supplied_digest = str(expected_manifest.pop("manifest_sha256", "") or "")
    if supplied_digest != _sha256_json(expected_manifest):
        raise ValueError("AUTHORED_SCAFFOLD_MANIFEST_HASH_MISMATCH")

    from .production_state_compiler import render_production_state_java
    from .project_edit import (
        ensure_main_initializer_call,
        inspect_fabric_project,
        write_text_files,
    )

    info = inspect_fabric_project(root)
    package_name = proposal.base_proposal.spec.package_name
    if info.package_name != package_name:
        raise ValueError(
            "AUTHORED_TYPED_PACKAGE_MISMATCH: "
            f"{info.package_name!r} != {package_name!r}"
        )

    typed_program = manifest.get("typed_program")
    if not isinstance(typed_program, Mapping):
        raise ValueError("AUTHORED_TYPED_PROGRAM_MANIFEST_MISSING")
    program_path = str(typed_program.get("path") or "").replace("\\", "/").strip()
    expected_program_path = (
        f"src/main/java/{package_name.replace('.', '/')}/AuthoredProgram.java"
    )
    if program_path != expected_program_path:
        raise ValueError("AUTHORED_TYPED_PROGRAM_PATH_DRIFT")

    program_target = root / program_path
    if program_target.exists():
        if not program_target.is_file() or program_target.is_symlink():
            raise ValueError("AUTHORED_TYPED_PROGRAM_TARGET_INVALID")
        current_program = program_target.read_text(encoding="utf-8")
        if "// MMM:TYPED_PLAN_OWNER" not in current_program:
            raise ValueError("AUTHORED_TYPED_PROGRAM_OWNERSHIP_CONFLICT")
    else:
        placeholder = (
            f"package {package_name};\n\n"
            "// MMM:TYPED_PLAN_OWNER\n"
            "public final class AuthoredProgram {\n"
            "    private AuthoredProgram() {}\n"
            "    public static void initialize() {}\n"
            "}\n"
        )
        write_text_files(
            info,
            {program_path: placeholder},
            replace_existing=False,
        )

    if bool(typed_program.get("state_required")):
        raw_state = design.get("_production_state_section")
        if not isinstance(raw_state, Mapping) or not raw_state:
            raise ValueError("AUTHORED_TYPED_STATE_AUTHORITY_MISSING")
        state_path = (
            f"src/main/java/{package_name.replace('.', '/')}/"
            "AuthoredStateModel.java"
        )
        state_target = root / state_path
        replace_state = False
        if state_target.exists():
            if not state_target.is_file() or state_target.is_symlink():
                raise ValueError("AUTHORED_TYPED_STATE_TARGET_INVALID")
            current_state = state_target.read_text(encoding="utf-8")
            if "// MMM:TYPED_PLAN_STATE_OWNER" not in current_state:
                raise ValueError("AUTHORED_TYPED_STATE_OWNERSHIP_CONFLICT")
            replace_state = True
        state_source = render_production_state_java(
            raw_state,
            package_name=package_name,
        )
        write_text_files(
            info,
            {state_path: state_source},
            replace_existing=replace_state,
        )

    ensure_main_initializer_call(
        info,
        import_line=f"import {package_name}.AuthoredProgram",
        call_line="AuthoredProgram.initialize()",
        marker="typed-plan",
    )
    return root


def _bound_target(design: Mapping[str, Any]) -> dict[str, str]:
    """Return the current host-selected platform target only."""

    selection = design.get("_platform_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("TYPED_PLATFORM_SELECTION_REQUIRED")
    target = selection.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("TYPED_PLATFORM_SELECTION_TARGET_REQUIRED")
    coordinates = target_coordinates_from_mapping(target)
    return {key: getattr(coordinates, key) for key in _TARGET_KEYS}


def _normalize_content_build_dependencies(
    modules: tuple[ProductionModule, ...],
    facts: tuple[Any, ...],
) -> tuple[ProductionModule, ...]:
    """Migrate saved content plans onto the semantic/build dependency boundary.

    content_design is host-authored semantic topology. Older saved plans projected
    gameplay relations (including requires, drops, recipe/tag membership and
    unlock-style edges) into ProductionModule.depends_on. Those edges are not
    source-generation prerequisites and can form valid gameplay cycles. Strip them
    before the CompleteProposal production DAG is validated so resume can consume
    an already-saved plan without replanning.
    """

    # Keep the persisted-facts parameter in this boundary for backward-compatible
    # callers. The migration is intentionally fact-independent: content_design
    # never owns production scheduling edges, regardless of relation vocabulary.
    _ = facts
    from .content_design_graph import _strip_semantic_content_build_dependencies

    return _strip_semantic_content_build_dependencies(modules)

def _canonicalize_saved_content_asset_id(value: Any, index: int) -> str:
    """Map legacy host-authored asset IDs onto the shared proposal ID contract."""

    return canonical_spec_id(
        str(value or ""),
        fallback=f"asset_{index}",
    )


def _defer_unbound_structured_content_assets(
    modules: tuple[ProductionModule, ...],
    assets: tuple[AssetRequest, ...],
    version_context: Any,
) -> tuple[tuple[ProductionModule, ...], tuple[AssetRequest, ...]]:
    """Migrate saved semantic plans that predate structured HOST asset ownership.

    gui.sprite and entity.fixed_uv cannot be synthesized from semantic colour/style
    hints alone. They require exact HOST geometry plus a concrete consumer binding.
    Older authored plans nevertheless emitted those requests unconditionally. When
    the target catalog has no binding at all for that subject, defer the asset and
    preserve its visual intent on the owning module so canonical source generation
    can still use the semantics. A present-but-invalid binding is retained and will
    fail closed in the normal resource contract validator.
    """

    from .resource_catalog import host_binding

    structured_render_kinds = {"gui.sprite", "entity.fixed_uv"}
    deferred_visuals: dict[str, dict[str, Any]] = {}
    kept: list[AssetRequest] = []

    for asset in assets:
        if asset.render_kind not in structured_render_kinds:
            kept.append(asset)
            continue
        binding = host_binding(version_context, asset.subject_id)
        if binding:
            # Do not hide malformed/partial HOST authority. The resource contract
            # owns validation once any binding is present.
            kept.append(asset)
            continue

        owner_id = str(asset.owner_module_id or asset.subject_id)
        if isinstance(asset.visual_spec, Mapping):
            deferred_visuals[owner_id] = deepcopy(dict(asset.visual_spec))

    if not deferred_visuals:
        return modules, tuple(kept)

    rewritten: list[ProductionModule] = []
    for module in modules:
        visual_spec = deferred_visuals.get(module.module_id)
        if visual_spec is None:
            rewritten.append(module)
            continue
        config = deepcopy(module.config)
        config.setdefault("visual_spec", visual_spec)
        updated = ProductionModule(
            module_id=module.module_id,
            kind=module.kind,
            config=config,
            depends_on=module.depends_on,
            required_gates=module.required_gates,
        )
        updated.validate()
        rewritten.append(updated)
    return tuple(rewritten), tuple(kept)


def _compile_content_artifact_graph(
    plan: AuthoredPlan,
    *,
    adapter: Any,
    mod_id: str,
    package_name: str,
) -> tuple[tuple[ProductionModule, ...], tuple[AssetRequest, ...], tuple[Any, ...]]:
    """Restore canonical content facts and lower them through the artifact graph."""

    raw = getattr(plan, "content_design", {})
    if not isinstance(raw, Mapping) or not raw:
        return (), (), ()
    planned_mod_id = str(raw.get("_mod_id") or "").strip()
    if planned_mod_id and planned_mod_id != mod_id:
        raise ValueError(
            f"CONTENT_MOD_ID_MISMATCH: planned={planned_mod_id!r} production={mod_id!r}"
        )

    raw_modules = raw.get("modules", ())
    raw_assets = raw.get("assets", ())
    raw_facts = raw.get("_implementation_facts", ())
    for label, values in (
        ("modules", raw_modules),
        ("assets", raw_assets),
        ("_implementation_facts", raw_facts),
    ):
        if not isinstance(values, list):
            raise ValueError(f"CONTENT_DESIGN_INVALID: {label} must be an array")

    module_fields = {
        "module_id",
        "kind",
        "config",
        "depends_on",
        "required_gates",
    }
    parsed_modules: list[ProductionModule] = []
    for index, item in enumerate(raw_modules):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"CONTENT_DESIGN_INVALID: modules[{index}] must be an object"
            )
        if set(item) != module_fields:
            raise ValueError(
                f"CONTENT_MODULE_FIELDS_INVALID: modules[{index}] "
                f"missing={sorted(module_fields - set(item))}, "
                f"unknown={sorted(set(item) - module_fields)}"
            )
        if not isinstance(item["module_id"], str) or not isinstance(item["kind"], str):
            raise TypeError(
                f"CONTENT_MODULE_IDENTITY_INVALID: modules[{index}]"
            )
        if not isinstance(item["config"], Mapping):
            raise TypeError(
                f"CONTENT_MODULE_CONFIG_INVALID: modules[{index}]"
            )
        for field_name in ("depends_on", "required_gates"):
            values = item[field_name]
            if not isinstance(values, list) or any(
                not isinstance(value, str) for value in values
            ):
                raise TypeError(
                    f"CONTENT_MODULE_ARRAY_INVALID: modules[{index}].{field_name}"
                )
        module = ProductionModule(
            module_id=item["module_id"],
            kind=item["kind"],
            config=deepcopy(dict(item["config"])),
            depends_on=tuple(item["depends_on"]),
            required_gates=tuple(item["required_gates"]),
        )
        module.validate()
        parsed_modules.append(module)
    modules = tuple(parsed_modules)

    asset_fields = {
        "asset_id",
        "kind",
        "visual_description",
        "render_kind",
        "subject_id",
        "owner_module_id",
        "container",
        "requested_width",
        "requested_height",
        "variant_count",
        "visual_spec",
    }
    parsed_assets: list[AssetRequest] = []
    seen_asset_ids: set[str] = set()
    for index, item in enumerate(raw_assets):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"CONTENT_DESIGN_INVALID: assets[{index}] must be an object"
            )
        if set(item) != asset_fields:
            raise ValueError(
                f"CONTENT_ASSET_FIELDS_INVALID: assets[{index}] "
                f"missing={sorted(asset_fields - set(item))}, "
                f"unknown={sorted(set(item) - asset_fields)}"
            )

        # Authored content assets are host-generated IDs. Plans saved before the
        # shared identifier contract was introduced may contain an otherwise
        # valid semantic asset whose "texture_<kind>_" prefix pushed it beyond
        # CompleteProposal's 64-character ID bound. Canonicalize exactly at this
        # saved-plan production boundary so resume/build does not require replan.
        asset_payload = deepcopy(dict(item))
        asset_payload["asset_id"] = _canonicalize_saved_content_asset_id(
            asset_payload.get("asset_id"),
            index,
        )
        if asset_payload["asset_id"] in seen_asset_ids:
            raise ValueError(
                "CONTENT_ASSET_ID_COLLISION_AFTER_CANONICALIZATION: "
                f"{asset_payload['asset_id']}"
            )
        seen_asset_ids.add(asset_payload["asset_id"])
        try:
            asset = AssetRequest(**asset_payload)
        except TypeError as exc:
            raise TypeError(
                f"CONTENT_ASSET_SHAPE_INVALID: assets[{index}]"
            ) from exc
        asset.validate()
        parsed_assets.append(asset)
    assets = tuple(parsed_assets)
    modules, assets = _defer_unbound_structured_content_assets(
        modules,
        assets,
        adapter.version_context,
    )

    from .implementation_fact import ImplementationFact
    from .artifact_expansion import expand_facts_to_jobs

    parsed_facts: list[ImplementationFact] = []
    for index, item in enumerate(raw_facts):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"CONTENT_DESIGN_INVALID: _implementation_facts[{index}] must be an object"
            )
        parsed_facts.append(ImplementationFact.from_dict(dict(item)))
    facts = tuple(parsed_facts)
    if modules and not facts:
        raise ValueError("CONTENT_ARTIFACT_FACTS_REQUIRED")

    modules = _normalize_content_build_dependencies(modules, facts)

    from .artifact_expansion import FACT_TO_CANONICAL_LEAVES
    from .platform_backend_contract import (
        missing_production_backend_capabilities,
        native_production_route_available,
    )
    from .prompt_fact_types import FactType

    # Older saved plans conflated every BLOCK_ENTITY_EXISTS fact with the
    # processing-machine backend. Canonicalize only when the machine-specific
    # contract is absent; real processing machines keep their original kind.
    block_entity_subjects = {
        fact.subject
        for fact in facts
        if fact.fact_type == FactType.BLOCK_ENTITY_EXISTS
    }
    normalized_modules: list[ProductionModule] = []
    machine_fields = {"input_item", "output_item", "output_count", "processing_ticks"}
    for module in modules:
        if (
            module.kind == "machine"
            and module.module_id in block_entity_subjects
            and not machine_fields.issubset(module.config)
        ):
            module = ProductionModule(
                module_id=module.module_id,
                kind="block_entity",
                config=deepcopy(module.config),
                depends_on=module.depends_on,
                required_gates=module.required_gates,
            )
            module.validate()
        normalized_modules.append(module)
    modules = tuple(normalized_modules)

    facts_by_subject: dict[str, list[ImplementationFact]] = {}
    for fact in facts:
        facts_by_subject.setdefault(fact.subject, []).append(fact)

    available_native_capabilities = deterministic_backend_capabilities(adapter)
    native_owner_ids: set[str] = set()
    for module in modules:
        owned_facts = facts_by_subject.get(module.module_id, ())
        blocked = [
            fact
            for fact in owned_facts
            if not _fact_artifact_route_ready(fact, adapter.version_context)
        ]
        if not blocked:
            continue
        missing_native = missing_production_backend_capabilities(
            available_native_capabilities, module.kind, module.config,
        )
        if (
            native_production_route_available(module.kind, module.config)
            and not missing_native
        ):
            native_owner_ids.add(module.module_id)
            continue
        blocked_leaves = sorted({
            leaf
            for fact in blocked
            for leaf in FACT_TO_CANONICAL_LEAVES.get(fact.fact_type, ())
        })
        raise ValueError(
            "CONTENT_NO_EXECUTABLE_BACKEND: "
            f"{module.module_id}/{module.kind} has no admitted artifact route "
            f"and no reviewed native backend for {adapter.minecraft_version}; "
            f"blocked_leaves={blocked_leaves}; "
            f"missing_native_capabilities={sorted(missing_native)}. "
            "A registered generator is not target support. Preserve the saved "
            "content semantics; this target needs an executable artifact binding "
            "or a reviewed native implementation."
        )

    if native_owner_ids:
        from .extended_content_generator import validate_extended_module_contract
        from .platform_backend_contract import EXTENDED_CONTENT_KINDS, SYSTEM_KIND_TO_PACK
        from .system_pack_validation import validate_system_modules

        system_modules: dict[str, list[dict[str, Any]]] = {}
        for module in modules:
            if module.module_id not in native_owner_ids:
                continue
            if module.kind in EXTENDED_CONTENT_KINDS:
                try:
                    validate_extended_module_contract(module)
                except Exception as exc:
                    raise ValueError(
                        "CONTENT_NATIVE_BACKEND_CONTRACT_INVALID: "
                        f"{module.module_id}/{module.kind}: {exc}"
                    ) from exc
            else:
                pack_id = SYSTEM_KIND_TO_PACK.get(module.kind)
                if pack_id is not None:
                    system_modules.setdefault(pack_id, []).append({
                        "module_id": module.module_id,
                        "kind": module.kind,
                        "config": module.config,
                        "depends_on": list(module.depends_on),
                        "required_gates": list(module.required_gates),
                    })
        for pack_id, members in system_modules.items():
            try:
                validate_system_modules(pack_id, members)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "CONTENT_NATIVE_BACKEND_CONTRACT_INVALID: "
                    f"{pack_id}: {exc}"
                ) from exc

    artifact_facts = tuple(
        fact for fact in facts if fact.subject not in native_owner_ids
    )
    jobs = tuple(
        expand_facts_to_jobs(
            artifact_facts,
            mod_id=mod_id,
            package_name=package_name,
            main_class=_main_class_name(mod_id),
            minecraft_version=adapter.minecraft_version,
            version_context=adapter.version_context,
        )
    )

    # Generator candidates persist their full semantic input so resume never has
    # to reconstruct it from a later model call. Enrich the fact-level contract
    # with the canonical module configuration and the normalized hard build deps.
    module_by_id = {module.module_id: module for module in modules}
    from .task_template_catalog import load_template
    for job in jobs:
        canonical_inputs = job.deterministic_inputs.get("_canonical_inputs")
        if not isinstance(canonical_inputs, dict):
            continue
        manifest = load_template(job.canonical_leaf)
        specification_ports = [
            str(port.get("name") or "")
            for port in manifest.get("inputs", ())
            if port.get("type") == "specification"
        ]
        if len(specification_ports) != 1:
            raise ValueError(
                f"CANONICAL_GENERATOR_SPEC_PORT_INVALID: {job.canonical_leaf}"
            )
        spec_input = canonical_inputs.get(specification_ports[0])
        if not isinstance(spec_input, dict):
            raise ValueError(
                f"CANONICAL_GENERATOR_SPEC_INPUT_MISSING: {job.job_id}"
            )
        bindings = spec_input.get("bindings")
        if not isinstance(bindings, dict):
            raise ValueError(
                f"CANONICAL_GENERATOR_BINDINGS_INVALID: {job.job_id}"
            )
        owner = module_by_id.get(job.owner_module)
        if owner is None:
            raise ValueError(f"CONTENT_ARTIFACT_FOREIGN_OWNER: {job.owner_module}")
        bindings["module_config_json"] = json.dumps(
            owner.config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        bindings["build_dependencies_json"] = json.dumps(
            list(owner.depends_on),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        bindings["required_gates_json"] = json.dumps(
            list(owner.required_gates),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        bindings["target_contract_json"] = json.dumps(
            adapter.version_context.to_dict().get("target", {}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    module_ids = {module.module_id for module in modules}
    from .artifact_job import validate_artifact_job_graph

    validate_artifact_job_graph(jobs, module_ids=module_ids)
    if len(module_ids) != len(modules):
        raise ValueError("CONTENT_ARTIFACT_DUPLICATE_MODULE_ID")
    owner_ids = {str(job.owner_module).strip() for job in jobs if str(job.owner_module).strip()}
    foreign = sorted(owner_ids - module_ids)
    if foreign:
        raise ValueError(
            "CONTENT_ARTIFACT_FOREIGN_OWNER: " + ", ".join(foreign)
        )
    artifact_owned_module_ids = module_ids - native_owner_ids
    unowned = sorted(artifact_owned_module_ids - owner_ids)
    if unowned:
        raise ValueError(
            "CONTENT_ARTIFACT_OWNER_MISSING: " + ", ".join(unowned)
        )
    unexpected_native_jobs = sorted(native_owner_ids & owner_ids)
    if unexpected_native_jobs:
        raise ValueError(
            "CONTENT_GENERATION_OWNERSHIP_CONFLICT: "
            + ", ".join(unexpected_native_jobs)
        )
    return modules, assets, jobs

def compile_authored_design(
    router: Any, plan: AuthoredPlan, *, existing_input_sha256: str = ""
) -> CompleteProposal:
    from .authored_structured_design import normalize_structured_sections
    from .production_state_compiler import compile_production_state_section

    if not isinstance(plan, AuthoredPlan):
        raise TypeError("compile_authored_design requires AuthoredPlan")
    if not getattr(plan, "typed_plan_ir", {}):
        raise ValueError(
            "TYPED_PLAN_REQUIRED: AuthoredPlan must contain Typed PlanIR."
        )

    structured = normalize_structured_sections(plan.structured_sections)
    if not structured:
        raise ValueError(
            "TYPED_STRUCTURED_AUTHORITY_REQUIRED: canonical structured design is required."
        )
    if structured != plan.structured_sections:
        raise ValueError(
            "TYPED_STRUCTURED_AUTHORITY_NONCANONICAL: planning must persist canonical records."
        )

    effective_existing = str(
        existing_input_sha256 or plan.existing_input_sha256 or ""
    ).strip()

    content_design = (
        plan.content_design
        if isinstance(getattr(plan, "content_design", None), Mapping)
        else {}
    )
    content_mod_id = str(content_design.get("_mod_id") or "").strip()
    mod_id = content_mod_id or ("authored_" + plan.calculate_hash()[:12])
    acceptance = (
        "Implement the behaviors in the saved authored design and exercise them in Minecraft.",
        "Build the project and verify that the mod loads and runs without errors.",
    )
    base = Proposal(
        schema_version="minecraft-mod-ai/proposal-v1",
        proposal_version=1,
        status=ProposalStatus.AWAITING_APPROVAL,
        requested_prompt=plan.requested_prompt,
        spec=ModSpec(
            mod_id=mod_id,
            mod_name="Authored Minecraft Mod",
            package_name=f"ai.minecraft.generated.{mod_id}",
            version="1.0.0",
            summary=plan.requested_prompt,
            contents=(),
        ),
        assumptions=(),
        exclusions=(),
        deferred_requests=(),
        acceptance_tests=acceptance,
        evidence_sources=(),
    )

    design = {"authored_plan": plan.to_dict()}
    design = bind_existing_project(router, design)
    from .typed_platform_ir import PLATFORM_HOST_KINDS

    platform_module_kinds = tuple(
        dict.fromkeys(
            str(item.get("kind") or "").strip()
            for item in plan.typed_plan_ir.get("platform_modules", ())
            if isinstance(item, Mapping)
            and str(item.get("kind") or "").strip()
            and str(item.get("kind") or "").strip() not in PLATFORM_HOST_KINDS
        )
    )

    # Target binding owns Typed PlatformIR host modules only. Content modules are
    # resolved after a target exists because their canonical artifact route is
    # target-specific. A GUI fact may use a registered canonical candidate generator
    # even when the target intentionally does not advertise the legacy gui-networking
    # system-pack capability. _compile_content_artifact_graph remains fail-closed
    # when neither that artifact route nor a reviewed native fallback is executable.
    target_module_kinds = platform_module_kinds
    design, base = bind_platform(
        router,
        plan.requested_prompt,
        design,
        base,
        module_kinds=target_module_kinds,
    )
    from .platform_catalog import adapter_for_lock_values

    adapter = adapter_for_lock_values(base.spec.platform)
    target = _bound_target(design)
    design = {**design, **target}

    production_state_section = compile_production_state_section(plan)
    modules, manifest = _compile_new_authored_modules(
        plan,
        mod_id=base.spec.mod_id,
        package_name=base.spec.package_name,
        target=target,
        production_state_section=production_state_section,
        deterministic_module_kinds=deterministic_backend_capabilities(adapter),
    )
    content_modules, content_assets, artifact_jobs = _compile_content_artifact_graph(
        plan,
        adapter=adapter,
        mod_id=base.spec.mod_id,
        package_name=base.spec.package_name,
    )
    typed_ids = {module.module_id for module in modules}
    duplicate_content_ids = sorted(
        typed_ids & {module.module_id for module in content_modules}
    )
    if duplicate_content_ids:
        raise ValueError(
            "CONTENT_TYPED_MODULE_COLLISION: "
            + ", ".join(duplicate_content_ids)
        )
    modules = (*content_modules, *modules)
    design = {
        **design,
        "_authored_execution_manifest": manifest,
        "_production_state_section": deepcopy(production_state_section),
        "_artifact_jobs": [job.to_dict() for job in artifact_jobs],
    }

    from .root_cause_trace import emit_root_cause

    emit_root_cause(
        "authored_design_lowering_selected",
        stage="production",
        operation="compile_authored_production",
        gate="typed_host_execution_only",
        result="PASS",
        details={
            "policy": manifest["policy"],
            "existing_input": bool(effective_existing),
            "module_ids": [module.module_id for module in modules],
            "source_text_sha256": manifest["source_text_sha256"],
            "manifest": manifest,
        },
    )
    return complete_proposal_from_parts(
        requested_prompt=plan.requested_prompt,
        base_proposal=base,
        game_design=design,
        modules=modules,
        assets=content_assets,
        acceptance_tests=tuple(
            dict.fromkeys(
                (
                    *acceptance,
                    *(
                        str(value)
                        for value in content_design.get("acceptance_tests", ())
                        if str(value).strip()
                    ),
                )
            )
        ),
        existing_input_sha256=effective_existing,
    )
