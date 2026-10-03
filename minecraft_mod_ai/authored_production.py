"""Pass a saved design to the implementation agent without planning it again."""

from __future__ import annotations

import hashlib
from copy import deepcopy
import json
from collections.abc import Mapping
from typing import Any

from .authored_plan import AuthoredPlan
from .complete_spec import (
    CompleteProposal,
    ProductionModule,
    complete_proposal_from_parts,
)
from .planning_pipeline import PlanningPipeline
from .spec import ModSpec, Proposal, ProposalStatus
from .target_contract import target_coordinates_from_mapping

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


def _compile_new_authored_modules(
    plan: AuthoredPlan,
    *,
    mod_id: str,
    package_name: str,
    target: Mapping[str, Any],
    production_state_section: Mapping[str, Any] | None = None,
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
    assert_typed_plan_host_support(
        plan.structured_sections,
        validated_plan,
    )

    raw_platform_modules = validated_plan.get("platform_modules", [])
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
            state_store_config = deepcopy(dict(item["config"]))
            continue
        if kind == "network_sync":
            if network_sync_config is not None:
                raise ValueError("TYPED_PLATFORM_NETWORK_SYNC_DUPLICATE")
            network_sync_config = {
                **deepcopy(dict(item["config"])),
                "__covers": list(item["covers"]),
            }
            continue
        if kind == "resource_policy":
            if resource_policy_config is not None:
                raise ValueError("TYPED_PLATFORM_RESOURCE_POLICY_DUPLICATE")
            resource_policy_config = deepcopy(dict(item["config"]))
            continue
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
    network_sync_needs_state = bool(
        network_sync_covers
        & {
            "authority_and_network.payloads",
            "authority_and_network.synchronization",
            "authority_and_network.reconnection",
        }
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
    task = _exact_authored_task(
        task_id=task_id,
        path=program_path,
        symbol=program_symbol,
        target=target,
        obligation=(
            "Compile the persisted Typed PlanIR exactly through the host Java backend. "
            "Do not invoke a coder or reinterpret authored behavior."
        ),
        semantic_outcome="Materialize the approved Typed PlanIR deterministically.",
        depends_on=(),
        consumes=(),
        provides=("authored_typed_program_ready",),
        worksheet={
            "typed_plan_ir": deepcopy(validated_plan),
            "typed_plan_source_sha256": source_sha,
        },
        required_gates=("target_compile",),
        target_status="host_reserved",
        execution_role="host_compiler",
    )
    module = ProductionModule(
        module_id=task_id,
        kind="typed_host",
        config={
            "evidence_task": task,
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
    manifest = {
        "schema_version": _AUTHORED_EXECUTION_SCHEMA,
        "policy": "host_typed_plan_ir",
        "source_text_sha256": source_sha,
        "source_bytes": len(plan.text.encode("utf-8")),
        "unit_count": 1 + len(platform_modules),
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

    import re
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


def compile_authored_design(
    router: Any, plan: AuthoredPlan, *, existing_input_sha256: str = ""
) -> CompleteProposal:
    from .authored_structured_design import normalize_structured_sections
    from .production_state_compiler import compile_production_state_section

    if not isinstance(plan, AuthoredPlan):
        raise TypeError("compile_authored_design requires AuthoredPlan")
    if not getattr(plan, "typed_plan_ir", {}):
        raise ValueError(
            "TYPED_PLAN_REQUIRED: legacy authored production routes have been removed."
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

    mod_id = "authored_" + plan.calculate_hash()[:12]
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
    binding = PlanningPipeline(router)
    design = binding._bind_existing_project(design)
    design, base, _, _ = binding._bind_platform(
        plan.requested_prompt,
        design,
        base,
    )
    target = _bound_target(design)
    design = {**design, **target}

    production_state_section = compile_production_state_section(plan)
    modules, manifest = _compile_new_authored_modules(
        plan,
        mod_id=base.spec.mod_id,
        package_name=base.spec.package_name,
        target=target,
        production_state_section=production_state_section,
    )
    design = {
        **design,
        "_authored_execution_manifest": manifest,
        "_production_state_section": deepcopy(production_state_section),
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
        acceptance_tests=acceptance,
        existing_input_sha256=effective_existing,
    )
