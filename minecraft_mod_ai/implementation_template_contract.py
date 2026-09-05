from __future__ import annotations

"""Dynamic implementation sketches for small-model Minecraft coding.

The host owns task decomposition, target compatibility, artifacts, dependencies, retrieval
references, and verification obligations. This module compiles those facts into a detailed,
stable template. The model may fill only named holes; it may not invent or remove holes,
change target coordinates, or rewrite host-owned dependency/artifact contracts.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

SCHEMA = "mmm/implementation-template-v1"
MODEL_FILL_FIELDS = frozenset(
    {
        "implementation_decision",
        "local_steps",
        "code_bindings",
        "reference_uses",
        "verification_intent",
        "uncertainties",
    }
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha(value: Any) -> str:
    data = value if isinstance(value, str) else _canonical(value)
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload[field] = ""
    return _sha(payload)


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raw: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw = value
    else:
        return ()
    return tuple(
        dict.fromkeys(
            text
            for item in raw
            if (text := str(item or "").strip())
        )
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _stable_hole_id(task_id: str, kind: str, subject: str, ordinal: int) -> str:
    digest = hashlib.sha256(
        _canonical(
            {
                "task_id": task_id,
                "kind": kind,
                "subject": subject,
                "ordinal": ordinal,
            }
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"hole_{kind}_{digest}"[:63]


def _artifact_records(task: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = task.get("artifact_obligations")
    if not isinstance(raw, list):
        return ()
    return tuple(dict(item) for item in raw if isinstance(item, Mapping))


def _target_constraints(task: Mapping[str, Any]) -> dict[str, Any]:
    target = _mapping(task.get("target_cell"))
    return {
        "minecraft_version": str(target.get("minecraft_version") or "").strip(),
        "loader": str(target.get("loader") or "").strip(),
        "mappings": str(
            target.get("mappings")
            or target.get("mapping_namespace")
            or target.get("source_api_family")
            or ""
        ).strip(),
        "java_version": str(
            target.get("java_version") or target.get("java") or ""
        ).strip(),
        "policy": (
            "Use only APIs compatible with this exact host-selected target. "
            "Reference code from another target is architectural evidence only until the host "
            "provides an exact compatibility receipt."
        ),
    }


def _minecraft_checklist(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = _artifact_records(task)
    artifact_kinds = {
        str(item.get("kind") or "").strip().casefold() for item in artifacts
    }
    capabilities = {
        item.casefold() for item in _strings(task.get("implementation_capabilities"))
    }
    gates = {item.casefold() for item in _strings(task.get("required_gates"))}
    predicates = {
        item.casefold() for item in _strings(task.get("conditional_predicates"))
    }
    checks: list[dict[str, Any]] = []

    def add(check_id: str, instruction: str, evidence: Sequence[str]) -> None:
        if any(item["check_id"] == check_id for item in checks):
            return
        checks.append(
            {
                "check_id": check_id,
                "instruction": instruction,
                "activated_by": list(dict.fromkeys(str(item) for item in evidence if item)),
            }
        )

    if "source_code" in artifact_kinds:
        add(
            "source_ownership",
            "Implement only inside host-owned source anchors; preserve package, source-set, and side boundaries.",
            ("artifact:source_code",),
        )
    if "registry_entry" in artifact_kinds or any("registry" in item for item in capabilities):
        add(
            "registry_lifecycle",
            "Use stable namespaced identifiers, register in the target loader lifecycle, and keep every code/resource reference consistent with the same identifier.",
            ("artifact:registry_entry", *sorted(capabilities)),
        )
    if "persistence_schema" in artifact_kinds or "needs_persistence" in predicates:
        add(
            "persistent_state_round_trip",
            "Define authoritative state ownership, encode/decode every persisted field, mark mutations dirty when required by the target API, and verify save/reload round trips.",
            ("artifact:persistence_schema", "predicate:needs_persistence"),
        )
    if "network_protocol" in artifact_kinds or "needs_network" in predicates:
        add(
            "server_authority",
            "Treat client input as a request only: decode symmetrically, validate on the server, mutate authoritative server state, then synchronize the observable result.",
            ("artifact:network_protocol", "predicate:needs_network"),
        )
        add(
            "network_side_safety",
            "Keep client-only classes out of common/server class-loading paths and register handlers/codecs on the correct side for the selected loader.",
            ("artifact:network_protocol",),
        )
    if (
        "data_or_client_resource" in artifact_kinds
        or "client_visual_or_ui_resource" in artifact_kinds
        or "needs_client_render" in predicates
    ):
        add(
            "resource_reference_closure",
            "Resolve namespace/path chains across models, textures, language keys, menus/screens, and code identifiers; no dangling resource identifier is allowed.",
            ("artifact:client_or_data_resource",),
        )
    if "generated_data_resource" in artifact_kinds or "generated_resource_validation" in gates:
        add(
            "datagen_reference_closure",
            "Generated recipes/tags/loot/models must resolve against registered identifiers and pass the host generated-resource validator.",
            ("artifact:generated_data_resource", "gate:generated_resource_validation"),
        )
    if "worldgen_data" in artifact_kinds or "needs_worldgen" in predicates:
        add(
            "worldgen_binding",
            "Keep configured/placed/biome-or-dimension bindings complete and verify target-version data/resource schemas before runtime validation.",
            ("artifact:worldgen_data", "predicate:needs_worldgen"),
        )
    if "loader_module_binding" in artifact_kinds or "needs_loader_leaf" in predicates:
        add(
            "loader_boundary",
            "Keep common gameplay contracts loader-neutral and place loader-specific registration/API glue only in the approved loader leaf.",
            ("artifact:loader_module_binding", "predicate:needs_loader_leaf"),
        )
    if "verification_artifact" in artifact_kinds or gates:
        add(
            "verification_from_behavior",
            "Verify the declared observable behavior plus failure/negative paths; compile success alone cannot satisfy player-facing acceptance.",
            tuple(f"gate:{item}" for item in sorted(gates)) or ("artifact:verification_artifact",),
        )
    if _strings(task.get("runtime_acceptance")) or _strings(task.get("public_acceptance")):
        add(
            "runtime_acceptance",
            "Bind implementation steps to the exact public/runtime acceptance statements and record how each is observable in-game or by an executable test.",
            ("public_acceptance", "runtime_acceptance"),
        )
    return checks


def _retrieval_fingerprint(task: Mapping[str, Any], kind: str, subject: str) -> dict[str, Any]:
    target = _target_constraints(task)
    semantic_terms = list(
        dict.fromkeys(
            [
                subject,
                str(task.get("semantic_outcome") or "").strip(),
                *_strings(task.get("implementation_capabilities")),
            ]
        )
    )
    artifact_terms = [
        str(item.get("kind") or "").strip()
        for item in _artifact_records(task)
        if str(item.get("kind") or "").strip()
    ]
    api_constraints = [
        value
        for value in (
            target["minecraft_version"],
            target["loader"],
            target["mappings"],
            target["java_version"],
        )
        if value
    ]
    return {
        "kind": kind,
        "semantic_terms": semantic_terms,
        "artifact_terms": list(dict.fromkeys(artifact_terms)),
        "required_gate_terms": list(_strings(task.get("required_gates"))),
        "target_terms": api_constraints,
        "selection_policy": (
            "Host retrieval selects compatible method/class/resource/test slices. "
            "Never send an entire reference repository to the small model when a bounded slice suffices."
        ),
    }


def _hole(
    task: Mapping[str, Any],
    *,
    kind: str,
    subject: str,
    ordinal: int,
    host_state: str,
    artifact_refs: Sequence[str] = (),
    acceptance_refs: Sequence[str] = (),
    evidence_refs: Sequence[str] = (),
) -> dict[str, Any]:
    task_id = str(task.get("task_id") or "").strip()
    return {
        "hole_id": _stable_hole_id(task_id, kind, subject, ordinal),
        "kind": kind,
        "subject": subject,
        "task_ref": task_id,
        "requirement_refs": list(_strings(task.get("requirement_refs"))),
        "host_state": host_state,
        "target_constraints": _target_constraints(task),
        "consumes": list(_strings(task.get("consumes"))),
        "provides": list(_strings(task.get("provides"))),
        "artifact_refs": list(dict.fromkeys(artifact_refs)),
        "acceptance_refs": list(dict.fromkeys(acceptance_refs)),
        "evidence_refs": list(
            dict.fromkeys([*_strings(task.get("reuse_refs")), *evidence_refs])
        ),
        "reference_slice_refs": [],
        "retrieval_fingerprint": _retrieval_fingerprint(task, kind, subject),
        "model_contract": {
            "allowed_fields": sorted(MODEL_FILL_FIELDS),
            "forbidden_authority": [
                "target_coordinates",
                "hole_identity_or_count",
                "dependency_edges",
                "owned_artifacts",
                "required_gates",
                "reference_compatibility",
            ],
            "instruction": (
                "Fill only this hole. Reuse host-provided references as implementation evidence; "
                "do not copy foreign names/constants/architecture that are not required by the MMM task."
            ),
        },
        "model_fill": {},
    }


def build_implementation_template(task: Mapping[str, Any]) -> dict[str, Any]:
    """Compile every host-known coding obligation into stable small-model holes."""
    task_id = str(task.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("implementation template requires task_id")

    holes: list[dict[str, Any]] = []
    ordinal = 0

    def add(kind: str, subject: str, state: str = "MODEL_FILL_REQUIRED", **kwargs: Any) -> None:
        nonlocal ordinal
        text = str(subject or "").strip()
        if not text:
            return
        holes.append(
            _hole(
                task,
                kind=kind,
                subject=text,
                ordinal=ordinal,
                host_state=state,
                **kwargs,
            )
        )
        ordinal += 1

    for capability in _strings(task.get("implementation_capabilities")):
        add("implementation_capability", capability)

    for obligation in _strings(task.get("design_resolution_obligations")):
        add("design_resolution", obligation)

    for artifact in _artifact_records(task):
        artifact_id = str(artifact.get("artifact_id") or "").strip()
        kind = str(artifact.get("kind") or "implementation_artifact").strip()
        locator = str(artifact.get("locator") or "").strip()
        purpose = str(artifact.get("purpose") or "").strip()
        subject = " | ".join(item for item in (kind, locator, purpose) if item)
        add(
            "artifact_implementation",
            subject,
            artifact_refs=(artifact_id,) if artifact_id else (),
        )

    for consumed in _strings(task.get("consumes")):
        add(
            "dataflow_input",
            consumed,
            state="HOST_BOUND_MODEL_REALIZATION_REQUIRED",
        )
    for provided in _strings(task.get("provides")):
        add(
            "dataflow_output",
            provided,
            state="HOST_BOUND_MODEL_REALIZATION_REQUIRED",
        )

    for gate in _strings(task.get("required_gates")):
        add(
            "verification_gate",
            gate,
            state="HOST_BOUND_MODEL_REALIZATION_REQUIRED",
        )

    public_acceptance = _strings(task.get("public_acceptance"))
    runtime_acceptance = _strings(task.get("runtime_acceptance"))
    for acceptance in public_acceptance:
        add(
            "public_acceptance",
            acceptance,
            acceptance_refs=(acceptance,),
        )
    for acceptance in runtime_acceptance:
        add(
            "runtime_acceptance",
            acceptance,
            acceptance_refs=(acceptance,),
        )

    for reuse_ref in _strings(task.get("reuse_refs")):
        add(
            "reference_adaptation",
            reuse_ref,
            evidence_refs=(reuse_ref,),
        )

    if not holes:
        add(
            "semantic_implementation",
            str(task.get("semantic_outcome") or task_id),
        )

    template: dict[str, Any] = {
        "schema_version": SCHEMA,
        "task_ref": task_id,
        "task_sha256_input": str(task.get("task_sha256") or ""),
        "semantic_outcome": str(task.get("semantic_outcome") or "").strip(),
        "requirement_refs": list(_strings(task.get("requirement_refs"))),
        "target_constraints": _target_constraints(task),
        "host_owned": {
            "owned_anchors": [
                dict(item)
                for item in task.get("owned_anchors", [])
                if isinstance(item, Mapping)
            ]
            if isinstance(task.get("owned_anchors"), list)
            else [],
            "depends_on": list(_strings(task.get("depends_on"))),
            "consumes": list(_strings(task.get("consumes"))),
            "provides": list(_strings(task.get("provides"))),
            "required_gates": list(_strings(task.get("required_gates"))),
            "artifact_obligations": list(_artifact_records(task)),
            "reuse_refs": list(_strings(task.get("reuse_refs"))),
        },
        "minecraft_checklist": _minecraft_checklist(task),
        "holes": holes,
        "completion_policy": {
            "operator": "all",
            "required_hole_ids": [item["hole_id"] for item in holes],
            "rule": (
                "No hole may be silently dropped because of token budget. If a model cannot fill a hole, "
                "it must leave it unresolved; the host schedules another bounded pass."
            ),
            "verification": (
                "Host compile/static/resource/GameTest/runtime gates are authoritative. Model self-rating "
                "or prose confidence cannot complete a hole."
            ),
        },
        "template_sha256": "",
    }
    template["template_sha256"] = _hash_without(template, "template_sha256")
    return template


def sanitize_hole_fills(
    implementation_template: Mapping[str, Any],
    value: Any,
) -> list[dict[str, Any]]:
    """Accept only model fields for host-created hole IDs; preserve host ordering."""
    holes = implementation_template.get("holes")
    if not isinstance(holes, list):
        return []
    known = [
        str(item.get("hole_id") or "")
        for item in holes
        if isinstance(item, Mapping) and str(item.get("hole_id") or "")
    ]
    raw_items = value if isinstance(value, list) else []
    by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            continue
        hole_id = str(raw.get("hole_id") or "").strip()
        if hole_id in known and hole_id not in by_id:
            by_id[hole_id] = raw

    result: list[dict[str, Any]] = []
    for hole_id in known:
        raw = by_id.get(hole_id)
        if raw is None:
            continue
        fill = {
            key: raw[key]
            for key in MODEL_FILL_FIELDS
            if key in raw
        }
        result.append({"hole_id": hole_id, **fill})
    return result


def _evidence_contracts(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Mapping):
        return ()
    return tuple(item for item in value.values() if isinstance(item, Mapping))


def install(
    *,
    planning_module: Any,
    planner_template_module: Any,
    task_capsule_module: Any,
) -> None:
    """Install one explicit template authority after task/artifact enrichment."""
    current_tasks = planning_module._compile_tasks
    if not getattr(current_tasks, "_mmm_detailed_implementation_template", False):

        @wraps(current_tasks)
        def compile_tasks(gaps, reuse, target, branches, ownership):
            tasks = current_tasks(gaps, reuse, target, branches, ownership)
            result: list[dict[str, Any]] = []
            for raw in tasks:
                task = dict(raw)
                task["implementation_template"] = build_implementation_template(task)
                task["task_sha256"] = ""
                task["task_sha256"] = planning_module._hash_without(task, "task_sha256")
                result.append(task)
            return tuple(result)

        compile_tasks._mmm_detailed_implementation_template = True
        planning_module._compile_tasks = compile_tasks

    existing_fields = tuple(task_capsule_module._COMPACT_TASK_FIELDS)
    task_capsule_module._COMPACT_TASK_FIELDS = tuple(
        dict.fromkeys(
            [
                *existing_fields,
                "implementation_capabilities",
                "design_resolution_obligations",
                "runtime_acceptance",
                "semantic_type",
                "unlock_policy",
                "implementation_template",
            ]
        )
    )

    planner_template_module.MODEL_TASK_DETAIL_KEYS = frozenset(
        {
            *planner_template_module.MODEL_TASK_DETAIL_KEYS,
            "hole_fills",
        }
    )

    current_skeleton = planner_template_module.build_batch_skeleton
    if not getattr(current_skeleton, "_mmm_detailed_implementation_template", False):

        @wraps(current_skeleton)
        def build_batch_skeleton(*args: Any, **kwargs: Any):
            result = current_skeleton(*args, **kwargs)
            contracts = _evidence_contracts(kwargs.get("host_module_contracts"))
            if not contracts and len(args) >= 7:
                contracts = _evidence_contracts(args[6])
            if not contracts:
                return result

            for module in result.get("modules", []):
                if not isinstance(module, dict):
                    continue
                config = module.get("config")
                if not isinstance(config, dict):
                    continue
                evidence_task = config.get("evidence_task")
                if isinstance(evidence_task, Mapping):
                    implementation_template = evidence_task.get("implementation_template")
                    if isinstance(implementation_template, Mapping):
                        config["implementation_template"] = dict(implementation_template)

            acceptance_explicit = bool(kwargs.get("acceptance_tests")) or (
                len(args) >= 8 and bool(args[7])
            )
            deliverables_explicit = bool(kwargs.get("deliverables")) or (
                len(args) >= 3 and bool(args[2])
            )
            if not acceptance_explicit:
                result["acceptance_tests"] = list(
                    dict.fromkeys(
                        item
                        for contract in contracts
                        for item in _strings(
                            contract.get("public_acceptance")
                            or contract.get("acceptance")
                            or contract.get("internal_invariants")
                        )
                    )
                )
            if not deliverables_explicit:
                result["completed_deliverables"] = list(
                    dict.fromkeys(
                        item
                        for contract in contracts
                        for item in _strings(contract.get("provides"))
                    )
                )
            return result

        build_batch_skeleton._mmm_detailed_implementation_template = True
        planner_template_module.build_batch_skeleton = build_batch_skeleton

    current_merge = planner_template_module.merge_model_output_into_skeleton
    if not getattr(current_merge, "_mmm_hole_fill_authority", False):

        @wraps(current_merge)
        def merge_model_output_into_skeleton(skeleton, model_output, valid_module_catalog):
            result = current_merge(skeleton, model_output, valid_module_catalog)
            for module in result.get("modules", []):
                if not isinstance(module, dict):
                    continue
                config = module.get("config")
                if not isinstance(config, dict):
                    continue
                implementation_template = config.get("implementation_template")
                if not isinstance(implementation_template, Mapping):
                    evidence_task = config.get("evidence_task")
                    implementation_template = (
                        evidence_task.get("implementation_template")
                        if isinstance(evidence_task, Mapping)
                        else None
                    )
                model_fill = config.get("model_fill")
                if not isinstance(model_fill, dict) or not isinstance(
                    implementation_template, Mapping
                ):
                    continue
                model_fill["hole_fills"] = sanitize_hole_fills(
                    implementation_template,
                    model_fill.get("hole_fills"),
                )
            return result

        merge_model_output_into_skeleton._mmm_hole_fill_authority = True
        planner_template_module.merge_model_output_into_skeleton = (
            merge_model_output_into_skeleton
        )


__all__ = [
    "MODEL_FILL_FIELDS",
    "SCHEMA",
    "build_implementation_template",
    "install",
    "sanitize_hole_fills",
]
