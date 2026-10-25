"""Compile-gated, host-assembled Java classes for the Fabric construction path.

The host owns package/class, generated imports and required entrypoint
registrations. The coder receives bounded declaration frames, never a request
for an entire source file or a free-form import catalogue.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .construction_direct_writer import (
    _claim_checkpoint,
    _digest,
    _read_checkpoint,
    _save_checkpoint,
)


class HostJavaConstructionError(ValueError):
    """A host-assembled compilation unit did not satisfy the target contract."""


def _repair_runtime_bindings(spec, receipts):
    from .runtime_mutation_author import RuntimeRepairBindings

    return RuntimeRepairBindings(json.loads(spec.get("bindings", {}).get(
        "runtime_state_bindings_json") or "{}"), receipts)


def _build_host_java_prompt(router, spec: dict, context: dict, *,
                            package: str, name: str, owner: str,
                            behaviors, source_context):
    """Assemble immutable checked API and behavior evidence once per file."""
    from .construction_file_author import _evidence
    from .host_prerequisite_api import read_prerequisite_apis
    from .host_typed_state_access import (
        state_access_contract, state_prompt_api, state_work_order_scope,
    )

    contract = context.get("authoring_contract") or {}
    requirement = str(spec.get("requirement", ""))
    if (contract.get("canonical_leaf") and contract.get("responsibility")
            and not spec.get("bindings", {}).get("request_feature_entrypoint")):
        requirement = str(contract["responsibility"])
    behavior_rows = [dict(row) for row in behaviors]
    # Event APIs are selected by the actual USER TRIGGER, not just the
    # broad feature title. Without the trigger, the host supplied zero
    # event APIs for "player opens trading" and the coder invented callback
    # parameters unrelated to any registered listener (production-logs 15).
    api_intent = " ".join((
        requirement,
        *(str(row.get(field) or "")
          for row in behavior_rows
          for field in ("intent", "trigger", "effects")),
        *(str(name)
          for row in behavior_rows
          for field in ("runtime_state_reads", "runtime_state_writes")
          for name in (row.get(field) or ())),
    ))
    prompt = {
        "requirement": requirement,
        "original_feature_goal": str(spec.get("requirement") or ""),
        "assigned_behaviors": behavior_rows,
        "feature_work_order": context.get("feature_work_order"),
        "target": context["target"],
        "package": package, "public_class": name,
        "output_path": spec["target_path"],
        "side": spec.get("side", "COMMON"),
        "construction_requirements": contract,
        "construction_bindings": context.get("binding_pages", {}),
        "implementation_requirements": spec.get("bindings", {}).get(
            "java_implementation_requirements", ""),
        "inspected_target_api": _evidence(
            context, api_intent, target_path=spec["target_path"],
            owner=owner, router=router,
            feature_entrypoint=bool(spec.get("bindings", {}).get(
                "request_feature_entrypoint"))),
        "materialized_prerequisite_api": read_prerequisite_apis(
            context, target_path=spec["target_path"],
            source_context=source_context),
    }
    state_access = state_access_contract(prompt["materialized_prerequisite_api"])
    prompt["materialized_prerequisite_api"], prompt["inspected_target_api"] = state_prompt_api(
        prompt["materialized_prerequisite_api"], prompt["inspected_target_api"], state_access,
    )
    prompt["host_state_access"] = state_access
    prompt["host_state_work_order_scope"] = state_work_order_scope(
        context.get("feature_work_order"), state_access,
    )
    return contract, requirement, behavior_rows, prompt, state_access


def compose_host_java_source(router, spec: dict, context: dict, *,
                             package: str, name: str, owner: str,
                             behaviors=(), source_context=()) -> str:
    from .construction_file_author import _verify_full_source
    from .construction_implementation import construction_implementation_fingerprint
    from .first_pass_java_source import compose_first_pass_java

    if str(context.get("target", {}).get("loader", "")).casefold() != "fabric":
        raise HostJavaConstructionError("JAVA_HOST_FABRIC_REQUIRED")
    if not context.get("java_classpath") or not context.get("project_root"):
        raise HostJavaConstructionError("JAVA_HOST_WORKSPACE_REQUIRED")
    root = Path(context["project_root"]).resolve()
    if not root.is_dir():
        raise HostJavaConstructionError("JAVA_HOST_WORKSPACE_NOT_FOUND")

    identifier = _digest({
        "schema": "mmm/host-java-compilation-unit-v1",
        "spec": spec, "owner": owner, "target": context["target"],
        "prerequisites": context.get("prerequisite_hashes", {}),
        "work_order": context.get("feature_work_order_sha256", ""),
        "implementation": construction_implementation_fingerprint(),
    })
    checkpoint = root / ".mmm" / "construction_host_files" / (identifier + ".json")
    candidate = root / ".mmm" / "construction_host_candidates" / (identifier + ".json")
    proof_path = root / ".mmm" / "construction_host_mutations" / (identifier + ".json")
    from .runtime_mutation_author import verify_native_runtime_mutation_coverage
    required_native = any(row.get("runtime_state_writes") for row in behaviors)
    mutation_receipts = []
    with _claim_checkpoint(checkpoint):
        saved = _read_checkpoint(checkpoint, identifier)
        if saved:
            if set(saved) != {"source"} or not isinstance(saved["source"], str):
                raise HostJavaConstructionError("JAVA_HOST_CHECKPOINT_INVALID")
            try:
                _verify_full_source(saved["source"], spec, context,
                    package=package, name=name, source_context=source_context)
                if required_native:
                    saved_proof = _read_checkpoint(proof_path, identifier)
                    if saved_proof.get("source_sha256") != hashlib.sha256(
                            saved["source"].encode()).hexdigest():
                        raise ValueError("NATIVE_MUTATION_CHECKPOINT_SOURCE_MISMATCH")
                    verify_native_runtime_mutation_coverage(
                        saved["source"], behaviors,
                        json.loads(saved_proof.get("receipts_json") or "null"))
            except (RuntimeError, ValueError, OSError):
                # Do not certify an old source after a prerequisite change.
                pass
            else:
                return saved["source"]

        rule_mode = bool(spec.get("bindings", {}).get("request_feature_entrypoint"))
        feature_route = None
        source_work = None
        if rule_mode:
            # Route by *executable* source evidence before any model decisions.
            # A planner-only world_state label is not an installed producer,
            # and an unnamed occurrence is not an authorized Fabric event.
            from .feature_execution_route import select_feature_execution_route
            from .host_fabric_callback_abi import callback_interfaces

            runtime_bindings = json.loads(spec.get("bindings", {}).get(
                "runtime_state_bindings_json") or "{}")
            available_callbacks = callback_interfaces(
                context.get("symbols") or {}, side=str(spec.get("side") or "COMMON"))
            feature_route = select_feature_execution_route(
                behaviors, runtime_bindings, context.get("symbols") or {},
                available_callbacks)
            if feature_route["route"] == "source_owned_java":
                rule_mode = False
            print("FEATURE_EXECUTION_ROUTE: " + json.dumps({
                "owner": owner,
                "route": feature_route["route"],
                "producer_obligations": feature_route["producer_obligations"],
                "verified_behavior_ids": feature_route["verified_behavior_ids"],
                "target_compilation_verified": False, "runtime_verified": False,
            }, ensure_ascii=False), flush=True)
        implementation_router = router
        if feature_route is not None and feature_route["route"] == "source_owned_java":
            from .feature_implementation_worker import require_feature_implementation_router
            implementation_router = require_feature_implementation_router(router)
        if rule_mode:
            # Rule authoring queries the authoritative typed symbol index in
            # bounded pages. Do not first ask for whole-feature API retrieval
            # or send the previous Java/body prompt to each small decision.
            from .host_prerequisite_api import read_prerequisite_apis
            from .host_typed_state_access import state_access_contract
            contract = context.get("authoring_contract") or {}
            requirement = str(spec.get("requirement") or "")
            behavior_rows = [dict(row) for row in behaviors]
            prerequisite_api = read_prerequisite_apis(
                context, target_path=spec["target_path"], source_context=source_context)
            state_access = state_access_contract(prerequisite_api)
            prompt = {"materialized_prerequisite_api": prerequisite_api,
                      "inspected_target_api": {}, "host_state_work_order_scope": {}}
        else:
            contract, requirement, behavior_rows, prompt, state_access = (
                _build_host_java_prompt(
                    implementation_router, spec, context, package=package, name=name,
                    owner=owner, behaviors=behaviors, source_context=source_context,
                )
            )
            if feature_route is not None:
                bindings = spec.get("bindings", {})
                prompt["source_owned_producer_obligations"] = feature_route[
                    "producer_obligations"]
                # Give the bounded source author actual inspected hook ABIs.
                # It owns a real registration expression and its receiver,
                # rather than being forced back into an earlier LLM hook
                # selection that already declared its source unimplemented.
                candidates = callback_interfaces(
                    context.get("symbols") or {},
                    side=str(spec.get("side") or "COMMON"))
                def _hook_priority(row):
                    scopes = ("ServerLevel", "ServerWorld", "MinecraftServer",
                              "ServerPlayer", "ServerPlayerEntity")
                    score = sum(any(str(kind).endswith(scope) for scope in scopes)
                                for kind in row.get("parameters", ()))
                    return (-score, str(row.get("event_id") or ""))
                prompt["source_owned_inspected_hooks"] = [
                    {key: row.get(key) for key in
                     ("event_id", "event_owner", "event_field",
                      "parameters", "return_type")}
                    for row in sorted(candidates, key=_hook_priority)[:32]
                ]
                prompt["runtime_state_bindings"] = {
                    key: value for key, value in runtime_bindings.items()
                    if any(key in (*row.get("runtime_state_reads", ()),
                                           *row.get("runtime_state_writes", ()))
                           for row in behavior_rows)
                }
                prompt["retained_typed_behavior"] = json.loads(
                    bindings.get("retained_typed_behavior_json") or "{}")
                prompt["owned_runtime_records"] = json.loads(
                    bindings.get("owned_runtime_records_json") or "[]")
                from .feature_source_work_graph import build_source_work_graph
                source_work = build_source_work_graph(
                    behavior_rows,
                    producer_obligations=feature_route["producer_obligations"],
                    inspected_hooks=candidates,
                    verified_behavior_ids=feature_route["verified_behavior_ids"],
                )
                # The full task ledger is host-owned. Never repeat the entire
                # inspected event catalogue or every behavior's dependency chain
                # on each 9B member request.
                print("FEATURE_SOURCE_WORK_GRAPH: " + json.dumps({
                    "owner": owner,
                    "task_count": len(source_work["tasks"]),
                    "unverified_obligations": source_work["unverified_obligations"],
                    "runtime_verified": False,
                }, ensure_ascii=False), flush=True)
                # These obligations are implementation work, NOT proof that
                # any state getter/hook already exists. The ordinary target
                # compiler/coverage contract remains mandatory.
        instructions = (
            "Implement the specified Fabric/Minecraft task using only inspected "
            "target APIs and materialized project source. The host builds every "
            "package/import/class wrapper, fixes the exact Java declaration "
            "signature and Fabric lifecycle. You only plan typed member roles "
            "and implement one bounded declaration at a time. Do not refer to "
            "nonexistent source, invent APIs, omit assigned behaviors, use "
            "placeholders, or copy external mod code. Model does not select "
            "imports. Official mappings, not Yarn, are selected."
        )
        effective_spec = dict(spec)
        effective_spec["requirement"] = requirement
        try:
            pending = _read_checkpoint(candidate, identifier)
            if required_native:
                # A compiler candidate saved before successful source proof is
                # never trusted to own real mutations on resume. Reconstruct
                # the checked calls and receipt together, not separately.
                pending = {}
            if pending:
                if set(pending) != {"source", "diagnostic"}:
                    raise HostJavaConstructionError("JAVA_HOST_CANDIDATE_INVALID")
                # A candidate is work in progress, never a successful receipt.
                # Recheck it with the real compiler before any repair or return.
                source = pending["source"]
            else:
                from .host_verified_entity_type import verified_entity_type_source
                if rule_mode:
                    from .host_rule_feature import compose_rule_feature
                    source = compose_rule_feature(
                        router, effective_spec, context, package=package, name=name,
                        behaviors=behavior_rows, state_access=state_access,
                        prerequisite_apis=prerequisite_api)
                else:
                    source = verified_entity_type_source(
                        effective_spec, contract, prompt["inspected_target_api"],
                        package=package, name=name, behaviors=behavior_rows,
                    )
                if source is None:
                    from .host_verified_entity_registry import (
                        verified_entity_registry_source,
                    )
                    source = verified_entity_registry_source(
                        effective_spec, contract, prompt["inspected_target_api"],
                        context.get("symbols") or {},
                        package=package, name=name, behaviors=behavior_rows,
                    )
                if source is None:
                    source = compose_first_pass_java(
                        implementation_router, effective_spec, [
                            {"role": "system", "content": instructions},
                            {"role": "user", "content": json.dumps(
                                prompt, ensure_ascii=False, default=str)},
                        ], source_context=source_context,
                        construction_context={
                            **context, "host_state_access": state_access,
                            "host_state_work_order_scope": prompt["host_state_work_order_scope"],
                            "host_source_owned": bool(
                                feature_route and feature_route["route"] == "source_owned_java"),
                            "source_owned_work_graph": source_work,
                            "runtime_mutation_receipts": mutation_receipts,
                        },
                        behaviors=behavior_rows,
                    )
            acceptance = (context.get("feature_work_order") or {}).get("acceptance") or {}
            budget = acceptance.get("max_compiler_repair_attempts", 2)
            if type(budget) is not int or budget < 0:
                raise HostJavaConstructionError("JAVA_HOST_REPAIR_BUDGET_INVALID")
            if rule_mode:
                # A compiler defect in host lowering is not a request for the
                # model to rewrite generated Java and escape the typed IR.
                # Retain source + diagnostic for host investigation.
                budget = 0
            from .construction_host_member_repair import repair_compiler_member
            from .construction_journal import JournaledConstructionRouter
            # Parsed declarations are not compiler-verified repairs. Replaying
            # a rejected correction would freeze a fresh session at the same
            # failure. Resume the whole candidate, then author only its failing
            # member against the latest compiler diagnostic.
            repair_router = router.router if isinstance(router, JournaledConstructionRouter) else router
            if feature_route is not None and feature_route["route"] == "source_owned_java":
                from .feature_implementation_worker import require_feature_implementation_router
                repair_router = require_feature_implementation_router(repair_router)
            for attempt in range(budget + 1):
                try:
                    _verify_full_source(source, spec, context, package=package,
                                        name=name, source_context=source_context)
                except (RuntimeError, ValueError, OSError) as compiler_error:
                    _save_checkpoint(candidate, identifier, {
                        "source": source,
                        "diagnostic": str(compiler_error) or type(compiler_error).__name__,
                    })
                    if attempt == budget:
                        raise
                    source = repair_compiler_member(
                        repair_router, source, str(compiler_error), filename=name + ".java",
                        target_path=spec["target_path"],
                        # The 26.2 index holds ~872K symbols. Repair reads
                        # it without mutation; cloning it per failed member
                        # needlessly multiplies memory pressure on Colab.
                        symbols=context.get("symbols") or {}, owner=owner,
                        api_evidence=prompt["inspected_target_api"],
                        prerequisite_api=prompt["materialized_prerequisite_api"],
                        state_access=state_access,
                        behaviors=behavior_rows,
                        runtime_state_bindings=_repair_runtime_bindings(spec, mutation_receipts),
                        attempt=attempt + 1,
                    )
                else:
                    # A compiler-valid edit may still delete a mandatory
                    # checked mutation. Re-prove final source after every
                    # repair and fail closed if a receipt's call disappeared.
                    if required_native:
                        verify_native_runtime_mutation_coverage(
                            source, behavior_rows, mutation_receipts)
                    break
        except (RuntimeError, ValueError, OSError) as exc:
            trace = root / ".mmm" / "construction_host_failures"
            trace.mkdir(parents=True, exist_ok=True)
            (trace / (identifier + ".json")).write_text(json.dumps({
                "target": spec["target_path"], "error": str(exc),
                "source": locals().get("source") or getattr(exc, "failed_source", None),
                "failed_response": getattr(exc, "failed_response", None),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        from .construction_coverage import record_authored_source_calls
        record_authored_source_calls(
            root, spec["target_path"], source, source_kind="direct")
        if required_native:
            _save_checkpoint(proof_path, identifier, {
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "receipts_json": json.dumps(
                    mutation_receipts, ensure_ascii=False, sort_keys=True),
            })
        _save_checkpoint(checkpoint, identifier, {"source": source})
        candidate.unlink(missing_ok=True)
        print("CONSTRUCTION HOST JAVA AUTHOR: " + json.dumps({
            "target_path": spec["target_path"], "owner": owner,
            "target_compilation_verified": True,
            "runtime_verified": False,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }, ensure_ascii=False), flush=True)
        return source

