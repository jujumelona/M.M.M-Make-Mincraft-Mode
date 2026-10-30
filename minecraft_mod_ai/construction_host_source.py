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
        "host_java_authoring": bool(context.get("host_direct_feature_source")),
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
    _verify_full_source = _verification_boundary()
    from .construction_implementation import construction_implementation_fingerprint
    from .incremental_java_pipeline import (compose_source, candidate_data, load_candidate, compile_attempts, exhausted)
    from .first_pass_java_source import compose_first_pass_java

    if str(context.get("target", {}).get("loader", "")).casefold() != "fabric":
        raise HostJavaConstructionError("JAVA_HOST_FABRIC_REQUIRED")
    if not context.get("java_classpath") or not context.get("project_root"):
        raise HostJavaConstructionError("JAVA_HOST_WORKSPACE_REQUIRED")
    root = Path(context["project_root"]).resolve()
    if not root.is_dir():
        raise HostJavaConstructionError("JAVA_HOST_WORKSPACE_NOT_FOUND")

    # A requested feature is new implementation work. Do not ask a capability
    # picker to prove that its state, event or original source method already
    # exists before allowing the source writer to create it.
    direct_feature = bool(spec.get("bindings", {}).get("request_feature_entrypoint"))
    context = {**context, "host_direct_feature_source": direct_feature}

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
    required_native = not direct_feature and any(
        row.get("runtime_state_writes") for row in behaviors)
    mutation_receipts = []
    incremental_state = {}
    context["incremental_state"] = incremental_state
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

        contract, requirement, behavior_rows, prompt, state_access = (
            _build_host_java_prompt(
                router, spec, context, package=package, name=name,
                owner=owner, behaviors=behaviors, source_context=source_context,
            )
        )
        context["host_state_access"] = state_access
        if direct_feature:
            from .host_fabric_callback_abi import callback_interfaces

            bindings = spec.get("bindings", {})
            # API signatures are coding context, not an existence/semantic
            # admission decision. Missing gameplay owners are authored below.
            prompt["available_fabric_callbacks"] = callback_interfaces(
                context.get("symbols") or {}, side=str(spec.get("side") or "COMMON"))
            prompt["runtime_state_bindings"] = json.loads(
                bindings.get("runtime_state_bindings_json") or "{}")
            prompt["retained_typed_behavior"] = json.loads(
                bindings.get("retained_typed_behavior_json") or "{}")
            prompt["owned_runtime_records"] = json.loads(
                bindings.get("owned_runtime_records_json") or "[]")
            print("FEATURE_SOURCE_AUTHOR: " + json.dumps({
                "owner": owner, "mode": "incremental_java_operations",
                "assigned_behaviors": len(behavior_rows),
            }, ensure_ascii=False), flush=True)
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
            pending = load_candidate(pending, incremental_state, direct_feature)
            if pending:
                # A candidate is work in progress, never a successful receipt.
                # Recheck it with the real compiler before any repair or return.
                source = pending["source"]
            else:
                from .host_verified_entity_type import verified_entity_type_source
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
                # Use a host-rendered semantic rule before opening the
                # per-Java-operand model loop, but only for exact, inspected
                # event bindings with no overlapping state/UI ownership.
                rule_template_used = False
                if source is None and direct_feature:
                    from .template_first_route import checked_rule_template_ready
                    if checked_rule_template_ready(
                        effective_spec, behavior_rows,
                        callbacks=prompt["available_fabric_callbacks"],
                        retained=prompt["retained_typed_behavior"],
                        owned_runtime_records=prompt["owned_runtime_records"],
                        source_ui_callbacks=json.loads(
                            bindings.get("source_ui_callbacks_json") or "[]"),
                    ):
                        from .host_rule_feature import compose_rule_feature
                        source = compose_rule_feature(
                            router, effective_spec, context,
                            package=package, name=name,
                            behaviors=behavior_rows, state_access=state_access,
                            prerequisite_apis=prompt["materialized_prerequisite_api"],
                        )
                        rule_template_used = True
                        print("FEATURE_SOURCE_AUTHOR: " + json.dumps({
                            "owner": owner, "mode": "checked_host_rule_template",
                            "assigned_behaviors": len(behavior_rows),
                        }, ensure_ascii=False), flush=True)
                if source is None:
                    source = compose_source(
                        router, effective_spec, [
                            {"role": "system", "content": instructions},
                            {"role": "user", "content": json.dumps(
                                prompt, ensure_ascii=False, default=str)},
                        ], source_context=source_context,
                        construction_context={
                            **context, "host_state_access": state_access,
                            "host_state_work_order_scope": prompt["host_state_work_order_scope"],
                            "runtime_mutation_receipts": mutation_receipts,
                            "incremental_state": incremental_state,
                        },
                        behaviors=behavior_rows, legacy_author=compose_first_pass_java,
                    )
            acceptance = (context.get("feature_work_order") or {}).get("acceptance") or {}
            budget = acceptance.get("max_compiler_repair_attempts", 2)
            if type(budget) is not int or budget < 0:
                raise HostJavaConstructionError("JAVA_HOST_REPAIR_BUDGET_INVALID")
            from .construction_host_member_repair import repair_compiler_member
            from .construction_journal import JournaledConstructionRouter
            # Parsed declarations are not compiler-verified repairs. Replaying
            # a rejected correction would freeze a fresh session at the same
            # failure. Resume the whole candidate, then author only its failing
            # member against the latest compiler diagnostic.
            repair_router = router.router if isinstance(router, JournaledConstructionRouter) else router
            # The source-owned fallback is a different compiler protocol,
            # not an incremental AST. Never send its Java compilation errors
            # to repair_incremental_program (there is no saved program), and
            # never give it the incremental route's unlimited per-cell loop.
            incremental_protocol = (direct_feature and
                                    incremental_state.get("strategy") != "host_source_owned")
            for attempt in compile_attempts(incremental_protocol, budget):
                try:
                    _verify_full_source(source, spec, context, package=package,
                                        name=name, source_context=source_context)
                except (RuntimeError, ValueError, OSError) as compiler_error:
                    # A host-rendered rule is not an incremental Java program.
                    # Never send its diagnostic to the unrelated member repair
                    # loop or silently replace it with a different trigger.
                    if locals().get("rule_template_used", False):
                        raise HostJavaConstructionError(
                            "HOST_RULE_TEMPLATE_COMPILATION_FAILED: "
                            + str(compiler_error)
                        ) from compiler_error
                    _save_checkpoint(candidate, identifier, candidate_data(
                        source, str(compiler_error) or type(compiler_error).__name__, incremental_state))
                    if exhausted(incremental_protocol, attempt, budget):
                        raise
                    repair_context = _repair_context(spec, context, name, owner, prompt,
                        state_access, behavior_rows, mutation_receipts, attempt)
                    if incremental_protocol:
                        from .incremental_java_pipeline import repair_member
                        source = repair_member(
                            repair_router, source, str(compiler_error),
                            context={**context, **repair_context, "implementation_context": prompt,
                                     "incremental_state": incremental_state, "incremental_spec": effective_spec,
                                     "incremental_messages": [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
                                     "host_state_access": state_access})
                    else:
                        source = repair_compiler_member(
                            repair_router, source, str(compiler_error), **repair_context)
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



def _verification_boundary():
    from functools import partial
    from .construction_file_author import _verify_full_source
    from .host_typed_state_access import validate_host_state_access, validate_behavior_state_provenance
    from .incremental_java_pipeline import verify_full_source
    return partial(verify_full_source, verifier=_verify_full_source,
                   state_validators=(validate_host_state_access, validate_behavior_state_provenance))


def _repair_context(spec, context, name, owner, prompt, state_access, behaviors, receipts, attempt):
    return {"filename": name + ".java", "target_path": spec["target_path"],
            "symbols": context.get("symbols") or {}, "owner": owner,
            "api_evidence": prompt["inspected_target_api"],
            "prerequisite_api": prompt["materialized_prerequisite_api"],
            "state_access": state_access, "behaviors": behaviors,
            "runtime_state_bindings": _repair_runtime_bindings(spec, receipts), "attempt": attempt + 1}
