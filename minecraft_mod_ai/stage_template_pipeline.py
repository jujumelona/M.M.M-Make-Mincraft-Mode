from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from typing import Any, Callable, Mapping

from .planner_operation import planner_operation
from .task_template_catalog import load_template

KNOWN_STAGES: tuple[str, ...] = ("code", "asset", "integration", "validation")


def _stage_binding(stage: str, identifier: str, context: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps([stage, identifier, context], sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def validate_stage_workflows() -> None:
    for stage in KNOWN_STAGES:
        workflow = load_template(f"{stage}/workflow")
        if workflow.get("id") != f"{stage}/workflow":
            raise ValueError(f"STAGE_WORKFLOW: invalid workflow identity for {stage}")
        steps = workflow.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"STAGE_WORKFLOW: {stage}/workflow has no steps")
        for identifier in steps:
            template = load_template(identifier)
            if template.get("id") != identifier:
                raise ValueError(f"STAGE_TEMPLATE: invalid template identity {identifier}")
            if not isinstance(template.get("input"), Mapping) or not isinstance(template.get("output"), Mapping):
                raise ValueError(f"STAGE_TEMPLATE: {identifier} must declare input and output")
            proof = template.get("proof")
            if not isinstance(proof, Mapping) or not str(proof.get("predicate") or "").strip():
                raise ValueError(f"STAGE_TEMPLATE: {identifier} must declare a proof predicate")


# --- Code stage evaluators ---

def _eval_code_file_plan(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    file_target = out.get("file_target") or ctx.get("file_target")
    reason = out.get("ownership_reason") or ctx.get("ownership_reason")
    if not file_target or (isinstance(file_target, dict) and not (file_target.get("owner") or file_target.get("path"))):
        return out, False, "Target file lacks explicit responsibility owner or path", "BLOCKED"
    if not reason:
        return out, False, "Missing ownership reason for target file", "BLOCKED"
    out["file_target"] = file_target
    out["ownership_reason"] = reason
    return out, True, "", "PASS"


def _eval_code_class_plan(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    contract = out.get("class_contract") or ctx.get("class_contract")
    if not isinstance(contract, Mapping):
        return out, False, "Missing or invalid class_contract", "BLOCKED"
    resp = contract.get("primary_responsibility") or contract.get("responsibility")
    owner = contract.get("file_owner") or ctx.get("file_target")
    if not resp or not owner:
        return out, False, "Class contract requires primary responsibility and verified file owner", "BLOCKED"
    out["class_contract"] = dict(contract)
    return out, True, "", "PASS"


def _eval_code_method_plan(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    contract = out.get("method_contract") or ctx.get("method_contract")
    if not contract:
        return out, False, "Method contract missing", "BLOCKED"
    methods = [contract] if isinstance(contract, Mapping) else list(contract)
    for m in methods:
        step = m.get("behavior_step") or ctx.get("behavior_step")
        if not step:
            return out, False, "Method contract must implement exactly one supplied behavior step", "BLOCKED"
    out["method_contract"] = contract
    return out, True, "", "PASS"


def _eval_code_field_plan(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    unres = out.get("unresolved_type") or ctx.get("unresolved_type") or []
    if unres:
        out["unresolved_type"] = unres
        return out, False, f"Unresolved field types present: {unres}", "BLOCKED"
    fields = out.get("field_contract") or ctx.get("field_contract") or []
    field_list = [fields] if isinstance(fields, Mapping) else list(fields)
    for f in field_list:
        if isinstance(f, Mapping):
            owner = f.get("state_owner") or ctx.get("class_contract", {}).get("name")
            purpose = f.get("purpose")
            if not owner or not purpose:
                return out, False, "Every field must have an explicit state owner and purpose", "BLOCKED"
    out["unresolved_type"] = []
    out["field_contract"] = fields
    return out, True, "", "PASS"


def _eval_code_registration_plan(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    blocked = out.get("blocked_reason") or ctx.get("blocked_reason")
    if blocked:
        out["blocked_reason"] = blocked
        return out, False, f"Registration blocked: {blocked}", "BLOCKED"
    reg = out.get("registration_operation") or ctx.get("registration_operation")
    if not isinstance(reg, Mapping):
        return out, False, "Registration operation contract missing", "BLOCKED"
    surface = reg.get("registry_surface") or reg.get("registry")
    ident = reg.get("identifier") or reg.get("id")
    init_point = reg.get("initialization_point") or reg.get("init_point")
    if not surface or not ident or not init_point:
        return out, False, "Registration requires verified registry surface, owned identifier, and initialization point", "BLOCKED"
    out["registration_operation"] = dict(reg)
    out["blocked_reason"] = ""
    return out, True, "", "PASS"


def _eval_code_call_graph_plan(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    edge = out.get("call_edge") or ctx.get("call_edge")
    if not edge:
        return out, False, "Call edge missing", "BLOCKED"
    edges = [edge] if isinstance(edge, Mapping) else list(edge)
    for e in edges:
        p = e.get("producer")
        c = e.get("consumer")
        r = e.get("reason")
        if not p or not c or not r:
            return out, False, "Call edge requires verified producer, consumer, and reason", "BLOCKED"
    out["call_edge"] = edge
    return out, True, "", "PASS"


def _eval_code_dependency_plan(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    unres = out.get("unresolved_dependency") or ctx.get("unresolved_dependency") or []
    if unres:
        out["unresolved_dependency"] = unres
        return out, False, f"Unresolved dependencies present: {unres}", "BLOCKED"
    deps = out.get("dependency") or ctx.get("dependency") or []
    dep_list = [deps] if isinstance(deps, Mapping) else list(deps)
    for d in dep_list:
        if isinstance(d, Mapping) and not (d.get("provider") or d.get("coordinates")):
            return out, False, "Every selected dependency must have a verified provider", "BLOCKED"
    out["dependency"] = deps
    out["unresolved_dependency"] = []
    return out, True, "", "PASS"


def _eval_code_import_plan(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    unres = out.get("unresolved_symbols") or ctx.get("unresolved_symbols") or []
    if unres:
        out["unresolved_symbols"] = unres
        return out, False, f"Unresolved symbols present: {unres}", "BLOCKED"
    out["imports"] = out.get("imports") or ctx.get("imports") or []
    out["unresolved_symbols"] = []
    return out, True, "", "PASS"


def _eval_code_generation_unit(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    blocked = out.get("blocked_reasons") or ctx.get("blocked_reasons") or []
    if blocked:
        out["blocked_reasons"] = blocked
        return out, False, f"Generation unit blocked: {blocked}", "BLOCKED"
    unit = out.get("generation_unit") or ctx.get("generation_unit")
    if not isinstance(unit, Mapping) or not (unit.get("owned_anchor") or unit.get("anchor") or ctx.get("file_target")):
        return out, False, "Generation unit must be bounded to one owned anchor", "BLOCKED"
    out["generation_unit"] = dict(unit)
    out["blocked_reasons"] = []
    return out, True, "", "PASS"


# --- Asset stage evaluators ---

def _eval_asset_asset_requirement(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    reqs = out.get("asset_requirements") or ctx.get("asset_requirements") or []
    req_list = [reqs] if isinstance(reqs, Mapping) else list(reqs)
    for r in req_list:
        if isinstance(r, Mapping):
            trace = r.get("trace_ref") or r.get("feature_id") or r.get("source_requirement")
            if not trace:
                return out, False, "Every asset requirement must trace to an explicit feature or resource requirement", "BLOCKED"
    out["asset_requirements"] = reqs
    return out, True, "", "PASS"


def _eval_asset_texture_plan(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    unres = out.get("unresolved_properties") or ctx.get("unresolved_properties") or []
    if unres:
        out["unresolved_properties"] = unres
        return out, False, f"Unresolved texture properties: {unres}", "BLOCKED"
    contract = out.get("texture_contract") or ctx.get("texture_contract")
    if contract is None:
        return out, False, "Texture contract missing", "BLOCKED"
    out["texture_contract"] = contract
    out["unresolved_properties"] = []
    return out, True, "", "PASS"


def _eval_asset_model_plan(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    unres = out.get("unresolved_properties") or ctx.get("unresolved_properties") or []
    if unres:
        out["unresolved_properties"] = unres
        return out, False, f"Unresolved model properties: {unres}", "BLOCKED"
    contract = out.get("model_contract") or ctx.get("model_contract")
    if contract is None:
        return out, False, "Model contract missing", "BLOCKED"
    out["model_contract"] = contract
    out["unresolved_properties"] = []
    return out, True, "", "PASS"


def _eval_asset_sound_plan(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    unres = out.get("unresolved_properties") or ctx.get("unresolved_properties") or []
    if unres:
        out["unresolved_properties"] = unres
        return out, False, f"Unresolved sound properties: {unres}", "BLOCKED"
    contract = out.get("sound_contract") or ctx.get("sound_contract")
    if contract is None:
        return out, False, "Sound contract missing", "BLOCKED"
    out["sound_contract"] = contract
    out["unresolved_properties"] = []
    return out, True, "", "PASS"


def _eval_asset_animation_plan(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    unres = out.get("unresolved_properties") or ctx.get("unresolved_properties") or []
    if unres:
        out["unresolved_properties"] = unres
        return out, False, f"Unresolved animation properties: {unres}", "BLOCKED"
    contract = out.get("animation_contract") or ctx.get("animation_contract")
    if contract is None:
        return out, False, "Animation contract missing", "BLOCKED"
    out["animation_contract"] = contract
    out["unresolved_properties"] = []
    return out, True, "", "PASS"


def _eval_asset_resource_validation(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    failed = list(out.get("failed_checks") or ctx.get("failed_checks") or [])
    explicit_valid = out.get("valid") if "valid" in out else ctx.get("valid")
    if failed or explicit_valid is False:
        out["valid"] = False
        out["failed_checks"] = failed or ["Explicit invalid resource state"]
        return out, False, f"Resource validation failed checks: {out['failed_checks']}", "FAIL"
    out["valid"] = True
    out["failed_checks"] = []
    return out, True, "", "PASS"


# --- Integration stage evaluators ---

def _eval_integration_registration(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    conflicts = list(out.get("conflicts") or ctx.get("conflicts") or [])
    ops = ctx.get("registration_operations") or out.get("registration_operations") or []
    if isinstance(ops, Mapping):
        ops = [ops]
    seen = {}
    for op in ops:
        ident = op.get("identifier") or op.get("id")
        if ident:
            if ident in seen:
                conflicts.append(f"Duplicate registration identifier: {ident}")
            seen[ident] = op
    out["conflicts"] = conflicts
    out["registration_edges"] = out.get("registration_edges") or ctx.get("registration_edges") or []
    if conflicts:
        return out, False, f"Registration conflicts detected: {conflicts}", "BLOCKED"
    return out, True, "", "PASS"


def _eval_integration_dependency_connect(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    cycles = list(out.get("cycles") or ctx.get("cycles") or [])
    edges = out.get("dependency_edges") or ctx.get("dependency_edges") or []
    graph = defaultdict(list)
    for edge in edges:
        u = edge.get("consumer") or edge.get("from")
        v = edge.get("provider") or edge.get("to")
        if u and v:
            graph[u].append(v)
        else:
            cycles.append(f"Invalid dependency edge missing provider or consumer: {edge}")
    visited = {}

    def dfs(node, path):
        visited[node] = 1
        for nxt in graph.get(node, []):
            if visited.get(nxt) == 1:
                cycle_str = " -> ".join(path + [nxt])
                cycles.append(f"Cycle detected: {cycle_str}")
            elif visited.get(nxt) != 2:
                dfs(nxt, path + [nxt])
        visited[node] = 2

    for node in list(graph.keys()):
        if visited.get(node) is None:
            dfs(node, [node])

    out["cycles"] = cycles
    out["dependency_edges"] = edges
    if cycles:
        return out, False, f"Dependency cycle or invalid edge detected: {cycles}", "BLOCKED"
    return out, True, "", "PASS"


def _eval_integration_feature_connect(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    unres = list(out.get("unresolved_connections") or ctx.get("unresolved_connections") or [])
    edges = out.get("feature_edges") or ctx.get("feature_edges") or []
    for edge in edges:
        if isinstance(edge, Mapping) and not (edge.get("contract") or edge.get("connection_contract")):
            unres.append(f"Edge missing contract: {edge}")
    out["unresolved_connections"] = unres
    out["feature_edges"] = edges
    if unres:
        return out, False, f"Unresolved feature connections: {unres}", "BLOCKED"
    return out, True, "", "PASS"


def _eval_integration_client_server_connect(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    violations = list(out.get("authority_violations") or ctx.get("authority_violations") or [])
    out["authority_violations"] = violations
    out["side_edges"] = out.get("side_edges") or ctx.get("side_edges") or []
    if violations:
        return out, False, f"Cross-side authority violations: {violations}", "BLOCKED"
    return out, True, "", "PASS"


def _eval_integration_resource_connect(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    missing = list(out.get("missing_resources") or ctx.get("missing_resources") or [])
    orphaned = list(out.get("orphaned_resources") or ctx.get("orphaned_resources") or [])
    refs = ctx.get("code_references") or []
    artifacts = ctx.get("resource_artifacts") or []
    if refs and artifacts:
        art_ids = {a.get("id") or a.get("path") for a in artifacts if isinstance(a, Mapping)}
        for ref in refs:
            target = ref.get("resource_id") or ref.get("target") or ref.get("path")
            if target and target not in art_ids:
                missing.append(target)
    out["missing_resources"] = missing
    out["orphaned_resources"] = orphaned
    out["resource_edges"] = out.get("resource_edges") or ctx.get("resource_edges") or []
    if missing or orphaned:
        return out, False, f"Resource connect errors: missing={missing}, orphaned={orphaned}", "BLOCKED"
    return out, True, "", "PASS"


def _eval_integration_call_graph_connect(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    invalid = list(out.get("invalid_edges") or ctx.get("invalid_edges") or [])
    units = ctx.get("generation_units") or []
    edges = out.get("call_edges") or ctx.get("call_edges") or []
    if units and edges:
        known = {u.get("id") or u.get("anchor") or u.get("name") for u in units if isinstance(u, Mapping)}
        for edge in edges:
            c = edge.get("caller") or edge.get("producer")
            callee = edge.get("callee") or edge.get("consumer")
            if c and c not in known:
                invalid.append(f"Caller '{c}' not in known units")
            if callee and callee not in known:
                invalid.append(f"Callee '{callee}' not in known units")
    out["invalid_edges"] = invalid
    out["call_graph"] = out.get("call_graph") or ctx.get("call_graph") or {}
    if invalid:
        return out, False, f"Invalid call graph edges: {invalid}", "BLOCKED"
    return out, True, "", "PASS"


# --- Validation stage evaluators ---

def _eval_validation_compile(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    diags = list(out.get("diagnostics") or ctx.get("diagnostics") or [])
    explicit_passed = out.get("passed") if "passed" in out else ctx.get("passed")
    exit_code = ctx.get("compile_exit_code") or ctx.get("exit_code", 0)
    has_error_diag = any(
        (isinstance(d, Mapping) and d.get("severity") == "error") or ("error" in str(d).lower())
        for d in diags
    )
    if explicit_passed is False or exit_code != 0 or has_error_diag:
        out["passed"] = False
        out["diagnostics"] = diags or [f"Compilation failed with exit code {exit_code}"]
        return out, False, f"Compilation failed with diagnostics: {out['diagnostics']}", "FAIL"
    out["passed"] = True
    out["diagnostics"] = diags
    return out, True, "", "PASS"


def _eval_validation_static_check(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    failed = list(out.get("failed_checks") or ctx.get("failed_checks") or [])
    explicit_passed = out.get("passed") if "passed" in out else ctx.get("passed")
    if explicit_passed is False or failed:
        out["passed"] = False
        out["failed_checks"] = failed or ["Static check failure"]
        return out, False, f"Static checks failed: {out['failed_checks']}", "FAIL"
    out["passed"] = True
    out["failed_checks"] = []
    return out, True, "", "PASS"


def _eval_validation_unit_test(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    failures = list(out.get("failures") or ctx.get("failures") or [])
    explicit_passed = out.get("passed") if "passed" in out else ctx.get("passed")
    if explicit_passed is False or failures:
        out["passed"] = False
        out["failures"] = failures or ["Unit test failed"]
        return out, False, f"Unit tests failed: {out['failures']}", "FAIL"
    out["passed"] = True
    out["failures"] = []
    return out, True, "", "PASS"


def _eval_validation_integration_test(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    failures = list(out.get("failures") or ctx.get("failures") or [])
    explicit_passed = out.get("passed") if "passed" in out else ctx.get("passed")
    if explicit_passed is False or failures:
        out["passed"] = False
        out["failures"] = failures or ["Integration test failed"]
        return out, False, f"Integration tests failed: {out['failures']}", "FAIL"
    out["passed"] = True
    out["failures"] = []
    return out, True, "", "PASS"


def _eval_validation_runtime_test(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    failures = list(out.get("failures") or ctx.get("failures") or [])
    explicit_passed = out.get("passed") if "passed" in out else ctx.get("passed")
    if explicit_passed is False or failures:
        out["passed"] = False
        out["failures"] = failures or ["Runtime test failed"]
        return out, False, f"Runtime tests failed: {out['failures']}", "FAIL"
    out["passed"] = True
    out["failures"] = []
    out["observations"] = out.get("observations") or ctx.get("observations") or []
    return out, True, "", "PASS"


def _eval_validation_gameplay_test(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    failed = list(out.get("failed_acceptance") or ctx.get("failed_acceptance") or [])
    explicit_passed = out.get("passed") if "passed" in out else ctx.get("passed")
    if explicit_passed is False or failed:
        out["passed"] = False
        out["failed_acceptance"] = failed or ["Gameplay acceptance failure"]
        return out, False, f"Gameplay acceptance failed: {out['failed_acceptance']}", "FAIL"
    out["passed"] = True
    out["failed_acceptance"] = []
    return out, True, "", "PASS"


def _eval_validation_requirement_trace(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    unproven = list(out.get("unproven_requirements") or ctx.get("unproven_requirements") or [])
    reqs = ctx.get("requirements") or []
    if reqs:
        impl = set(ctx.get("implementation_receipts") or [])
        test = set(ctx.get("test_receipts") or [])
        for r in reqs:
            rid = r if isinstance(r, str) else r.get("id")
            if rid and (rid not in impl or rid not in test):
                unproven.append(rid)
    out["unproven_requirements"] = unproven
    out["traces"] = out.get("traces") or ctx.get("traces") or []
    if unproven:
        return out, False, f"Unproven mandatory requirements: {unproven}", "BLOCKED"
    return out, True, "", "PASS"


def _eval_validation_feature_trace(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    unproven = list(out.get("unproven_features") or ctx.get("unproven_features") or [])
    feats = ctx.get("atomic_features") or []
    if feats:
        artifacts = set(ctx.get("artifact_receipts") or [])
        tests = set(ctx.get("test_receipts") or [])
        for f in feats:
            fid = f if isinstance(f, str) else f.get("feature_id")
            if fid and (fid not in artifacts or fid not in tests):
                unproven.append(fid)
    out["unproven_features"] = unproven
    out["traces"] = out.get("traces") or ctx.get("traces") or []
    if unproven:
        return out, False, f"Atomic features lack artifact or test proof: {unproven}", "BLOCKED"
    return out, True, "", "PASS"


def _eval_validation_resource_trace(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    missing = list(out.get("missing_resources") or ctx.get("missing_resources") or [])
    unowned = list(out.get("unowned_resources") or ctx.get("unowned_resources") or [])
    reqs = ctx.get("resource_requirements") or []
    if reqs:
        gen = {r if isinstance(r, str) else r.get("id") for r in ctx.get("generated_resources", [])}
        for r in reqs:
            rid = r if isinstance(r, str) else r.get("id")
            if rid and rid not in gen:
                missing.append(rid)
    out["missing_resources"] = missing
    out["unowned_resources"] = unowned
    out["traces"] = out.get("traces") or ctx.get("traces") or []
    if missing or unowned:
        return out, False, f"Resource trace errors: missing={missing}, unowned={unowned}", "BLOCKED"
    return out, True, "", "PASS"


def _eval_validation_completeness(ctx: Mapping[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any], bool, str, str]:
    blockers = list(out.get("blockers") or ctx.get("blockers") or [])
    validation_receipts = ctx.get("validation_receipts") or []
    for r in validation_receipts:
        if r.get("template_id") == "validation/completeness":
            continue
        p = r.get("proof") or {}
        if not p.get("passed") or r.get("status") != "PASS":
            blockers.append(f"{r.get('template_id')}: {p.get('reason') or r.get('status')}")
    for k in ("unproven_requirements", "unproven_features", "missing_resources", "unowned_resources"):
        val = ctx.get(k)
        if val:
            blockers.append(f"Unresolved {k}: {val}")
    out["blockers"] = blockers
    if blockers:
        out["complete"] = False
        return out, False, f"Completeness blocked: {blockers}", "BLOCKED"
    out["complete"] = True
    return out, True, "", "PASS"


STAGE_EVALUATORS: dict[str, Callable[[Mapping[str, Any], dict[str, Any]], tuple[dict[str, Any], bool, str, str]]] = {
    "code/file_plan": _eval_code_file_plan,
    "code/class_plan": _eval_code_class_plan,
    "code/method_plan": _eval_code_method_plan,
    "code/field_plan": _eval_code_field_plan,
    "code/registration_plan": _eval_code_registration_plan,
    "code/call_graph_plan": _eval_code_call_graph_plan,
    "code/dependency_plan": _eval_code_dependency_plan,
    "code/import_plan": _eval_code_import_plan,
    "code/generation_unit": _eval_code_generation_unit,
    "asset/asset_requirement": _eval_asset_asset_requirement,
    "asset/texture_plan": _eval_asset_texture_plan,
    "asset/model_plan": _eval_asset_model_plan,
    "asset/sound_plan": _eval_asset_sound_plan,
    "asset/animation_plan": _eval_asset_animation_plan,
    "asset/resource_validation": _eval_asset_resource_validation,
    "integration/registration": _eval_integration_registration,
    "integration/dependency_connect": _eval_integration_dependency_connect,
    "integration/feature_connect": _eval_integration_feature_connect,
    "integration/client_server_connect": _eval_integration_client_server_connect,
    "integration/resource_connect": _eval_integration_resource_connect,
    "integration/call_graph_connect": _eval_integration_call_graph_connect,
    "validation/compile": _eval_validation_compile,
    "validation/static_check": _eval_validation_static_check,
    "validation/unit_test": _eval_validation_unit_test,
    "validation/integration_test": _eval_validation_integration_test,
    "validation/runtime_test": _eval_validation_runtime_test,
    "validation/gameplay_test": _eval_validation_gameplay_test,
    "validation/requirement_trace": _eval_validation_requirement_trace,
    "validation/feature_trace": _eval_validation_feature_trace,
    "validation/resource_trace": _eval_validation_resource_trace,
    "validation/completeness": _eval_validation_completeness,
}


def _evaluate_step(
    identifier: str,
    context: Mapping[str, Any],
    template: Mapping[str, Any],
    initial_output: dict[str, Any],
) -> tuple[dict[str, Any], bool, str, str]:
    evaluator = STAGE_EVALUATORS.get(identifier)
    if evaluator is not None:
        return evaluator(context, initial_output)

    # Fallback generic evaluator
    output = dict(initial_output)
    for blocker_key in ("blocked_reason", "blocked_reasons", "failed_checks", "failures", "cycles", "conflicts"):
        val = output.get(blocker_key) or context.get(blocker_key)
        if val:
            return output, False, f"Blocked by {blocker_key}: {val}", "BLOCKED"
    return output, True, "", "PASS"


def execute_stage_step(
    stage: str,
    identifier: str,
    *,
    context: Mapping[str, Any],
    progress: dict[str, Any] | None = None,
    checkpoint=None,
) -> dict[str, Any]:
    template = load_template(identifier)
    binding = _stage_binding(stage, identifier, context)
    saved = (progress or {}).get(binding)
    if saved is not None:
        return deepcopy(saved)

    proof = template["proof"]
    predicate = str(proof.get("predicate") or "").strip()

    output_keys = list(template.get("output", {}).keys())
    output: dict[str, Any] = {}
    for key in output_keys:
        val = context.get(key)
        output[key] = deepcopy(val) if val is not None else []

    output, passed, reason, status = _evaluate_step(identifier, context, template, output)

    proof_payload: dict[str, Any] = {"passed": passed, "predicate": predicate}
    if not passed and reason:
        proof_payload["reason"] = reason

    receipt = {
        "template_id": identifier,
        "status": status,
        "output": output,
        "proof": proof_payload,
    }

    if checkpoint is not None:
        checkpoint(binding, deepcopy(receipt))
    return receipt


def run_stage_pipeline(
    stage_name: str,
    context: Mapping[str, Any],
    *,
    progress: dict[str, Any] | None = None,
    checkpoint=None,
) -> dict[str, Any]:
    if stage_name not in KNOWN_STAGES:
        raise ValueError(f"STAGE_PIPELINE: unknown stage {stage_name}")

    workflow = load_template(f"{stage_name}/workflow")
    receipts: list[dict[str, Any]] = []
    accumulated: dict[str, Any] = dict(context)

    for identifier in workflow["steps"]:
        with planner_operation(identifier):
            receipt = execute_stage_step(
                stage_name,
                identifier,
                context=accumulated,
                progress=progress,
                checkpoint=checkpoint,
            )
            receipts.append(receipt)
            accumulated.update(receipt.get("output", {}))
            if stage_name == "validation":
                accumulated["validation_receipts"] = list(receipts)

    return {
        "stage": stage_name,
        "receipts": receipts,
        "results": accumulated,
    }
