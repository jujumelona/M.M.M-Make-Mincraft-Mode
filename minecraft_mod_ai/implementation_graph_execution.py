"""Transactional execution of host-admitted implementation source units."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .complete_spec import ProductionModule
from .implementation_ir import (
    ImplementationGraphError,
    OutputBudgetExhausted,
    admissible_tokens,
    compile_authored_graph,
    digest,
    node_cost,
    ordered_nodes,
    refine_node,
    validate_node,
)
from .implementation_lifecycle import ACTIVATION_API, activation_call
from .project_write_lock import project_write_lock
from .root_cause_trace import emit_root_cause


def public_api_errors(source: str, node: Mapping[str, Any]) -> tuple[str, ...]:
    # Strip comments so a declaration in an explanation cannot satisfy the contract.
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)
    normalized = re.sub(r"\s+", " ", code)
    errors = []
    for api in node["public_api"]:
        declaration = re.sub(r"\s+", " ", api.strip().rstrip(";"))
        if not re.search(re.escape(declaration) + r"\s*(?:\{|;|=|throws\b)", normalized):
            errors.append(f"Frozen implementation API missing or changed: {api}")
    return tuple(errors)


def _bind_atomic_leaf_contract(
    task: dict[str, Any], node: Mapping[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    from .authored_execution_schema import concern_contracts, section_for_symbol
    from .authored_production import _task_sha

    section = section_for_symbol(str(node.get("symbol") or ""))
    concerns = list(concern_contracts(section))
    if concerns:
        task["implementation_obligations"] = list(node["obligations"])
        task["task_sha256"] = _task_sha(task)
    return section, concerns

def _leaf_module(node: dict[str, Any], graph: dict[str, Any], request: dict[str, Any]) -> ProductionModule:
    from .authored_production import _exact_authored_task

    by_symbol = {n["symbol"]: n for n in graph["nodes"]}
    dependencies = []
    for symbol in node["depends_on"]:
        dependency = by_symbol[symbol]
        # Explicit dependency contracts never disappear behind a file-count/context cap.
        dependencies.append({"symbol": symbol, "path": dependency["path"],
                             "public_api": dependency["public_api"],
                             "responsibility": dependency["responsibility"]})
    requirements = {r: graph["requirements"][r] for r in node["requirements"]}
    task = _exact_authored_task(
        task_id="ir_" + node["symbol"].lower(), path=node["path"], symbol=node["symbol"],
        target=request["target"], obligation=json.dumps({
            "responsibility": node["responsibility"], "obligations": node["obligations"],
            "source_requirements": requirements, "public_api": node["public_api"],
            "dependencies": dependencies,
            "rules": (
                "Implement only this responsibility. Preserve every frozen API. Use dependency APIs; "
                "do not duplicate their state. Do not add entrypoints or sibling files. "
                + (
                    f"The host invokes {activation_call(node['symbol'])} from the mod entrypoint. "
                    "Implement this node's runtime registration in that hook. An onInitialize member "
                    "is an ordinary method, not a separately registered Fabric entrypoint."
                    if node["activation"] else ""
                )
            ),
        }, ensure_ascii=False), semantic_outcome=node["responsibility"],
        depends_on=tuple("ir_" + symbol.lower() for symbol in node["depends_on"]),
        consumes=tuple(node["depends_on"]), provides=(node["symbol"],),
        worksheet={"implementation_ir_node": node}, required_gates=("target_compile",),
        target_status="host_reserved",
    )
    section, atomic_concerns = _bind_atomic_leaf_contract(task, node)
    return ProductionModule(
        module_id="ir_" + node["symbol"].lower(), kind="custom_java",
        config={"implementation": "custom", "evidence_task": task,
                "implementation_ir_node": node,
                "implementation_section": section,
                "implementation_atomic_concerns": atomic_concerns,
                "implementation_dependency_context": json.dumps(dependencies, ensure_ascii=False),
                **request["target"]}, required_gates=("target_compile",),
    )


def execute_implementation_graph(generator: Any, project_root: str | Path, *,
                                 module: ProductionModule,
                                 execution_feedback: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from . import custom_module_generator as direct

    module.validate(policy=generator.policy)
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ImplementationGraphError("IMPLEMENTATION_IR_PROJECT_REQUIRED")
    request = module.config["implementation_graph_request"]
    target = request["target"]
    package, mod_id = request["package"], request["mod_id"]
    entry = direct._safe_target(root, direct._normalize_project_path(request["entrypoint_path"]))
    if not entry.is_file():
        raise ImplementationGraphError("IMPLEMENTATION_IR_ENTRYPOINT_MISSING")
    cache = direct._safe_target(root, f".minecraft_ai/implementation-ir/{digest(request)}.json")
    originals: dict[str, bytes | None] = {}
    receipts: list[dict[str, Any]] = []

    def remember(path: str) -> Path:
        dest = direct._safe_target(root, direct._normalize_project_path(path))
        # Reject symlink parents too; resolving a path alone would hide an in-root alias.
        unresolved = root / path
        if any(p.is_symlink() for p in (unresolved, *unresolved.parents) if p != root):
            raise ImplementationGraphError("IMPLEMENTATION_IR_SYMLINK_TARGET")
        if path not in originals:
            originals[path] = dest.read_bytes() if dest.is_file() else None
        return dest

    def save() -> None:
        if "graph" in state:
            state["graph_hash"] = digest(state["graph"])
        direct._atomic_write(cache, json.dumps(state, ensure_ascii=False, indent=2))

    def save_compilation(draft: dict[str, Any]) -> None:
        state["compilation"] = draft
        save()

    def save_refinement(pending: dict[str, Any]) -> None:
        state["refinement_pending"] = pending
        save()

    with project_write_lock(root):
        if cache.is_file():
            state = json.loads(cache.read_text(encoding="utf-8"))
            if state.get("request_hash") != digest(request):
                raise ImplementationGraphError("IMPLEMENTATION_IR_CHECKPOINT_DRIFT")
        else:
            state = {"request_hash": digest(request), "blocked_decodes": [], "refinements": 0}
        if "graph" in state:
            graph = state["graph"]
            if state.get("graph_hash") != digest(graph) or graph.get("source_text") != request["text"]:
                raise ImplementationGraphError("IMPLEMENTATION_IR_CHECKPOINT_DRIFT")
            # A valid hash is not proof that an older frontend admitted legal APIs.
            # Recheck contracts before any cached node can create or mutate source.
            for cached_node in graph["nodes"]:
                admitted = validate_node({k: v for k, v in cached_node.items() if k != "path"},
                                         package=package, mod_id=mod_id, refs=set(graph["requirements"]))
                if admitted != cached_node:
                    raise ImplementationGraphError("IMPLEMENTATION_IR_CHECKPOINT_DRIFT")
            graph["nodes"] = ordered_nodes(graph["nodes"])
        else:
            graph = compile_authored_graph(generator.router, text=request["text"], package=package,
                                  mod_id=mod_id, target=target,
                                  context=entry.read_text(encoding="utf-8"),
                                  resume=state.get("compilation"), checkpoint=save_compilation)
            state["graph"] = graph
            state.pop("compilation", None)
            save()
        if any(n["path"].casefold() == request["entrypoint_path"].casefold() for n in graph["nodes"]):
            raise ImplementationGraphError("IMPLEMENTATION_IR_ENTRYPOINT_RESERVED")
        runtime_budget = admissible_tokens(generator.router)
        emit_root_cause("implementation_graph_compiled", stage="production", result="PASS",
                        details={"graph": graph, "output_allowance": runtime_budget,
                                 "estimated_output_sizes": {n["symbol"]: node_cost(n) for n in graph["nodes"]}})
        completed: set[str] = set()
        try:
            while len(completed) < len(graph["nodes"]):
                node = next(n for n in graph["nodes"] if n["symbol"] not in completed
                            and set(n["depends_on"]) <= completed)
                fingerprint = digest(node)
                reason = ""
                if fingerprint in state["blocked_decodes"]:
                    reason = "OUTPUT_BUDGET_EXHAUSTED"
                elif runtime_budget is not None and node_cost(node) > runtime_budget:
                    reason = "preflight_output_budget"
                if not reason:
                    if node["path"] == request["entrypoint_path"]:
                        raise ImplementationGraphError("IMPLEMENTATION_IR_ENTRYPOINT_RESERVED")
                    dest = remember(node["path"])
                    before = originals[node["path"]]
                    try:
                        if node["kind"] == "resource":
                            payload = direct._call_coder(generator.router, [
                                {"role": "system", "content": (
                                    'Return {"content":"<complete JSON resource>","summary":"..."}. '
                                    "Implement this exact resource only, using the selected platform format."
                                )},
                                {"role": "user", "content": json.dumps({
                                    "node": node, "platform": target,
                                    "requirements": {r: graph["requirements"][r] for r in node["requirements"]},
                                    "execution_feedback": direct._bounded_execution_feedback(execution_feedback),
                                }, ensure_ascii=False)},
                            ])
                            json.loads(payload["content"])
                            direct._atomic_write(dest, payload["content"])
                        else:
                            leaf = _leaf_module(node, graph, request)
                            if not dest.exists():
                                source = f"package {package};\npublic final class {node['symbol']} {{\n"
                                if node["activation"]:
                                    source += ACTIVATION_API + " { /* MMM_AUTHORED_FEATURE_BODY */ }\n"
                                else:
                                    source += "// MMM_AUTHORED_FEATURE_BODY\n"
                                direct._atomic_write(dest, source + "}\n")
                            result = generator.generate(root, module=leaf,
                                                        minecraft_version=target["minecraft_version"],
                                                        loader=target["loader"],
                                                        execution_feedback=execution_feedback)
                            receipts.append(result)
                    except OutputBudgetExhausted:
                        # The failed whole-file attempt is never compiler-repaired or
                        # resubmitted, even when the outer work ledger resumes this job.
                        state["blocked_decodes"].append(fingerprint)
                        save()
                        if before is None:
                            dest.unlink(missing_ok=True)
                        else:
                            direct._atomic_write(dest, before)
                        reason = "OUTPUT_BUDGET_EXHAUSTED"
                    else:
                        completed.add(node["symbol"])
                if reason:
                    emit_root_cause("implementation_node_decomposition", stage="production", result="START",
                                    details={"reason": reason, "symbol": node["symbol"],
                                             "node_hash": fingerprint, "same_decode_retry": False})
                    if not state.get("refinement_pending"):
                        state["refinements"] += 1
                        save()
                    # Halve the admitted workload after an observed exhaustion. Merely
                    # lowering the model's estimate cannot make the same request eligible.
                    runtime_budget = admissible_tokens(generator.router)
                    budget = (
                        min(runtime_budget, node_cost(node) // 2)
                        if reason == "OUTPUT_BUDGET_EXHAUSTED" and runtime_budget is not None
                        else runtime_budget
                    )
                    refined = refine_node(generator.router, node, nodes=graph["nodes"],
                                          package=package, mod_id=mod_id,
                                          requirements=graph["requirements"], reason=reason, budget=budget,
                                          pending=state.get("refinement_pending"), checkpoint=save_refinement)
                    graph["nodes"] = refined
                    state.pop("refinement_pending", None)
                    save()

            main = remember(request["entrypoint_path"])
            source = main.read_text(encoding="utf-8")
            calls = "\n".join(f"        {activation_call(n['symbol'])}" for n in graph["nodes"] if n["activation"])
            start, end = "// MMM_IR_ACTIVATION_START", "// MMM_IR_ACTIVATION_END"
            block = start + "\n" + calls + "\n        " + end + "\n"
            if start in source:
                source, count = re.subn(re.escape(start) + r".*?" + re.escape(end), lambda _: block,
                                        source, count=1, flags=re.DOTALL)
            else:
                source, count = re.subn(r"(public\s+void\s+onInitialize\s*\(\s*\)\s*\{)",
                                        lambda m: m[1] + "\n        " + block, source, count=1)
            if count != 1:
                raise ImplementationGraphError("IMPLEMENTATION_IR_ENTRYPOINT_BINDING_FAILED")
            direct._atomic_write(main, source)
            report = direct.GradleRunner(generator._cache_dir(root)).compile_java(root)
            if getattr(report, "status", "") != "PASS":
                raise ImplementationGraphError("IMPLEMENTATION_IR_INTEGRATION_COMPILE_FAILED: " + direct._compile_log(report))
            operations = []
            for path, before in originals.items():
                dest = direct._safe_target(root, path)
                # Exhausted parent tasks can disappear only when their stable facade
                # is retained; never report an abandoned or partial file as generated.
                if not dest.is_file():
                    continue
                operations.append({"operation": "create" if before is None else "replace", "path": path,
                                   "before_sha256": direct._sha256_text(before.decode()) if before is not None else "",
                                   "after_sha256": direct._sha256_text(dest.read_text(encoding="utf-8"))})
            paths = [op["path"] for op in operations]
            return {
                "schema_version": "mmm/custom-module-result-v3", "module_id": module.module_id,
                "kind": module.kind, "status": "SOURCE_GENERATED", "touched_paths": paths,
                "patch_receipt": {"schema_version": "mmm/direct-source-write-v1", "status": "APPLIED",
                                  "operations": operations, "touched_paths": paths},
                "operation_count": len(operations), "implementation_ir": graph,
                "source_unit_receipts": receipts, "required_gates": list(module.required_gates),
                "generation_verification": {"status": "PASS", "mode": "gradle_compile_java"},
                "runtime_tests": ["Execute the requested GameTest/runtime gates."],
                "output_exhaustion_continuations": 0, "decomposition_count": state["refinements"],
            }
        except BaseException:
            for path, before in reversed(tuple(originals.items())):
                dest = direct._safe_target(root, path)
                if before is None:
                    dest.unlink(missing_ok=True)
                else:
                    direct._atomic_write(dest, before)
            raise
