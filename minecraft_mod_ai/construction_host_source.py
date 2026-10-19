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
    _claim_checkpoint, _digest, _read_checkpoint, _save_checkpoint,
)


class HostJavaConstructionError(ValueError):
    """A host-assembled compilation unit did not satisfy the target contract."""


def compose_host_java_source(router, spec: dict, context: dict, *,
                             package: str, name: str, owner: str,
                             behaviors=(), source_context=()) -> str:
    from .construction_file_author import (
        _evidence, _read_materialized_context, _verify_full_source,
    )
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
        "implementation": __import__(
            "minecraft_mod_ai.construction_implementation",
            fromlist=["construction_implementation_fingerprint"],
        ).construction_implementation_fingerprint(),
    })
    checkpoint = root / ".mmm" / "construction_host_files" / (identifier + ".json")
    with _claim_checkpoint(checkpoint):
        saved = _read_checkpoint(checkpoint, identifier)
        if saved:
            if set(saved) != {"source"} or not isinstance(saved["source"], str):
                raise HostJavaConstructionError("JAVA_HOST_CHECKPOINT_INVALID")
            try:
                _verify_full_source(saved["source"], spec, context,
                    package=package, name=name, source_context=source_context)
            except (RuntimeError, ValueError, OSError):
                # Do not certify an old source after a prerequisite change.
                pass
            else:
                return saved["source"]

        contract = context.get("authoring_contract") or {}
        requirement = str(spec.get("requirement", ""))
        if (contract.get("canonical_leaf") and contract.get("responsibility")
                and not spec.get("bindings", {}).get("request_feature_entrypoint")):
            requirement = str(contract["responsibility"])
        behavior_rows = [dict(row) for row in behaviors]
        api_intent = requirement + " " + " ".join(
            str(row.get("intent") or "") for row in behavior_rows)
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
            "materialized_project_source": _read_materialized_context(
                context, target_path=spec["target_path"],
                source_context=source_context),
        }
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
            source = compose_first_pass_java(
                router, effective_spec, [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": json.dumps(
                        prompt, ensure_ascii=False, default=str)},
                ], source_context=source_context,
                construction_context=context, behaviors=behavior_rows,
            )
            _verify_full_source(source, spec, context, package=package,
                                name=name, source_context=source_context)
        except (RuntimeError, ValueError, OSError) as exc:
            trace = root / ".mmm" / "construction_host_failures"
            trace.mkdir(parents=True, exist_ok=True)
            (trace / (identifier + ".json")).write_text(json.dumps({
                "target": spec["target_path"], "error": str(exc),
                "source": locals().get("source"),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        from .construction_coverage import record_authored_source_calls
        record_authored_source_calls(
            root, spec["target_path"], source, source_kind="direct")
        _save_checkpoint(checkpoint, identifier, {"source": source})
        print("CONSTRUCTION HOST JAVA AUTHOR: " + json.dumps({
            "target_path": spec["target_path"], "owner": owner,
            "target_compilation_verified": True,
            "runtime_verified": False,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }, ensure_ascii=False), flush=True)
        return source
