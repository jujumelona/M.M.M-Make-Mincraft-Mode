"""Host-owned source-reuse handoff for the default authored production lane.

Discovery candidates are never executable code. Only proven donor decisions
may be materialized into the final workspace. The small model gets a bounded
receipt, never authority to invent or select a donor.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def source_discovery_algorithm_version() -> str:
    """Change when provider discovery or donor handoff semantics change."""
    return "shared-source-candidate-retrieval-v2"


def source_reference_fingerprint(reference_repositories: Sequence[str]) -> str:
    """Bind optional source references to the exact reuse decision."""
    return "sha256:" + _sha(list(reference_repositories))


def source_candidate_fingerprint(
    repository_candidates: Sequence[Mapping[str, Any]],
) -> str:
    """Invalidate reuse decisions when host-provided research inputs change."""
    return "sha256:" + _sha([
        dict(item) for item in repository_candidates if isinstance(item, Mapping)
    ])


def authored_capability_graph(
    requested_prompt: str,
    structured_sections: Mapping[str, Any],
    *, request_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if request_scope and request_scope.get("requirements"):
        nodes = list(dict.fromkeys(row["search_query"] for row in request_scope["requirements"]))
        return {
            "nodes": nodes,
            "search_terms": [{"capability": node, "terms": [f"minecraft {node} mod", node]}
                             for node in nodes],
            "requirement_bindings": [{"requirement_id": row["requirement_id"],
                                      "capability": row["search_query"]}
                                     for row in request_scope["requirements"]],
            "retrieval_policy_version": "request-features-v4-exact-source-partial-fit",
            "coverage_policy": "original request features remain authoritative",
        }
    from .minecraft_knowledge_nodes import detect_features

    # Known taxonomy only supplies search keywords, never the request's scope.
    # Include the actual authored request to preserve uncatalogued mechanics.
    features = [
        str(name).replace("_", " ")
        for name in detect_features(requested_prompt)
        if str(name) != "base_mod"
    ]
    full_request = " ".join(requested_prompt.split())

    # Do not search a Korean/free-form multi-feature request as one GitHub
    # keyword query. The approved structured plan already identifies concrete
    # systems and interactions in provider-searchable technical vocabulary.
    # These are *retrieval hints*, not newly invented gameplay requirements.
    import re

    hints: list[str] = []
    for section, collection, keys in (
        ("reuse_assessment", "adaptations", ("part",)),
        ("resources_and_ui", "assets", ("purpose", "production_owner")),
        ("integration", "entry_points", ("boundary",)),
        ("behavior_contract", "outputs", ("name",)),
    ):
        raw_section = structured_sections.get(section, {})
        spec = raw_section.get("specification", {}) if isinstance(raw_section, Mapping) else {}
        records = spec.get(collection, ()) if isinstance(spec, Mapping) else ()
        if not isinstance(records, (list, tuple)):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            for key in keys:
                phrase = re.sub(r"[_./-]+", " ", str(record.get(key) or "").casefold())
                phrase = " ".join(phrase.split())
                words = phrase.split()
                if (
                    1 <= len(words) <= 6
                    and len(phrase) <= 70
                    and phrase not in {"player", "host", "game system", "game"}
                    and phrase not in hints
                ):
                    hints.append(phrase)
    nodes = list(dict.fromkeys(
        ["custom gameplay " + full_request[:150]]
        + ["minecraft fabric " + name for name in features]
        + ["minecraft fabric " + hint for hint in hints]
    ))
    return {
        "nodes": nodes,
        "search_terms": [
            {
                "capability": node,
                "terms": (
                    [f"minecraft {node.removeprefix('minecraft fabric ')} mod", node]
                    if node.startswith("minecraft fabric ")
                    else [*("minecraft " + hint + " mod" for hint in hints), node]
                ),
            }
            for node in nodes
        ],
        "retrieval_policy_version": "structured-intent-v3-exact-source-partial-fit",
        "coverage_policy": "discovery_hints_only; authored design remains authoritative",
    }


def resolve_authored_source_reuse(
    requested_prompt: str,
    structured_sections: Mapping[str, Any],
    *,
    minecraft_version: str,
    loader: str,
    reference_repositories: Sequence[str] = (),
    request_scope: Mapping[str, Any] | None = None,
    repository_candidates: Sequence[Mapping[str, Any]] = (),
    requested_curseforge_dependencies: Sequence[Mapping[str, Any]] = (),
    prior_reuse_plan: Mapping[str, Any] | None = None,
    router: Any | None = None,
) -> dict[str, Any]:
    from .grounded_source_reuse import build_repository_reuse_plan

    graph = authored_capability_graph(requested_prompt, structured_sections, request_scope=request_scope)
    frozen = {
        "plan_sha256": "sha256:" + _sha(graph),
        "capability_graph": graph,
    }
    # Reusing pinned source observations is safe only for EXACTLY the same
    # authored capability graph, platform and source-discovery contract.
    # It eliminates the second CurseForge search on resume; execution still
    # independently re-downloads pinned bytes and proves installation.
    target = {"minecraft_version": minecraft_version, "loader": loader}
    authored_input_sha = "sha256:" + _sha({
        "requested_prompt": requested_prompt,
        "structured_sections": structured_sections,
        "request_scope": request_scope,
        "reference_repositories": list(reference_repositories),
        "target": target,
    })
    if isinstance(prior_reuse_plan, Mapping) and (
        prior_reuse_plan.get("schema_version") == "mmm/grounded-repository-reuse-plan-v2"
        and prior_reuse_plan.get("source_plan_sha256") == frozen["plan_sha256"]
        and prior_reuse_plan.get("capability_graph") == graph
        and prior_reuse_plan.get("authored_input_sha256") == authored_input_sha
        and prior_reuse_plan.get("bound_target") == target
        and prior_reuse_plan.get("pre_design_candidates_sha256")
            == source_candidate_fingerprint(repository_candidates)
        and prior_reuse_plan.get("discovery_algorithm_version")
            == source_discovery_algorithm_version()
        and prior_reuse_plan.get("requested_curseforge_dependencies") == [
            dict(x) for x in requested_curseforge_dependencies if isinstance(x, Mapping)
        ]
    ):
        from copy import deepcopy
        cached = deepcopy(dict(prior_reuse_plan))
        cached["planning_cache_reused"] = True
        cached["incremental_source_reuse_costs"] = {
            "source_downloads": 0, "model_judgement_calls": 0,
            "inspected_candidate_count": 0,
        }
        return cached
    design = {
        "_pre_retrieval_plan": frozen,
        "_requested_curseforge_dependencies": [
            dict(item) for item in requested_curseforge_dependencies
            if isinstance(item, Mapping)
        ],
        # Debug-authored references have zero reuse authority. The normal host
        # inspector and target compile proof still decide source_transplant.
        "_preselected_reference_repositories": list(reference_repositories),
        # Candidates must be host-stamped; the existing reader validates
        # reference_only / verification_required before an inspection starts.
        "_planning_state": {
            "repository_candidates": [
                dict(item) for item in repository_candidates
                if isinstance(item, Mapping)
            ],
        },
        "_platform_selection": {
            "target": {
                "minecraft_version": minecraft_version,
                "loader": loader,
            },
        },
    }
    # Preserve the single-argument host planner ABI when no live model router
    # is provided (debug/fixture callers); real production passes the router.
    receipt = (
        build_repository_reuse_plan(design, router=router)
        if router is not None
        else build_repository_reuse_plan(design)
    )
    return {
        **receipt,
        "pre_design_candidates_sha256": source_candidate_fingerprint(repository_candidates),
        "authored_input_sha256": authored_input_sha,
        "discovery_algorithm_version": source_discovery_algorithm_version(),
        "reference_repositories_sha256": source_reference_fingerprint(reference_repositories),
        "bound_target": {
            "minecraft_version": minecraft_version,
            "loader": loader,
        },
        "origin": "default_authored_production",
    }


def verified_reuse_context(plan: Mapping[str, Any]) -> str:
    """Compact, authority-preserving context for a low-memory local model."""
    decisions = plan.get("capabilities", ())
    summaries = []
    for row in decisions if isinstance(decisions, list) else ():
        if not isinstance(row, Mapping):
            continue
        source_id = str(row.get("source_id") or "")
        summaries.append({
            "capability": str(row.get("capability") or "")[:160],
            "source": source_id if row.get("mode") == "source_transplant" else "",
            "mode": row.get("mode", "fresh"),
        })
    references = []
    for row in list(plan.get("reference_sources", ()))[:8]:
        if not isinstance(row, Mapping) or row.get("reference_only") is not True:
            continue
        references.append({k: row.get(k) for k in (
            "capability", "source_url", "pinned_sha", "license_spdx",
            "selected_file", "selected_symbol", "file_sha256",
            "semantic_score", "rationale", "residual_work", "source_excerpt")})
    return (
        "\n\nHOST SOURCE-REUSE RECEIPTS (read-only facts, not new requests):\n"
        + json.dumps(summaries, ensure_ascii=False, separators=(",", ":"))
        + "\nSEMANTIC SOURCE REFERENCES (UNVERIFIED, NOT INSTALLED):\n"
        + json.dumps(references, ensure_ascii=False, separators=(",", ":"))
        + "\nOnly source_transplant rows have passed code inspection and compile proof. "
        "Fresh rows need an independent implementation. Never treat a repository search "
        "result or a feature description as working code.\n"
    )


def _verified_donor_license_notice(
    donor: Mapping[str, Any],
    blobs: Mapping[str, bytes],
) -> tuple[str, bytes]:
    """Retain a pinned donor license notice; never invent an attribution."""
    license_text = next((
        content for path, content in blobs.items()
        if path.casefold().split("/")[-1] in {
            "license", "license.txt", "license.md",
            "copying", "copying.txt", "copying.md",
        }
    ), None)
    if not license_text:
        raise RuntimeError(
            "SOURCE_REUSE_LICENSE_NOTICE_REQUIRED: " + str(donor["repository"])
        )
    filename = (
        str(donor["repository"]).replace("/", "__").replace(".", "_")
        + "-" + str(donor["commit_sha"])[:12] + ".LICENSE"
    )
    return "src/main/resources/META-INF/mmm-third-party/" + filename, license_text


def audit_authored_source_reuse(
    reuse_plan: Mapping[str, Any],
    *,
    installed_donor_count: int | None = None,
) -> dict[str, Any]:
    """Distinguish search, verified reuse and fresh code in BOTH run modes.

    An inspected repository or a successful Gradle build does not prove that
    donor code was reused. Fresh selection is permitted, but must remain
    explicitly visible as an independent implementation obligation.
    """
    if reuse_plan.get("schema_version") != "mmm/grounded-repository-reuse-plan-v2":
        raise ValueError("SOURCE_REUSE_AUDIT_SCHEMA_INVALID")
    graph = reuse_plan.get("capability_graph")
    nodes = graph.get("nodes") if isinstance(graph, Mapping) else None
    decisions = reuse_plan.get("capabilities")
    if not isinstance(nodes, list) or not isinstance(decisions, list):
        raise ValueError("SOURCE_REUSE_AUDIT_GRAPH_INVALID")
    intended = [str(node) for node in nodes]
    seen: set[str] = set()
    verified: list[str] = []
    fresh: list[str] = []
    for row in decisions:
        if not isinstance(row, Mapping):
            raise ValueError("SOURCE_REUSE_AUDIT_DECISION_INVALID")
        capability = str(row.get("capability") or "")
        if capability not in intended or capability in seen:
            raise ValueError("SOURCE_REUSE_AUDIT_CAPABILITY_MISMATCH")
        seen.add(capability)
        mode = row.get("mode")
        if mode == "fresh":
            if row.get("source_id"):
                raise ValueError("SOURCE_REUSE_AUDIT_FRESH_HAS_DONOR")
            fresh.append(capability)
        elif mode == "source_transplant":
            proof = row.get("proof_receipt")
            if (
                not isinstance(proof, Mapping)
                or not proof.get("compile_passed")
                or not str(row.get("source_id") or "").startswith("host-donor:")
            ):
                raise ValueError("SOURCE_REUSE_AUDIT_UNVERIFIED_TRANSPLANT")
            verified.append(capability)
        else:
            raise ValueError("SOURCE_REUSE_AUDIT_MODE_INVALID")
    if len(seen) != len(intended):
        raise ValueError("SOURCE_REUSE_AUDIT_UNCOVERED_CAPABILITIES")
    # One selected donor may cover several capabilities. Materialization
    # operates on selected rows, so validate that count exactly.
    if installed_donor_count is not None and installed_donor_count != len(verified):
        raise ValueError("SOURCE_REUSE_AUDIT_DONOR_NOT_INSTALLED")
    inspections = reuse_plan.get("inspection_receipts", ())
    proofs = reuse_plan.get("proof_receipts", ())
    return {
        "schema_version": "mmm/authored-source-reuse-audit-v1",
        "status": (
            "VERIFIED_REUSE" if verified and not fresh
            else "MIXED_REUSE_AND_FRESH" if verified
            else "FRESH_IMPLEMENTATION_REQUIRED"
        ),
        "capability_count": len(intended),
        "verified_transplant_count": len(verified),
        "fresh_implementation_count": len(fresh),
        "verified_transplant_capabilities": verified,
        "fresh_implementation_capabilities": fresh,
        "reference_source_count": len(reuse_plan.get('reference_sources', ())),
        "reference_capabilities": sorted({row['capability']
                                          for row in reuse_plan.get('reference_sources', ())}),
        "inspection_failure_count": sum(
            isinstance(row, Mapping) and row.get("status") in {"inspection_error", "inspection_rejected"}
            for row in inspections if isinstance(inspections, (list, tuple))
        ),
        "proof_failure_count": sum(
            isinstance(row, Mapping) and row.get("status") in {"proof_error", "rejected"}
            for row in proofs if isinstance(proofs, (list, tuple))
        ),
        "behavior_contract_unavailable_count": sum(
            isinstance(row, Mapping) and row.get("status") == "behavior_contract_unavailable"
            for row in proofs if isinstance(proofs, (list, tuple))
        ),
        "compile_only_behavior_unverified_count": sum(
            isinstance(row, Mapping) and row.get("status") == "compile_only_unverified_behavior"
            for row in proofs if isinstance(proofs, (list, tuple))
        ),
        # Preserve the actual rejected candidates and their stage-specific
        # reasons so a final download can reveal where reuse stopped.
        "inspection_failures": [
            {
                "capability": str(row.get("capability") or ""),
                "repository": str(row.get("repository") or ""),
                "status": str(row.get("status") or ""),
                "error": str(row.get("error") or row.get('rejection_reason') or ""),
            }
            for row in inspections
            if isinstance(row, Mapping)
            and row.get("status") in {"inspection_error", "inspection_rejected"}
        ] if isinstance(inspections, (list, tuple)) else [],
        "proof_failures": [
            {
                "capability": str(row.get("capability") or ""),
                "candidate_id": str(row.get("candidate_id") or ""),
                "status": str(row.get("status") or ""),
                "error": str(row.get("error") or row.get("failure_message") or ""),
                "failure_code": str(row.get("failure_code") or ""),
            }
            for row in proofs
            if isinstance(row, Mapping)
            and row.get("status") in {"proof_error", "rejected"}
        ] if isinstance(proofs, (list, tuple)) else [],
        "installed_donor_count": installed_donor_count,
        "fresh_requires_independent_production_verification": bool(fresh),
    }


def materialize_verified_authored_sources(
    project_root: str,
    reuse_plan: Mapping[str, Any],
    *,
    minecraft_version: str,
    loader: str,
    package_name: str = "",
    mod_id: str = "",
) -> dict[str, Any]:
    """Install only proof-bound donor source into the real production workspace.

    Reuse without source files in the build is not reuse. Donor code is
    hash-verified again when downloaded, deterministically adapted, and never
    permitted to overwrite an existing project's Java or resource files.
    """
    from pathlib import Path
    from .source_transplant import materialize_source_slices
    from .reuse_adapters import apply_deterministic_adapters

    bound = reuse_plan.get("bound_target")
    if bound != {"minecraft_version": minecraft_version, "loader": loader}:
        raise ValueError("SOURCE_REUSE_TARGET_MISMATCH: donor proof is for another target")

    # Catalogue hits cannot auto-install a mod. Only explicit provider file
    # selections with local bytes and an acknowledged license policy enter this
    # lane. The provider response is fetched again HERE, never trusted from a
    # model-generated or persisted plan.
    requested_deps = reuse_plan.get("requested_curseforge_dependencies", ())
    if not isinstance(requested_deps, (list, tuple)) or len(requested_deps) > 8:
        raise ValueError("SOURCE_REUSE_DEPENDENCY_REQUESTS_INVALID")
    installed_dependencies = []
    if requested_deps:
        from .curseforge_dependency_adoption import (
            fetch_exact_curseforge_file_metadata, verify_curseforge_dependency,
        )
        from .curseforge_dependency_project import install_verified_fabric_dependency
        for requested in requested_deps:
            if not isinstance(requested, Mapping):
                raise ValueError("SOURCE_REUSE_DEPENDENCY_REQUEST_INVALID")
            project_id = requested.get("project_id")
            file_id = requested.get("file_id")
            if type(project_id) is not int or type(file_id) is not int:
                raise ValueError("SOURCE_REUSE_DEPENDENCY_FILE_ID_INVALID")
            exact_metadata = fetch_exact_curseforge_file_metadata(project_id, file_id)
            verified = verify_curseforge_dependency(
                str(requested.get("jar_path") or ""),
                project_id=project_id, file_id=file_id,
                file_metadata=exact_metadata,
                minecraft_version=minecraft_version, loader=loader,
                license_policy_confirmed=(
                    requested.get("license_policy_confirmed") is True
                ),
            )
            installed_dependencies.append(install_verified_fabric_dependency(
                project_root, verified, minecraft_version=minecraft_version
            ))

    chosen = [
        row for row in reuse_plan.get("capabilities", ())
        if isinstance(row, Mapping) and row.get("mode") == "source_transplant"
    ]
    notices = reuse_plan.get("reference_license_notices", {})
    if notices:
        root = Path(project_root).resolve()
        for relative, notice in notices.items():
            target = (root / relative).resolve()
            if not target.is_relative_to(root) or not str(relative).startswith("src/main/resources/META-INF/"):
                raise ValueError("SOURCE_REUSE_LICENSE_PATH_INVALID")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(notice, encoding="utf-8")
    if not chosen:
        return {"schema_version": "mmm/authored-source-install-v1", "donor_count": 0,
                "files": [], "dependency_count": len(installed_dependencies),
                "dependencies": installed_dependencies}
    if not package_name.strip() or not mod_id.strip():
        raise ValueError(
            "SOURCE_REUSE_TARGET_IDENTITY_REQUIRED: verified transplant needs "
            "the actual mod package and mod ID for deterministic adaptation."
        )

    root = Path(project_root).resolve()
    downloaded = materialize_source_slices(root, reuse_plan)
    donors = downloaded.get("donors", ())
    if len(donors) != len(chosen):
        raise RuntimeError("SOURCE_REUSE_MATERIALIZATION_INCOMPLETE")

    # All files are inspected before writing any donor into the production tree.
    prepared: dict[str, tuple[bytes, str]] = {}
    supported = (
        "src/main/java/", "src/main/resources/",
        "src/client/java/", "src/client/resources/",
    )
    from .platform_catalog import adapter_for_target
    adapter = adapter_for_target(minecraft_version, loader)
    target_context = {
        "minecraft_version": adapter.minecraft_version,
        "loader": adapter.loader,
        "mappings": adapter.yarn_mappings,
        "mappings_kind": adapter.mappings_kind,
        "java_version": adapter.java_version,
        "fabric_loader": adapter.fabric_loader,
        "fabric_api": adapter.fabric_api,
        "fabric_loom": adapter.fabric_loom,
        "gradle": adapter.gradle,
        "target_package": package_name,
        "target_modid": mod_id,
    }
    attribution: list[dict[str, Any]] = []
    for chosen_row, donor in zip(chosen, donors):
        if donor.get("capability") != chosen_row.get("capability"):
            raise RuntimeError("SOURCE_REUSE_DONOR_RECEIPT_MISMATCH")
        dependencies = donor.get("required_dependencies") or ()
        if dependencies:
            # A compiled isolated donor does not automatically install its Maven
            # dependencies into the final project. Refuse a false success.
            raise RuntimeError(
                "SOURCE_REUSE_DEPENDENCY_INTEGRATION_REQUIRED: "
                + str(donor.get("repository"))
            )
        blobs: dict[str, bytes] = {}
        for item in donor.get("files", ()):
            relative = str(item.get("source_path") or "")
            if not relative:
                raise ValueError("SOURCE_REUSE_INVALID_PATH")
            blobs[relative] = Path(item["path"]).read_bytes()
        notice_path, notice_bytes = _verified_donor_license_notice(donor, blobs)
        prepared[notice_path] = (notice_bytes, str(donor["repository"]))
        for extra_path, extra_bytes in blobs.items():
            if extra_path.casefold() in {"notice", "notice.txt", "notice.md", "copyright"}:
                document_name = str(donor["repository"]).replace("/", "__").replace(".", "_")
                dest = ("src/main/resources/META-INF/mmm-third-party/" + document_name
                        + "-" + str(donor["commit_sha"])[:12] + "." + extra_path.replace(".", "_"))
                if dest in prepared and prepared[dest][0] != extra_bytes:
                    raise RuntimeError("SOURCE_REUSE_NOTICE_COLLISION")
                prepared[dest] = (extra_bytes, str(donor["repository"]))
        adapted, _ = apply_deterministic_adapters(blobs, target_context)
        copied_for_donor = 0
        for relative, data in adapted.items():
            relative = str(relative).replace(chr(92), "/")
            if not relative.startswith(supported) or relative.endswith("/"):
                continue
            if not isinstance(data, (str, bytes)):
                raise ValueError("SOURCE_REUSE_NON_TEXT_OR_BYTES")
            raw = data.encode("utf-8") if isinstance(data, str) else data
            if relative in prepared and prepared[relative][0] != raw:
                raise RuntimeError("SOURCE_REUSE_ARTIFACT_COLLISION: " + relative)
            prepared[relative] = (raw, str(donor["repository"]))
            copied_for_donor += 1
        if not copied_for_donor:
            raise RuntimeError(
                "SOURCE_REUSE_NO_PRODUCTION_FILES: " + str(donor.get("repository"))
            )
        attribution.append({
            "repository": donor["repository"],
            "commit_sha": donor["commit_sha"],
            "license_id": donor["license_id"],
            "capability": donor["capability"],
        })

    # Validate path ownership and collisions before the first file write.
    for relative, (raw, _owner) in prepared.items():
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("SOURCE_REUSE_PATH_ESCAPE: " + relative) from exc
        if target.exists() and (not target.is_file() or target.read_bytes() != raw):
            raise RuntimeError("SOURCE_REUSE_EXISTING_FILE_CONFLICT: " + relative)

    for relative, (raw, _owner) in sorted(prepared.items()):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(raw)
    provenance_file = root / ".minecraft_ai" / "reuse" / "source_provenance.json"
    provenance_file.parent.mkdir(parents=True, exist_ok=True)
    provenance_file.write_text(
        json.dumps(attribution, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Bind every installed artifact to its exact bytes. File presence alone is
    # insufficient for resumption or for authoring callable donor integration.
    file_hashes = {
        relative: hashlib.sha256(raw).hexdigest()
        for relative, (raw, _owner) in sorted(prepared.items())
    }
    file_owners = {relative: owner for relative, (_raw, owner) in sorted(prepared.items())}
    installed = [
        {"path": relative, "sha256": file_hashes[relative],
         "repository": file_owners[relative],
         "capabilities": sorted({row["capability"] for row in attribution
                                 if row["repository"] == file_owners[relative]})}
        for relative in sorted(prepared)
        if relative.startswith(("src/main/java/", "src/client/java/"))
        and relative.endswith(".java")
    ]
    manifest_path = root / ".minecraft_ai" / "reuse" / "installed_sources.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "mmm/installed-source-members-v1",
        "sources": installed,
    }, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "schema_version": "mmm/authored-source-install-v1",
        "donor_count": len(attribution),
        "files": sorted(prepared),
        "file_hashes": file_hashes,
        "file_owners": file_owners,
        "installed_manifest": str(manifest_path),
        "provenance": str(provenance_file),
        "dependency_count": len(installed_dependencies),
        "dependencies": installed_dependencies,
    }



def installed_donor_source_context(project_root, capabilities):
    """Read only installed source members bound to the current behavior.

    A pinned donor's declared source path is never assumed to survive mapping
    adaptation; the materialization manifest lists the actual installed paths.
    """
    from pathlib import Path
    root = Path(project_root).resolve()
    manifest = root / ".minecraft_ai" / "reuse" / "installed_sources.json"
    wanted = set(capabilities)
    if not wanted:
        return ()
    if not manifest.is_file():
        raise ValueError("CONSTRUCTION_INSTALLED_DONOR_MANIFEST_MISSING")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "mmm/installed-source-members-v1":
        raise ValueError("CONSTRUCTION_INSTALLED_DONOR_MANIFEST_SCHEMA")
    result = []
    covered = set()
    for row in payload.get("sources", ()):
        matched = wanted & set(row.get("capabilities", ()))
        if not matched:
            continue
        relative = str(row.get("path") or "")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not relative.startswith(
            ("src/main/java/", "src/client/java/")
        ) or not path.is_file():
            raise ValueError("CONSTRUCTION_INSTALLED_DONOR_PATH_INVALID: " + relative)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row.get("sha256"):
            raise ValueError("CONSTRUCTION_INSTALLED_DONOR_HASH_MISMATCH: " + relative)
        result.append({"path": relative, "sha256": digest,
                       "authority": "installed_target_adapted_source",
                       "repository": row.get("repository", ""),
                       "capabilities": sorted(matched)})
        covered.update(matched)
    if covered != wanted:
        raise ValueError("CONSTRUCTION_INSTALLED_DONOR_COVERAGE_MISSING: " +
                         ",".join(sorted(wanted - covered)))
    return tuple(result)


__all__ = [
    "authored_capability_graph",
    "resolve_authored_source_reuse",
    "verified_reuse_context",
    "materialize_verified_authored_sources",
    "audit_authored_source_reuse",
    "installed_donor_source_context",
]
