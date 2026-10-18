"""Small, task-owned code context from pinned and target-proven donor files."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .rag_index import ProjectRAGIndex


class VerifiedSourceRAG:
    def __init__(self):
        self.sources: list[tuple[str, ProjectRAGIndex, Path, Mapping[str, Any]]] = []
        self.license_notices: dict[str, str] = {}
        self.reference_failures: list[dict[str, str]] = []

    def context(self, capabilities: Sequence[str], query: str) -> list[dict[str, Any]]:
        """Retrieve real complete donor declarations, without silent index misses.

        The snippet lookup first reads ranked RAG hits. If the index surfaces
        only class headers or no callable hit, inspect the pinned source slice
        itself. Only capabilities assigned to this owner may cross this gate.
        A source-only fallback is explicitly marked, not treated as semantic
        or target-compile evidence.
        """
        import re
        from .construction_context import java_source_units

        def words(text):
            split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
            return set(re.findall(r"[a-z]{3,}|[가-힣]{2,}", split.casefold())) - {
                "minecraft", "fabric", "forge", "the", "and", "with", "from",
                "mod", "code", "source", "implementation", "register", "method",
            }

        selected, seen = [], set()
        for capability, index, root, donor in self.sources:
            if capability not in capabilities:
                continue
            root = root.resolve()
            wanted = words(query + " " + capability)
            parsed = {}
            choices = {}

            def examine(path):
                path = Path(path).resolve()
                if not path.is_relative_to(root) or path.suffix != ".java":
                    return
                if path in parsed:
                    units = parsed[path]
                else:
                    if not path.is_file():
                        raise ValueError("SOURCE_RAG_PINNED_FILE_MISSING: " + str(path))
                    try:
                        units = java_source_units(path.read_text(encoding="utf-8"),
                                                  path=str(path.relative_to(root)))
                    except (ValueError, RuntimeError) as exc:
                        self.reference_failures.append({
                            "capability": capability, "repository": donor["repository"],
                            "status": "source_parse_failed", "error": str(exc)[:1200],
                        })
                        units = []
                    parsed[path] = units
                for unit in units:
                    if unit["kind"] not in {"method", "constructor", "field"}:
                        continue
                    identity = (str(path), unit["start_line"], unit["end_line"])
                    name = words(unit.get("name", ""))
                    signature = words(unit.get("signature", ""))
                    owner = words(unit.get("owner", ""))
                    # Rank actual member boundaries and call sites, not only
                    # the nearest index hit. Names/signatures dominate code
                    # body words so a generic comment cannot outrank a true API.
                    code = words(unit.get("code", ""))
                    score = (6 * len(wanted & name) + 3 * len(wanted & signature)
                             + 2 * len(wanted & owner) + len(wanted & code))
                    choices[identity] = (score, path, unit)

            hits = list(index.search(query + " " + capability, limit=8))
            for hit in hits:
                examine(hit.source_path)
            # A top-eight text hit is NOT exhaustive code relevance. Search
            # the entire *pinned, proof-bounded* Java source slice for method
            # declarations; otherwise a loosely matching class header hides
            # the actual feature implementation in a different file.
            for path in sorted(root.rglob("*.java")):
                examine(path)
            ordered = sorted(choices.values(), key=lambda row: (
                -row[0], str(row[1]), row[2]["start_line"], row[2]["end_line"],
            ))
            if not ordered:
                self.reference_failures.append({
                    "capability": capability, "repository": donor["repository"],
                    "status": "no_callable_java_source",
                    "error": "The pinned source contains no parseable Java member.",
                })
            for score, path, unit in ordered[:8]:
                identity = (str(path), unit["start_line"], unit["end_line"])
                bound_identity = (capability, *identity)
                if bound_identity in seen:
                    continue
                seen.add(bound_identity)
                selected.append({
                    "capability": capability, "repository": donor["repository"],
                    "source_commit": donor["commit_sha"], "license": donor["license_id"],
                    "path": str(path.relative_to(root)), "start_line": unit["start_line"],
                    "end_line": unit["end_line"], "owner": unit["owner"],
                    "symbol": unit["name"], "signature": unit["signature"],
                    "source_unit_kind": unit["kind"],
                    "adapted_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "code": unit["code"],
                    "source_minecraft_version": (donor.get("compatibility_evidence") or {}).get("minecraft_version", ""),
                    "source_closure_complete": donor.get("closure_complete", True),
                    "authority": donor.get("_authority", "installed_target_adapted_source"),
                    "retrieval_method": "verified_semantic_match" if score else "inspected_source_fallback",
                    "retrieval_score": score,
                    **({"migration_diagnostics": donor["_migration_diagnostics"]}
                       if donor.get("_migration_diagnostics") else {}),
                })
            print("SOURCE_RAG_HANDOFF: " + json.dumps({
                "capability": capability, "repository": donor["repository"],
                "ranked_index_hits": len(hits), "parsed_java_paths": len(parsed),
                "callable_candidates": len(ordered),
                "delivered_members": min(len(ordered), 8),
                "best_match_score": ordered[0][0] if ordered else 0,
                "authority": donor.get("_authority", "installed_target_adapted_source"),
            }, ensure_ascii=False), flush=True)
        return selected

@contextmanager
def verified_source_rag(reuse_plan: Mapping[str, Any], *, adapter: Any,
                        package_name: str, mod_id: str):
    """Download/hash-check the proven slice once, then index its adapted code.

    The same materializer and adapters install these files into the final mod.
    Target-proven adapted code and hash-checked licensed migration references
    have separate authority labels. Search descriptions are not executable code,
    and unrelated donor APIs never enter an owner's task.
    """
    from .authored_reuse_bridge import _verified_donor_license_notice
    from .reuse_adapters import apply_deterministic_adapters
    from .source_transplant import (
        DonorSlice,
        materialize_pinned_donor,
        materialize_source_slices,
    )

    rag = VerifiedSourceRAG()
    chosen = [row for row in reuse_plan.get("capabilities", ())
              if isinstance(row, Mapping) and row.get("mode") == "source_transplant"]
    references = reuse_plan.get("reference_sources", ())
    if not chosen and not references:
        yield rag
        return
    target = {"minecraft_version": adapter.minecraft_version, "loader": adapter.loader}
    if reuse_plan.get("bound_target") != target:
        raise ValueError("SOURCE_REUSE_TARGET_MISMATCH")
    mapping_namespace = adapter.mappings_kind
    if mapping_namespace == "mojang" or not adapter.mappings_applicable:
        mapping_namespace = "official"
    target_context = {
        **target, "mappings": adapter.yarn_mappings, "mappings_kind": adapter.mappings_kind,
        "java_version": adapter.java_version, "fabric_loader": adapter.fabric_loader,
        "fabric_api": adapter.fabric_api, "fabric_loom": adapter.fabric_loom,
        "gradle": adapter.gradle, "target_package": package_name, "target_modid": mod_id,
    }
    for decision in chosen:
        evidence = (decision.get("donor") or {}).get("compatibility_evidence") or {}
        if evidence.get("minecraft_version") != adapter.minecraft_version:
            raise ValueError("SOURCE_REUSE_DONOR_VERSION_MISMATCH")
    with TemporaryDirectory(prefix="mmm_verified_source_rag_") as directory:
        workspace = Path(directory)
        downloaded = materialize_source_slices(workspace, reuse_plan) if chosen else {"donors": []}
        donors = downloaded.get("donors", ())
        if len(donors) != len(chosen):
            raise RuntimeError("SOURCE_REUSE_MATERIALIZATION_INCOMPLETE")
        for ordinal, (decision, donor) in enumerate(zip(chosen, donors)):
            if donor.get("capability") != decision.get("capability"):
                raise RuntimeError("SOURCE_REUSE_DONOR_RECEIPT_MISMATCH")
            if donor.get("required_dependencies"):
                raise RuntimeError("SOURCE_REUSE_DEPENDENCY_INTEGRATION_REQUIRED")
            blobs = {item["source_path"]: Path(item["path"]).read_bytes()
                     for item in donor.get("files", ())}
            adapted, _ = apply_deterministic_adapters(blobs, target_context)
            root = workspace / str(ordinal) / "source"
            for relative, content in adapted.items():
                path = (root / relative).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError("SOURCE_REUSE_PATH_ESCAPE")
                if not str(relative).startswith(("src/main/", "src/client/")):
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content.encode() if isinstance(content, str) else content)
            index = ProjectRAGIndex(workspace / str(ordinal) / "source.sqlite")
            index.build([root], metadata={
                **target, "mapping_namespace": mapping_namespace,
                "java_version": str(adapter.java_version), "license": donor["license_id"],
                "source_commit": donor["commit_sha"], "source_kind": "verified_adapted_donor",
            })
            rag.sources.append((decision["capability"], index, root, donor))
        # Same-version mods remain useful references when dependency closure or
        # deterministic target compilation fails. Their original APIs are not
        # target-callable authority and their files are never installed here.
        reference_counts: dict[str, int] = {}
        installed_capabilities = {row[0] for row in rag.sources}
        for ordinal, reference in enumerate(references):
            capability = reference["capability"]
            if capability in installed_capabilities:
                continue
            try:
                if isinstance(reference.get("donor"), Mapping):
                    donor_slice = DonorSlice.from_dict(reference["donor"])
                    if (donor_slice.compatibility_evidence is None or
                            not donor_slice.compatibility_evidence.minecraft_version or
                            not donor_slice.compatibility_evidence.loader):
                        raise ValueError("SOURCE_REFERENCE_VERSION_EVIDENCE_MISSING")
                    blobs = materialize_pinned_donor(donor_slice)
                    donor = donor_slice.to_dict()
                else:
                    # Current CurseForge semantic reviews carry pinned file
                    # identities, not the legacy nested DonorSlice envelope.
                    # Verify those exact bytes/license before indexing them.
                    from .curseforge_reuse_production import read_pinned_source_reference
                    donor, blobs = read_pinned_source_reference(reference)
                notice_path, notice_bytes = _verified_donor_license_notice(donor, blobs)
                root = workspace / ("reference_" + str(ordinal)) / "source"
                for relative, content in blobs.items():
                    path = (root / relative).resolve()
                    if not path.is_relative_to(root.resolve()):
                        raise ValueError("SOURCE_REUSE_PATH_ESCAPE")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
                index = ProjectRAGIndex(root.parent / "source.sqlite")
                evidence = donor.get("compatibility_evidence") or {}
                index.build([root], metadata={
                    "minecraft_version": evidence.get("minecraft_version") or "donor-unbound",
                    "loader": evidence.get("loader") or "donor-unbound",
                    "mapping_namespace": "unknown", "java_version": "donor-unbound",
                    "license": donor["license_id"], "source_commit": donor["commit_sha"],
                    "source_kind": "pinned_reference_requires_target_adaptation",
                })
                notice = notice_bytes.decode("utf-8")
            except (OSError, ValueError, RuntimeError, KeyError, sqlite3.Error) as exc:
                # Optional migration guidance has no installed-source authority.
                # Unavailable or invalid evidence must not block fresh authoring.
                rag.reference_failures.append({
                    "capability": capability,
                    "repository": str((reference.get("donor") or {}).get("repository")
                                      or reference.get("source_url") or ""),
                    "status": "reference_unavailable", "error": f"{type(exc).__name__}: {exc}"[:1200],
                })
                print(f"SOURCE_RAG: reference unavailable for {capability} ({type(exc).__name__})",
                      flush=True)
                continue
            rag.license_notices[notice_path] = notice
            donor["_authority"] = "reference_only_requires_target_adaptation"
            donor["_migration_diagnostics"] = reference.get("migration_diagnostics", "")
            rag.sources.append((capability, index, root, donor))
            reference_counts[capability] = reference_counts.get(capability, 0) + 1
        yield rag


def bind_owner_source_context(bindings: dict[str, Any], *, owner: Any,
                             request_scope: Mapping[str, Any], reuse_plan: Mapping[str, Any],
                             rag: VerifiedSourceRAG, requirement: str) -> None:
    """Persist per-owner feature intent and real callable code in coder bindings."""
    refs = set(owner.config.get("requirement_refs", ()))
    features = [row for row in request_scope.get("requirements", ())
                if row.get("requirement_id") in refs]
    if features:
        bindings["request_features_json"] = json.dumps(
            [{"requirement_id": row["requirement_id"], "statement": row["statement"]}
             for row in features], ensure_ascii=False, separators=(",", ":"))
    graph = reuse_plan.get("capability_graph", {})
    capabilities = [row["capability"] for row in graph.get("requirement_bindings", ())
                    if row.get("requirement_id") in refs]
    if not request_scope:
        capabilities = list(dict.fromkeys(
            [row["capability"] for row in reuse_plan.get("capabilities", ())
             if row.get("mode") == "source_transplant"] +
            [row["capability"] for row in reuse_plan.get("reference_sources", ())]))
    # Catalogue findings are bounded design references, never installed Java.
    # Keep them separate from verified_source_code_json so a small model
    # cannot mistake search metadata for a callable dependency.
    catalogue_hints = []
    for decision in reuse_plan.get("capabilities", ()):
        if decision.get("capability") not in capabilities:
            continue
        for hint in decision.get("catalogue_evidence", ())[:3]:
            if not isinstance(hint, Mapping):
                continue
            catalogue_hints.append({
                "capability": decision["capability"],
                "title": str(hint.get("title") or "")[:120],
                "summary": str(hint.get("summary") or "")[:260],
                "source_id": str(hint.get("source_id") or ""),
                "catalog_url": str(hint.get("catalog_url") or ""),
                "authority": "design_reference_only_not_java_source",
            })
    if catalogue_hints:
        bindings["catalogue_research_json"] = json.dumps(
            catalogue_hints[:12], ensure_ascii=False, separators=(",", ":")
        )
    snippets = rag.context(capabilities, requirement + " " + owner.module_id)
    # A target-proven Java donor is a mandatory handoff, not an optional RAG
    # suggestion. Resource-only transplants need no Java method, but an
    # installed .java donor must never disappear silently at this boundary.
    proven_java = {
        row["capability"]
        for row in reuse_plan.get("capabilities", ())
        if row.get("mode") == "source_transplant"
        and row.get("capability") in capabilities
        and any(str(item.get("path") or "").endswith(".java")
                for item in (row.get("donor") or {}).get("files", ()))
    }
    delivered_java = {
        row["capability"] for row in snippets
        if row.get("authority") == "installed_target_adapted_source"
    }
    missing = sorted(proven_java - delivered_java)
    if missing:
        raise ValueError("SOURCE_RAG_VERIFIED_DONOR_NOT_BOUND: " + ", ".join(missing))
    bindings["source_reuse_binding_receipt_json"] = json.dumps({
        "owner": owner.module_id, "required_capabilities": list(dict.fromkeys(capabilities)),
        "verified_java_expected": sorted(proven_java),
        "verified_java_delivered": sorted(delivered_java),
        "snippet_count": len(snippets),
        "reference_snippets": sum(
            row.get("authority") == "reference_only_requires_target_adaptation"
            for row in snippets),
        "reference_failures": [row for row in rag.reference_failures
                              if row.get("capability") in capabilities],
    }, ensure_ascii=False, separators=(",", ":"))
    print("SOURCE_RAG_OWNER_BINDING: " + bindings["source_reuse_binding_receipt_json"], flush=True)
    if snippets:
        bindings["verified_source_code_json"] = json.dumps(snippets, ensure_ascii=False, separators=(",", ":"))
        bindings["source_integration_instruction"] = (
            "Only installed_target_adapted_source files will be installed in this mod. Call their actual "
            "classes/methods where relevant, implement the missing glue and residual behavior, "
            "and bind the requested player/world interaction. Do not regenerate donor files "
            "or substitute a registration or display screen for the stated behavior."
            " Reference-only code uses the donor's original APIs: port its logic into this "
            "target's generated class, fixing the supplied migration diagnostics. Never import "
            "uninstalled donor classes or copy obsolete mapping/API names. Retain source attribution."
        )
