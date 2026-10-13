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
        """Return whole AST declarations, never mid-method text fragments.

        Hits remain discovery hints. A donor becomes callable only when the
        exact target's installed output and inspected symbols are bound later.
        """
        from .construction_context import java_source_units

        selected, seen = [], set()
        for capability, index, root, donor in self.sources:
            if capability not in capabilities:
                continue
            for hit in index.search(query + " " + capability, limit=8):
                path = Path(hit.source_path).resolve()
                if not path.is_relative_to(root.resolve()) or path.suffix != ".java":
                    continue
                if not path.is_file():
                    raise ValueError("SOURCE_RAG_PINNED_FILE_MISSING: " + str(path))
                raw = path.read_text(encoding="utf-8")
                try:
                    units = java_source_units(raw, path=str(path.relative_to(root)))
                except (ValueError, RuntimeError) as exc:
                    self.reference_failures.append({
                        "capability": capability, "repository": donor["repository"],
                        "status": "source_parse_failed", "error": str(exc)[:1200],
                    })
                    continue
                first = int(hit.start_line)
                last = int(getattr(hit, "end_line", None) or first)
                callable_units = [unit for unit in units
                                  if unit["kind"] in {"method", "constructor", "field"}]
                members = [unit for unit in callable_units
                           if unit["start_line"] <= last and unit["end_line"] >= first]
                if not members and callable_units:
                    # RAG can match a class header, import or nearby comment.
                    # Select the nearest complete declaration, never a slice
                    # of a method or an unrelated full compilation unit.
                    members = [min(callable_units, key=lambda unit: (
                        max(unit["start_line"] - last, first - unit["end_line"], 0),
                        unit["start_line"],
                    ))]
                for unit in members:
                    identity = (str(path), unit["start_line"], unit["end_line"])
                    if identity in seen:
                        continue
                    seen.add(identity)
                    selected.append({
                        "capability": capability, "repository": donor["repository"],
                        "source_commit": donor["commit_sha"], "license": donor["license_id"],
                        "path": str(path.relative_to(root)), "start_line": unit["start_line"],
                        "end_line": unit["end_line"], "owner": unit["owner"],
                        "symbol": unit["name"], "signature": unit["signature"],
                        "source_unit_kind": unit["kind"],
                        "adapted_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "code": unit["code"],
                        "authority": donor.get("_authority", "installed_target_adapted_source"),
                        **({"migration_diagnostics": donor["_migration_diagnostics"]}
                           if donor.get("_migration_diagnostics") else {}),
                    })
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
        # Older mods that need model adaptation are useful source evidence even
        # when deterministic target compilation fails. Never install this code
        # unchanged or label its API as valid for the requested target.
        reference_counts: dict[str, int] = {}
        installed_capabilities = {row[0] for row in rag.sources}
        for ordinal, reference in enumerate(references):
            capability = reference["capability"]
            if capability in installed_capabilities:
                continue
            donor_slice = DonorSlice.from_dict(reference["donor"])
            if not donor_slice.closure_complete:
                continue
            try:
                blobs = materialize_pinned_donor(donor_slice)
                donor = donor_slice.to_dict()
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
            except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
                # Optional migration guidance has no installed-source authority.
                # Unavailable or invalid evidence must not block fresh authoring.
                rag.reference_failures.append({
                    "capability": capability, "repository": donor_slice.repository,
                    "status": "reference_unavailable", "error": f"{type(exc).__name__}: {exc}"[:1200],
                })
                print(f"SOURCE_RAG: reference unavailable {donor_slice.repository} ({type(exc).__name__})",
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
        capabilities = [row["capability"] for row in reuse_plan.get("capabilities", ())
                        if row.get("mode") == "source_transplant"]
    snippets = rag.context(capabilities, requirement + " " + owner.module_id)
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
