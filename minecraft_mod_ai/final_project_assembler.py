from __future__ import annotations

"""Authoritative Final Project Assembler and Integration Orchestrator.

Combines all verified reused donor slices, partial residual components, and freshly
generated modules into a single canonical Minecraft mod project layout with typed merges.

Typed Merge Rules (each has a dedicated merge function):
1. Java/Kotlin source: Normalize packages via deterministic adapters, preserve immutable reused symbols.
2. Resource namespace: Translate donor modid paths (assets/<donor>/*, data/<donor>/*) to target_modid.
3. Mod Metadata: Combine entrypoints in fabric.mod.json / neoforge.mods.toml across all donors.
4. Mixin config: Union all mixin class entries into a single <modid>.mixins.json.
5. Access Widener: Concatenate all AW declarations into a single <modid>.accesswidener.
6. Build script: Merge dependency coordinates and repositories into build.gradle.
7. Tags/recipes/loot: Namespace-aware merge of data-pack JSON files.
8. Emits FinalProjectAssemblyResult and reuse-manifest.json SBOM.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .composition_solver import generate_reuse_manifest
from .reuse_adapters import apply_deterministic_adapters
from .reuse_planner import TargetImplementationPlan
from .reuse_proof_executor import ReuseProofReceipt, ResidualWorkOrder
from .source_transplant import DonorSlice, materialize_pinned_donor
from .verified_scaffold_registry import apply_verified_scaffold

_RESOURCE_NS_RE = re.compile(
    r"^(src/main/resources/(?:assets|data)/)([^/]+)/(.+)$"
)
_MIXIN_CONFIG_RE = re.compile(r"\.mixins?\.json$", re.IGNORECASE)


@dataclass(frozen=True)
class FinalProjectAssemblyResult:
    project_name: str
    target_loader: str
    target_minecraft: str
    total_files: int
    reused_file_count: int
    residual_file_count: int
    fresh_file_count: int
    staged_paths: tuple[str, ...]
    manifest_sbom: dict[str, Any] = field(default_factory=dict)
    work_orders: tuple[ResidualWorkOrder, ...] = ()
    is_valid: bool = True
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/final-project-assembly-result-v1",
            "project_name": self.project_name,
            "target_loader": self.target_loader,
            "target_minecraft": self.target_minecraft,
            "total_files": self.total_files,
            "reused_file_count": self.reused_file_count,
            "residual_file_count": self.residual_file_count,
            "fresh_file_count": self.fresh_file_count,
            "staged_paths": list(self.staged_paths),
            "manifest_sbom": self.manifest_sbom,
            "work_orders": [w.to_dict() for w in self.work_orders],
            "is_valid": self.is_valid,
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# Typed merge helpers
# ---------------------------------------------------------------------------

def _merge_fabric_mod_json(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    target_modid: str,
) -> dict[str, Any]:
    """Merge two fabric.mod.json dicts: combine entrypoints, mixins, depends."""
    merged = dict(existing)
    merged["id"] = target_modid
    merged["name"] = target_modid.replace("_", " ").title()

    # Entrypoints: union all lists per key
    ep_merged = dict(merged.get("entrypoints", {}))
    for ep_key, ep_val in incoming.get("entrypoints", {}).items():
        curr = ep_merged.get(ep_key, [])
        if not isinstance(curr, list):
            curr = [curr] if curr else []
        adds = ep_val if isinstance(ep_val, list) else ([ep_val] if ep_val else [])
        ep_merged[ep_key] = list(dict.fromkeys(curr + adds))
    if ep_merged:
        merged["entrypoints"] = ep_merged

    # Mixins: union mixin config file references
    m_curr = merged.get("mixins", [])
    m_new = incoming.get("mixins", [])
    if not isinstance(m_curr, list):
        m_curr = [m_curr]
    if not isinstance(m_new, list):
        m_new = [m_new]
    merged["mixins"] = list(dict.fromkeys(m_curr + m_new))

    # Depends: union dependency entries, keep highest version constraint
    d_curr = dict(merged.get("depends", {}))
    for dep_id, dep_ver in incoming.get("depends", {}).items():
        if dep_id not in d_curr:
            d_curr[dep_id] = dep_ver
    if d_curr:
        merged["depends"] = d_curr

    return merged


def _merge_mixin_configs(existing_json: str, incoming_json: str) -> str:
    """Merge two mixin JSON configs: union mixin class arrays."""
    try:
        a = json.loads(existing_json)
        b = json.loads(incoming_json)
    except Exception:
        return existing_json

    for key in ("mixins", "client", "server"):
        a_list = a.get(key, [])
        b_list = b.get(key, [])
        if not isinstance(a_list, list):
            a_list = []
        if not isinstance(b_list, list):
            b_list = []
        merged = list(dict.fromkeys(a_list + b_list))
        if merged:
            a[key] = merged

    return json.dumps(a, indent=2)


def _merge_access_widener(existing: str, incoming: str) -> str:
    """Merge two access widener files: union all declarations, keep header."""
    lines_a = existing.strip().splitlines()
    lines_b = incoming.strip().splitlines()

    header = ""
    decls: list[str] = []
    seen: set[str] = set()

    for line in lines_a + lines_b:
        stripped = line.strip()
        if stripped.startswith("accessWidener") and not header:
            header = stripped
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if stripped not in seen:
            seen.add(stripped)
            decls.append(stripped)

    if not header:
        header = "accessWidener v2 named"
    return header + "\n" + "\n".join(decls) + "\n"


def _merge_build_gradle_dependencies(existing: str, incoming_deps: Sequence[str]) -> str:
    """Inject additional dependency lines into build.gradle dependencies block."""
    if not incoming_deps:
        return existing

    dep_block_re = re.compile(r"(dependencies\s*\{)(.*?)(})", re.DOTALL)
    match = dep_block_re.search(existing)
    if not match:
        new_block = "dependencies {\n" + "\n".join(f"    {d}" for d in incoming_deps) + "\n}\n"
        return existing + "\n" + new_block

    existing_block = match.group(2)
    new_lines: list[str] = []
    for dep in incoming_deps:
        if dep.strip() not in existing_block:
            new_lines.append(f"    {dep.strip()}")
    if not new_lines:
        return existing

    return (
        existing[:match.end(2)]
        + "\n".join(new_lines) + "\n"
        + existing[match.end(2):]
    )


def _merge_data_json(existing_json: str, incoming_json: str) -> str:
    """Merge two data-pack JSON files (tags, recipes, loot): union values arrays."""
    try:
        a = json.loads(existing_json)
        b = json.loads(incoming_json)
    except Exception:
        return existing_json

    if not isinstance(a, dict) or not isinstance(b, dict):
        return existing_json

    # Tags: merge "values" arrays
    if "values" in a or "values" in b:
        va = a.get("values", [])
        vb = b.get("values", [])
        if isinstance(va, list) and isinstance(vb, list):
            a["values"] = list(dict.fromkeys(va + vb))
        if "replace" in b:
            a["replace"] = b["replace"]
        return json.dumps(a, indent=2)

    # Generic: shallow merge keys
    for k, v in b.items():
        if k not in a:
            a[k] = v
    return json.dumps(a, indent=2)


def _rewrite_resource_namespace(
    path: str,
    target_modid: str,
) -> str:
    """Rewrite assets/<donor_modid>/... or data/<donor_modid>/... to target_modid."""
    m = _RESOURCE_NS_RE.match(path)
    if m:
        prefix, _donor_ns, rest = m.group(1), m.group(2), m.group(3)
        return f"{prefix}{target_modid}/{rest}"
    return path


# ---------------------------------------------------------------------------
# FinalProjectAssembler
# ---------------------------------------------------------------------------

class FinalProjectAssembler:
    """Authoritative assembler for multi-donor reused components and fresh modules."""

    def __init__(
        self,
        target_workspace: Path | str,
        target_context: Mapping[str, Any],
    ) -> None:
        self.workspace_path = Path(target_workspace)
        self.target_context = dict(target_context)
        self.target_loader = str(target_context.get("loader") or "fabric").lower().strip()
        self.target_minecraft = str(target_context.get("minecraft_version") or "1.21.1").strip()
        self.target_modid = str(target_context.get("target_modid") or "generated_mod").strip()
        self.target_package = str(target_context.get("target_package") or "ai.minecraft.generated.mod").strip()

    def assemble_plan(
        self,
        plan: TargetImplementationPlan,
        *,
        proof_receipts: Mapping[str, ReuseProofReceipt] | None = None,
        residual_files: Mapping[str, str | bytes] | None = None,
        fresh_files: Mapping[str, str | bytes] | None = None,
    ) -> FinalProjectAssemblyResult:
        """Assemble a full project from an implementation plan and verification receipts."""
        proof_map = proof_receipts or {}
        donors_to_reuse: list[DonorSlice] = []
        work_orders: list[ResidualWorkOrder] = []

        for decision in plan.capabilities:
            cap = decision.capability
            donor_slice = decision.donor_slice
            if donor_slice is None and isinstance(decision.donor, Mapping) and "repository" in decision.donor:
                from .source_transplant import DonorFile
                dfiles = tuple(
                    DonorFile(
                        path=f.get("path", ""),
                        blob_sha=f.get("blob_sha", ""),
                        sha256=f.get("sha256", ""),
                        size=f.get("size", 0),
                        symbols=tuple(f.get("symbols", ())),
                    )
                    for f in decision.donor.get("files", ())
                )
                donor_slice = DonorSlice(
                    capability=decision.donor.get("capability", cap),
                    repository=decision.donor.get("repository", ""),
                    commit_sha=decision.donor.get("commit_sha", ""),
                    license_id=decision.donor.get("license_id", "MIT"),
                    source_url=decision.donor.get("source_url", ""),
                    target_compatibility=decision.donor.get("target_compatibility", "exact"),
                    files=dfiles,
                    seed_files=tuple(decision.donor.get("seed_files", ())),
                    source_symbols=tuple(decision.donor.get("source_symbols", ())),
                    required_dependencies=tuple(decision.donor.get("required_dependencies", ())),
                    donor_tests=tuple(decision.donor.get("donor_tests", ())),
                    confidence=decision.donor.get("confidence", 0.9),
                    adaptation_cost=decision.donor.get("adaptation_cost", 0.0),
                    closure_complete=decision.donor.get("closure_complete", True),
                )

            if donor_slice is not None:
                rcpt = decision.proof_receipt or proof_map.get(cap)
                if rcpt and rcpt.proof_level in {"COMPILE_VERIFIED", "BEHAVIOR_VERIFIED", "PARTIAL_REUSE", "SUBGRAPH_COMPILE_VERIFIED"}:
                    donors_to_reuse.append(donor_slice)
                elif decision.mode in {"same_project", "mmm_verified"}:
                    donors_to_reuse.append(donor_slice)

            if decision.proof_receipt and decision.proof_receipt.work_order:
                work_orders.append(decision.proof_receipt.work_order)
            elif cap in proof_map and proof_map[cap].work_order:
                work_orders.append(proof_map[cap].work_order)

        return self.assemble(
            reused_donors=donors_to_reuse,
            residual_files=residual_files,
            fresh_files=fresh_files,
            work_orders=work_orders,
        )

    def _stage_file(
        self,
        staged: dict[str, str | bytes],
        norm_path: str,
        content: str | bytes,
        errors: list[str],
        *,
        source_label: str,
    ) -> bool:
        """Stage a single file with typed merge for known metadata paths.

        Returns True if the file was staged (new or merged), False if rejected.
        """
        # Resource namespace rewrite
        norm_path = _rewrite_resource_namespace(norm_path, self.target_modid)

        if norm_path not in staged:
            staged[norm_path] = content
            return True

        existing = staged[norm_path]

        # Typed merge: fabric.mod.json
        if norm_path == "src/main/resources/fabric.mod.json":
            try:
                a = json.loads(str(existing))
                b = json.loads(str(content))
                if isinstance(a, dict) and isinstance(b, dict):
                    staged[norm_path] = json.dumps(
                        _merge_fabric_mod_json(a, b, self.target_modid), indent=2
                    )
                    return True
            except Exception:
                pass

        # Typed merge: mixin configs
        fname = PurePosixPath(norm_path).name
        if _MIXIN_CONFIG_RE.search(fname) and isinstance(existing, str) and isinstance(content, str):
            staged[norm_path] = _merge_mixin_configs(existing, content)
            return True

        # Typed merge: access widener
        if fname.endswith(".accesswidener") and isinstance(existing, str) and isinstance(content, str):
            staged[norm_path] = _merge_access_widener(existing, content)
            return True

        # Typed merge: data-pack JSON (tags, recipes, loot tables)
        if norm_path.startswith("src/main/resources/data/") and fname.endswith(".json"):
            if isinstance(existing, str) and isinstance(content, str):
                staged[norm_path] = _merge_data_json(existing, content)
                return True

        # No typed merge available for this path — collision
        errors.append(f"{source_label}: {norm_path}")
        return False

    def assemble(
        self,
        *,
        reused_donors: Sequence[DonorSlice] = (),
        reused_adapted_files: Mapping[str, str | bytes] | None = None,
        residual_files: Mapping[str, str | bytes] | None = None,
        fresh_files: Mapping[str, str | bytes] | None = None,
        work_orders: Sequence[ResidualWorkOrder] = (),
    ) -> FinalProjectAssemblyResult:
        """Assemble all project components into the target workspace with typed merge."""
        staged: dict[str, str | bytes] = {}
        reused_count = 0
        residual_count = 0
        fresh_count = 0
        errors: list[str] = []
        all_donor_deps: list[str] = []

        # 1. Stage verified reused files
        if reused_donors:
            for donor in reused_donors:
                raw_files: dict[str, str | bytes] = {}
                try:
                    raw_map = materialize_pinned_donor(donor)
                    if not raw_map and donor.files:
                        errors.append(f"DONOR_MATERIALIZATION_FAILED: {donor.repository}@{donor.commit_sha}")
                        continue
                    for rel_path, raw_bytes in raw_map.items():
                        try:
                            raw_files[rel_path] = raw_bytes.decode("utf-8")
                        except UnicodeDecodeError:
                            raw_files[rel_path] = raw_bytes
                except Exception as exc:
                    errors.append(f"DONOR_MATERIALIZATION_ERROR: {donor.repository}@{donor.commit_sha} - {exc}")
                    continue

                adapted, _ = apply_deterministic_adapters(raw_files, self.target_context)
                for rel_path, content in adapted.items():
                    norm_path = rel_path.replace("\\", "/").strip("/")
                    ok = self._stage_file(staged, norm_path, content, errors, source_label="DUPLICATE_REUSED_PATH")
                    if ok:
                        reused_count += 1

                # Collect donor dependencies for build.gradle merge
                for dep in donor.required_dependencies:
                    if dep and dep not in all_donor_deps:
                        all_donor_deps.append(dep)

        elif reused_adapted_files:
            for rel_path, content in reused_adapted_files.items():
                norm_path = rel_path.replace("\\", "/").strip("/")
                staged[norm_path] = content
                reused_count += 1

        # 2. Stage residual generated files (must not overwrite verified reused files)
        if residual_files:
            for rel_path, content in residual_files.items():
                norm_path = rel_path.replace("\\", "/").strip("/")
                ok = self._stage_file(staged, norm_path, content, errors, source_label="RESIDUAL_OVERWRITE_PROTECTED_FILE")
                if ok:
                    residual_count += 1

        # 3. Stage fresh generated files
        if fresh_files:
            for rel_path, content in fresh_files.items():
                norm_path = rel_path.replace("\\", "/").strip("/")
                ok = self._stage_file(staged, norm_path, content, errors, source_label="FRESH_OVERWRITE_PROTECTED_FILE")
                if ok:
                    fresh_count += 1

        # 4. Typed merge: fabric.mod.json final normalization
        fmj_key = "src/main/resources/fabric.mod.json"
        if fmj_key in staged and self.target_loader == "fabric":
            try:
                mod_meta = json.loads(str(staged[fmj_key]))
                if isinstance(mod_meta, dict):
                    mod_meta["id"] = self.target_modid
                    mod_meta["name"] = self.target_modid.replace("_", " ").title()
                    # Deduplicate entrypoints
                    entrypoints = mod_meta.get("entrypoints", {})
                    if isinstance(entrypoints, dict):
                        for ep_key in ("main", "client", "server"):
                            curr = entrypoints.get(ep_key, [])
                            if isinstance(curr, list):
                                entrypoints[ep_key] = list(dict.fromkeys(curr))
                    # Ensure mixin config references target modid
                    mixins = mod_meta.get("mixins", [])
                    if isinstance(mixins, list):
                        canonical_mixin = f"{self.target_modid}.mixins.json"
                        if canonical_mixin not in mixins:
                            # Rewrite donor mixin refs to canonical
                            mod_meta["mixins"] = list(dict.fromkeys(
                                [canonical_mixin if _MIXIN_CONFIG_RE.search(m) else m for m in mixins]
                                if mixins else [canonical_mixin]
                            ))
                    staged[fmj_key] = json.dumps(mod_meta, indent=2)
            except Exception:
                pass

        # 5. Typed merge: build.gradle dependency injection
        if all_donor_deps and "build.gradle" in staged and isinstance(staged["build.gradle"], str):
            dep_lines = [f'modImplementation "{d}"' for d in all_donor_deps]
            staged["build.gradle"] = _merge_build_gradle_dependencies(
                staged["build.gradle"], dep_lines
            )

        is_valid = len(errors) == 0

        # Atomic materialization: Only write to disk if pre-write validation passed completely
        if is_valid:
            self.workspace_path.mkdir(parents=True, exist_ok=True)
            apply_verified_scaffold(self.workspace_path, self.target_context)

            for rel_path, content in staged.items():
                dest = self.workspace_path / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, bytes):
                    dest.write_bytes(content)
                else:
                    dest.write_text(str(content), encoding="utf-8")

            manifest = generate_reuse_manifest(reused_donors, project_name=self.target_modid)
            manifest_path = self.workspace_path / "reuse-manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        else:
            manifest = {}

        return FinalProjectAssemblyResult(
            project_name=self.target_modid,
            target_loader=self.target_loader,
            target_minecraft=self.target_minecraft,
            total_files=len(staged) if is_valid else 0,
            reused_file_count=reused_count if is_valid else 0,
            residual_file_count=residual_count if is_valid else 0,
            fresh_file_count=fresh_count if is_valid else 0,
            staged_paths=tuple(sorted(staged.keys())) if is_valid else (),
            manifest_sbom=manifest,
            work_orders=tuple(work_orders),
            is_valid=is_valid,
            errors=tuple(errors),
        )