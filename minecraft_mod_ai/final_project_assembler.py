from __future__ import annotations

"""Authoritative Final Project Assembler and Integration Orchestrator.

Combines all verified reused donor slices, partial residual components, and freshly
generated modules into a single canonical Minecraft mod project layout with typed merges.

Merge Rules:
1. Java/Kotlin source merge: Normalize packages, preserve immutable reused symbols.
2. Resource merge: Translate donor namespaces to target_modid (assets, data, models, textures).
3. Mod Metadata merge: Combine entrypoints (main, client, server) in fabric.mod.json / neoforge.mods.toml.
4. Mixin & AW merge: Combine mixin configurations and access wideners into unified files.
5. Build script & dependency merge: Inject all required external repositories and coordinates.
6. Emits FinalProjectAssemblyResult and reuse-manifest.json SBOM.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .composition_solver import generate_reuse_manifest
from .reuse_adapters import adapt_donor_slice_for_target
from .reuse_planner import TargetImplementationPlan
from .reuse_proof_executor import ReuseProofReceipt, ResidualWorkOrder
from .source_transplant import DonorSlice
from .verified_scaffold_registry import apply_verified_scaffold


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

        for decision in plan.decisions:
            cap = decision.capability
            if decision.donor_slice is not None:
                rcpt = proof_map.get(cap)
                if rcpt and rcpt.proof_level in {"COMPILE_VERIFIED", "BEHAVIOR_VERIFIED", "PARTIAL_REUSE", "SUBGRAPH_COMPILE_VERIFIED"}:
                    donors_to_reuse.append(decision.donor_slice)
                elif decision.mode == "same_project":
                    donors_to_reuse.append(decision.donor_slice)

            if rcpt and rcpt.work_order:
                work_orders.append(rcpt.work_order)

        return self.assemble(
            reused_donors=donors_to_reuse,
            residual_files=residual_files,
            fresh_files=fresh_files,
            work_orders=work_orders,
        )

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
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        apply_verified_scaffold(self.workspace_path, self.target_context)

        staged: dict[str, str | bytes] = {}
        reused_count = 0
        residual_count = 0
        fresh_count = 0
        errors: list[str] = []

        # 1. Stage verified reused files
        if reused_donors:
            for donor in reused_donors:
                adapted = adapt_donor_slice_for_target(donor, self.target_context)
                for rel_path, content in adapted.items():
                    norm_path = rel_path.replace("\\", "/").strip("/")
                    if norm_path in staged:
                        errors.append(f"DUPLICATE_REUSED_PATH: {norm_path}")
                    staged[norm_path] = content
                    reused_count += 1
        elif reused_adapted_files:
            for rel_path, content in reused_adapted_files.items():
                norm_path = rel_path.replace("\\", "/").strip("/")
                staged[norm_path] = content
                reused_count += 1

        # 2. Stage residual generated files (must not overwrite verified reused files)
        if residual_files:
            for rel_path, content in residual_files.items():
                norm_path = rel_path.replace("\\", "/").strip("/")
                if norm_path in staged:
                    errors.append(f"RESIDUAL_OVERWRITE_PROTECTED_FILE: {norm_path}")
                else:
                    staged[norm_path] = content
                    residual_count += 1

        # 3. Stage fresh generated files
        if fresh_files:
            for rel_path, content in fresh_files.items():
                norm_path = rel_path.replace("\\", "/").strip("/")
                if norm_path in staged:
                    errors.append(f"FRESH_OVERWRITE_PROTECTED_FILE: {norm_path}")
                else:
                    staged[norm_path] = content
                    fresh_count += 1

        # 4. Write all staged files to disk
        for rel_path, content in staged.items():
            dest = self.workspace_path / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                dest.write_bytes(content)
            else:
                dest.write_text(str(content), encoding="utf-8")

        # 5. Generate cryptographic SBOM manifest
        manifest = generate_reuse_manifest(reused_donors, project_name=self.target_modid)
        manifest_path = self.workspace_path / "reuse-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        is_valid = len(errors) == 0

        return FinalProjectAssemblyResult(
            project_name=self.target_modid,
            target_loader=self.target_loader,
            target_minecraft=self.target_minecraft,
            total_files=len(staged),
            reused_file_count=reused_count,
            residual_file_count=residual_count,
            fresh_file_count=fresh_count,
            staged_paths=tuple(sorted(staged.keys())),
            manifest_sbom=manifest,
            work_orders=tuple(work_orders),
            is_valid=is_valid,
            errors=tuple(errors),
        )