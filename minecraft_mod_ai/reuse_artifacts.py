from __future__ import annotations

"""Universal Reusable Artifact Bundle & Multi-Origin Normalizer.

Unifies external donors, same-project components, and MMM-verified registry assets
into a single authoritative representation. Downstream subsystems (composition solver,
residual coder, final assembler, manifest generator) operate exclusively on this
common artifact bundle.
"""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .source_transplant import DonorSlice, materialize_pinned_donor


@dataclass(frozen=True)
class ReusableArtifactBundle:
    bundle_id: str
    capability: str
    origin_kind: str  # "external_donor" | "same_project" | "mmm_verified"
    source_ref: str
    files: Mapping[str, bytes] = field(default_factory=dict)
    file_hashes: Mapping[str, str] = field(default_factory=dict)  # path -> sha256
    requirement_ids: tuple[str, ...] = ()
    protected_paths: tuple[str, ...] = ()
    protected_symbols: tuple[str, ...] = ()
    dependency_receipts: tuple[Any, ...] = ()
    target_compatibility: str = "exact"
    proof_receipt: Any | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    owned_namespaces: tuple[str, ...] = ()
    external_namespaces: tuple[str, ...] = ("minecraft", "c", "fabric", "neoforge", "forge")

    @classmethod
    def from_donor_slice(
        cls,
        donor: DonorSlice,
        *,
        proof_receipt: Any | None = None,
        requirement_ids: Sequence[str] = (),
        raw_files: Mapping[str, bytes] | None = None,
    ) -> ReusableArtifactBundle:
        """Construct a ReusableArtifactBundle from a materialized DonorSlice."""
        files = dict(raw_files) if raw_files is not None else {}
        if not files:
            try:
                raw_map = materialize_pinned_donor(donor)
                files = dict(raw_map)
            except Exception:
                files = {}

        file_hashes: dict[str, str] = {}
        for path, bdata in files.items():
            file_hashes[path] = hashlib.sha256(bdata).hexdigest()

        protected_paths = tuple(sorted(files.keys()))
        protected_symbols = tuple(donor.source_symbols)

        # Detect owned namespace from donor files or repository name
        repo_name = donor.repository.split("/")[-1].lower().replace("-", "_") if donor.repository else ""
        owned_ns: list[str] = []
        if repo_name:
            owned_ns.append(repo_name)
        for p in files:
            if p.startswith("src/main/resources/assets/"):
                parts = p.split("/")
                if len(parts) > 4 and parts[4] not in ("minecraft", "c", "fabric", "neoforge", "forge"):
                    if parts[4] not in owned_ns:
                        owned_ns.append(parts[4])
            elif p.startswith("src/main/resources/data/"):
                parts = p.split("/")
                if len(parts) > 4 and parts[4] not in ("minecraft", "c", "fabric", "neoforge", "forge"):
                    if parts[4] not in owned_ns:
                        owned_ns.append(parts[4])

        return cls(
            bundle_id=f"donor:{donor.repository}@{donor.commit_sha}:{donor.capability}",
            capability=donor.capability,
            origin_kind="external_donor",
            source_ref=f"{donor.repository}@{donor.commit_sha}",
            files=files,
            file_hashes=file_hashes,
            requirement_ids=tuple(requirement_ids) or (donor.capability,),
            protected_paths=protected_paths,
            protected_symbols=protected_symbols,
            dependency_receipts=tuple(donor.required_dependencies),
            target_compatibility=donor.target_compatibility,
            proof_receipt=proof_receipt,
            provenance={
                "repository": donor.repository,
                "commit_sha": donor.commit_sha,
                "license_id": donor.license_id,
                "source_url": donor.source_url,
                "donor_slice": donor.to_dict(),
            },
            owned_namespaces=tuple(owned_ns),
        )

    @classmethod
    def from_same_project(
        cls,
        capability: str,
        files: Mapping[str, bytes],
        *,
        symbols: Sequence[str] = (),
        requirement_ids: Sequence[str] = (),
    ) -> ReusableArtifactBundle:
        """Construct a bundle for existing project assets."""
        file_hashes = {p: hashlib.sha256(b).hexdigest() for p, b in files.items()}
        return cls(
            bundle_id=f"same_project:{capability}",
            capability=capability,
            origin_kind="same_project",
            source_ref="current_workspace",
            files=dict(files),
            file_hashes=file_hashes,
            requirement_ids=tuple(requirement_ids) or (capability,),
            protected_paths=tuple(sorted(files.keys())),
            protected_symbols=tuple(symbols),
            provenance={"source": "current_project"},
        )

    @classmethod
    def from_verified_component(
        cls,
        component_id: str,
        capability: str,
        files: Mapping[str, bytes],
        *,
        requirement_ids: Sequence[str] = (),
        dependencies: Sequence[str] = (),
    ) -> ReusableArtifactBundle:
        """Construct a bundle from a verified MMM registry component."""
        file_hashes = {p: hashlib.sha256(b).hexdigest() for p, b in files.items()}
        return cls(
            bundle_id=f"mmm_verified:{component_id}",
            capability=capability,
            origin_kind="mmm_verified",
            source_ref=component_id,
            files=dict(files),
            file_hashes=file_hashes,
            requirement_ids=tuple(requirement_ids) or (capability,),
            protected_paths=tuple(sorted(files.keys())),
            protected_symbols=(),
            dependency_receipts=tuple(dependencies),
            provenance={"component_id": component_id},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/reusable-artifact-bundle-v1",
            "bundle_id": self.bundle_id,
            "capability": self.capability,
            "origin_kind": self.origin_kind,
            "source_ref": self.source_ref,
            "file_hashes": dict(self.file_hashes),
            "requirement_ids": list(self.requirement_ids),
            "protected_paths": list(self.protected_paths),
            "protected_symbols": list(self.protected_symbols),
            "dependency_receipts": list(self.dependency_receipts),
            "target_compatibility": self.target_compatibility,
            "proof_receipt": self.proof_receipt.to_dict() if hasattr(self.proof_receipt, "to_dict") else self.proof_receipt,
            "provenance": dict(self.provenance),
            "owned_namespaces": list(self.owned_namespaces),
            "external_namespaces": list(self.external_namespaces),
        }
