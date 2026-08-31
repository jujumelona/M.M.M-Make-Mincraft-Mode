from __future__ import annotations

"""Universal Reusable Artifact Bundle & Multi-Origin Normalizer.

Unifies external donors, same-project components, and MMM-verified registry assets
into a single authoritative representation. Downstream subsystems (composition solver,
residual coder, final assembler, manifest generator) operate exclusively on this
common artifact bundle.
"""

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .reuse_license import is_reusable_source_license
from .source_transplant import (
    DonorSlice,
    SourceTransplantError,
    materialize_pinned_donor,
)


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _normalized_path(value: Any) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw or raw.startswith("/") or re.match(r"^[A-Za-z]:/", raw):
        return ""
    parts = tuple(part for part in raw.split("/") if part not in {"", "."})
    if not parts or ".." in parts:
        return ""
    return "/".join(parts)


def _normalized_sha256(value: Any) -> str:
    digest = str(value or "").strip().casefold()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest):
        return "sha256:" + digest
    return ""


def _normalized_hash_mapping(values: Mapping[str, Any]) -> dict[str, str] | None:
    """Normalize an artifact map without silently dropping malformed receipt rows."""

    normalized: dict[str, str] = {}
    for raw_path, raw_digest in values.items():
        path = _normalized_path(raw_path)
        digest = _normalized_sha256(raw_digest)
        if not path or not digest or path in normalized:
            return None
        normalized[path] = digest
    return normalized


def _receipt_value(receipt: Any, key: str, default: Any = None) -> Any:
    if isinstance(receipt, Mapping):
        return receipt.get(key, default)
    return getattr(receipt, key, default)


def _hashes_match(
    expected: Mapping[str, Any], bundle: ReusableArtifactBundle
) -> bool:
    normalized_expected = _normalized_hash_mapping(expected)
    normalized_bundle = _normalized_hash_mapping(bundle.file_hashes)
    protected_paths = tuple(_normalized_path(path) for path in bundle.protected_paths)
    if (
        normalized_expected is None
        or normalized_bundle is None
        or not all(protected_paths)
        or len(set(protected_paths)) != len(protected_paths)
        or set(normalized_expected) != set(protected_paths)
        or set(normalized_bundle) != set(protected_paths)
    ):
        return False
    return all(
        normalized_expected[path] == normalized_bundle[path]
        for path in protected_paths
    )


def _proof_dependencies_are_exact(values: Sequence[Any]) -> bool:
    for raw in values:
        if not isinstance(raw, Mapping):
            return False
        if raw.get("is_resolved") is not True:
            return False
        if not str(raw.get("resolved_coordinate") or "").strip():
            return False
        if not str(raw.get("repository") or "").strip():
            return False
        if not str(raw.get("gradle_configuration") or "").strip():
            return False
        fingerprint = str(raw.get("resolution_fingerprint") or "").strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", fingerprint):
            return False
    return True


def bundle_proof_allows_reuse(
    bundle: ReusableArtifactBundle, receipt: Any,
) -> bool:
    """Accept reuse only when a receipt binds this exact bundle and its bytes."""

    from .proof_level import ProofLevel

    level = ProofLevel.from_value(_receipt_value(receipt, "proof_level"))
    if not level.allows_reuse():
        return False
    if str(_receipt_value(receipt, "capability", "")).strip() != bundle.capability:
        return False
    if not bundle.protected_paths or not _hashes_match(bundle.file_hashes, bundle):
        return False

    if bundle.origin_kind == "external_donor":
        if _receipt_value(receipt, "authoritative_compile") is not True:
            return False
        if str(_receipt_value(receipt, "candidate_id", "")).strip() != bundle.source_ref:
            return False
        repository = str(bundle.provenance.get("repository") or "").strip()
        commit_sha = str(bundle.provenance.get("commit_sha") or "").strip()
        license_id = str(bundle.provenance.get("license_id") or "").strip()
        if (
            not repository
            or not re.fullmatch(r"[0-9a-f]{40,64}", commit_sha)
            or not is_reusable_source_license(license_id)
            or bundle.source_ref != f"{repository}@{commit_sha}"
        ):
            return False
        if level.is_verified() and _receipt_value(receipt, "compile_passed") is not True:
            return False
        if level.is_partial() and not tuple(
            _receipt_value(receipt, "verified_artifacts", ()) or ()
        ):
            return False
        proof_dependencies = tuple(
            _receipt_value(receipt, "dependency_receipts", ()) or ()
        )
        if tuple(bundle.dependency_receipts) != proof_dependencies:
            return False
        donor_payload = bundle.provenance.get("donor_slice")
        declared_dependencies = (
            tuple(donor_payload.get("required_dependencies", ()) or ())
            if isinstance(donor_payload, Mapping)
            else ()
        )
        if declared_dependencies and not proof_dependencies:
            return False
        if not _proof_dependencies_are_exact(proof_dependencies):
            return False
        contract = _receipt_value(receipt, "contract")
        protected = _receipt_value(contract, "protected_artifacts", {})
        return isinstance(protected, Mapping) and _hashes_match(protected, bundle)

    expected_schema = {
        "same_project": "mmm/same-project-proof-receipt-v1",
        "mmm_verified": "mmm/registry-component-proof-receipt-v1",
    }.get(bundle.origin_kind)
    if expected_schema is None or not isinstance(receipt, Mapping):
        return False
    if receipt.get("schema_version") != expected_schema:
        return False
    if str(receipt.get("bundle_id") or "") != bundle.bundle_id:
        return False
    if str(receipt.get("source_ref") or "") != bundle.source_ref:
        return False
    hashes = receipt.get("file_hashes")
    return isinstance(hashes, Mapping) and _hashes_match(hashes, bundle)


class BundleMaterializationError(RuntimeError):
    """Raised when a proof-bound bundle cannot supply its protected bytes."""


def _verified_content_hashes(
    content: Mapping[str, str | bytes],
    declared: Mapping[str, str] | None,
) -> tuple[dict[str, str | bytes], dict[str, str]]:
    files: dict[str, str | bytes] = {}
    hashes = _normalized_hash_mapping(declared or {})
    if hashes is None:
        raise ValueError("Reusable artifact hashes must be SHA-256 values.")
    for raw_path, value in content.items():
        path = _normalized_path(raw_path)
        if not path or not isinstance(value, (str, bytes)):
            raise ValueError("Reusable artifact files require safe paths and text/bytes.")
        raw = value.encode("utf-8") if isinstance(value, str) else value
        actual = _sha256(raw)
        if path in hashes and hashes[path] != actual:
            raise ValueError(f"Reusable artifact content hash mismatch: {path}")
        files[path] = value
        hashes[path] = actual
    return files, hashes


@dataclass(frozen=True)
class ReusableArtifactBundle:
    bundle_id: str
    capability: str
    origin_kind: str  # "external_donor" | "same_project" | "mmm_verified"
    source_ref: str
    files: Mapping[str, str | bytes] = field(default_factory=dict)
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
        raw_files: Mapping[str, str | bytes] | None = None,
        protected_artifacts: Mapping[str, str] | None = None,
        materialize: bool = False,
        discovery_client: Any = None,
    ) -> ReusableArtifactBundle:
        """Normalize a proven donor without repeating proof-time downloads.

        Planning is metadata-only by default. ``materialize=True`` is an explicit
        downstream choice for callers that actually own file staging. For partial
        reuse, ``protected_artifacts`` must be the proof receipt's verified subset;
        the unverified remainder stays outside this bundle and in the residual
        generation contract.
        """

        files = dict(raw_files) if raw_files is not None else {}
        if materialize and raw_files is None:
            files = dict(
                materialize_pinned_donor(
                    donor,
                    discovery_client=discovery_client,
                )
            )

        if protected_artifacts is not None and not isinstance(
            protected_artifacts, Mapping
        ):
            raise ValueError("Protected donor artifacts must be a path-to-hash mapping.")
        proven_hashes = _normalized_hash_mapping(protected_artifacts or {})
        if proven_hashes is None:
            raise ValueError("Protected donor artifacts require safe SHA-256 bindings.")
        if files:
            verified_files, observed_hashes = _verified_content_hashes(
                files, proven_hashes or None
            )
            if not proven_hashes:
                proven_hashes = observed_hashes
            files = {
                path: value
                for path, value in verified_files.items()
                if path in proven_hashes
            }
        protected_paths = tuple(sorted(proven_hashes))
        protected_symbols = tuple(
            _receipt_value(proof_receipt, "verified_symbols", ()) or ()
        )
        dependency_receipts = tuple(
            _receipt_value(proof_receipt, "dependency_receipts", ()) or ()
        )

        # Detect owned namespace from donor files or repository name
        repo_name = donor.repository.split("/")[-1].lower().replace("-", "_") if donor.repository else ""
        owned_ns: list[str] = []
        if repo_name:
            owned_ns.append(repo_name)
        for path in protected_paths:
            if not path.startswith((
                "src/main/resources/assets/",
                "src/main/resources/data/",
            )):
                continue
            parts = path.split("/")
            if (
                len(parts) > 4
                and parts[4] not in ("minecraft", "c", "fabric", "neoforge", "forge")
                and parts[4] not in owned_ns
            ):
                owned_ns.append(parts[4])

        return cls(
            bundle_id=f"donor:{donor.repository}@{donor.commit_sha}:{donor.capability}",
            capability=donor.capability,
            origin_kind="external_donor",
            source_ref=f"{donor.repository}@{donor.commit_sha}",
            files=files,
            file_hashes=proven_hashes,
            requirement_ids=tuple(requirement_ids) or (donor.capability,),
            protected_paths=protected_paths,
            protected_symbols=protected_symbols,
            dependency_receipts=dependency_receipts,
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
        files: Mapping[str, str | bytes] | None = None,
        *,
        file_hashes: Mapping[str, str] | None = None,
        symbols: Sequence[str] = (),
        requirement_ids: Sequence[str] = (),
        proof_receipt: Any | None = None,
        source_ref: str = "current_workspace",
        provenance: Mapping[str, Any] | None = None,
    ) -> ReusableArtifactBundle:
        """Construct a bundle for existing project assets."""
        content, hashes = _verified_content_hashes(files or {}, file_hashes)
        return cls(
            bundle_id=f"same_project:{capability}",
            capability=capability,
            origin_kind="same_project",
            source_ref=source_ref,
            files=content,
            file_hashes=hashes,
            requirement_ids=tuple(requirement_ids) or (capability,),
            protected_paths=tuple(sorted(hashes)),
            protected_symbols=tuple(symbols),
            proof_receipt=proof_receipt,
            provenance=dict(provenance or {"source": "current_project"}),
        )

    @classmethod
    def from_verified_component(
        cls,
        component_id: str,
        capability: str,
        files: Mapping[str, str | bytes] | None = None,
        *,
        file_hashes: Mapping[str, str] | None = None,
        requirement_ids: Sequence[str] = (),
        dependencies: Sequence[str] = (),
        symbols: Sequence[str] = (),
        proof_receipt: Any | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> ReusableArtifactBundle:
        """Construct a bundle from a verified MMM registry component."""
        content, hashes = _verified_content_hashes(files or {}, file_hashes)
        return cls(
            bundle_id=f"mmm_verified:{component_id}",
            capability=capability,
            origin_kind="mmm_verified",
            source_ref=component_id,
            files=content,
            file_hashes=hashes,
            requirement_ids=tuple(requirement_ids) or (capability,),
            protected_paths=tuple(sorted(hashes)),
            protected_symbols=tuple(symbols),
            dependency_receipts=tuple(dependencies),
            proof_receipt=proof_receipt,
            provenance=dict(provenance or {"component_id": component_id}),
        )

    def materialize_for_target(
        self,
        *,
        workspace_root: str | Path,
        target_context: Mapping[str, Any],
        discovery_client: Any = None,
    ) -> dict[str, str | bytes]:
        """Return target-adapted bytes through the bundle's single materializer."""

        files: dict[str, str | bytes] = dict(self.files)
        if not files and self.origin_kind == "external_donor":
            donor_payload = self.provenance.get("donor_slice")
            if not isinstance(donor_payload, Mapping):
                raise BundleMaterializationError("Pinned donor provenance is missing.")
            try:
                donor = DonorSlice.from_dict(donor_payload)
                files = dict(
                    materialize_pinned_donor(
                        donor, discovery_client=discovery_client
                    )
                )
            except (SourceTransplantError, KeyError, TypeError, ValueError) as exc:
                raise BundleMaterializationError("Pinned donor materialization failed.") from exc
        elif not files and self.origin_kind == "same_project":
            root = Path(workspace_root).expanduser().resolve()
            for protected_path in self.protected_paths:
                path = _normalized_path(protected_path)
                if not path:
                    raise BundleMaterializationError("Same-project bundle path is unsafe.")
                target = root.joinpath(*path.split("/"))
                try:
                    resolved = target.resolve()
                    resolved.relative_to(root)
                except (OSError, RuntimeError, ValueError) as exc:
                    raise BundleMaterializationError("Same-project bundle escaped its root.") from exc
                if not resolved.is_file() or resolved.is_symlink():
                    continue
                files[path] = resolved.read_bytes()
        elif not files and self.origin_kind == "mmm_verified":
            artifact = self.provenance.get("artifact")
            raw_files = artifact.get("files") if isinstance(artifact, Mapping) else None
            if isinstance(raw_files, Mapping):
                for raw_path, raw_value in raw_files.items():
                    path = _normalized_path(raw_path)
                    if not path:
                        continue
                    value = raw_value
                    if isinstance(raw_value, Mapping):
                        value = raw_value.get("content", raw_value.get("text"))
                    if isinstance(value, (str, bytes)):
                        files[path] = value

        if self.origin_kind == "external_donor":
            from .reuse_adapters import apply_deterministic_adapters

            files, _ = apply_deterministic_adapters(files, target_context)
        normalized_files: dict[str, str | bytes] = {}
        for raw_path, value in files.items():
            path = _normalized_path(raw_path)
            if not path:
                raise BundleMaterializationError("Materialized bundle contains an unsafe path.")
            if path in normalized_files:
                raise BundleMaterializationError("Materialized bundle contains duplicate paths.")
            normalized_files[path] = value
        return normalized_files

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
