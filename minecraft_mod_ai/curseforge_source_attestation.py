"""Exact per-file provenance between pinned donor bytes and Gradle-proven outputs.

This module is intentionally separate from semantic ranking. It is invoked AFTER
MMM's authoritative compile proof, not as a replacement for compilation.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import re
from typing import Any


def _sha256(raw: bytes | str) -> str:
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def attest_compiled_donor_files(donor: Any, raw_files: Mapping[str, bytes],
                                proof: Mapping[str, Any], target_context: Mapping[str, Any]) -> list[dict[str, str]]:
    """Map every pinned original to one actual *compiled* adapted artifact.

    Package relocation preserves Java basenames; if ambiguous, decline reuse.
    This intentionally refuses any donor containing unproved files, including
    license/NOTICE files required for redistribution.
    """
    from .reuse_adapters import apply_deterministic_adapters
    if proof.get("authoritative_compile") is not True or proof.get("compile_passed") is not True:
        raise ValueError("SOURCE_ATTESTATION_REAL_COMPILE_REQUIRED")
    verified = proof.get("verified_artifacts") or ()
    if not isinstance(verified, (list, tuple)) or not all(isinstance(p, str) for p in verified):
        raise ValueError("SOURCE_ATTESTATION_BAD_VERIFIED_ARTIFACTS")
    verified_set = set(verified)
    contract = proof.get("contract")
    protected = contract.get("protected_artifacts") if isinstance(contract, Mapping) else None
    if not isinstance(protected, Mapping):
        raise ValueError("SOURCE_ATTESTATION_MISSING_PROOF_HASHES")
    inputs = {item.path: raw_files[item.path] for item in donor.files}
    adapted, _receipts = apply_deterministic_adapters(inputs, target_context)
    original_names = Counter(path.rsplit("/", 1)[-1] for path in inputs)
    output_names = Counter(path.rsplit("/", 1)[-1] for path in adapted)
    seen_outputs: set[str] = set()
    mappings: list[dict[str, str]] = []
    for item in donor.files:
        original = item.path
        raw = inputs[original]
        if _sha256(raw) != item.sha256:
            raise ValueError("SOURCE_ATTESTATION_ORIGINAL_CHANGED")
        if original in adapted:
            output = original
        elif original.endswith(".java"):
            basename = original.rsplit("/", 1)[-1]
            if original_names[basename] != 1 or output_names[basename] != 1:
                raise ValueError("SOURCE_ATTESTATION_AMBIGUOUS_RELOCATION: " + basename)
            matches = [p for p in adapted if p.endswith("/" + basename)]
            if len(matches) != 1:
                raise ValueError("SOURCE_ATTESTATION_UNMAPPED_JAVA: " + original)
            output = matches[0]
        else:
            raise ValueError("SOURCE_ATTESTATION_UNMAPPED_RESOURCE: " + original)
        if output in seen_outputs or output not in verified_set:
            raise ValueError("SOURCE_ATTESTATION_UNVERIFIED_OUTPUT: " + output)
        if protected.get(output) != _sha256(adapted[output]):
            raise ValueError("SOURCE_ATTESTATION_COMPILED_HASH_MISMATCH: " + output)
        seen_outputs.add(output)
        mappings.append({
            "source_path": original,
            "source_sha256": item.sha256,
            "verified_path": output,
            "verified_sha256": _sha256(adapted[output]),
        })
    return mappings


def check_attested_files(donor: Any, proof: Mapping[str, Any]) -> None:
    """Fail closed unless exactly one complete and distinct mapping exists."""
    raw = proof.get("verified_file_attestations")
    if not isinstance(raw, list) or len(raw) != len(donor.files):
        raise ValueError("SOURCE_ATTESTATION_MISSING_OR_INCOMPLETE")
    verified = set(proof.get("verified_artifacts") or ())
    contract = proof.get("contract")
    protected = contract.get("protected_artifacts") if isinstance(contract, Mapping) else None
    if not isinstance(protected, Mapping):
        raise ValueError("SOURCE_ATTESTATION_MISSING_PROOF_HASHES")
    donor_map = {item.path: item for item in donor.files}
    if len(donor_map) != len(donor.files):
        raise ValueError("SOURCE_ATTESTATION_DUPLICATE_DONOR")
    mapped_src: set[str] = set()
    mapped_output: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("SOURCE_ATTESTATION_ENTRY_INVALID")
        source = str(item.get("source_path") or "")
        dest = str(item.get("verified_path") or "")
        expected = donor_map.get(source)
        if (expected is None or source in mapped_src or dest in mapped_output
                or str(item.get("source_sha256") or "") != expected.sha256
                or dest not in verified
                or protected.get(dest) != item.get("verified_sha256")
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(item.get("verified_sha256") or ""))):
            raise ValueError("SOURCE_ATTESTATION_BINDING_INVALID")
        if source != dest and source.rsplit("/", 1)[-1] != dest.rsplit("/", 1)[-1]:
            raise ValueError("SOURCE_ATTESTATION_UNEXPECTED_RENAME")
        mapped_src.add(source)
        mapped_output.add(dest)
    if mapped_src != set(donor_map):
        raise ValueError("SOURCE_ATTESTATION_DONOR_COVERAGE_INVALID")
