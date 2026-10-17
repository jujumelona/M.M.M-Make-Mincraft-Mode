"""Regression tests for the production donor source attestation contract."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from minecraft_mod_ai.curseforge_source_attestation import (
    attest_compiled_donor_files, check_attested_files,
)


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _fixture():
    blobs = {
        "src/main/java/example/Alpha.java": b"package example; class Alpha {}",
        "src/main/java/example/Beta.java": b"package example; class Beta {}",
        "LICENSE": b"SPDX-License-Identifier: MIT\n",
    }
    donor = SimpleNamespace(files=tuple(SimpleNamespace(path=p, sha256=_sha(b)) for p, b in blobs.items()))
    proof = {
        "authoritative_compile": True,
        "compile_passed": True,
        "verified_artifacts": list(blobs),
        "contract": {"protected_artifacts": {p: _sha(b) for p, b in blobs.items()}},
    }
    return donor, blobs, proof


def test_attests_every_original_file_not_just_one():
    donor, blobs, proof = _fixture()
    with patch("minecraft_mod_ai.reuse_adapters.apply_deterministic_adapters", return_value=(dict(blobs), ())):
        attestations = attest_compiled_donor_files(donor, blobs, proof, {})
    assert len(attestations) == 3
    check_attested_files(donor, {**proof, "verified_file_attestations": attestations})


def test_missing_compiled_file_cannot_borrow_another_file_receipt():
    donor, blobs, proof = _fixture()
    proof["verified_artifacts"] = ["src/main/java/example/Alpha.java", "LICENSE"]
    with patch("minecraft_mod_ai.reuse_adapters.apply_deterministic_adapters", return_value=(dict(blobs), ())):
        with pytest.raises(ValueError, match="UNVERIFIED_OUTPUT"):
            attest_compiled_donor_files(donor, blobs, proof, {})


def test_missing_authoritative_compile_fails_closed():
    donor, blobs, proof = _fixture()
    proof["authoritative_compile"] = False
    with pytest.raises(ValueError, match="REAL_COMPILE_REQUIRED"):
        attest_compiled_donor_files(donor, blobs, proof, {})


def test_wrong_protected_artifact_hash_fails_closed():
    donor, blobs, proof = _fixture()
    proof["contract"]["protected_artifacts"]["src/main/java/example/Alpha.java"] = "sha256:" + "0" * 64
    with patch("minecraft_mod_ai.reuse_adapters.apply_deterministic_adapters", return_value=(dict(blobs), ())):
        with pytest.raises(ValueError, match="COMPILED_HASH_MISMATCH"):
            attest_compiled_donor_files(donor, blobs, proof, {})


def test_tampered_attestation_is_rejected():
    donor, blobs, proof = _fixture()
    with patch("minecraft_mod_ai.reuse_adapters.apply_deterministic_adapters", return_value=(dict(blobs), ())):
        attestations = attest_compiled_donor_files(donor, blobs, proof, {})
    attestations[0]["source_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="BINDING_INVALID"):
        check_attested_files(donor, {**proof, "verified_file_attestations": attestations})
