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


def test_namespace_relocation_attested_for_json_and_png():
    from minecraft_mod_ai.reuse_adapters import apply_deterministic_adapters

    blobs = {
        "src/main/resources/assets/oldmod/lang/ko_kr.json": b'{"item.oldmod.key":"Title"}',
        "src/main/resources/assets/oldmod/textures/item/key.png": b"\\x89PNG\\r\\n\\x1a\\nPNGDATA",
        "LICENSE": b"MIT License",
    }
    donor = SimpleNamespace(files=tuple(SimpleNamespace(path=p, sha256=_sha(raw))
                                        for p, raw in blobs.items()))
    context = {"target_modid": "newmod", "donor_modid": "oldmod"}
    adapted, _ = apply_deterministic_adapters(blobs, context)
    proof = {"authoritative_compile": True, "compile_passed": True,
             "verified_artifacts": list(adapted),
             "contract": {"protected_artifacts": {
                 path: _sha(raw.encode("utf-8") if isinstance(raw, str) else raw)
                 for path, raw in adapted.items()
             }}}
    mapping = attest_compiled_donor_files(donor, blobs, proof, context)
    check_attested_files(donor, {**proof, "verified_file_attestations": mapping})
    mapped = {row["source_path"]: row["verified_path"] for row in mapping}
    assert mapped["src/main/resources/assets/oldmod/lang/ko_kr.json"] == (
        "src/main/resources/assets/newmod/lang/ko_kr.json"
    )
    assert mapped["src/main/resources/assets/oldmod/textures/item/key.png"] == (
        "src/main/resources/assets/newmod/textures/item/key.png"
    )


def test_resource_attestation_cannot_swap_unrelated_resource_directory():
    donor, blobs, proof = _fixture()
    old_path = "src/main/resources/assets/oldmod/lang/ko_kr.json"
    new_path = "src/main/resources/assets/newmod/loot_tables/ko_kr.json"
    blobs[old_path] = b"{}"
    donor.files = tuple(SimpleNamespace(path=p, sha256=_sha(b)) for p, b in blobs.items())
    proof["verified_artifacts"].append(new_path)
    proof["contract"]["protected_artifacts"][new_path] = _sha(b"{}")
    with pytest.raises(ValueError, match="UNEXPECTED_RENAME"):
        check_attested_files(donor, {
            **proof, "verified_file_attestations": [
                {"source_path": p, "source_sha256": _sha(raw),
                 "verified_path": new_path if p == old_path else p,
                 "verified_sha256": _sha(raw)}
                for p, raw in blobs.items()
            ],
        })


def test_text_and_utf8_decodable_binary_share_relocated_namespace():
    from minecraft_mod_ai.reuse_adapters import apply_deterministic_adapters

    files = {
        "src/main/resources/assets/oldmod/lang/en_us.json": b'{"item.oldmod.key":"Value"}',
        "src/main/resources/assets/oldmod/textures/item/key.png": b"ABCD",
    }
    adapted, _ = apply_deterministic_adapters(files, {"target_modid": "newmod"})
    assert "src/main/resources/assets/oldmod/lang/en_us.json" not in adapted
    assert "src/main/resources/assets/oldmod/textures/item/key.png" not in adapted
    assert "src/main/resources/assets/newmod/lang/en_us.json" in adapted
    assert adapted["src/main/resources/assets/newmod/textures/item/key.png"] == b"ABCD"
