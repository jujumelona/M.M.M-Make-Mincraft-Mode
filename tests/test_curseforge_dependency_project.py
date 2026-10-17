import hashlib
import json

import pytest

from minecraft_mod_ai.curseforge_dependency_project import (
    install_verified_fabric_dependency,
)


def _setup(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    jar = tmp_path / "external.jar"
    jar.write_bytes(b"dummy artifact for installer unit test")
    (root / "src/main/resources").mkdir(parents=True)
    (root / "build.gradle").write_text("plugins { id 'fabric-loom' }\n",
                                      encoding="utf-8")
    (root / "src/main/resources/fabric.mod.json").write_text(json.dumps({
        "schemaVersion": 1, "id": "my_mod",
        "depends": {"minecraft": "26.2", "fabricloader": "*"},
    }), encoding="utf-8")
    proof = {
        "schema_version": "mmm/curseforge-dependency-proof-v1",
        "mode": "dependency", "minecraft_version": "26.2",
        "loader": "fabric", "local_integrity_passed": True,
        "license_policy_confirmed": True, "redistribution_authorized": False,
        "jar_path": str(jar), "sha256": hashlib.sha256(jar.read_bytes()).hexdigest(),
        "fabric_mod_id": "alienmod", "project_id": 456, "file_id": 123,
    }
    return root, proof


def test_installs_external_dependency_idempotently(tmp_path):
    root, proof = _setup(tmp_path)
    receipt = install_verified_fabric_dependency(
        root, proof, minecraft_version="26.2")
    assert receipt["status"] == "INSTALLED"
    assert receipt["redistributed"] is False
    assert "modImplementation files(" in (root/"build.gradle").read_text()
    depends = json.loads((root/"src/main/resources/fabric.mod.json").read_text())["depends"]
    assert depends["alienmod"] == "*"
    assert install_verified_fabric_dependency(root, proof, minecraft_version="26.2")["status"] == "ALREADY_INSTALLED"


def test_rejects_wrong_version_or_forged_checksum(tmp_path):
    root, proof = _setup(tmp_path)
    with pytest.raises(ValueError, match="PROOF_INVALID"):
        install_verified_fabric_dependency(root, proof, minecraft_version="1.21.8")
    with pytest.raises(ValueError, match="SHA256_MISMATCH"):
        install_verified_fabric_dependency(root, {**proof, "sha256": "a"*64},
                                           minecraft_version="26.2")


def test_does_not_copy_binary_into_release_project(tmp_path):
    root, proof = _setup(tmp_path)
    install_verified_fabric_dependency(root, proof, minecraft_version="26.2")
    assert not (root/"libs").exists()
    assert len(list(root.rglob("*.jar"))) == 0
