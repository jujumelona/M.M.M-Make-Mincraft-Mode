"""CurseForge candidates become dependencies only after byte-level proof."""
import hashlib
import json
from zipfile import ZipFile

import pytest

from minecraft_mod_ai.curseforge_dependency_adoption import (
    CurseForgeDependencyError,
    verify_curseforge_dependency,
)


def _fixture(tmp_path, *, version="26.2", loader="Fabric"):
    jar = tmp_path / "alien.jar"
    with ZipFile(jar, "w") as z:
        z.writestr("fabric.mod.json", json.dumps({
            "schemaVersion": 1, "id": "alienmod",
            "version": "1.0.0", "depends": {"minecraft": version}
        }))
        z.writestr("mod/Alien.class", b"test-bytecode")
    info = {
        "id": 123, "modId": 456, "isAvailable": True, "fileName": "alien.jar",
        "gameVersions": [version, loader],
        "hashes": [{"algo": 1, "value": hashlib.sha1(
            jar.read_bytes(), usedforsecurity=False).hexdigest()}],
    }
    return jar, info


def _verify(jar, meta, **kwargs):
    return verify_curseforge_dependency(
        jar, project_id=456, file_id=123, file_metadata=meta,
        minecraft_version=kwargs.pop("minecraft_version", "26.2"),
        loader=kwargs.pop("loader", "fabric"),
        license_policy_confirmed=kwargs.pop("license_policy_confirmed", True),
    )


def test_byte_verified_curseforge_jar_produces_dependency_not_source_proof(tmp_path):
    jar, meta = _fixture(tmp_path)
    evidence = _verify(jar, meta)
    assert evidence["mode"] == "dependency"
    assert evidence["local_integrity_passed"]
    assert not evidence["runtime_verified"]
    assert not evidence["source_reuse"]
    assert not evidence["redistribution_authorized"]
    assert evidence["file_id"] == 123


def test_wrong_version_loader_and_hash_are_rejected(tmp_path):
    jar, meta = _fixture(tmp_path)
    with pytest.raises(CurseForgeDependencyError, match="VERSION_LOADER"):
        _verify(jar, meta, minecraft_version="1.20.1")
    with pytest.raises(CurseForgeDependencyError, match="HASH_MISMATCH"):
        _verify(jar, {**meta, "hashes": [{"algo": 1, "value": "a"*40}]})
    with pytest.raises(CurseForgeDependencyError, match="PROVIDER_FILE_MISMATCH"):
        _verify(jar, {**meta, "id": 999})


def test_unknown_license_is_never_auto_adopted(tmp_path):
    jar, meta = _fixture(tmp_path)
    with pytest.raises(CurseForgeDependencyError, match="LICENSE"):
        _verify(jar, meta, license_policy_confirmed=False)


def test_non_mod_jar_cannot_be_reused(tmp_path):
    jar = tmp_path / "not-a-mod.jar"
    with ZipFile(jar, "w") as z:
        z.writestr("pack.mcmeta", "{}")
    meta = {"id": 123, "modId": 456, "isAvailable": True,
            "fileName": "not-a-mod.jar",
            "gameVersions": ["26.2", "Fabric"],
            "hashes": [{"algo": 1, "value": hashlib.sha1(
                jar.read_bytes(), usedforsecurity=False).hexdigest()}]}
    with pytest.raises(CurseForgeDependencyError, match="NOT_FABRIC_MOD"):
        _verify(jar, meta)
