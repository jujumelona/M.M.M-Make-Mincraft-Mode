from __future__ import annotations

"""REG-034: a successful source build cannot excuse a missing packaged resource."""

import json
import zipfile

from minecraft_mod_ai.spec import ModSpec
from minecraft_mod_ai.toolchain_contract import fabric_dependency_predicates
from minecraft_mod_ai.validator import validate_jar


def test_packaged_jar_missing_required_runtime_resource_fails(
    tmp_path, synthetic_platform_lock
):
    spec = ModSpec(
        mod_id="pack_probe",
        mod_name="Package Probe",
        package_name="example.pack",
        version="1.0.0",
        summary="Package-content regression fixture.",
        contents=(),
        platform=synthetic_platform_lock,
    )
    metadata = {
        "schemaVersion": 1,
        "id": spec.mod_id,
        "version": spec.version,
        "environment": "*",
        "depends": fabric_dependency_predicates(spec.platform),
        "entrypoints": {"main": ["example.pack.PackProbeMod"]},
    }
    jar_path = tmp_path / "pack-probe.jar"
    with zipfile.ZipFile(jar_path, "w") as archive:
        archive.writestr("fabric.mod.json", json.dumps(metadata, sort_keys=True))
        archive.writestr("example/pack/PackProbeMod.class", b"\xca\xfe\xba\xbe")
        archive.writestr("assets/pack_probe/lang/en_us.json", "{}")
        # Deliberately omit assets/pack_probe/lang/ko_kr.json.

    report = validate_jar(jar_path, spec)
    assert report.status == "FAIL"
    missing = [finding for finding in report.findings
               if finding.code == "JAR_RESOURCE_MISSING"]
    assert any(finding.path == "assets/pack_probe/lang/ko_kr.json"
               for finding in missing)
