"""Regression for production-logs (3)(5): key-only Fabric artifacts and Java JSON trailers.

Exercises graph receipt -> startup wiring, not just the startup helper, and
the exact decoded payload shape from the failed production correction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.artifact_graph_executor import _wire_validated_java_startup
from minecraft_mod_ai.construction_source_repair import assemble_java_implementation
from minecraft_mod_ai.host_fabric_initializer_wiring import wire_completed_fabric_java


def _write(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


@pytest.mark.parametrize("key_template", [
    "fabric/block/key",
    "fabric/item/key",
    "fabric/item/key_identifier",
    "fabric/item/key_resource_location",
])
def test_graph_avoids_bootstrapping_registry_key_definition(tmp_path, key_template):
    _write(tmp_path, "src/main/resources/fabric.mod.json",
           json.dumps({"id": "test", "entrypoints": {"main": []}}))
    name = "ModBlockIds" if key_template.startswith("fabric/block/") else "ModItemIds"
    key_rel = f"src/main/java/proof/registry/{name}.java"
    reg_rel = "src/main/java/proof/registry/ActualRegistry.java"
    key = _write(tmp_path, key_rel, f"""package proof.registry;
public final class {name} {{
    public static final String ORE_KEY = "ore";
}}
""")
    registry = _write(tmp_path, reg_rel, """package proof.registry;
public final class ActualRegistry {
    public static void initialize() {}
}
""")
    jobs = [
        SimpleNamespace(canonical_leaf="minecraft/block/registry",
                        template_id=key_template),
        SimpleNamespace(canonical_leaf="minecraft/block/registry",
                        template_id="fabric/block/register_basic"),
    ]
    receipts = [
        {"status": "PASS", "materialization": {
            "path": str(path),
            "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }}
        for path in (key, registry)
    ]
    _wire_validated_java_startup(
        jobs, receipts, context={"project_root": str(tmp_path)}, base_dir=tmp_path,
    )
    metadata = json.loads(
        (tmp_path / "src/main/resources/fabric.mod.json").read_text(encoding="utf-8")
    )
    bootstraps = [entry for entry in metadata["entrypoints"]["main"]
                  if "MmmRegistryBootstrap" in entry]
    assert len(bootstraps) == 1
    bootstrap = tmp_path / "src/main/java" / (
        bootstraps[0].replace(".", "/") + ".java"
    )
    source = bootstrap.read_text(encoding="utf-8")
    assert "proof.registry.ActualRegistry.initialize();" in source
    assert f"{name}.initialize()" not in source


def test_unknown_registry_template_cannot_bypass_missing_initializer(tmp_path):
    _write(tmp_path, "src/main/resources/fabric.mod.json",
           json.dumps({"id": "test", "entrypoints": {"main": []}}))
    relative = "src/main/java/proof/registry/ModBlockIds.java"
    _write(tmp_path, relative, """package proof.registry;
public final class ModBlockIds {
    public static final String KEY = "ore";
}
""")
    with pytest.raises(ValueError, match="FABRIC_INIT_REGISTRY_ENTRYPOINT_MISSING"):
        wire_completed_fabric_java(
            tmp_path, [(relative, "minecraft/block/registry", "unknown/registry")]
        )


_FRAME = {
    "kind": "method",
    "prefix": "private boolean check(Object value) {\n",
    "suffix": "\n}",
}


def test_decoded_json_closer_cannot_be_embedded_in_java_member():
    # The JSON is syntactically valid. Its *string value* contains the stray
    # JSON closer, exactly as in the failing llama.cpp production response.
    response = json.dumps({"implementation": 'return value != null;"}'})
    assert assemble_java_implementation(response, _FRAME) == (
        _FRAME["prefix"] + "return value != null;" + _FRAME["suffix"]
    )


def test_java_string_with_brace_is_not_truncated():
    response = json.dumps({"implementation": 'return value != null && "}".length() == 1;'})
    result = assemble_java_implementation(response, _FRAME)
    assert result == (_FRAME["prefix"] +
                      'return value != null && "}".length() == 1;' +
                      _FRAME["suffix"])


def test_json_trailer_repair_cannot_hide_invalid_java():
    response = json.dumps({"implementation": 'return value == ;"}'})
    with pytest.raises(ValueError) as exc:
        assemble_java_implementation(response, _FRAME)
    assert exc.value.failed_response == response


def test_trailer_recovery_still_enforces_state_access_contract():
    frame = {**_FRAME, "host_state_access": [{
        "owner": "proof.AuthoredStateModel",
        "methods": [{"name": "readState_balance",
                     "signature": "public static int readState_balance();"}],
    }]}
    response = json.dumps({"implementation":
        'return AuthoredStateModel.getState("missing") != null;"}'})
    with pytest.raises(ValueError, match="JAVA_HOST_RAW_STATE_ACCESS") as exc:
        assemble_java_implementation(response, frame)
    assert exc.value.failed_response == response
    assert exc.value.failed_source == (
        _FRAME["prefix"] +
        'return AuthoredStateModel.getState("missing") != null;' +
        _FRAME["suffix"]
    )
