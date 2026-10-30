"""Every original-occurrence producer belongs to compiler admission, not inference."""
from __future__ import annotations

import json
import pytest

from minecraft_mod_ai.host_source_publication import source_publication_transaction
from minecraft_mod_ai.incremental_java_pipeline import candidate_data, load_candidate

_SYMBOLS = {
    "class": {"kind": "CLASS", "owner": "example.Transaction",
              "name": "Transaction", "access": 1},
    "purchase": {"kind": "METHOD", "owner": "example.Transaction",
                 "name": "applyPurchase", "parameters": ["java.lang.String", "int"],
                 "return_type": "boolean", "access": 1},
}
_ARGS = {"source_site_ref": "purchase", "behavior_id": "test_behavior",
         "package": "example.generated", "at": "HEAD", "side": "COMMON",
         "java_version": 25}


def _workspace(tmp_path):
    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    original = '{"schemaVersion":1,"id":"custommod","mixins":["existing.json"]}'
    manifest.write_text(original)
    return manifest, original


def test_rejected_compiler_or_provenance_restores_manifest_mixin_and_receipt(tmp_path):
    from minecraft_mod_ai.source_owned_custom_event import plan_verified_custom_event
    manifest, original = _workspace(tmp_path)
    plan = plan_verified_custom_event(symbols=_SYMBOLS, mod_id="custommod", **_ARGS)
    records = [{"kind": "custom", "args": dict(_ARGS)}]
    with pytest.raises(ValueError, match="javac symbol failure"):
        with source_publication_transaction(
            project_root=tmp_path, symbols=_SYMBOLS, records=records):
            assert manifest.read_text() != original
            assert all((tmp_path / name).is_file() for name in plan["files"])
            raise ValueError("javac symbol failure")
    assert manifest.read_text() == original
    assert all(not (tmp_path / name).exists() for name in plan["files"])
    assert not list((tmp_path / ".mmm/source-producers").glob("*.json"))


def test_accepted_compiler_and_proof_keep_installed_mixin(tmp_path):
    from minecraft_mod_ai.source_owned_custom_event import plan_verified_custom_event
    manifest, original = _workspace(tmp_path)
    plan = plan_verified_custom_event(symbols=_SYMBOLS, mod_id="custommod", **_ARGS)
    records = [{"kind": "custom", "args": dict(_ARGS)}]
    with source_publication_transaction(
        project_root=tmp_path, symbols=_SYMBOLS, records=records):
        assert manifest.read_text() != original
    assert all((tmp_path / name).is_file() for name in plan["files"])
    assert list((tmp_path / ".mmm/source-producers").glob("*.json"))


def test_source_owned_candidate_roundtrip_reinstalls_exact_original_source_plan(tmp_path):
    manifest, original = _workspace(tmp_path)
    state = {"strategy": "host_source_owned", "native_mutation_receipts": [],
             "source_publications": [{"kind": "custom", "args": dict(_ARGS)}]}
    pending = candidate_data("package example; class Bridge {}", "compilation failed", state)
    restored = {}
    assert load_candidate(pending, restored, True) == pending
    assert restored["source_publications"] == state["source_publications"]
    with pytest.raises(RuntimeError, match="failed after replay"):
        with source_publication_transaction(
            project_root=tmp_path, symbols=_SYMBOLS,
            records=restored["source_publications"]):
            assert manifest.read_text() != original
            raise RuntimeError("failed after replay")
    assert manifest.read_text() == original
    legacy = {key: value for key, value in pending.items()
              if key != "source_publications_json"}
    assert load_candidate(legacy, {}, True) == {}


def test_source_publication_invalid_plan_has_no_writes(tmp_path):
    manifest, original = _workspace(tmp_path)
    with pytest.raises(ValueError, match="HOST_SOURCE_PUBLICATION_CUSTOM_INVALID"):
        with source_publication_transaction(
            project_root=tmp_path, symbols=_SYMBOLS,
            records=[{"kind": "custom", "args": {**_ARGS, "symbols": {}}}]):
            pytest.fail("invalid plan reached compilation")
    assert manifest.read_text() == original


def test_ui_producer_rollback_restores_original_manifest_and_all_files(tmp_path):
    from test_source_owned_ui_event import _plan, SYMBOLS
    planned = _plan()
    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    original = json.dumps({"id": "allocation", "entrypoints": {
        "main": ["example.Preexisting"], "client": []}})
    manifest.write_text(original)
    with pytest.raises(ValueError, match="semantic condition rejected"):
        with source_publication_transaction(
            project_root=tmp_path, symbols=SYMBOLS,
            records=[{"kind": "ui", "plan": planned}]):
            assert all((tmp_path / name).exists() for name in planned["files"])
            raise ValueError("semantic condition rejected")
    assert manifest.read_text() == original
    assert all(not (tmp_path / name).exists() for name in planned["files"])
    assert not list((tmp_path / ".mmm/typed-ui-producers").glob("*.json"))
