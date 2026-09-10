from __future__ import annotations

import json
from pathlib import Path
import pytest

from minecraft_mod_ai.artifact_job import ArtifactJob
from minecraft_mod_ai.artifact_materializer import (
    MaterializeError,
    materialize_binary_asset,
    materialize_java_patch,
    materialize_job_output,
    materialize_json_merge,
    materialize_whole_file,
)


def test_materialize_whole_file(tmp_path: Path):
    target = tmp_path / "hello.txt"
    r1 = materialize_whole_file(target, "hello world")
    assert r1.status == "SUCCESS"
    assert r1.before_sha256 is None
    assert r1.after_sha256 is not None
    assert target.read_text(encoding="utf-8") == "hello world"

    # Overwrite
    r2 = materialize_whole_file(target, "hello modified")
    assert r2.status == "SUCCESS"
    assert r2.before_sha256 == r1.after_sha256
    assert target.read_text(encoding="utf-8") == "hello modified"


def test_materialize_java_patch(tmp_path: Path):
    target = tmp_path / "ModItems.java"
    target.write_text(
        "public class ModItems {\n"
        "    /* MMM:item_registry */\n"
        "}\n",
        encoding="utf-8",
    )

    # Patch with keep_anchor=True
    r1 = materialize_java_patch(
        target,
        "/* MMM:item_registry */",
        "public static final Item RAW_LUNITE = ...;",
        keep_anchor=True,
    )
    assert r1.status == "SUCCESS"
    content = target.read_text(encoding="utf-8")
    assert "public static final Item RAW_LUNITE = ...;" in content
    assert "/* MMM:item_registry */" in content

    # Error on missing anchor
    with pytest.raises(MaterializeError, match="ANCHOR_NOT_FOUND"):
        materialize_java_patch(target, "/* NON_EXISTENT */", "code")


def test_materialize_json_merge(tmp_path: Path):
    target = tmp_path / "en_us.json"
    r1 = materialize_json_merge(target, {"item.space.lunite": "Lunite"})
    assert r1.status == "SUCCESS"
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data == {"item.space.lunite": "Lunite"}

    # Merge another key
    r2 = materialize_json_merge(target, {"item.space.raw_lunite": "Raw Lunite"})
    assert r2.status == "SUCCESS"
    assert r2.before_sha256 == r1.after_sha256
    data2 = json.loads(target.read_text(encoding="utf-8"))
    assert data2 == {
        "item.space.lunite": "Lunite",
        "item.space.raw_lunite": "Raw Lunite",
    }


def test_materialize_binary_asset(tmp_path: Path):
    target = tmp_path / "texture.png"
    fake_png = b"\x89PNG\r\n\x1a\nfake_image_data"
    r = materialize_binary_asset(target, fake_png)
    assert r.status == "SUCCESS"
    assert target.read_bytes() == fake_png


def test_materialize_job_output_dispatch(tmp_path: Path):
    # Test java_patch dispatch
    java_file = tmp_path / "ModItems.java"
    java_file.write_text("class ModItems { /* MMM:reg */ }", encoding="utf-8")
    job = ArtifactJob(
        job_id="test.reg",
        template_id="fabric/item/register_basic",
        owner_module="raw_lunite",
        target_path=str(java_file),
        anchor="/* MMM:reg */",
    )
    receipt = materialize_job_output(job, "Item RAW_LUNITE;", base_dir=tmp_path)
    assert receipt.operation == "java_patch"
    assert "Item RAW_LUNITE;" in java_file.read_text(encoding="utf-8")


def test_materialize_whole_file_expected_sha256(tmp_path: Path):
    target = tmp_path / "guarded.txt"
    r1 = materialize_whole_file(target, "version 1")
    assert r1.status == "SUCCESS"

    # Mismatch raises
    with pytest.raises(MaterializeError, match="SHA_MISMATCH"):
        materialize_whole_file(target, "version 2", expected_sha256="bad_sha")

    # Match succeeds
    r2 = materialize_whole_file(target, "version 2", expected_sha256=r1.after_sha256)
    assert r2.status == "SUCCESS"
    assert target.read_text(encoding="utf-8") == "version 2"


def test_materialize_java_patch_ambiguous_anchor(tmp_path: Path):
    target = tmp_path / "Ambiguous.java"
    target.write_text("/* MMM:dup */\n/* MMM:dup */", encoding="utf-8")
    with pytest.raises(MaterializeError, match="ANCHOR_AMBIGUOUS"):
        materialize_java_patch(target, "/* MMM:dup */", "code")


def test_materialize_json_merge_corrupted_fails_closed(tmp_path: Path):
    target = tmp_path / "corrupted.json"
    target.write_text("{ not valid json !!!", encoding="utf-8")
    with pytest.raises(MaterializeError, match="JSON_CORRUPTED"):
        materialize_json_merge(target, {"key": "val"})


def test_materialize_json_merge_conflict_fails_closed(tmp_path: Path):
    target = tmp_path / "lang.json"
    target.write_text(json.dumps({"item.test": "Existing"}), encoding="utf-8")
    with pytest.raises(MaterializeError, match="JSON_KEY_CONFLICT"):
        materialize_json_merge(target, {"item.test": "Conflicting"})


def test_ensure_artifact_scaffolding_includes_blocks(tmp_path: Path):
    from minecraft_mod_ai.artifact_materializer import ensure_artifact_scaffolding

    ensure_artifact_scaffolding(
        tmp_path,
        mod_id="space",
        package_name="net.space",
        main_class="SpaceMod",
    )
    block_ids = tmp_path / "src" / "main" / "java" / "net" / "space" / "registry" / "ModBlockIds.java"
    assert block_ids.is_file()
    assert "/* MMM:block_keys */" in block_ids.read_text(encoding="utf-8")

    blocks = tmp_path / "src" / "main" / "java" / "net" / "space" / "registry" / "ModBlocks.java"
    assert blocks.is_file()
    assert "/* MMM:block_registry */" in blocks.read_text(encoding="utf-8")

    main_class = tmp_path / "src" / "main" / "java" / "net" / "space" / "SpaceMod.java"
    assert main_class.is_file()
    main_content = main_class.read_text(encoding="utf-8")
    assert "ModBlocks.initialize();" in main_content
    assert "ModItems.initialize();" in main_content

