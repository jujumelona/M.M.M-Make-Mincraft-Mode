"""Targeted regressions for CurseForge-origin source relevance and resource closure."""
import hashlib
import io
import zipfile

from minecraft_mod_ai.curseforge_semantic_source_review import (
    _safe_archive_files, inventory_from_archive,
)
from minecraft_mod_ai.curseforge_reuse_production import donor_from_reference


MIT = (
    b"MIT License\nPermission is hereby granted, free of charge, to any person "
    b"obtaining a copy of this software and associated documentation files "
    b"(the Software), to deal in the Software without restriction."
)


def _archive(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as zf:
        for name, raw in files.items():
            zf.writestr("donor-abcdef/" + name, raw)
    return output.getvalue()


def test_korean_requirement_uses_curseforge_search_terms_and_late_java_body():
    java = (
        "package example;\npublic class Teleport {\n"
        + ("  // filler documentation for old prefix-only ranking\n" * 150)
        + "  public void teleportAction() { System.out.println(\"teleport\"); }\n"
        + "}\n"
    ).encode()
    archive = _archive({
        "LICENSE": MIT,
        "src/main/java/example/Teleport.java": java,
        "src/main/resources/assets/example/lang/en_us.json": b'{"item.example.key":"Home"}',
    })
    files = _safe_archive_files(archive)
    assert "src/main/resources/assets/example/lang/en_us.json" in files
    inventory = inventory_from_archive("기지로 순간 이동", archive,
                                       search_terms=("teleport", "travel"))
    assert inventory["status"] == "inspected"
    row = inventory["evidence"][0]
    assert row["rank"] > 0
    assert "teleportAction" in row["snippet"]


def test_real_dependency_metadata_and_assets_are_in_donor_manifest():
    source = b"package example; public class Teleport { public void teleport() {} }"
    files = {
        "LICENSE": MIT,
        "build.gradle": b"dependencies { modImplementation 'software.bernie.geckolib:geckolib-fabric:4.6.0' }",
        "src/main/java/example/Teleport.java": source,
        "src/main/resources/assets/example/lang/en_us.json": b'{"item.example.key":"Home"}',
    }
    reference = {
        "capability": "teleport", "selected_file": "src/main/java/example/Teleport.java",
        "selected_symbol": "Teleport",
        "file_sha256": "sha256:" + hashlib.sha256(source).hexdigest(),
        "source_url": "https://github.com/example/teleport-mod",
        "pinned_sha": "a" * 40,
        "license_spdx": "MIT", "semantic_score": 90,
    }
    manifest = donor_from_reference(reference, files)
    paths = {row["path"] for row in manifest["files"]}
    assert "src/main/java/example/Teleport.java" in paths
    assert "src/main/resources/assets/example/lang/en_us.json" in paths
    assert "geckolib" in manifest["required_dependencies"]
