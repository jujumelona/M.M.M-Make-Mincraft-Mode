from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai.small_model_rag_relations import derive_relations


def test_dependency_graph_tracks_types_registry_mixins_and_gradle(tmp_path: Path) -> None:
    root = tmp_path / "project"
    java = root / "src/main/java/demo"
    mixin_java = java / "mixin"
    resources = root / "src/main/resources"
    model = resources / "assets/demo/models/item/widget.json"
    mixin_config = resources / "demo.mixins.json"
    fabric_mod = resources / "fabric.mod.json"
    version_catalog = root / "gradle/libs.versions.toml"

    mixin_java.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    version_catalog.parent.mkdir(parents=True)

    target = java / "Target.java"
    target.write_text(
        "package demo; public final class Target {}\n",
        encoding="utf-8",
    )
    registry_user = java / "RegistryUser.java"
    registry_user.write_text(
        "package demo;\n"
        "final class RegistryUser {\n"
        "  Target target;\n"
        '  Object id = Identifier.of("demo", "item/widget");\n'
        "}\n",
        encoding="utf-8",
    )
    target_mixin = mixin_java / "TargetMixin.java"
    target_mixin.write_text(
        "package demo.mixin;\n"
        "import demo.Target;\n"
        "@Mixin(Target.class) final class TargetMixin {}\n",
        encoding="utf-8",
    )
    model.write_text('{"parent":"minecraft:item/generated"}\n', encoding="utf-8")
    mixin_config.write_text(
        json.dumps({"package": "demo.mixin", "mixins": ["TargetMixin"]}),
        encoding="utf-8",
    )
    fabric_mod.write_text(
        json.dumps({"schemaVersion": 1, "id": "demo", "mixins": ["demo.mixins.json"]}),
        encoding="utf-8",
    )
    (root / "gradle.properties").write_text("org.gradle.jvmargs=-Xmx1G\n", encoding="utf-8")
    settings = root / "settings.gradle"
    settings.write_text('rootProject.name = "demo"\n', encoding="utf-8")
    build = root / "build.gradle"
    build.write_text("dependencies { implementation libs.fabric.loader }\n", encoding="utf-8")
    version_catalog.write_text("[versions]\nfabric = \"1.0\"\n", encoding="utf-8")

    edges = derive_relations([root])
    triples = {(row["source"], row["target"], row["kind"]) for row in edges}

    assert (str(registry_user.resolve()), str(target.resolve()), "java_type") in triples
    assert (str(registry_user.resolve()), str(model.resolve()), "registry_ref") in triples
    assert (str(target_mixin.resolve()), str(target.resolve()), "mixin_target") in triples
    assert (str(mixin_config.resolve()), str(target_mixin.resolve()), "mixin_class") in triples
    assert (str(fabric_mod.resolve()), str(mixin_config.resolve()), "fabric_mixin_config") in triples
    assert (str(build.resolve()), str((root / "gradle.properties").resolve()), "gradle_properties") in triples
    assert (str(build.resolve()), str(settings.resolve()), "gradle_settings") in triples
    assert (str(build.resolve()), str(version_catalog.resolve()), "gradle_version_catalog") in triples
