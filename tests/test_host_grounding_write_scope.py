from __future__ import annotations

from minecraft_mod_ai import host_grounding


def test_coder_grounding_publishes_exact_patch_write_scope(monkeypatch) -> None:
    monkeypatch.setattr(
        host_grounding,
        "skills_for_model_role",
        lambda _role: (
            "ground-production-with-live-evidence",
            "patch-existing-project",
            "generate-fabric-core",
        ),
    )
    monkeypatch.setattr(
        host_grounding,
        "reviewed_mcp_servers_for_model_role",
        lambda _stage, _role: (),
    )

    grounding = host_grounding.build_coder_grounding(
        module_kind="custom_java",
        source_observation_receipt={
            "schema_version": "mmm/source-observation-receipt-v1",
            "project_sha256": "sha256:project",
        },
        research_context={"schema_version": "mmm/research-context-v1"},
        minecraft_version="1.21.1",
        loader="fabric",
        mappings="yarn",
    )

    scope = grounding["write_scope"]
    assert scope == host_grounding.custom_module_write_scope()
    assert scope["allowed_prefixes"] == [
        "src/main/java/",
        "src/main/resources/",
        "src/test/java/",
        "src/gametest/",
        ".minecraft_ai/",
    ]
    assert scope["allowed_files"] == [
        "build.gradle",
        "gradle.properties",
        "settings.gradle",
    ]
    assert scope["protected_prefixes"] == [
        ".minecraft_ai/research",
        ".minecraft_ai/context-observations",
    ]
    assert "README.md" in scope["examples_rejected"]


def test_custom_module_path_policy_has_one_fail_closed_authority() -> None:
    for path in (
        "src/main/java/example/Feature.java",
        "src/main/resources/fabric.mod.json",
        "src/test/java/example/FeatureTest.java",
        "src/gametest/resources/test.snbt",
        ".minecraft_ai/generated/receipt.json",
        "build.gradle",
        "gradle.properties",
        "settings.gradle",
    ):
        assert host_grounding.custom_module_path_allowed(path) is True
        assert host_grounding.custom_module_path_protected(path) is False

    for path in (
        ".minecraft_ai/research/ledger.json",
        ".minecraft_ai/context-observations/page.json",
    ):
        assert host_grounding.custom_module_path_protected(path) is True
        assert host_grounding.custom_module_path_allowed(path) is False

    for path in (
        "README.md",
        "LICENSE",
        "docs/design.md",
        "gradlew",
        "../README.md",
        "src/main/java/../../README.md",
        ".minecraft_ai/generated/../../../README.md",
        "/tmp/Feature.java",
        r"src\main\java\..\..\README.md",
    ):
        assert host_grounding.custom_module_path_allowed(path) is False
