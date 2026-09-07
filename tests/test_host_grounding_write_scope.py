from __future__ import annotations

from minecraft_mod_ai import host_grounding


def test_coder_grounding_publishes_source_only_outer_write_scope(monkeypatch) -> None:
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
    ]
    assert scope["allowed_files"] == []
    assert scope["protected_prefixes"] == [".minecraft_ai"]
    assert "README.md" in scope["examples_rejected"]
    assert "build.gradle" in scope["examples_rejected"]
    assert "task capsule writable_paths" in scope["policy"]


def test_custom_module_path_policy_keeps_build_and_host_state_read_only() -> None:
    for path in (
        "src/main/java/example/Feature.java",
        "src/main/resources/fabric.mod.json",
        "src/test/java/example/FeatureTest.java",
        "src/gametest/resources/test.snbt",
    ):
        assert host_grounding.custom_module_path_allowed(path) is True
        assert host_grounding.custom_module_path_protected(path) is False

    for path in (
        ".minecraft_ai/research/ledger.json",
        ".minecraft_ai/context-observations/page.json",
        ".minecraft_ai/generated/receipt.json",
    ):
        assert host_grounding.custom_module_path_protected(path) is True
        assert host_grounding.custom_module_path_allowed(path) is False

    for path in (
        "build.gradle",
        "gradle.properties",
        "settings.gradle",
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
