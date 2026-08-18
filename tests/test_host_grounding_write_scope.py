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
