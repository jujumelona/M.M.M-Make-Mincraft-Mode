from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_test_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """Keep unit tests deterministic while production caches remain durable."""

    monkeypatch.setenv("MMM_ECOSYSTEM_DISCOVERY", "off")
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path / "planner-checkpoints"))
    monkeypatch.setenv("MMM_PLANNER_TRACE_DIR", str(tmp_path / "planner-traces"))
    monkeypatch.setenv(
        "MMM_RESEARCH_CHECKPOINT_ROOT",
        str(tmp_path / "research-checkpoints"),
    )
    monkeypatch.setenv(
        "MMM_RESEARCH_DOCUMENT_DIR",
        str(tmp_path / "research-evidence"),
    )

    # Technology-radar unit tests exercise capability/gate semantics directly.
    # Supply an explicit executable fixture target there instead of depending on
    # the removed production-wide historical platform default.
    if request.module.__name__ == "test_technology_radar":
        from minecraft_mod_ai.platform_catalog import adapter_for_target

        adapter = adapter_for_target("1.20.1", "fabric")
        target = {
            "edition": adapter.edition,
            "minecraft_version": adapter.minecraft_version,
            "loader": adapter.loader,
            "mappings": adapter.yarn_mappings,
            "java_version": adapter.java_version,
            "fabric_loader": adapter.fabric_loader,
            "fabric_api": adapter.fabric_api,
        }
        original = request.module.build_technology_radar

        def build_with_explicit_test_target(*args, **kwargs):
            kwargs.setdefault("target", target)
            return original(*args, **kwargs)

        monkeypatch.setattr(
            request.module,
            "build_technology_radar",
            build_with_explicit_test_target,
        )
