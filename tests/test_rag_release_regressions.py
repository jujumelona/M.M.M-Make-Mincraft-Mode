"""Regression tests for authored release integrity."""
from __future__ import annotations

from minecraft_mod_ai.final_artifact import _authored_feature_semantic_findings


def test_authored_coverage_rejects_nonfunctional_gui_placeholder(tmp_path) -> None:
    source = (
        tmp_path / "src" / "client" / "java" / "demo"
        / "generated" / "FabricationGuiScreenRegistration.java"
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        'public class FabricationGuiScreenRegistration { '
        'String title = "Planned: Implement gameplay obligation ship assembly"; }',
        encoding="utf-8",
    )
    findings = _authored_feature_semantic_findings(tmp_path, ())
    assert any("nonfunctional requirement placeholder" in item for item in findings)


def test_authored_coverage_does_not_reject_real_gui_label(tmp_path) -> None:
    source = (
        tmp_path / "src" / "client" / "java" / "demo"
        / "generated" / "FabricationGuiScreenRegistration.java"
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        'public class FabricationGuiScreenRegistration { '
        'String title = "Fabricate ship module"; }',
        encoding="utf-8",
    )
    assert _authored_feature_semantic_findings(tmp_path, ()) == []



