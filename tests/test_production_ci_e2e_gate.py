"""Fail-closed regression checks for the real production CI admission gate."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_partial_ci_gate_requires_actual_gradle_and_gametest_e2e() -> None:
    workflow = (ROOT / ".github/workflows/partial-ci.yml").read_text(
        encoding="utf-8"
    )
    assert "production_e2e:" in workflow
    assert "Real production Gradle + GameTest E2E" in workflow
    assert "tests/test_deterministic_debug_full_pipeline_e2e.py" in workflow
    assert "needs: [impact, python, tests, production_e2e]" in workflow
    assert 'if [ "$PRODUCTION_E2E_REQUIRED" = true ]; then' in workflow
    assert 'test "$PRODUCTION_E2E" = success || conclusion=failure' in workflow
    assert 'test "$PRODUCTION_E2E" = skipped || conclusion=failure' in workflow


def test_partial_ci_e2e_selection_includes_colab_and_real_verifier() -> None:
    workflow = (ROOT / ".github/workflows/partial-ci.yml").read_text(
        encoding="utf-8"
    )
    for required_path in (
        "M.M.M_Make_Mincraft_Mode_Colab.ipynb",
        "minecraft_mod_ai/runner",
        "minecraft_mod_ai/complete_orchestrator",
        "minecraft_mod_ai/colab_",
        "minecraft_mod_ai/resource_image_pipeline",
        "minecraft_mod_ai/resource_alpha_segmentation",
        "tests/test_gametest_eula_consent_forwarding.py",
        "tests/test_deterministic_debug_full_pipeline_e2e.py",
    ):
        assert required_path in workflow
    assert 'handle.write("run_e2e="' in workflow
    assert "run_e2e: ${{ steps.select.outputs.run_e2e }}" in workflow
    assert "if: ${{ needs.impact.outputs.run_e2e == 'true' }}" in workflow
