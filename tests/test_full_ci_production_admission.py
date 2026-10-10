from __future__ import annotations

"""Prevent a green CI Gate from bypassing executable production evidence."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / ".github" / "workflows" / "main-ci.yml"
E2E = ROOT / ".github" / "workflows" / "deterministic-debug-full-e2e.yml"


def _workflow(path: Path) -> dict:
    # BaseLoader preserves the YAML 1.1 "on" key as text.
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(value, dict)
    return value


def test_full_ci_triggers_on_the_complete_production_change_surface():
    workflow = _workflow(MAIN)
    triggers = workflow["on"]["push"]
    assert triggers["branches"] == ["main"]
    changed = set(triggers["paths"])
    assert {
        "minecraft_mod_ai/**",
        "tests/**",
        "tools/**",
        "integrations/**",
        "plugins/**",
        "skills/**",
        "schemas/**",
        ".github/**",
        "M.M.M_Make_Mincraft_Mode_Colab.ipynb",
        "pyproject.toml",
        "requirements-colab.txt",
        "download_resources.py",
    } <= changed
    # Rapid follow-up patches must not erase the previous SHA's failed logs.
    concurrency = workflow["concurrency"]
    assert concurrency["cancel-in-progress"] == "false"
    assert "github.sha" in concurrency["group"]


def test_real_production_e2e_is_an_unskippable_main_ci_dependency():
    workflow = _workflow(MAIN)
    jobs = workflow["jobs"]
    proof = jobs["deterministic-production"]
    assert proof["uses"] == "./.github/workflows/deterministic-debug-full-e2e.yml"
    assert "if" not in proof and "continue-on-error" not in proof
    gate = jobs["ci-gate"]
    needs = gate["needs"]
    assert {"audit", "tests", "python313", "model-realistic-replay", "deterministic-production"} <= set(needs)
    assert gate["if"] == "${{ always() }}"
    assert "continue-on-error" not in gate
    step = gate["steps"][0]
    assert step["env"]["PRODUCTION_PROOF"] == "${{ needs.deterministic-production.result }}"
    assert 'test "$PRODUCTION_PROOF" = success' in step["run"]


@pytest.mark.parametrize("rejected", ["skipped", "cancelled", "failure", "pending", ""])
def test_ci_gate_rejects_all_non_success_production_results(rejected):
    gate = _workflow(MAIN)["jobs"]["ci-gate"]["steps"][0]
    env = {
        **os.environ,
        "AUDIT": "success",
        "TESTS": "success",
        "PY313": "success",
        "MODEL_REPLAY": "success",
        "PRODUCTION_PROOF": rejected,
    }
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", gate["run"]],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0, rejected


def test_real_e2e_performs_actual_build_and_preserves_failure_artifacts():
    workflow = _workflow(E2E)
    assert "workflow_call" in workflow["on"]
    job = workflow["jobs"]["deterministic-debug-full-e2e"]
    assert "if" not in job and "continue-on-error" not in job
    setup_java = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/setup-java@"))
    assert setup_java["with"]["java-version"] == "21"
    steps = job["steps"]
    execution = next(step for step in steps if "Run production Gradle" in step.get("name", ""))
    assert "continue-on-error" not in execution
    command = execution["run"]
    assert "tools/pytest_diagnostics.py" in command
    assert "--log" in command and "--junit" in command
    assert "tests/test_gametest_eula_consent_forwarding.py" in command
    assert "tests/test_deterministic_debug_full_pipeline_e2e.py" in command
    assert "|| true" not in command and "--ignore" not in command
    artifact = next(step for step in steps if "actions/upload-artifact@" in step.get("uses", ""))
    assert artifact["if"] == "always()"
    assert artifact["with"]["if-no-files-found"] == "error"
