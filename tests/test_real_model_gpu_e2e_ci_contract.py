from __future__ import annotations

"""Guard the real GPU execution path against mock or CPU-only green CI."""

import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GPU_WORKFLOW = ROOT / ".github/workflows/real-model-colab-e2e.yml"
MAIN_WORKFLOW = ROOT / ".github/workflows/main-ci.yml"


def _workflow(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _job():
    return _workflow(GPU_WORKFLOW)["jobs"]["colab-real-model-e2e"]


def _script(step: dict) -> str:
    command = step["run"].splitlines()
    starts = [index for index, line in enumerate(command) if "python - <<'PY'" in line]
    assert len(starts) == 1
    start = starts[0]
    end = command.index("PY", start + 1)
    script = "\n".join(command[start + 1:end]) + "\n"
    ast.parse(script)
    return script


def test_real_gpu_workflow_is_reusable_and_requires_dedicated_t4_runner():
    workflow = _workflow(GPU_WORKFLOW)
    assert "workflow_call" in workflow["on"]
    assert workflow["on"]["push"]["branches"] == ["main"]
    job = _job()
    assert job["runs-on"] == ["self-hosted", "linux", "x64", "t4"]
    assert "if" not in job and "continue-on-error" not in job
    assert workflow["concurrency"]["cancel-in-progress"] == "false"
    assert "github.sha" in workflow["concurrency"]["group"]


def test_hardware_preflight_rejects_mock_and_non_t4_profiles():
    job = _job()
    preflight = next(step for step in job["steps"] if "Verify real local foundation" in step.get("name", ""))
    source = _script(preflight)
    assert "ModelRegistry().load_profile" in source
    assert '("planner", "researcher", "coder", "coder_safe")' in source
    assert 'adapter != "llama_cpp"' in source
    assert 'model_id.startswith("mock/")' in source
    assert "nvidia-smi" in source
    assert 'if not any("T4" in line for line in devices)' in source
    assert "GPU_PREFLIGHT.json" in source


def test_gpu_e2e_proves_model_work_and_native_cuda_residency():
    job = _job()
    step = next(step for step in job["steps"] if "Run normal planner plus DebugToken" in step.get("name", ""))
    command = step["run"]
    assert "set -o pipefail" in command
    assert "--query-compute-apps=pid,process_name,used_gpu_memory" in command
    assert "gpu-process-samples.log" in command
    assert "normal-and-debug-e2e.log" in command
    assert 'if not proposal.modules:' in command
    assert 'jar_validation.get("status") != "PASS"' in command
    assert "GPU_E2E_INFERENCE_OFFLOAD_NOT_PROVEN" in command
    assert "llama-server" in command
    assert ">= 512" in command
    assert "run_gametest=True" in command
    assert "successful_build_commands" in command
    assert "gametest_report" in command
    assert "|| true" not in command and "continue-on-error" not in step


def test_gpu_evidence_must_be_uploaded_even_when_e2e_fails():
    artifact = next(
        step for step in _job()["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert artifact["if"] == "always()"
    assert artifact["with"]["if-no-files-found"] == "error"
    paths = artifact["with"]["path"]
    for expected in (
        "GPU_PREFLIGHT.json", "gpu-process-samples.log",
        "normal-and-debug-e2e.log", "REAL_MODEL_COLAB_E2E.json",
        "DEBUG_TOKEN_REAL_BUILD_E2E.json",
    ):
        # The debug receipt is under the included debug-fixture subtree.
        if expected == "DEBUG_TOKEN_REAL_BUILD_E2E.json":
            assert "debug-fixture/**" in paths
        else:
            assert expected in paths


def test_real_gpu_e2e_remains_standalone_without_blocking_generic_ci():
    gpu = _workflow(GPU_WORKFLOW)
    # A self-hosted T4 is not a generally available CI prerequisite.
    assert "workflow_dispatch" in gpu["on"]
    assert "push" in gpu["on"]
    assert "workflow_call" in gpu["on"]
    assert _job()["runs-on"] == ["self-hosted", "linux", "x64", "t4"]

    generic = _workflow(MAIN_WORKFLOW)
    jobs = generic["jobs"]
    gate = jobs["ci-gate"]
    assert "real-t4-model-production" not in jobs
    assert "real-t4-model-production" not in gate["needs"]
    step = gate["steps"][0]
    assert "REAL_T4_E2E" not in step["env"]
    assert "REAL_T4_E2E" not in step["run"]
    # Keep reproducible CPU-side verification mandatory.
    assert jobs["deterministic-production"]["uses"] == (
        "./.github/workflows/deterministic-debug-full-e2e.yml"
    )
    assert "deterministic-production" in gate["needs"]
