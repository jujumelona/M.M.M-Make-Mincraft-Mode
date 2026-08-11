import json
import runpy
from pathlib import Path

import pytest


def test_notebooks_are_registry_driven_and_have_no_silent_fallback() -> None:
    forbidden = (
        "Qwen/Qwen3.5-9B-Instruct",
        "Qwen/Qwen3.5-4B-Instruct",
        "built-in 폴백",
        "deterministic-fallback",
        "model_name_or_path=LOCAL_MODEL_ID",
    )
    for path in (
        Path("M.M.M_Make_Mincraft_Mode_Colab.ipynb"),
        Path("Minecraft_Multimodal_Mod_AI_Architecture_v6.ipynb"),
    ):
        raw = path.read_text(encoding="utf-8")
        json.loads(raw)
        assert "MODEL_PROFILE" in raw
        assert "config/model_registry.yaml" in raw
        assert all(token not in raw for token in forbidden)


def test_notebook_builder_matches_checked_in_notebooks() -> None:
    module = runpy.run_path("tools/build_colab_notebook.py")
    rendered = module["serialize_notebook"](module["build_notebook"]())
    assert Path("M.M.M_Make_Mincraft_Mode_Colab.ipynb").read_text(encoding="utf-8") == rendered
    assert Path("Minecraft_Multimodal_Mod_AI_Architecture_v6.ipynb").read_text(encoding="utf-8") == rendered


def test_existing_zip_upload_is_explicit_and_opt_in() -> None:
    module = runpy.run_path("tools/build_colab_notebook.py")
    notebook = module["build_notebook"]()
    cells = {cell["id"]: cell["source"] for cell in notebook.cells}

    assert "PATCH_EXISTING" in cells["configuration"]
    assert "if PATCH_EXISTING:" in cells["existing-input"]
    assert "colab_files.upload()" in cells["existing-input"]
    assert "len(uploaded) != 1" in cells["existing-input"]
    assert 'suffix.lower() != ".zip"' in cells["existing-input"]
    assert "inspect_existing_project_archive(EXISTING_INPUT)" in cells["existing-input"]
    assert "existing_report.has_sources" in cells["existing-input"]
    assert "existing_report.has_gradle_project" in cells["existing-input"]
    assert "existing_input=EXISTING_INPUT" in cells["plan"]
    assert "approval_hash" not in "\n".join(cells.values())
    assert "complete_proposal" not in "\n".join(cells.values())


def test_local_colab_profiles_require_verified_qwen_fast_kernels() -> None:
    module = runpy.run_path("tools/build_colab_notebook.py")
    notebook = module["build_notebook"]()
    cells = {cell["id"]: cell["source"] for cell in notebook.cells}
    setup_source = Path("tools/colab_runtime_setup.py").read_text(encoding="utf-8")

    assert 'MODEL_PROFILE = "Qwen3.5-9B_6GB"' in cells["configuration"]
    assert '["Qwen3.5-9B_6GB", "Gemma4-12B_7GB", "Gemma4-26B_14GB", "Qwen3.6-35B_23GB", "Qwen3.6-27B_18GB", "Qwen3.6-27B_14GB", "mini_mod", "fast_test"]' in cells["configuration"]
    assert '"tools" / "colab_runtime_setup.py"' in cells["setup"]
    assert "spec_from_file_location" in cells["setup"]
    assert "USED_COMMIT[:12]" in cells["setup"]
    assert '"+main:refs/remotes/origin/main"' in cells["setup"]
    assert '"merge",' in cells["setup"]
    assert '"pull",' not in cells["setup"]
    assert "refs/remotes/origin/main" in cells["setup"]
    assert '"--untracked-files=no"' in cells["setup"]
    assert "setup_colab_runtime(" in cells["setup"]
    assert "engine_module_file=engine_module_file" in cells["setup"]
    assert 'SETUP_STATE["receipt"]' in cells["setup"]
    assert cells["setup"].index('print("GitHub commit:"') < cells["setup"].index(
        "setup_colab_runtime("
    )
    assert "flash-linear-attention" not in cells["setup"]

    assert "LOCAL_PROFILES = frozenset(" in setup_source
    assert '"Qwen3.5-9B_6GB"' in setup_source
    assert 'REMOTE_PROJECT_INSTALL_TARGET = (\n    ".[ui,rag,' in setup_source
    assert 'LOCAL_PROJECT_INSTALL_TARGET = (\n    ".[ui,local-model,rag,' in setup_source
    assert "_install_project(local_profile=profile in LOCAL_PROFILES)" in setup_source
    assert '"--no-build-isolation"' in setup_source
    assert "flash-linear-attention[cuda,conv1d]>=0.5.1,<0.6" in setup_source
    assert "from causal_conv1d import causal_conv1d_fn, causal_conv1d_update" in setup_source
    assert "chunk_gated_delta_rule" in setup_source
    assert "fused_recurrent_gated_delta_rule" in setup_source
    assert "modeling_qwen3_5" in setup_source
    assert "is_fast_path_available" in setup_source
    assert "causal_conv1d_fn(" in setup_source
    assert "causal_conv1d_update(" in setup_source
    assert "torch.cuda.synchronize()" in setup_source
    assert "use_qk_l2norm_in_kernel=True" in setup_source
    assert "initial_state=recurrent_state" in setup_source
    assert "without embedded" in setup_source
    assert "loaded from a different checkout" in setup_source
    assert "restart the Colab " in setup_source
    assert "runtime and rerun from cell 1" in setup_source
    assert "Qwen3.5 fast path: unavailable; using standard PyTorch" in setup_source


def test_notebook_checks_setup_fingerprint_and_prints_resolved_planner() -> None:
    module = runpy.run_path("tools/build_colab_notebook.py")
    notebook = module["build_notebook"]()
    cells = {cell["id"]: cell["source"] for cell in notebook.cells}

    assert "def assert_current_colab_setup" in cells["registry"]
    assert "COLAB_SETUP_MODULE.assert_setup_state(" in cells["registry"]
    assert "planner_config = registry_manager.role" in cells["registry"]
    assert "planner_config.model_id" in cells["registry"]
    assert "planner_config.quantization" in cells["registry"]
    assert "planner_config.max_context" in cells["registry"]
    assert "planner_config.max_input_tokens" in cells["registry"]
    assert "planner_config.max_new_tokens" in cells["registry"]
    assert "기획 native context:" in cells["registry"]
    assert "기획 page input:" in cells["registry"]
    assert "기획 page output:" in cells["registry"]
    assert "assert_current_colab_setup()" in cells["plan"]
    assert "assert_current_colab_setup()" in cells["build"]


def test_colab_setup_fingerprint_is_stable_and_receipt_excludes_secrets() -> None:
    setup = runpy.run_path("tools/colab_runtime_setup.py")
    fingerprint = setup["setup_request_fingerprint"]
    build_receipt = setup["_build_receipt"]
    validate_remote_url = setup["_validated_remote_url"]
    request = {
        "repo_dir": Path(".").resolve(),
        "used_commit": "a" * 40,
        "model_profile": "remote_quality",
        "save_to_google_drive": False,
        "remote_base_url": "https://user:TOPSECRET@example.test/v1?api_key=TOPSECRET",
        "remote_text_model": "planner-model",
        "remote_image_model": "image-model",
        "remote_speech_model": "speech-model",
    }

    first = fingerprint(**request)
    assert first == fingerprint(**request)
    assert first != fingerprint(**{**request, "model_profile": "t4_local"})
    receipt = build_receipt(
        **request,
        output_root="/content/mmm-output",
        setup_fingerprint=first,
        torch=None,
    )
    serialized = json.dumps(receipt, sort_keys=True)
    assert "TOPSECRET" not in serialized
    assert "api_key" not in serialized.lower()
    assert receipt["remote"]["base_url"] == "https://example.test/v1"
    assert receipt["setup_fingerprint"] == first
    assert validate_remote_url("https://example.test/v1") == "https://example.test/v1"
    with pytest.raises(ValueError, match="without embedded"):
        validate_remote_url("https://user:TOPSECRET@example.test/v1")
    with pytest.raises(ValueError, match="without embedded"):
        validate_remote_url("https://example.test/v1?api_key=TOPSECRET")


def test_remote_runtime_check_detects_environment_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = runpy.run_path("tools/colab_runtime_setup.py")
    assert_remote_environment = setup["_assert_remote_environment"]
    endpoint = "https://example.test/v1"
    for role in setup["REMOTE_TEXT_ROLES"]:
        monkeypatch.setenv(f"MMM_{role}_BASE_URL", endpoint)
        monkeypatch.setenv(f"MMM_{role}_MODEL", "planner-model")
        monkeypatch.setenv(f"MMM_{role}_API_KEY", "TOPSECRET")
    monkeypatch.setenv("MMM_IMAGE_BASE_URL", endpoint)
    monkeypatch.setenv("MMM_IMAGE_MODEL", "image-model")
    monkeypatch.setenv("MMM_IMAGE_API_KEY", "TOPSECRET")
    monkeypatch.setenv("MMM_SPEECH_BASE_URL", endpoint)
    monkeypatch.setenv("MMM_SPEECH_MODEL", "speech-model")
    monkeypatch.setenv("MMM_SPEECH_API_KEY", "TOPSECRET")
    request = {
        "remote_base_url": endpoint,
        "remote_text_model": "planner-model",
        "remote_image_model": "image-model",
        "remote_speech_model": "speech-model",
    }

    assert_remote_environment(**request)
    monkeypatch.setenv("MMM_PLANNER_BASE_URL", "https://changed.example/v1")
    with pytest.raises(RuntimeError, match="environment changed"):
        assert_remote_environment(**request)


def test_qwen_fastpath_extra_is_linux_only_and_includes_fixed_fla() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "qwen-fastpath = [" in pyproject
    assert "flash-linear-attention[cuda,conv1d]>=0.5.1,<0.6" in pyproject
    assert "sys_platform == 'linux'" in pyproject


def test_static_colab_requirements_leave_cuda_fastpath_to_runtime_preflight() -> None:
    requirements = Path("requirements-colab.txt").read_text(encoding="utf-8")

    assert "ui,local-model,rag,image,speech,production-audio,training" in requirements
    assert ".[qwen-fastpath]" not in requirements


def test_notebook_code_cells_compile_top_to_bottom() -> None:
    module = runpy.run_path("tools/build_colab_notebook.py")
    notebook = module["build_notebook"]()

    for cell in notebook.cells:
        if cell["cell_type"] == "code":
            compile(cell["source"], f"<colab:{cell['id']}>", "exec")
