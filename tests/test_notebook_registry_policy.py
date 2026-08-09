import json
import runpy
from pathlib import Path


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

    assert "PATCH_EXISTING = False" in cells["configuration"]
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

    assert 'MODEL_PROFILE = "t4_quality"' in cells["configuration"]
    assert '["t4_quality", "t4_local", "remote_quality"]' in cells["configuration"]
    assert 'MODEL_PROFILE in {"t4_quality", "t4_local"}' in cells["setup"]
    assert '"--no-build-isolation"' in cells["setup"]
    assert "flash-linear-attention[cuda,conv1d]>=0.5.2,<0.6" in cells["setup"]
    assert "from causal_conv1d import causal_conv1d_fn, causal_conv1d_update" in cells["setup"]
    assert "chunk_gated_delta_rule" in cells["setup"]
    assert "fused_recurrent_gated_delta_rule" in cells["setup"]
    assert "modeling_qwen3_5" in cells["setup"]
    assert "is_fast_path_available" in cells["setup"]
    assert "causal_conv1d_fn(" in cells["setup"]
    assert "causal_conv1d_update(" in cells["setup"]
    assert "torch.cuda.synchronize()" in cells["setup"]
    assert "restart the Colab " in cells["setup"]
    assert "runtime and rerun from cell 1" in cells["setup"]
    assert "silently continue" in cells["setup"]


def test_qwen_fastpath_extra_is_linux_only_and_includes_fixed_fla() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "qwen-fastpath = [" in pyproject
    assert "flash-linear-attention[cuda,conv1d]>=0.5.2,<0.6" in pyproject
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
