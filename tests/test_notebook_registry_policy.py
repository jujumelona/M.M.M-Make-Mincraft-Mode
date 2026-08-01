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


def test_notebook_code_cells_compile_top_to_bottom() -> None:
    module = runpy.run_path("tools/build_colab_notebook.py")
    notebook = module["build_notebook"]()

    for cell in notebook.cells:
        if cell["cell_type"] == "code":
            compile(cell["source"], f"<colab:{cell['id']}>", "exec")
