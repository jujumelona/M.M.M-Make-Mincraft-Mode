import json
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
    import runpy

    module = runpy.run_path("tools/build_colab_notebook.py")
    rendered = module["serialize_notebook"](module["build_notebook"]())
    assert Path("M.M.M_Make_Mincraft_Mode_Colab.ipynb").read_text(encoding="utf-8") == rendered
    assert Path("Minecraft_Multimodal_Mod_AI_Architecture_v6.ipynb").read_text(encoding="utf-8") == rendered
