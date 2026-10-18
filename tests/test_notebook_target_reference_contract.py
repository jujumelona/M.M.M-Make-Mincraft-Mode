from __future__ import annotations

import json
from pathlib import Path


from minecraft_mod_ai.platform_selection_pipeline import (
    _host_retarget_requested,
    _host_target_constraints,
)

NOTEBOOK_PATH = Path("M.M.M_Make_Mincraft_Mode_Colab.ipynb")


def _notebook_cells() -> dict[str, str]:
    raw = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return {
        str(cell["id"]): "".join(cell.get("source") or ())
        for cell in raw.get("cells", ())
    }


def test_notebook_exposes_target_and_reference_inputs_to_complete_session() -> None:
    cells = _notebook_cells()
    configuration = cells["configuration"]
    plan = cells["plan"]
    assert 'MINECRAFT_VERSION = "26.2" #@param [' in configuration
    assert 'MINECRAFT_VERSION = "Auto"' not in configuration
    for version in (
        "26.2",
        "26.1.1",
        "26.1",
        "1.21.10",
        "1.21.8",
        "1.21.5",
        "1.21.4",
        "1.21.1",
        "1.20.6",
        "1.20.4",
        "1.20.1",
    ):
        assert f'"{version}"' in configuration
    assert 'MOD_LOADER = "Auto"' in configuration
    assert "REFERENCE_MOD_URLS" not in configuration
    assert "MMM_REFERENCE_MOD_URLS" not in configuration
    assert "minecraft_version=MINECRAFT_VERSION" in plan
    assert "loader=MOD_LOADER" in plan


def test_host_target_markers_are_parsed_without_semantic_guessing() -> None:
    text = (
        "user prompt\n"
        "[HOST_TARGET_CONSTRAINT Minecraft 1.21.1]\n"
        "[HOST_LOADER_CONSTRAINT Fabric]"
    )
    assert _host_target_constraints(text) == ("1.21.1", "fabric")


def test_explicit_notebook_target_retargets_existing_project_when_it_differs() -> None:
    assert _host_retarget_requested(
        existing_version="1.20.1",
        existing_loader="fabric",
        host_version="1.21.1",
        host_loader="fabric",
    )
    assert _host_retarget_requested(
        existing_version="1.21.1",
        existing_loader="forge",
        host_version="1.21.1",
        host_loader="fabric",
    )
    assert not _host_retarget_requested(
        existing_version="1.21.1",
        existing_loader="fabric",
        host_version="1.21.1",
        host_loader="fabric",
    )


def test_notebook_target_selection_does_not_require_repository_search():
    configuration = _notebook_cells()["configuration"]
    assert "REFERENCE_MOD_URLS" not in configuration
    assert "CURSEFORGE_API_KEY" not in configuration
