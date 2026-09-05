from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_mod_ai.grounded_source_reuse import (
    _grounded_repository_cards,
    _normalize_reference_repository,
)
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
    assert 'MINECRAFT_VERSION = "Auto"' in configuration
    assert 'MOD_LOADER = "Auto"' in configuration
    assert 'REFERENCE_MOD_URLS = ""' in configuration
    assert 'os.environ["MMM_REFERENCE_MOD_URLS"] = reference_mod_urls' in configuration
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


def test_reference_mod_urls_are_host_owned_repository_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "MMM_REFERENCE_MOD_URLS",
        "https://github.com/owner/trade-mod, owner/second-mod; "
        "https://github.com/owner/trade-mod.git",
    )
    cards = _grounded_repository_cards({"_pre_design_research": {"domain_notes": []}})

    assert [item["repository"] for item in cards] == [
        "owner/trade-mod",
        "owner/second-mod",
    ]
    assert all(item["explicit_reference"] for item in cards)
    assert all(item["page_refs"] == ["host:reference_mod"] for item in cards)


def test_reference_mod_input_rejects_non_github_web_pages() -> None:
    with pytest.raises(ValueError, match="GitHub repository"):
        _normalize_reference_repository("https://example.com/not-a-source-repository")
