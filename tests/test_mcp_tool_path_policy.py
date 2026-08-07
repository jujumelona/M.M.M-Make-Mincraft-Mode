import json
from pathlib import Path

import pytest

from minecraft_mod_ai.complete_spec import (
    ProductionModule,
    complete_proposal_from_parts,
)
from minecraft_mod_ai.mcp_tools import MMMToolService
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.spec import SpecValidationError


def _proposal():
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan(
        "Create one frost item"
    )
    return complete_proposal_from_parts(
        requested_prompt="Create one verified item",
        base_proposal=base,
        game_design={"title": "Path policy"},
        modules=(ProductionModule("verified_item", "item"),),
        acceptance_tests=("The item is registered.",),
    )


def _json_payload(proposal) -> dict:
    return json.loads(json.dumps(proposal.to_dict()))


@pytest.mark.parametrize(
    ("options", "existing_input"),
    [
        ({"server_launcher": "{outside}"}, None),
        ({"screenshot_paths": ["{outside}"]}, None),
        ({}, "{outside}"),
    ],
)
def test_complete_execution_rejects_mcp_file_inputs_outside_workspace(
    tmp_path: Path,
    options: dict,
    existing_input: str | None,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"untrusted external path")
    service = MMMToolService(workspace_root=workspace)
    proposal = _proposal()
    rendered_options = {
        key: (
            [
                str(outside) if value == "{outside}" else value
                for value in raw
            ]
            if isinstance(raw, list)
            else str(outside) if raw == "{outside}" else raw
        )
        for key, raw in options.items()
    }
    rendered_existing = (
        str(outside)
        if existing_input == "{outside}"
        else existing_input
    )

    with pytest.raises(
        SpecValidationError,
        match="escaped the configured workspace",
    ):
        service.execute_complete_project(
            _json_payload(proposal),
            proposal.calculate_hash(),
            "outside-path",
            options=rendered_options,
            existing_input=rendered_existing,
        )

    assert not (workspace / "outside-path").exists()


def test_complete_execution_rejects_relative_parent_escape(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.zip"
    outside.write_bytes(b"outside")
    service = MMMToolService(workspace_root=workspace)
    proposal = _proposal()

    with pytest.raises(
        SpecValidationError,
        match="escaped the configured workspace",
    ):
        service.execute_complete_project(
            _json_payload(proposal),
            proposal.calculate_hash(),
            "relative-escape",
            existing_input="../outside.zip",
        )


@pytest.mark.parametrize(
    "method_name",
    ("plan_game", "plan_complete_game"),
)
def test_mcp_planning_media_cannot_escape_workspace(
    tmp_path: Path,
    method_name: str,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "reference.png"
    outside.write_bytes(b"outside")
    service = MMMToolService(
        workspace_root=workspace,
        router_factory=lambda: object(),
    )

    with pytest.raises(
        SpecValidationError,
        match="escaped the configured workspace",
    ):
        getattr(service, method_name)(
            "Plan from this reference.",
            media_paths=[str(outside)],
        )
