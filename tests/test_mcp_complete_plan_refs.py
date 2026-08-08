from __future__ import annotations

import base64
from dataclasses import dataclass
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
from minecraft_mod_ai.production_contract import (
    compile_production_contract,
    evaluate_quality_contract,
    persist_quality_report,
)
from minecraft_mod_ai.scale_policy import ScalePolicy
from minecraft_mod_ai.spec import SpecValidationError


def _proposal(module_count: int = 73):
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan(
        "Create one frost item."
    )
    return complete_proposal_from_parts(
        requested_prompt="Create a scalable frost content mod.",
        base_proposal=base,
        game_design={
            "title": "Frost Archive",
            "pitch": "A scalable collection of frost artifacts.",
            "core_loop": ["Discover", "Craft", "Test"],
        },
        modules=tuple(
            ProductionModule(f"frost_module_{index:05d}", "item")
            for index in range(module_count)
        ),
        acceptance_tests=("Every requested frost module is registered.",),
    )


def _quality_proposal():
    legacy = _proposal(2)
    compiled = compile_production_contract(
        requested_prompt=legacy.requested_prompt,
        game_design=legacy.game_design,
        modules=legacy.modules,
        acceptance_tests=legacy.acceptance_tests,
    )
    return complete_proposal_from_parts(
        requested_prompt=legacy.requested_prompt,
        base_proposal=legacy.base_proposal,
        game_design={
            **legacy.game_design,
            "_production_contract": compiled.contract,
        },
        modules=legacy.modules,
        acceptance_tests=compiled.acceptance_tests,
    )


def test_complete_plan_uses_opaque_ref_and_paged_sections(
    tmp_path: Path,
    monkeypatch,
) -> None:
    proposal = _proposal()
    monkeypatch.setattr(
        "minecraft_mod_ai.mcp_tools.CompleteGameDesignPlanner.plan",
        lambda self, prompt, media_paths=(), existing_input_sha256="": proposal,
    )
    service = MMMToolService(
        workspace_root=tmp_path / "workspace",
        router_factory=lambda: object(),
    )

    result = service.plan_complete_game("Create the frost archive.")

    assert result["schema_version"] == "mmm/complete-plan-result-v3"
    assert "complete_proposal" not in result
    assert "game_design" not in result
    assert result["proposal_ref"].startswith("plan_")
    assert result["counts"]["modules"] == 73

    first = service.read_complete_plan_section(
        result["proposal_ref"],
        "modules",
        limit=11,
    )
    second = service.read_complete_plan_section(
        result["proposal_ref"],
        "modules",
        cursor=first["next_cursor"],
        limit=11,
    )
    assert len(first["items"]) == 11
    assert first["remaining"] == 62
    assert second["items"][0]["module_id"] == "frost_module_00011"

    approved = service.approve_complete_plan(
        approval_hash=result["approval_hash"],
        proposal_ref=result["proposal_ref"],
    )
    assert approved["status"] == "approved"
    assert "complete_proposal" not in approved
    assert approved["proposal_ref"] == result["proposal_ref"]


def test_complete_execution_accepts_ref_without_inline_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    proposal = _proposal(3)
    service = MMMToolService(workspace_root=tmp_path / "workspace")
    proposal_ref = service._store_complete_proposal(proposal)
    captured = {}

    @dataclass
    class _Result:
        def to_dict(self):
            return {"status": "captured"}

    def fake_execute(self, parsed, **kwargs):
        captured["proposal"] = parsed
        captured.update(kwargs)
        return _Result()

    monkeypatch.setattr(
        "minecraft_mod_ai.mcp_tools.CompleteProductionOrchestrator.execute",
        fake_execute,
    )

    result = service.execute_complete_project(
        approval_hash=proposal.calculate_hash(),
        run_name="ref-run",
        proposal_ref=proposal_ref,
    )

    assert result == {"status": "captured"}
    assert captured["proposal"].calculate_hash() == proposal.calculate_hash()
    assert captured["approval_hash"] == proposal.calculate_hash()


def test_quality_contract_and_run_status_are_bounded_read_only_views(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    service = MMMToolService(workspace_root=workspace)
    proposal = _quality_proposal()
    proposal_ref = service._store_complete_proposal(proposal)

    summary = service.read_quality_contract(proposal_ref)

    assert summary["catalog_stats"]["requirements"] >= 1
    quality_dimension_ids = {
        item["dimension_id"] for item in summary["quality_dimensions"]
    }
    assert {
        "correctness",
        "build",
        "research",
        "runtime",
    } <= quality_dimension_ids
    assert "contract_sha256" not in summary
    contract = proposal.game_design["_production_contract"]
    report = evaluate_quality_contract(
        contract,
        {},
        proposal.calculate_hash(),
    )
    persist_quality_report(
        workspace / "quality-run/.minecraft_ai/quality-convergence.json",
        report,
    )

    status = service.quality_status("quality-run")

    assert status["overall_status"] == "MISSING"
    assert status["run_name"] == "quality-run"
    assert set(status["unresolved_dimension_ids"]) == quality_dimension_ids
    with pytest.raises(SpecValidationError, match="run_name"):
        service.quality_status("../quality-run")


@pytest.mark.parametrize(
    "proposal_ref",
    (
        "../complete-proposal.json",
        "plan_" + ("a" * 63),
        "plan_" + ("A" * 64),
        "plan_" + ("a" * 64) + "/extra",
    ),
)
def test_complete_plan_ref_is_not_a_caller_controlled_path(
    tmp_path: Path,
    proposal_ref: str,
) -> None:
    service = MMMToolService(workspace_root=tmp_path / "workspace")

    with pytest.raises(SpecValidationError, match="valid M.M.M plan reference"):
        service.read_complete_plan_section(proposal_ref)


def test_complete_plan_requires_exactly_one_transport_form(
    tmp_path: Path,
) -> None:
    service = MMMToolService(workspace_root=tmp_path / "workspace")
    proposal = _proposal(1)

    with pytest.raises(SpecValidationError, match="exactly one"):
        service.approve_complete_plan(
            complete_proposal=proposal.to_dict(),
            proposal_ref="plan_" + ("a" * 64),
            approval_hash=proposal.calculate_hash(),
        )


def test_complete_plan_ref_binds_the_exact_root_index_bytes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    service = MMMToolService(workspace_root=workspace)
    proposal_ref = service._store_complete_proposal(_proposal(2))
    semantic_digest = proposal_ref.split("_")[1]
    index = (
        workspace
        / ".minecraft_ai"
        / "plans"
        / semantic_digest
        / "complete-proposal.json"
    )
    index.write_text(
        index.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    with pytest.raises(
        SpecValidationError,
        match="index does not match its opaque reference",
    ):
        service.read_complete_plan_section(proposal_ref)


def test_complete_plan_page_is_byte_bounded_and_resumes_large_item(
    tmp_path: Path,
) -> None:
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan(
        "Create one frost item."
    )
    proposal = complete_proposal_from_parts(
        requested_prompt="Create one item with a large authored configuration.",
        base_proposal=base,
        game_design={"title": "Large item"},
        modules=(
            ProductionModule(
                "large_authored_item",
                "item",
                config={"lore": "ice" * 20_000},
            ),
        ),
        acceptance_tests=("The large item is registered.",),
    )
    service = MMMToolService(
        workspace_root=tmp_path / "workspace",
        policy=ScalePolicy(mcp_page_bytes=8192),
    )
    proposal_ref = service._store_complete_proposal(proposal)
    cursor = ""
    chunks: list[bytes] = []

    while True:
        page = service.read_complete_plan_section(
            proposal_ref,
            "modules",
            cursor=cursor,
            limit=1,
        )
        assert len(
            json.dumps(
                page,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ) <= 8192
        fragment = page["item_fragment"]
        assert fragment is not None
        chunks.append(base64.b64decode(fragment["data"]))
        cursor = page["next_cursor"]
        if not cursor:
            break

    decoded = json.loads(b"".join(chunks).decode("utf-8"))
    assert decoded["module_id"] == "large_authored_item"
    assert decoded["config"]["lore"] == "ice" * 20_000
    assert fragment["complete"]


def test_complete_plan_cursor_cannot_be_forged_to_skip_items(
    tmp_path: Path,
) -> None:
    service = MMMToolService(workspace_root=tmp_path / "workspace")
    proposal_ref = service._store_complete_proposal(_proposal(4))
    first = service.read_complete_plan_section(
        proposal_ref,
        "modules",
        limit=1,
    )
    forged = first["next_cursor"].replace("p_1_", "p_0_", 1)
    assert forged != first["next_cursor"]

    with pytest.raises(
        SpecValidationError,
        match="does not match this proposal section",
    ):
        service.read_complete_plan_section(
            proposal_ref,
            "modules",
            cursor=forged,
            limit=1,
        )
