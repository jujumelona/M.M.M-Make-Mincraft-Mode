from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from minecraft_mod_ai.complete_spec import ProductionModule, complete_proposal_from_parts
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
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan("Create one frost item.")
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
        game_design={**legacy.game_design, "_production_contract": compiled.contract},
        modules=legacy.modules,
        acceptance_tests=compiled.acceptance_tests,
    )


def test_complete_plan_uses_opaque_ref_and_paged_sections(tmp_path: Path, monkeypatch) -> None:
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

    assert result["schema_version"] == "mmm/complete-plan-result-v4"
    assert "complete_proposal" not in result
    assert "game_design" not in result
    assert result["proposal_ref"].startswith("plan_")
    assert result["counts"]["modules"] == 73

    first = service.read_complete_plan_section(result["proposal_ref"], "modules", limit=11)
    second = service.read_complete_plan_section(
        result["proposal_ref"], "modules", cursor=first["next_cursor"], limit=11
    )
    assert len(first["items"]) == 11
    assert first["remaining"] == 62
    assert second["items"][0]["module_id"] == "frost_module_00011"

    approved = service.approve_complete_plan(
        approval_hash=result["approval_hash"], proposal_ref=result["proposal_ref"]
    )
    assert approved["status"] == "approved"
    assert "complete_proposal" not in approved
    assert approved["proposal_ref"] == result["proposal_ref"]


def test_complete_execution_accepts_ref_without_inline_payload(tmp_path: Path, monkeypatch) -> None:
    proposal = _proposal(2)
    service = MMMToolService(
        workspace_root=tmp_path / "workspace",
        router_factory=lambda: object(),
    )
    proposal_ref = service._store_complete_proposal(proposal)
    called = {}

    class Result:
        def to_dict(self):
            return {"summary": "done"}

    class Orchestrator:
        def __init__(self, **_kwargs):
            pass

        def execute(self, value, **kwargs):
            called["proposal"] = value
            called["kwargs"] = kwargs
            return Result()

    monkeypatch.setattr("minecraft_mod_ai.mcp_tools.CompleteProductionOrchestrator", Orchestrator)
    result = service.execute_complete_project(
        proposal_ref=proposal_ref,
        approval_hash=proposal.calculate_hash(),
        run_name="ref-execution",
    )

    assert called["proposal"].calculate_hash() == proposal.calculate_hash()
    assert result["summary"] == "done"


def test_quality_contract_and_run_status_are_bounded_read_only_views(tmp_path: Path) -> None:
    proposal = _quality_proposal()
    service = MMMToolService(
        workspace_root=tmp_path / "workspace",
        router_factory=lambda: object(),
    )
    proposal_ref = service._store_complete_proposal(proposal)
    quality = service.read_quality_contract(proposal_ref)

    contract = proposal.game_design["_production_contract"]
    report = evaluate_quality_contract(contract, {}, proposal.calculate_hash())
    report_path = service.workspace_root / "run" / ".minecraft_ai" / "quality-convergence.json"
    persist_quality_report(report_path, report)
    status = service.quality_status("run")

    assert quality["proposal_ref"] == proposal_ref
    assert quality["quality_dimensions"]
    assert status["run_name"] == "run"
    assert status["overall_status"] in {"MISSING", "FAIL", "PASS"}


def test_complete_plan_ref_is_not_a_caller_controlled_path(tmp_path: Path) -> None:
    service = MMMToolService(
        workspace_root=tmp_path / "workspace",
        router_factory=lambda: object(),
    )
    for invalid in (
        "../complete-proposal.json",
        "plan_" + "a" * 63,
        "plan_" + "A" * 64,
        "plan_" + "a" * 64 + "/extra",
    ):
        with pytest.raises((ValueError, SpecValidationError)):
            service.read_complete_plan_section(invalid, "modules")


def test_complete_plan_requires_exactly_one_transport_form(tmp_path: Path) -> None:
    service = MMMToolService(
        workspace_root=tmp_path / "workspace",
        router_factory=lambda: object(),
    )
    proposal = _proposal(2)
    ref = service._store_complete_proposal(proposal)

    with pytest.raises(SpecValidationError, match="exactly one"):
        service.execute_complete_project(
            complete_proposal=proposal.to_dict(),
            proposal_ref=ref,
            approval_hash=proposal.calculate_hash(),
            run_name="invalid-transport",
        )


def test_complete_plan_ref_binds_the_exact_root_index_bytes(tmp_path: Path) -> None:
    service = MMMToolService(
        workspace_root=tmp_path / "workspace",
        router_factory=lambda: object(),
    )
    ref = service._store_complete_proposal(_proposal(2))
    digest = ref.split("_", 2)[1]
    index = service.workspace_root / ".minecraft_ai" / "plans" / digest / "complete-proposal.json"
    index.write_bytes(index.read_bytes() + b" ")

    with pytest.raises(SpecValidationError, match="opaque reference"):
        service.read_complete_plan_section(ref, "modules")


def test_complete_plan_page_is_byte_bounded_and_resumes_large_item(tmp_path: Path) -> None:
    policy = ScalePolicy(mcp_page_bytes=8 * 1024)
    legacy = _proposal(1)
    proposal = complete_proposal_from_parts(
        requested_prompt=legacy.requested_prompt,
        base_proposal=legacy.base_proposal,
        game_design=legacy.game_design,
        modules=(ProductionModule("huge", "item", {"description": "x" * 32_000}),),
        acceptance_tests=legacy.acceptance_tests,
    )
    service = MMMToolService(
        workspace_root=tmp_path / "workspace",
        router_factory=lambda: object(),
        policy=policy,
    )
    ref = service._store_complete_proposal(proposal)

    cursor = ""
    chunks: list[bytes] = []
    item_sha = ""
    for _ in range(20):
        page = service.read_complete_plan_section(ref, "modules", cursor=cursor, limit=16)
        encoded = json.dumps(
            page, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        assert len(encoded) <= policy.mcp_page_bytes
        fragment = page["item_fragment"]
        assert fragment is not None
        if item_sha:
            assert fragment["item_sha256"] == item_sha
        item_sha = fragment["item_sha256"]
        chunks.append(base64.b64decode(fragment["data"]))
        cursor = page["next_cursor"]
        if not cursor:
            break
    else:
        raise AssertionError("large item fragment cursor did not terminate")

    reconstructed = json.loads(b"".join(chunks).decode("utf-8"))
    assert reconstructed["module_id"] == "huge"
    assert len(reconstructed["config"]["description"]) == 32_000


def test_complete_plan_cursor_cannot_be_forged_to_skip_items(tmp_path: Path) -> None:
    service = MMMToolService(
        workspace_root=tmp_path / "workspace",
        router_factory=lambda: object(),
    )
    ref = service._store_complete_proposal(_proposal(3))
    first = service.read_complete_plan_section(ref, "modules", limit=1)
    assert first["next_cursor"]
    forged = first["next_cursor"][:-1] + (
        "A" if first["next_cursor"][-1] != "A" else "B"
    )
    with pytest.raises(SpecValidationError):
        service.read_complete_plan_section(ref, "modules", cursor=forged, limit=1)
