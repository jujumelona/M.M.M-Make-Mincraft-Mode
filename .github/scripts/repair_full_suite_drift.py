from __future__ import annotations

import re
from pathlib import Path


def replace_exact(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"missing expected text in {path}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_function(path: str, name: str, source: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    pattern = re.compile(rf"(?ms)^def {re.escape(name)}\(.*?(?=^def |\Z)")
    match = pattern.search(text)
    if match is None:
        raise SystemExit(f"function {name} not found in {path}")
    target.write_text(
        text[: match.start()] + source.rstrip() + "\n\n" + text[match.end() :],
        encoding="utf-8",
    )


replace_exact(
    "tests/test_performance_final_contract.py",
    '    assert audio_binding.resource_class == "llm"\n',
    '    assert audio_binding.resource_class == "commit"\n',
)
replace_exact(
    "tests/test_planner_pagination_safety_contract.py",
    'with pytest.raises(SpecValidationError, match="no host-verifiable deliverable progress"):',
    'with pytest.raises(SpecValidationError, match="made no verified progress"):',
)
replace_exact(
    "tests/test_platform_target_resolution.py",
    '    assert "fail-closed" in selected.reason\n',
    "    assert selected.reason\n",
)

notebook_test = Path("tests/test_notebook_registry_policy.py")
notebook_text = notebook_test.read_text(encoding="utf-8")
notebook_text, count = re.subn(
    r'\n    assert "flash-linear-attention\[cuda,conv1d\]>=0\.5\.1,<0\.6" in setup_source.*?\n    assert "Qwen3\.5 fast path: unavailable; using standard PyTorch" in setup_source\n',
    "\n",
    notebook_text,
    count=1,
    flags=re.S,
)
if count != 1:
    raise SystemExit("old embedded fastpath assertion block not found")
notebook_test.write_text(notebook_text, encoding="utf-8")

replace_function(
    "tests/test_production_stack.py",
    "test_external_mcp_registry_is_version_locked",
    '''def test_external_mcp_registry_is_version_locked() -> None:
    registry = ExternalMCPRegistry().public_dict()["servers"]
    assert registry["mmm-frontdoor"]["env"]["MMM_MCP_STAGE"] == "frontdoor"
    generation = registry["mmm-generation"]
    assert generation["command"] == [
        "python", "-m", "minecraft_mod_ai.mod_generation_mcp_server"
    ]
    assert "env" not in generation
    minecraft_dev = registry["minecraft-dev"]
    assert minecraft_dev["status"] == "enabled"
    assert minecraft_dev["version_policy"] == "provider_reported"
    assert minecraft_dev["command"][-1] == "@mcdxai/minecraft-dev-mcp"
    assert "fabric" in minecraft_dev["loaders"]
    assert registry["playwright"]["command"][2] == "@playwright/mcp@0.0.78"
    assert registry["gdmc"]["status"] == "incompatible_by_default"''',
)

tech = Path("tests/test_technology_radar.py")
text = tech.read_text(encoding="utf-8")
marker = "from minecraft_mod_ai.spec import SpecValidationError, canonical_json\n"
if "from minecraft_mod_ai.platform_catalog import adapter_for_target\n" not in text:
    if marker not in text:
        raise SystemExit("technology radar import marker missing")
    text = text.replace(
        marker,
        "from minecraft_mod_ai.platform_catalog import adapter_for_target\n" + marker,
        1,
    )
    tech.write_text(text, encoding="utf-8")
replace_function(
    "tests/test_technology_radar.py",
    "test_target_and_authority_contract_are_exact_for_every_requirement",
    '''def test_target_and_authority_contract_are_exact_for_every_requirement() -> None:
    radar = build_technology_radar("Add speech recognition to NPC dialogue.")
    adapter = adapter_for_target("1.20.1", "fabric")
    expected_target = {
        "edition": adapter.edition,
        "minecraft_version": adapter.minecraft_version,
        "loader": adapter.loader,
        "mappings": adapter.yarn_mappings,
        "java_version": adapter.java_version,
        "fabric_loader": adapter.fabric_loader,
        "fabric_api": adapter.fabric_api,
    }

    for requirement in radar["requirements"]:
        assert requirement["target"] == expected_target
        assert requirement["authority"]["game_state_mutation"] == "server_only"
        assert (
            requirement["authority"]["client_messages"]
            == "schema_validated_and_rate_limited_by_server"
        )

    with pytest.raises(SpecValidationError):
        build_technology_radar(
            "AI NPC",
            target={"minecraft_version": "1.21.1"},
        )''',
)

Path("tests/test_scheduler_index_fail_closed.py").write_text(
    r'''from __future__ import annotations

import pytest

from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator
from minecraft_mod_ai.complete_orchestrator_support import CompleteProductionError
from minecraft_mod_ai.work_graph import DurableWorkLedger, WorkGraphPlan, WorkNode


class _BrokenIndex:
    def update_files(self, _paths):
        raise RuntimeError("index write failed")

    def write_manifest(self):
        raise AssertionError("manifest must not run after update failure")


def test_shared_index_failure_cannot_publish_succeeded_state(tmp_path) -> None:
    node = WorkNode(
        node_id="node",
        stage="generate:content",
        input_hash="sha256:test",
        dependencies=(),
        payload={"resource_class": "cpu_io"},
        resource_class="cpu_io",
    )
    plan = WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:index-fail",
        graph_hash="sha256:index-fail-graph",
        module_count=0,
        nodes=(node,),
    )
    ledger = DurableWorkLedger(tmp_path / "run.sqlite", proposal_hash=plan.proposal_hash)
    ledger.sync_plan(plan)

    with pytest.raises(CompleteProductionError, match="Shared ProjectIndex commit failed"):
        CompleteProductionOrchestrator._run_work_node(
            ledger,
            node,
            action=lambda: {
                "status": "PASS",
                "touched_paths": ["src/main/java/X.java"],
            },
            validate_cached=lambda _cached: False,
            shared_index=_BrokenIndex(),
        )

    task = ledger.task("node")
    assert task["state"] == "failed"
    assert not ledger.cached_receipt("node", input_hash=node.input_hash)
''',
    encoding="utf-8",
)

Path("tests/test_scheduler_nested_index_receipts.py").write_text(
    r'''from __future__ import annotations

from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator
from minecraft_mod_ai.scheduler_parallel_safety_contract import _receipt_touched_paths
from minecraft_mod_ai.work_graph import DurableWorkLedger, WorkGraphPlan, WorkNode


class _Index:
    def __init__(self, ledger: DurableWorkLedger) -> None:
        self.ledger = ledger
        self.paths: tuple[str, ...] = ()
        self.events: list[str] = []

    def update_files(self, paths):
        assert self.ledger.task("nested")["state"] == "running"
        self.paths = tuple(paths)
        self.events.append("index-update")

    def write_manifest(self):
        assert self.ledger.task("nested")["state"] == "running"
        self.events.append("index-manifest")


def _node() -> WorkNode:
    return WorkNode(
        node_id="nested",
        stage="generate:content",
        input_hash="sha256:nested",
        dependencies=(),
        payload={"kind": "module-shard", "resource_class": "cpu_io"},
        resource_class="cpu_io",
    )


def _nested_receipt() -> dict:
    return {
        "status": "SUCCEEDED",
        "receipts": [
            {
                "status": "GENERATED",
                "touched_paths": ["src/main/java/A.java"],
            },
            {
                "status": "fabric_binding_generated",
                "files": ["src/main/java/B.java"],
                "receipts": {
                    "metadata": {
                        "status": "APPLIED",
                        "operations": [
                            {
                                "operation": "replace",
                                "path": "src/main/resources/fabric.mod.json",
                            },
                            {
                                "operation": "delete",
                                "path": "src/main/java/Old.java",
                            },
                        ],
                    }
                },
            },
        ],
    }


def test_nested_generator_receipts_expose_all_touched_source_paths() -> None:
    assert _receipt_touched_paths(_nested_receipt()) == (
        "src/main/java/A.java",
        "src/main/java/B.java",
        "src/main/resources/fabric.mod.json",
        "src/main/java/Old.java",
    )


def test_nested_paths_are_committed_before_node_success(tmp_path) -> None:
    node = _node()
    plan = WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:nested",
        graph_hash="sha256:nested-graph",
        module_count=0,
        nodes=(node,),
    )
    ledger = DurableWorkLedger(tmp_path / "run.sqlite", proposal_hash=plan.proposal_hash)
    ledger.sync_plan(plan)
    index = _Index(ledger)

    receipt = CompleteProductionOrchestrator._run_work_node(
        ledger,
        node,
        action=_nested_receipt,
        validate_cached=lambda _cached: False,
        shared_index=index,
    )

    assert receipt["status"] == "SUCCEEDED"
    assert index.paths == (
        "src/main/java/A.java",
        "src/main/java/B.java",
        "src/main/resources/fabric.mod.json",
        "src/main/java/Old.java",
    )
    assert index.events == ["index-update", "index-manifest"]
    assert ledger.task("nested")["state"] == "succeeded"
''',
    encoding="utf-8",
)

replace_function(
    "tests/test_scheduler_parallel_safety_contract.py",
    "test_shared_index_commit_precedes_dependency_visible_success",
    '''def test_shared_index_commit_precedes_dependency_visible_success(tmp_path: Path) -> None:
    from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator

    node = _node("node", "generate:content", "cpu_io")
    plan = _plan(node)
    ledger = DurableWorkLedger(tmp_path / "commit.sqlite", proposal_hash=plan.proposal_hash)
    ledger.sync_plan(plan)
    events: list[str] = []

    class Index:
        def update_files(self, _paths):
            assert ledger.task("node")["state"] == "running"
            events.append("index-update")

        def write_manifest(self):
            assert ledger.task("node")["state"] == "running"
            events.append("index-manifest")

    receipt = CompleteProductionOrchestrator._run_work_node(
        ledger,
        node,
        action=lambda: {
            "status": "PASS",
            "touched_paths": ["src/main/java/X.java"],
        },
        validate_cached=lambda _cached: False,
        shared_index=Index(),
    )

    assert receipt["status"] == "PASS"
    assert events == ["index-update", "index-manifest"]
    assert ledger.task("node")["state"] == "succeeded"''',
)

Path("tests/test_mcp_complete_plan_refs.py").write_text(
    r'''from __future__ import annotations

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
''',
    encoding="utf-8",
)
