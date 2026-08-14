from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import minecraft_mod_ai.agentic_optimization_contract as agentic
import minecraft_mod_ai.complete_planner as complete_planner
import minecraft_mod_ai.custom_generation_search_contract as custom_search
import minecraft_mod_ai.llama_parallel_runtime_contract as llama_parallel
import minecraft_mod_ai.llama_server_hardware_policy as llama_hardware
import minecraft_mod_ai.max_efficiency_runtime_contract as max_efficiency
import minecraft_mod_ai.performance_final_contract as performance
import minecraft_mod_ai.scheduler_parallel_safety_contract as safety
from minecraft_mod_ai.model_registry import ModelRegistry
from minecraft_mod_ai.model_router import ModelRouter
from minecraft_mod_ai.work_graph import DurableWorkLedger, WorkGraphPlan, WorkNode


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _node(node_id: str, resource_class: str) -> WorkNode:
    return WorkNode(
        node_id=node_id,
        stage="generate:custom" if resource_class == "llm" else "generate:assets",
        input_hash=f"sha256:{node_id}",
        dependencies=(),
        payload={"kind": "test", "resource_class": resource_class},
        resource_class=resource_class,
    )


def test_candidate_receipts_use_prefixed_hashes_and_validate_after_state(tmp_path: Path) -> None:
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    base.mkdir()
    candidate.mkdir()
    (base / "replace.txt").write_text("old", encoding="utf-8")
    (base / "delete.txt").write_text("gone", encoding="utf-8")
    (candidate / "replace.txt").write_text("new", encoding="utf-8")
    result = {
        "patch_receipt": {
            "operations": [
                {
                    "path": "replace.txt",
                    "operation": "replace",
                    "before_sha256": _sha(b"old"),
                    "after_sha256": _sha(b"new"),
                },
                {
                    "path": "delete.txt",
                    "operation": "delete",
                    "before_sha256": _sha(b"gone"),
                    "after_sha256": None,
                },
            ]
        }
    }
    capture = max_efficiency._candidate_patch_capture(
        base_root=base, candidate_root=candidate, result=result
    )
    assert capture["operations"][0]["expected_sha256"] == _sha(b"old")
    assert capture["operations"][1]["expected_sha256"] == _sha(b"gone")

    (candidate / "delete.txt").write_text("still here", encoding="utf-8")
    with pytest.raises(RuntimeError, match="delete output still exists"):
        max_efficiency._candidate_patch_capture(
            base_root=base, candidate_root=candidate, result=result
        )


def test_candidate_after_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    base.mkdir()
    candidate.mkdir()
    (base / "x.txt").write_text("old", encoding="utf-8")
    (candidate / "x.txt").write_text("new", encoding="utf-8")
    result = {
        "patch_receipt": {
            "operations": [
                {
                    "path": "x.txt",
                    "operation": "replace",
                    "before_sha256": _sha(b"old"),
                    "after_sha256": _sha(b"wrong"),
                }
            ]
        }
    }
    with pytest.raises(RuntimeError, match="after hash drifted"):
        max_efficiency._candidate_patch_capture(
            base_root=base, candidate_root=candidate, result=result
        )


def test_parallel_candidate_workspaces_and_model_routers_are_isolated(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "A.java").write_text("class A {}", encoding="utf-8")
    left_root = max_efficiency._clone_candidate_snapshot(
        base, candidate_index=0, performance_module=performance
    )
    right_root = max_efficiency._clone_candidate_snapshot(
        base, candidate_index=1, performance_module=performance
    )
    try:
        assert left_root.parent != right_root.parent
        router = ModelRouter(profile="Qwen3.5-9B_6GB")
        left = custom_search._fork_router_for_candidate(router)
        right = custom_search._fork_router_for_candidate(router)
        assert left is not right and left is not router and right is not router
        assert left.registry is router.registry and right.registry is router.registry
        left_workspace = tmp_path / "left-workspace"
        right_workspace = tmp_path / "right-workspace"
        left_workspace.mkdir()
        right_workspace.mkdir()
        left.bind_agent_workspace(left_workspace)
        right.bind_agent_workspace(right_workspace)
        assert left._agent_workspace_root == left_workspace.resolve()
        assert right._agent_workspace_root == right_workspace.resolve()
        assert router._agent_workspace_root is None
    finally:
        shutil.rmtree(left_root.parent, ignore_errors=True)
        shutil.rmtree(right_root.parent, ignore_errors=True)


def test_scheduler_resource_priority_precedes_node_name(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    plan = WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:priority",
        graph_hash="sha256:priority-graph",
        module_count=2,
        nodes=(_node("a-image", "image_gpu"), _node("z-llm", "llm")),
    )
    ledger = DurableWorkLedger(tmp_path / "work.sqlite", proposal_hash=plan.proposal_hash)
    ledger.sync_plan(plan)
    token = safety._SHARED_LOCAL_GPU_LANE.set(True)
    try:
        claimed = ledger.claim_ready(
            "mmm-orchestrator",
            stages=("generate:custom", "generate:assets"),
            lease_seconds=60,
        )
        assert claimed is not None
        assert claimed["node_id"] == "z-llm"
        assert ledger.task("a-image")["state"] == "pending"
    finally:
        safety._SHARED_LOCAL_GPU_LANE.reset(token)


def test_custom_search_propagates_keyboard_interrupt(monkeypatch, tmp_path: Path) -> None:
    class Router:
        def bind_agent_workspace(self, *_args, **_kwargs):
            return self

    class Generator:
        def __init__(self):
            self.router = Router()
            self._cached_index = None
            self._cached_root = None

        def generate(self, *_args, **_kwargs):
            raise AssertionError("capture stub should own this test")

    fake_module = SimpleNamespace(CustomModuleGenerator=Generator)
    monkeypatch.setattr(custom_search, "_width", lambda _module: 2)

    def capture(*_args, **_kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(custom_search, "_capture_candidate", capture)
    custom_search.install(fake_module)
    with pytest.raises(KeyboardInterrupt):
        Generator().generate(tmp_path, module=SimpleNamespace())


def test_parallel_planner_search_propagates_keyboard_interrupt(monkeypatch) -> None:
    def current(*_args, **_kwargs):
        return {"fallback": True}

    def base(_router, *, system_prompt: str, **_kwargs):
        if "Candidate 1 of 2" in system_prompt:
            raise KeyboardInterrupt()
        return {"ok": True}

    current._mmm_verifier_plan_search = True
    current.__wrapped__ = base
    monkeypatch.setattr(complete_planner, "_generate_json_page_with_repair", current)
    monkeypatch.setattr(agentic, "_planner_candidate_count", lambda _request, _stage: 2)
    monkeypatch.setattr(agentic, "_score_plan_page", lambda _page: (1.0, {}))
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    router = SimpleNamespace(
        profile="test",
        registry=SimpleNamespace(
            role=lambda _profile, _role: SimpleNamespace(
                exclusive_gpu=True, provider="local", adapter="llama_cpp"
            )
        ),
    )
    llama_parallel._install_planner_search_parallelism()
    with pytest.raises(KeyboardInterrupt):
        complete_planner._generate_json_page_with_repair(
            router,
            system_prompt="system",
            request={},
            media_paths=(),
            expected_contracts=(),
            stage="unit",
        )


def test_t4_aliases_resolve_to_actual_qwen35_9b() -> None:
    for path in ("config/model_registry.yaml", "minecraft_mod_ai/config/model_registry.yaml"):
        registry = ModelRegistry(path)
        for profile in ("t4_local", "t4_quality"):
            planner = registry.role(profile, "planner")
            assert planner.model_id == "unsloth/Qwen3.5-9B-MTP-GGUF"
            assert planner.extra["gguf_filename"] == "Qwen3.5-9B-UD-Q4_K_XL.gguf"
            assert planner.max_context == 32768


def test_bounded_section_budget_does_not_cap_paginated_qwen_json(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_BOUNDED_SECTION_MAX_TOKENS", raising=False)
    monkeypatch.delenv("MMM_QWEN35_MAX_OUTPUT_TOKENS", raising=False)
    adapter = SimpleNamespace(
        config=SimpleNamespace(
            model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
            extra={"gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf"},
            max_new_tokens=8192,
        )
    )
    section = SimpleNamespace(
        messages=({"role": "user", "content": "bounded"},),
        response_format="json",
        response_schema={
            "type": "object",
            "properties": {"section": {"type": "string"}},
            "required": ["section"],
            "additionalProperties": False,
        },
        tools=(),
        tool_choice=None,
    )
    paged = SimpleNamespace(
        messages=({"role": "user", "content": "paged"},),
        response_format="json",
        response_schema=None,
        tools=(),
        tool_choice=None,
    )
    assert llama_hardware._server_payload(adapter, section)["max_tokens"] == 2048
    assert llama_hardware._server_payload(adapter, paged)["max_tokens"] == 8192
