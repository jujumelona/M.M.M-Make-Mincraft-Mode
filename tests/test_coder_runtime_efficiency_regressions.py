from __future__ import annotations

import json
import threading
from types import SimpleNamespace

from minecraft_mod_ai.llama_tool_output_budget import _tool_max_tokens, tool_output_budget
from minecraft_mod_ai.model_registry import ModelRegistry
from minecraft_mod_ai.platform_custom_coder_contract import (
    _capture_agent_binding,
    _restore_agent_binding,
)
from minecraft_mod_ai.small_model_compacting_adapter import (
    _IMPLEMENTATION_SOURCE_SEED_BYTES,
    _compact_implementation_seed,
    _json_bytes,
)


def test_qwen35_coder_uses_registry_declared_precise_agent_policy() -> None:
    config = ModelRegistry().role("t4_local", "coder")

    assert config.extra["runtime_contract"] == "qwen"
    assert config.extra["agent_thinking"] is True
    precise = config.extra["sampling_profiles"]["precise_coding"]
    assert precise == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    }


def test_tool_turn_default_budget_is_bounded_below_full_generation(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_TOOL_MAX_TOKENS", raising=False)

    assert _tool_max_tokens() == 4096
    assert tool_output_budget(SimpleNamespace(max_new_tokens=8192)) == 4096

    monkeypatch.setenv("MMM_LLAMA_TOOL_MAX_TOKENS", "6000")
    assert tool_output_budget(SimpleNamespace(max_new_tokens=8192)) == 6000


def test_custom_coder_agent_binding_is_transaction_scoped() -> None:
    original_runtime = object()
    router = SimpleNamespace(
        _generation_lock=threading.RLock(),
        _agent_workspace_root="before",
        _agent_tool_runtime=original_runtime,
        _agent_require_fresh_evidence=False,
    )
    proxy = SimpleNamespace(_router=router)

    snapshot = _capture_agent_binding(proxy)
    router._agent_workspace_root = "custom-stage"
    router._agent_tool_runtime = object()
    router._agent_require_fresh_evidence = True
    _restore_agent_binding(snapshot)

    assert router._agent_workspace_root == "before"
    assert router._agent_tool_runtime is original_runtime
    assert router._agent_require_fresh_evidence is False


def test_implementation_seed_removes_duplicate_receipts_and_bounds_exact_source() -> None:
    receipt = {
        "project_sha256": "sha256:project",
        "observations_sha256": "sha256:observations",
    }
    records = [
        {
            "observation_id": f"obs-{index}",
            "path": f"src/main/java/example/C{index}.java",
            "sha256": f"sha256:{index}",
            "content_start_bytes": 0,
            "content_end_bytes": 3000,
            "text": "x" * 3000,
        }
        for index in range(10)
    ]
    payload = {
        "phase": "implement_module",
        "project_manifest": {"sha256": "duplicate-manifest"},
        "source_observation_receipt": dict(receipt),
        "research_context": {"facts": ["duplicate research"] * 256},
        "host_grounding": {
            "schema_version": "mmm/host-owned-coder-grounding-v1",
            "evidence_bindings": {
                "project_exact_rag": {"receipt": dict(receipt)},
            },
        },
        "initial_exact_source_context": {
            "schema_version": "mmm/source-observation-context-v1",
            "ledger_receipt": dict(receipt),
            "global_anchor_count": 5,
            "global_anchors": records[:5],
            "page_observations": records[5:],
        },
    }
    messages = (
        {"role": "system", "content": "coder"},
        {"role": "user", "content": json.dumps(payload)},
    )

    compacted = _compact_implementation_seed(messages)
    result = json.loads(compacted[1]["content"])

    assert "project_manifest" not in result
    assert "source_observation_receipt" not in result
    assert "research_context" not in result
    assert result["host_grounding"] == payload["host_grounding"]
    source = result["initial_exact_source_context"]
    assert _json_bytes(source) <= _IMPLEMENTATION_SOURCE_SEED_BYTES + 512
    assert source["model_seed_compaction"]["omitted_record_count"] > 0
    assert source["model_seed_compaction"]["supplemental_retrieval_available"] is True
