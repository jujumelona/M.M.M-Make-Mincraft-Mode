from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.completion_boundary_work_recovery import install as install_work_recovery
from minecraft_mod_ai.llama_finish_reason_contract import (
    CONTEXT_PRESSURE,
    OUTPUT_EXHAUSTED,
    LlamaCompletionBoundaryError,
    _length_error,
    completion_boundary_kind,
)
from minecraft_mod_ai.llama_tool_output_budget import install as install_tool_output_budget
from minecraft_mod_ai.mcp_stdio_support import install_mcp_protocol_print_guard
from minecraft_mod_ai.model_adapters.base import (
    ModelBackendError,
    ModelConfigurationError,
)


def _tool(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object"}},
    }


def test_finish_reason_classifier_distinguishes_context_and_output() -> None:
    output = _length_error(
        {"usage": {"prompt_tokens": 100, "completion_tokens": 4095}},
        {"max_tokens": 4096},
    )
    context = _length_error(
        {"usage": {"prompt_tokens": 32000, "completion_tokens": 128}},
        {"max_tokens": 8192},
    )

    assert output.kind == OUTPUT_EXHAUSTED
    assert context.kind == CONTEXT_PRESSURE


def test_boundary_kind_survives_model_backend_wrapper() -> None:
    boundary = LlamaCompletionBoundaryError("exhausted", kind=OUTPUT_EXHAUSTED)
    wrapped = ModelBackendError(role="coder", model_id="model", cause=boundary)

    assert completion_boundary_kind(wrapped) == OUTPUT_EXHAUSTED


def test_source_mutation_keeps_authoritative_output_budget() -> None:
    hardware = SimpleNamespace()

    def server_payload(adapter, request):
        return {"max_tokens": adapter.config.max_new_tokens}

    hardware._server_payload = server_payload
    install_tool_output_budget(hardware)
    adapter = SimpleNamespace(config=SimpleNamespace(max_new_tokens=8192))

    edit_request = SimpleNamespace(tools=(_tool("apply_source_edit"),))
    patch_request = SimpleNamespace(tools=(_tool("apply_source_patch"),))
    search_request = SimpleNamespace(tools=(_tool("search_code_rag"),))

    assert hardware._server_payload(adapter, edit_request)["max_tokens"] == 8192
    assert hardware._server_payload(adapter, patch_request)["max_tokens"] == 8192
    assert hardware._server_payload(adapter, search_request)["max_tokens"] == 4096


def test_work_node_preserves_output_exhaustion_without_whole_action_replay() -> None:
    class Ledger:
        def __init__(self) -> None:
            self.state = "running"
            self.retries = 0

        def task(self, _node_id):
            return {"state": self.state}

        def retry(self, _node_id):
            self.retries += 1
            self.state = "pending"

    class FakeOrchestrator:
        calls = 0
        error = None

        @staticmethod
        def _run_work_node(ledger, node, *, action, validate_cached, shared_index=None):
            FakeOrchestrator.calls += 1
            ledger.state = "failed"
            boundary = LlamaCompletionBoundaryError("exhausted", kind=OUTPUT_EXHAUSTED)
            FakeOrchestrator.error = ModelBackendError(
                role="coder", model_id="model", cause=boundary
            )
            raise FakeOrchestrator.error from boundary

    install_work_recovery(FakeOrchestrator)
    ledger = Ledger()
    with pytest.raises(ModelBackendError) as caught:
        FakeOrchestrator._run_work_node(
            ledger,
            SimpleNamespace(node_id="generate-custom-1"),
            action=lambda: {},
            validate_cached=lambda value: True,
        )

    assert completion_boundary_kind(caught.value) == OUTPUT_EXHAUSTED
    assert caught.value is FakeOrchestrator.error
    assert FakeOrchestrator.calls == 1
    assert ledger.retries == 0


def test_install_peels_legacy_whole_node_replay_wrapper() -> None:
    class Ledger:
        retries = 0

        def task(self, _node_id):
            return {"state": "failed"}

        def retry(self, _node_id):
            self.retries += 1

    class FakeOrchestrator:
        calls = 0

        @staticmethod
        def _run_work_node(ledger, node, *, action, validate_cached, shared_index=None):
            FakeOrchestrator.calls += 1
            boundary = LlamaCompletionBoundaryError("exhausted", kind=OUTPUT_EXHAUSTED)
            raise ModelBackendError(role="coder", model_id="model", cause=boundary) from boundary

    original = FakeOrchestrator._run_work_node

    def legacy_replay(ledger, node, *, action, validate_cached, shared_index=None):
        try:
            return original(
                ledger,
                node,
                action=action,
                validate_cached=validate_cached,
                shared_index=shared_index,
            )
        except ModelBackendError:
            ledger.retry(node.node_id)
            return original(
                ledger,
                node,
                action=action,
                validate_cached=validate_cached,
                shared_index=shared_index,
            )

    legacy_replay.__wrapped__ = original
    legacy_replay._mmm_completion_boundary_work_recovery_v1 = True
    FakeOrchestrator._run_work_node = staticmethod(legacy_replay)

    install_work_recovery(FakeOrchestrator)
    ledger = Ledger()
    with pytest.raises(ModelBackendError) as caught:
        FakeOrchestrator._run_work_node(
            ledger,
            SimpleNamespace(node_id="generate-custom-1"),
            action=lambda: {},
            validate_cached=lambda value: True,
        )

    assert completion_boundary_kind(caught.value) == OUTPUT_EXHAUSTED
    assert FakeOrchestrator.calls == 1
    assert ledger.retries == 0


def test_work_node_does_not_retry_unrelated_failure() -> None:
    class FakeOrchestrator:
        calls = 0

        @staticmethod
        def _run_work_node(ledger, node, *, action, validate_cached, shared_index=None):
            FakeOrchestrator.calls += 1
            raise RuntimeError("unrelated")

    install_work_recovery(FakeOrchestrator)
    with pytest.raises(RuntimeError, match="unrelated"):
        FakeOrchestrator._run_work_node(
            SimpleNamespace(),
            SimpleNamespace(node_id="node"),
            action=lambda: {},
            validate_cached=lambda value: True,
        )
    assert FakeOrchestrator.calls == 1


def test_work_node_does_not_blindly_retry_exhausted_causal_resync() -> None:
    """A model-protocol failure alone is not a safe durable retry receipt.

    Custom-module tool calls run in a disposable staging workspace.  Until the
    producer proves that workspace was discarded and no external side effect
    escaped it, replaying the whole node would silently repeat expensive model and
    retrieval work from the beginning.
    """

    class Ledger:
        def __init__(self) -> None:
            self.state = "running"
            self.retries = 0

        def task(self, _node_id):
            return {"state": self.state}

        def retry(self, _node_id):
            self.retries += 1
            self.state = "pending"

    class FakeOrchestrator:
        calls = 0

        @staticmethod
        def _run_work_node(ledger, node, *, action, validate_cached, shared_index=None):
            FakeOrchestrator.calls += 1
            ledger.state = "failed"
            raise ModelConfigurationError(
                "Model failed the single causal-frontier re-synchronization attempt; "
                "forced='search_project_rag' rejected=apply_source_edit "
                "visible=search_project_rag,java_workspace_symbols,search_code_rag"
            )

    install_work_recovery(FakeOrchestrator)
    ledger = Ledger()
    with pytest.raises(ModelConfigurationError, match="causal-frontier"):
        FakeOrchestrator._run_work_node(
            ledger,
            SimpleNamespace(node_id="generate-custom-00000000"),
            action=lambda: {},
            validate_cached=lambda value: True,
        )

    assert FakeOrchestrator.calls == 1
    assert ledger.retries == 0


def test_mcp_print_guard_keeps_default_print_off_stdout(capsys) -> None:
    original = builtins.print
    try:
        install_mcp_protocol_print_guard()
        print("runtime diagnostic")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "runtime diagnostic" in captured.err
    finally:
        builtins.print = original


def test_first_party_mcp_config_uses_protocol_safe_launcher() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / ".mcp.json").read_text(encoding="utf-8"))
    first_party = {
        name: value
        for name, value in config["mcpServers"].items()
        if name.startswith("mmm-")
    }

    assert first_party
    for server in first_party.values():
        args = server["args"]
        assert args[:2] == ["-m", "minecraft_mod_ai.mcp_stdio_entrypoint"]
        assert args[2] in {
            "minecraft_mod_ai.mcp_server",
            "minecraft_mod_ai.mod_generation_mcp_server",
        }
