"""Actual MCP tools/call envelopes must be decoded before UV attestation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_mod_ai import complete_orchestrator_services as services
from minecraft_mod_ai.complete_orchestrator_support import CompleteProductionError


def _response(operation: str, result: dict) -> dict:
    return {
        "schema_version": "mmm/blockbench-call-result-v1",
        "operation": operation,
        "result": result,
    }


def test_uv_decodes_structured_and_text_mcp_receipts() -> None:
    structured = _response(
        "validate_uv", {"structuredContent": {"status": "PASS", "overlaps": 0}}
    )
    assert services._blockbench_tool_payload(structured, "validate_uv")["status"] == "PASS"
    text_content = _response(
        "validate_uv",
        {"content": [{"type": "text", "text": json.dumps({"status": "OK"})}]},
    )
    assert services._blockbench_tool_payload(text_content, "validate_uv")["status"] == "OK"


def test_blockbench_mcp_rejects_failed_and_mismatched_tool_receipts() -> None:
    with pytest.raises(CompleteProductionError, match="isError"):
        services._blockbench_tool_payload(
            _response("validate_uv", {"isError": True}), "validate_uv"
        )
    with pytest.raises(CompleteProductionError, match="invalid MCP receipt"):
        services._blockbench_tool_payload(_response("render_preview", {}), "validate_uv")


def test_blockbench_review_checks_real_uv_and_png_and_closes_project(
    monkeypatch, tmp_path: Path
) -> None:
    geo = tmp_path / "model.geo.json"
    geo.write_text("{}", encoding="utf-8")
    calls: list[str] = []

    class FakeMCP:
        def __init__(self, *, workspace_root):
            assert workspace_root == tmp_path

        def call(self, operation, arguments):
            calls.append(operation)
            if operation == "validate_uv":
                return _response(operation, {"structuredContent": {"status": "PASS"}})
            if operation == "render_preview":
                Path(arguments["output_path"]).write_bytes(
                    bytes.fromhex("89504e470d0a1a0a") + b"test"
                )
            return _response(operation, {"isError": False})

        def close(self):
            calls.append("close-transport")

    monkeypatch.setattr(services, "BlockbenchMCPClient", FakeMCP)
    result = services.blockbench_review(
        {"entity_id": "test_entity", "files": [str(geo)]}, tmp_path
    )
    assert result["uv"]["status"] == "PASS"
    assert result["preview_sha256"].startswith("sha256:")
    assert calls == [
        "open_project", "validate_uv", "render_preview",
        "close_project", "close-transport",
    ]


def test_blockbench_rejects_non_png_preview_without_marking_uv_as_passed(
    monkeypatch, tmp_path: Path
) -> None:
    geo = tmp_path / "model.geo.json"
    geo.write_text("{}", encoding="utf-8")

    class FakeMCP:
        def __init__(self, *, workspace_root):
            pass

        def call(self, operation, arguments):
            if operation == "validate_uv":
                return _response(operation, {"structuredContent": {"status": "PASS"}})
            if operation == "render_preview":
                Path(arguments["output_path"]).write_bytes(b"not-an-image")
            return _response(operation, {})

        def close(self):
            pass

    monkeypatch.setattr(services, "BlockbenchMCPClient", FakeMCP)
    with pytest.raises(CompleteProductionError, match="valid PNG"):
        services.blockbench_review(
            {"entity_id": "test_entity", "files": [str(geo)]}, tmp_path
        )
