from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from minecraft_mod_ai.blockbench_client import (
    BlockbenchMCPClient,
    BlockbenchMCPError,
    allowed_blockbench_operations,
)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:3000@evil.example/bb-mcp",
        "http://user@127.0.0.1:3000/bb-mcp",
        "http://127.0.0.1.evil.example:3000/bb-mcp",
        "http://2130706433:3000/bb-mcp",
        "http://[::2]:3000/bb-mcp",
        "ftp://127.0.0.1:3000/bb-mcp",
        "http://127.0.0.1:3000/bb-mcp#remote",
        "http://127.0.0.1:99999/bb-mcp",
        "http://127.0.0.1:3000\\@evil.example/bb-mcp",
    ],
)
def test_rejects_deceptive_or_non_loopback_urls(
    tmp_path: Path, url: str
) -> None:
    with pytest.raises(BlockbenchMCPError):
        BlockbenchMCPClient(url=url, workspace_root=tmp_path)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:3000/bb-mcp",
        "https://localhost:3443/bb-mcp",
        "http://[::1]:3000/bb-mcp",
    ],
)
def test_accepts_only_explicit_loopback_forms(tmp_path: Path, url: str) -> None:
    client = BlockbenchMCPClient(url=url, workspace_root=tmp_path)
    try:
        assert client.url == url
        assert client.workspace_root == tmp_path.resolve()
    finally:
        client.close()


def _ready_client(tmp_path: Path) -> BlockbenchMCPClient:
    client = BlockbenchMCPClient(workspace_root=tmp_path)
    client.session_id = "test-session"
    client.list_tools = lambda: [  # type: ignore[method-assign]
        {"name": name} for name in allowed_blockbench_operations()
    ]
    return client


def test_normalizes_reviewed_paths_before_transport(tmp_path: Path) -> None:
    project = tmp_path / "models" / "entity.bbmodel"
    project.parent.mkdir()
    project.write_text("{}", encoding="utf-8")
    client = _ready_client(tmp_path)
    payloads: list[dict[str, Any]] = []

    def record(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        payloads.append(payload)
        return {"content": []}, {}

    client._post = record  # type: ignore[method-assign]
    arguments = {"path": "models/entity.bbmodel"}
    try:
        client.call("open_project", arguments)
    finally:
        client.close()

    assert arguments == {"path": "models/entity.bbmodel"}
    sent = payloads[-1]["params"]["arguments"]
    assert sent == {"path": str(project.resolve())}


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        ("open_project", {"path": "../outside.bbmodel"}),
        ("render_preview", {"output_path": "../outside.png"}),
        ("export_bbmodel", {"output_path": "../outside.bbmodel"}),
        (
            "export_geckolib",
            {"model_output_path": "../outside.geo.json"},
        ),
    ],
)
def test_rejects_input_and_output_path_traversal_before_transport(
    tmp_path: Path,
    operation: str,
    arguments: dict[str, Any],
) -> None:
    client = BlockbenchMCPClient(workspace_root=tmp_path)
    client.initialize = lambda: pytest.fail(  # type: ignore[method-assign]
        "transport must not run for a rejected path"
    )
    try:
        with pytest.raises(BlockbenchMCPError, match="escapes"):
            client.call(operation, arguments)
    finally:
        client.close()


def test_rejects_symlink_escape_for_inputs_and_outputs(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir(exist_ok=True)
    outside_model = outside / "outside.bbmodel"
    outside_model.write_text("{}", encoding="utf-8")
    input_link = tmp_path / "input.bbmodel"
    output_link = tmp_path / "output-link"
    try:
        input_link.symlink_to(outside_model)
        output_link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks are unavailable on this host: {exc}")

    client = BlockbenchMCPClient(workspace_root=tmp_path)
    client.initialize = lambda: pytest.fail(  # type: ignore[method-assign]
        "transport must not run for a symlink escape"
    )
    try:
        with pytest.raises(BlockbenchMCPError, match="escapes"):
            client.call("open_project", {"path": str(input_link)})
        with pytest.raises(BlockbenchMCPError, match="escapes"):
            client.call(
                "render_preview",
                {"output_path": str(output_link / "preview.png")},
            )
    finally:
        client.close()


def test_workspace_root_can_be_configured_by_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MMM_BLOCKBENCH_WORKSPACE_ROOT", str(tmp_path))
    client = BlockbenchMCPClient()
    try:
        assert client.workspace_root == tmp_path.resolve()
    finally:
        client.close()


@pytest.mark.parametrize(
    ("operation", "arguments", "message"),
    [
        ("validate_uv", {"unexpected": True}, "unsupported arguments"),
        ("close_project", {"save": True}, "unsupported arguments"),
        (
            "create_cube",
            {
                "name": "body",
                "from": [0, 0, 0],
                "to": [1, 1, 1],
                "script": "dangerous()",
            },
            "unsupported arguments",
        ),
        (
            "create_animation",
            {
                "name": "idle",
                "bones": {
                    "body": [
                        {
                            "time": 0,
                            "rotation": [0, 0, 0],
                            "execute": "dangerous()",
                        }
                    ]
                },
            },
            "unsupported arguments",
        ),
        (
            "set_uv",
            {"element": "body", "uv": [0, 0, 16]},
            "exactly 4",
        ),
        (
            "export_bbmodel",
            {
                "path": "one.bbmodel",
                "output_path": "two.bbmodel",
            },
            "exactly one",
        ),
    ],
)
def test_operation_schemas_are_closed_and_checked_before_transport(
    tmp_path: Path,
    operation: str,
    arguments: dict[str, Any],
    message: str,
) -> None:
    client = BlockbenchMCPClient(workspace_root=tmp_path)
    client.initialize = lambda: pytest.fail(  # type: ignore[method-assign]
        "transport must not run for invalid arguments"
    )
    try:
        with pytest.raises(BlockbenchMCPError, match=message):
            client.call(operation, arguments)
    finally:
        client.close()


def test_valid_closed_schema_call_keeps_output_inside_workspace(
    tmp_path: Path,
) -> None:
    client = _ready_client(tmp_path)
    payloads: list[dict[str, Any]] = []

    def record(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        payloads.append(payload)
        return {"content": []}, {}

    client._post = record  # type: ignore[method-assign]
    try:
        client.call(
            "create_cube",
            {
                "name": "body",
                "from": [0, 0, 0],
                "to": [4, 8, 4],
                "origin": [2, 4, 2],
                "mirror": False,
                "uv": {"north": [0, 0, 4, 8]},
            },
        )
        client.call(
            "render_preview",
            {
                "output_path": "previews/body.png",
                "width": 512,
                "height": 512,
                "background": "#00000000",
            },
        )
    finally:
        client.close()

    assert len(payloads) == 2
    preview_path = Path(
        payloads[-1]["params"]["arguments"]["output_path"]
    )
    assert preview_path == (tmp_path / "previews" / "body.png").resolve()
    preview_path.relative_to(tmp_path.resolve())


def test_non_object_arguments_are_rejected_before_transport(
    tmp_path: Path,
) -> None:
    client = BlockbenchMCPClient(workspace_root=tmp_path)
    client.initialize = lambda: pytest.fail(  # type: ignore[method-assign]
        "transport must not run for invalid arguments"
    )
    try:
        with pytest.raises(BlockbenchMCPError, match="must be an object"):
            client.call("validate_uv", [])  # type: ignore[arg-type]
    finally:
        client.close()
