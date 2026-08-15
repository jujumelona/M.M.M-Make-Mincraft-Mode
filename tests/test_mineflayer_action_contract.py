from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from minecraft_mod_ai.mineflayer_bridge import MineflayerBridge


def _bridge_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "integrations"
        / "mineflayer"
        / "bridge.mjs"
    )


def _bridge_source() -> str:
    return _bridge_path().read_text(encoding="utf-8")


def _dispatched_actions(source: str) -> set[str]:
    match = re.search(r"const actions = \{(?P<body>.*?)\n\};", source, re.DOTALL)
    assert match is not None, "Mineflayer action registry is missing"
    dispatched: set[str] = set()
    for entry in match.group("body").split(","):
        token = entry.strip()
        if not token:
            continue
        dispatched.add(token.split(":", 1)[0].strip())
    return dispatched


def test_python_and_node_action_registries_match_exactly() -> None:
    assert _dispatched_actions(_bridge_source()) == set(MineflayerBridge.ACTIONS)


def test_required_runtime_assertion_is_real_and_bounded() -> None:
    source = _bridge_source()
    assert "async function waitFor" in source
    assert "wait_for: waitFor" in source
    assert "Unsupported wait_for condition" in source
    assert "timeout_ms" in source
    assert "30000" in source
    for condition in (
        "inventory_contains",
        "held_item",
        "health",
        "food",
        "position_near",
        "block_at",
        "entity_present",
        "window_open",
    ):
        assert f'type === "{condition}"' in source


def test_mineflayer_bridge_javascript_parses_when_node_is_available() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed on this test host")
    completed = subprocess.run(
        [node, "--check", str(_bridge_path())],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
