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
        / "mineflayer-1201"
        / "bridge.mjs"
    )


def _bridge_source() -> str:
    return _bridge_path().read_text(encoding="utf-8")


def test_every_python_allowlisted_action_has_node_dispatch_case() -> None:
    source = _bridge_source()
    dispatched = set(re.findall(r'case\s+"([a-z_]+)"\s*:', source))
    assert MineflayerBridge.ACTIONS <= dispatched


def test_required_runtime_assertion_is_real_and_bounded() -> None:
    source = _bridge_source()
    assert 'case "wait_for"' in source
    assert 'Unsupported wait_for condition' in source
    assert 'timeout_ms' in source
    assert '30000' in source
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
