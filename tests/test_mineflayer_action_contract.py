from __future__ import annotations

import re
from pathlib import Path

from minecraft_mod_ai.mineflayer_bridge import MineflayerBridge


def _bridge_source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "integrations"
        / "mineflayer-1201"
        / "bridge.mjs"
    ).read_text(encoding="utf-8")


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
