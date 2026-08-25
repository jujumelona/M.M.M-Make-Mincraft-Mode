from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from minecraft_mod_ai.custom_module_generator import (
    CustomModuleGenerator,
    _output_exhaustion_continuation_messages,
)
from minecraft_mod_ai.complete_spec import ProductionModule


def test_output_exhaustion_continuation_messages_structure() -> None:
    """Verify that continuation messages carry forward preserved source state SHA and paths."""
    module = ProductionModule(
        module_id="custom_sword",
        kind="item",
        config={"name": "Custom Sword"},
        depends_on=(),
        required_gates=(),
    )

    messages = _output_exhaustion_continuation_messages(
        module=module,
        minecraft_version="1.20.1",
        loader="fabric",
        mappings="yarn",
        java_version=17,
        continuation_index=1,
        state_sha256="sha256:abc123state",
        touched_paths=("src/main/java/com/example/CustomSwordItem.java",),
        discarded_paths=(),
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    content = json.loads(messages[1]["content"])
    assert content["phase"] == "implement_module"
    assert content["continuation"]["continuation_index"] == 1
    assert content["continuation"]["preserved_source_state_sha256"] == "sha256:abc123state"
    assert "src/main/java/com/example/CustomSwordItem.java" in content["continuation"]["preserved_paths_preview"]


def test_custom_module_generator_continuation_preserves_checkpoints(tmp_path: Path) -> None:
    """Verify that CustomModuleGenerator initializes with checkpoint root and preserves state."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    checkpoint_dir = tmp_path / "checkpoints"

    router = MagicMock()
    router.profile = "t4_local"

    generator = CustomModuleGenerator(
        router,
        checkpoint_root=checkpoint_dir,
    )

    assert generator._checkpoint_root == checkpoint_dir.resolve()
