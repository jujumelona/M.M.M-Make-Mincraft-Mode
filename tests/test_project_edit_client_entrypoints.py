from __future__ import annotations

import json

from minecraft_mod_ai.project_edit import (
    ensure_fabric_client_entrypoints,
    inspect_fabric_project,
)


def test_generated_client_entrypoints_are_merged_atomically(tmp_path) -> None:
    root = tmp_path / "project"
    main_java = root / "src/main/java/example/ExampleMod.java"
    metadata = root / "src/main/resources/fabric.mod.json"
    main_java.parent.mkdir(parents=True)
    metadata.parent.mkdir(parents=True)
    main_java.write_text(
        "package example;\npublic final class ExampleMod {}\n",
        encoding="utf-8",
    )
    metadata.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "id": "example",
                "entrypoints": {
                    "main": ["example.ExampleMod"],
                    "client": ["example.client.ExistingClient"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    info = inspect_fabric_project(root)
    first = ensure_fabric_client_entrypoints(
        info,
        entrypoints=(
            "example.client.generated.MarketClient",
            "example.client.generated.BlueprintClient",
        ),
    )
    second = ensure_fabric_client_entrypoints(
        info,
        entrypoints=("example.client.generated.MarketClient",),
    )

    assert first["status"] == "UPDATED"
    assert second["status"] == "UNCHANGED"
    saved = json.loads(metadata.read_text(encoding="utf-8"))
    assert saved["entrypoints"]["client"] == [
        "example.client.ExistingClient",
        "example.client.generated.BlueprintClient",
        "example.client.generated.MarketClient",
    ]
