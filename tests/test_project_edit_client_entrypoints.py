from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

from minecraft_mod_ai.artifact_job import (
    ArtifactJob,
    canonical_client_entrypoints,
)
from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator
from minecraft_mod_ai.implementation_identity import ExecutorType
from minecraft_mod_ai.project_edit import (
    ensure_fabric_client_entrypoints,
    inspect_fabric_project,
)
from minecraft_mod_ai.validator import _complete_client_required


def _canonical_client_job() -> ArtifactJob:
    return ArtifactJob(
        job_id="shipyard.screen_handler",
        template_id="",
        owner_module="shipyard",
        executor_type=ExecutorType.PYTHON_GENERATOR,
        deterministic_inputs={
            "_canonical_inputs": {
                "screen": {
                    "side": "CLIENT",
                    "bindings": {
                        "package_name": "example.client.generated",
                        "class_name": "ShipyardScreen",
                    },
                }
            }
        },
    )


def test_canonical_artifact_job_is_client_approval_source_of_truth() -> None:
    job = _canonical_client_job()

    assert canonical_client_entrypoints((job,)) == (
        "example.client.generated.ShipyardScreen",
    )
    proposal = SimpleNamespace(
        modules=(),
        game_design={"_artifact_jobs": [job.to_dict()]},
    )
    assert _complete_client_required(proposal) is True


def test_client_metadata_mutation_runs_inside_generation_work_node() -> None:
    source = inspect.getsource(
        CompleteProductionOrchestrator._execute_generation_work
    )
    node_action = source.index("def module_node_action(")
    metadata_edit = source.index("ensure_fabric_client_entrypoints(")
    assert node_action < metadata_edit
    assert '"src/main/resources/fabric.mod.json"' in source
    assert '"module_ids": metadata_owner_ids' in source
    assert "client_metadata_owner_ids" not in source


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
