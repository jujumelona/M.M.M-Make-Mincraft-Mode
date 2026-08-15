from __future__ import annotations

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from minecraft_mod_ai.remote_trajectory_store import (
    _manifest,
    _normalize_repo,
    _remote_hash_valid,
    _stamp_remote_record,
)
from minecraft_mod_ai.trajectory_memory import (
    append_trajectory,
    build_work_trajectory,
    execution_context_from_messages,
    relevant_trajectories,
)
from minecraft_mod_ai.trajectory_record_integrity import (
    record_remote_eligible,
    validate_trajectory_record,
)


def _task(*, minecraft_version: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": "custom_java",
        "members": [{"module_id": "demo"}],
    }
    if minecraft_version is not None:
        payload["minecraft_version"] = minecraft_version
        payload["loader"] = "fabric"
        payload["java_version"] = "21"
    return {
        "node_id": "repair-demo",
        "stage": "repair",
        "payload": payload,
    }


def _strong_row(*, minecraft_version: str | None = None) -> dict[str, object]:
    return build_work_trajectory(
        _task(minecraft_version=minecraft_version),
        outcome="SUCCESS",
        receipt={
            "build": {
                "commands": [
                    {"name": "clean_build", "exit_code": 0, "timed_out": False},
                    {"name": "gametest", "exit_code": 0, "timed_out": False},
                ]
            }
        },
    )


def _schema(name: str) -> dict[str, object]:
    return json.loads((Path("schemas") / name).read_text(encoding="utf-8"))


def test_v3_json_schema_accepts_local_and_sanitized_remote_records() -> None:
    schema = _schema("verified_trajectory_v3.schema.json")
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    row = _strong_row(minecraft_version="future-a")
    validator.validate(row)
    assert row["execution_context"] == {
        "java_version": "21",
        "loader": "fabric",
        "minecraft_version": "future-a",
    }
    assert validate_trajectory_record(row)

    remote = _stamp_remote_record(row)
    validator.validate(remote)
    assert _remote_hash_valid(remote)
    assert validate_trajectory_record(remote)
    assert record_remote_eligible(remote)


def test_remote_content_hash_rejects_mutation() -> None:
    remote = _stamp_remote_record(_strong_row())
    assert _remote_hash_valid(remote)
    remote["stage"] = "tampered"
    assert not _remote_hash_valid(remote)


def test_stored_eligibility_flags_are_not_trusted() -> None:
    row = _strong_row()
    tampered = copy.deepcopy(row)
    verification = tampered["verification"]
    assert isinstance(verification, dict)
    verification["remote_eligible"] = False
    remote = _stamp_remote_record(tampered)
    # The remote content hash is internally consistent, but the verifier-chain
    # re-derivation still rejects the forged cached eligibility flag.
    assert _remote_hash_valid(remote)
    assert not validate_trajectory_record(remote)
    assert not record_remote_eligible(remote)


def test_verified_experience_hard_rejects_incompatible_or_contextless_target(tmp_path) -> None:
    matching = _strong_row(minecraft_version="future-a")
    wrong_target = _strong_row(minecraft_version="future-b")
    contextless = _strong_row()
    assert append_trajectory(tmp_path, matching)
    assert append_trajectory(tmp_path, wrong_target)
    assert append_trajectory(tmp_path, contextless)

    rows = relevant_trajectories(
        tmp_path,
        "repair custom_java demo",
        task_class="repair",
        current_context={
            "target_version": "future-a",
            "loader": "fabric",
            "jdk_version": "21",
        },
        limit=8,
    )
    assert [row["trajectory_id"] for row in rows] == [matching["trajectory_id"]]


def test_execution_context_reads_only_structured_message_state() -> None:
    messages = [
        {
            "role": "system",
            "content": json.dumps(
                {
                    "platform_lock": {
                        "target_version": "future-a",
                        "loader": "fabric",
                        "mapping_version": "map-a",
                        "jdk_version": 21,
                    },
                    "diagnostic": {"error_code": "E100"},
                }
            ),
        },
        {
            "role": "assistant",
            "content": "minecraft_version=future-b loader=neoforge should never become host state",
        },
    ]
    assert execution_context_from_messages(messages) == {
        "error_code": "E100",
        "java_version": 21,
        "loader": "fabric",
        "mappings_version": "map-a",
        "minecraft_version": "future-a",
    }


def test_repository_links_normalize_to_backend_ids() -> None:
    assert _normalize_repo(
        "https://github.com/example/trajectory-memory.git",
        "github",
    ) == "example/trajectory-memory"
    assert _normalize_repo(
        "https://huggingface.co/datasets/example/trajectory-memory",
        "huggingface",
    ) == "example/trajectory-memory"
    assert _normalize_repo("example/trajectory-memory", "github") == "example/trajectory-memory"


def test_manifest_matches_declared_store_schema() -> None:
    schema = _schema("trajectory_store_manifest_v1.schema.json")
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(_manifest())
    assert _manifest()["shard_pattern"] == "memory/v3/{task_class}.jsonl"
    assert _manifest()["success_policy"] == "remote-success-requires-L3-or-higher"
