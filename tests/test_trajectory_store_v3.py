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
from minecraft_mod_ai.trajectory_memory import build_work_trajectory
from minecraft_mod_ai.trajectory_record_integrity import (
    record_remote_eligible,
    validate_trajectory_record,
)


def _task() -> dict[str, object]:
    return {
        "node_id": "repair-demo",
        "stage": "repair",
        "payload": {"kind": "custom_java", "members": [{"module_id": "demo"}]},
    }


def _strong_row() -> dict[str, object]:
    return build_work_trajectory(
        _task(),
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

    row = _strong_row()
    validator.validate(row)
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
