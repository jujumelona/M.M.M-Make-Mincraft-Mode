from __future__ import annotations

import pytest

from minecraft_mod_ai.source_patch import (
    SourcePatchError,
    TransactionalSourcePatcher,
    sha256_file,
)


def test_known_cross_operation_fields_are_canonicalized(tmp_path) -> None:
    patcher = TransactionalSourcePatcher(tmp_path)
    target = tmp_path / "src" / "Example.java"

    patcher.apply(
        [
            {
                "operation": "create",
                "path": "src/Example.java",
                "content": "alpha\n",
                "expected_sha256": "copied-from-mixed-contract",
                "replacements": [{"old": "unused", "new": "unused"}],
            }
        ]
    )
    assert target.read_text(encoding="utf-8") == "alpha\n"

    patcher.apply(
        [
            {
                "operation": "replace",
                "path": "src/Example.java",
                "expected_sha256": sha256_file(target),
                "content": "beta\n",
                "replacements": [{"old": "unused", "new": "unused"}],
            }
        ]
    )
    assert target.read_text(encoding="utf-8") == "beta\n"

    patcher.apply(
        [
            {
                "operation": "edit",
                "path": "src/Example.java",
                "expected_sha256": sha256_file(target),
                "replacements": [{"old": "beta", "new": "gamma", "count": 1}],
                "content": "copied-from-mixed-contract",
            }
        ]
    )
    assert target.read_text(encoding="utf-8") == "gamma\n"

    patcher.apply(
        [
            {
                "operation": "delete",
                "path": "src/Example.java",
                "expected_sha256": sha256_file(target),
                "content": "copied-from-mixed-contract",
                "replacements": [{"old": "unused", "new": "unused"}],
            }
        ]
    )
    assert not target.exists()


def test_arbitrary_unknown_fields_remain_strictly_rejected(tmp_path) -> None:
    patcher = TransactionalSourcePatcher(tmp_path)

    with pytest.raises(SourcePatchError, match="Unknown create fields"):
        patcher.apply(
            [
                {
                    "operation": "create",
                    "path": "Example.java",
                    "content": "alpha\n",
                    "unexpected": "must-not-be-silently-dropped",
                }
            ]
        )


def test_canonicalization_does_not_synthesize_required_fields(tmp_path) -> None:
    patcher = TransactionalSourcePatcher(tmp_path)

    with pytest.raises(SourcePatchError, match="Create content must be text"):
        patcher.apply(
            [
                {
                    "operation": "create",
                    "path": "Example.java",
                    "expected_sha256": "copied-from-mixed-contract",
                    "replacements": [{"old": "unused", "new": "unused"}],
                }
            ]
        )
