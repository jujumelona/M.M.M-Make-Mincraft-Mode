from __future__ import annotations

import pytest

from minecraft_mod_ai.source_edit_scalar_protocol_contract import (
    _reject_empty_java_materialization,
)
from minecraft_mod_ai.source_set_boundary_contract import (
    SourceSetBoundaryError,
    assert_server_safe_source_sets,
    source_set_boundary_errors,
)


class _Runtime:
    class AgentToolRuntimeError(RuntimeError):
        pass


def test_source_edit_rejects_empty_java_materialization() -> None:
    patch = {
        "project_root": ".",
        "operations": [
            {
                "operation": "replace",
                "path": "src/main/java/dev/mmm/debugfixture/DebugToken.java",
                "content": "",
            }
        ],
    }

    with pytest.raises(
        _Runtime.AgentToolRuntimeError,
        match="may not materialize an empty source file",
    ):
        _reject_empty_java_materialization(_Runtime, patch)


def test_source_edit_allows_explicit_java_delete() -> None:
    patch = {
        "project_root": ".",
        "operations": [
            {
                "operation": "delete",
                "path": "src/main/java/dev/mmm/debugfixture/DebugToken.java",
                "expected_sha256": "sha256:deadbeef",
            }
        ],
    }

    _reject_empty_java_materialization(_Runtime, patch)


def test_source_set_preflight_rejects_empty_java_source(tmp_path) -> None:
    project = tmp_path / "project"
    target = project / "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    target.parent.mkdir(parents=True)
    target.write_text(" \n\t", encoding="utf-8")

    errors = source_set_boundary_errors(project)

    assert errors == (
        "src/main/java/dev/mmm/debugfixture/DebugToken.java: common Java source is empty",
    )
    with pytest.raises(SourceSetBoundaryError, match="Java source is empty"):
        assert_server_safe_source_sets(project)
