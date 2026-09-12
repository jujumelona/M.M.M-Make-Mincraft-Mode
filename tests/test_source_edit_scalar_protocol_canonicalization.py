from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_mod_ai.source_edit_scalar_protocol_contract import materialize_model_source_edit


class _Runtime:
    class AgentToolRuntimeError(RuntimeError):
        pass

    _MODEL_SOURCE_PREFIXES = (
        "src/main/java/",
        "src/main/resources/",
        "src/test/java/",
        "src/gametest/",
    )

    @staticmethod
    def _discover_model_project_root(workspace_root: str | Path) -> tuple[Path, str]:
        return Path(workspace_root).resolve(), "."


def test_create_file_collapses_expanded_compatibility_payload(tmp_path: Path) -> None:
    content = "DebugToken\n"
    payload = {
        "operation": "create_file",
        "path": "src/main/resources/debug-token.txt",
        "file": "src/main/resources/debug-token.txt",
        "target_path": "src/main/resources/debug-token.txt",
        "target_file": "src/main/resources/debug-token.txt",
        "content": content,
        "text": content,
        "new": content,
        "new_text": content,
        "new_content": content,
        "replacement": content,
        "code": content,
        "body": content,
        # Compatibility-expanded helper slots for other operations must not poison an
        # explicitly selected create_file action.
        "old": "adapter-helper",
        "old_text": "adapter-helper",
        "anchor": "adapter-helper",
        "count": 7,
        "package_name": "adapter.helper",
        "declaration": "AdapterHelper",
        "import_name": "adapter.Helper",
        "member": "adapter-helper",
    }

    result = materialize_model_source_edit(_Runtime, tmp_path, payload)

    assert result == {
        "project_root": ".",
        "operations": [
            {
                "operation": "create",
                "path": "src/main/resources/debug-token.txt",
                "content": content,
            }
        ],
    }


def test_create_file_accepts_genuinely_new_java_source(tmp_path: Path) -> None:
    content = (
        "package dev.mmm.debugfixture;\n\n"
        "public final class DebugToken {\n"
        "    private DebugToken() {}\n"
        "}\n"
    )

    result = materialize_model_source_edit(
        _Runtime,
        tmp_path,
        {
            "operation": "create_file",
            "path": "src/main/java/dev/mmm/debugfixture/DebugToken.java",
            "content": content,
        },
    )

    assert result == {
        "project_root": ".",
        "operations": [
            {
                "operation": "create",
                "path": "src/main/java/dev/mmm/debugfixture/DebugToken.java",
                "content": content,
            }
        ],
    }


def test_create_file_rejects_conflicting_content_aliases(tmp_path: Path) -> None:
    with pytest.raises(_Runtime.AgentToolRuntimeError, match="Conflicting.*content"):
        materialize_model_source_edit(
            _Runtime,
            tmp_path,
            {
                "operation": "create_file",
                "path": "src/main/resources/debug-token.txt",
                "content": "DebugToken\n",
                "replacement": "DifferentToken\n",
            },
        )


def test_create_file_still_rejects_unknown_fields(tmp_path: Path) -> None:
    with pytest.raises(
        _Runtime.AgentToolRuntimeError,
        match="Unknown model-facing source-write fields",
    ):
        materialize_model_source_edit(
            _Runtime,
            tmp_path,
            {
                "operation": "create_file",
                "path": "src/main/resources/debug-token.txt",
                "content": "DebugToken\n",
                "unexpected_adapter_field": "must stay fail-closed",
            },
        )


def test_standard_aliases_are_consumed_when_canonical_value_already_exists(
    tmp_path: Path,
) -> None:
    target = tmp_path / "src/main/resources/example.txt"
    target.parent.mkdir(parents=True)
    target.write_text("before\n", encoding="utf-8")

    result = materialize_model_source_edit(
        _Runtime,
        tmp_path,
        {
            "operation": "replace_exact",
            "path": "src/main/resources/example.txt",
            "target_path": "src/main/resources/example.txt",
            "old": "before\n",
            "old_text": "before\n",
            "new": "after\n",
            "new_text": "after\n",
            "replacement": "after\n",
        },
    )

    operation = result["operations"][0]
    assert operation["operation"] == "edit"
    assert operation["replacements"] == [
        {"old": "before\n", "new": "after\n", "count": 1}
    ]
