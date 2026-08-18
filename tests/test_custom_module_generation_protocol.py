from __future__ import annotations

import pytest

from minecraft_mod_ai.custom_module_generator import (
    CustomModuleGenerationError,
    CustomModuleGenerator,
    _canonicalize_generation_payload,
    _generation_fragment_action,
    _repair_generation_messages,
)


def test_final_metadata_only_completion_accepts_accumulated_operations() -> None:
    payload = _canonicalize_generation_payload(
        {
            "operations": [],
            "complete": True,
            "next_cursor": "",
            "context_page_complete": True,
        }
    )

    assert (
        _generation_fragment_action(
            payload,
            is_last_page=True,
            has_accumulated_operations=True,
            current_cursor="",
            seen_cursors=set(),
        )
        == "page_complete"
    )


def test_metadata_only_advancing_cursor_is_protocol_progress() -> None:
    payload = _canonicalize_generation_payload(
        {
            "operations": [],
            "complete": False,
            "next_cursor": "cursor-2",
            "context_page_complete": False,
        }
    )

    assert (
        _generation_fragment_action(
            payload,
            is_last_page=False,
            has_accumulated_operations=False,
            current_cursor="cursor-1",
            seen_cursors={"cursor-1"},
        )
        == "cursor"
    )


def test_range_only_fragment_fails_without_becoming_fake_completion() -> None:
    payload = _canonicalize_generation_payload({"start": 0, "end": 4096})

    with pytest.raises(CustomModuleGenerationError, match="no protocol progress"):
        _generation_fragment_action(
            payload,
            is_last_page=True,
            has_accumulated_operations=False,
            current_cursor="",
            seen_cursors=set(),
        )


def test_repeated_cursor_is_rejected() -> None:
    payload = _canonicalize_generation_payload(
        {
            "operations": [],
            "complete": False,
            "next_cursor": "same-cursor",
            "context_page_complete": False,
        }
    )

    with pytest.raises(CustomModuleGenerationError, match="repeated next_cursor"):
        _generation_fragment_action(
            payload,
            is_last_page=False,
            has_accumulated_operations=True,
            current_cursor="same-cursor",
            seen_cursors={"same-cursor"},
        )


def test_root_fabric_metadata_create_is_canonicalized_before_scope_validation() -> None:
    payload = _canonicalize_generation_payload(
        {
            "operations": [
                {
                    "operation": "create",
                    "path": "fabric.mod.json",
                    "content": "{}",
                }
            ],
            "complete": True,
        }
    )

    operation = payload["operations"][0]
    assert operation["path"] == "src/main/resources/fabric.mod.json"
    generator = object.__new__(CustomModuleGenerator)
    generator._validate_operations([operation])


def test_non_create_root_fabric_metadata_is_not_rewritten() -> None:
    payload = _canonicalize_generation_payload(
        {
            "operations": [
                {
                    "operation": "replace",
                    "path": "fabric.mod.json",
                    "expected_sha256": "0" * 64,
                    "content": "{}",
                }
            ],
            "complete": True,
        }
    )

    generator = object.__new__(CustomModuleGenerator)
    with pytest.raises(CustomModuleGenerationError, match="outside the .* scope"):
        generator._validate_operations(payload["operations"])


def test_repair_history_omits_invalid_assistant_payload() -> None:
    base = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "stable request"},
    ]
    repaired = _repair_generation_messages(base, "received {start,end}")

    assert repaired[:2] == base
    assert [message["role"] for message in repaired] == ["system", "user", "user"]
    assert all(message["role"] != "assistant" for message in repaired)
    assert "invalid assistant payload is intentionally omitted" in repaired[-1]["content"]


def test_empty_final_page_without_any_operation_is_rejected() -> None:
    payload = _canonicalize_generation_payload(
        {
            "operations": [],
            "complete": True,
            "context_page_complete": True,
        }
    )

    with pytest.raises(CustomModuleGenerationError, match="before any patch operation"):
        _generation_fragment_action(
            payload,
            is_last_page=True,
            has_accumulated_operations=False,
            current_cursor="",
            seen_cursors=set(),
        )
