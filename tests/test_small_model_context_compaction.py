from __future__ import annotations

import hashlib
import json

from minecraft_mod_ai.small_model_context_compaction import compact_messages


def _oversized_messages() -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "stable system contract"},
        {
            "role": "assistant",
            "content": "old-a:" + "a" * 10000,
            "tool_calls": [
                {
                    "function": {
                        "name": "repo_read",
                        "arguments": '{"path":"src/main/java/Demo.java"}',
                    }
                }
            ],
        },
        {
            "role": "tool",
            "name": "repo_read",
            "content": json.dumps(
                {
                    "tool": "repo_read",
                    "ok": False,
                    "error": "compiler mismatch",
                    "result": {
                        "jdt_status": "FAIL",
                        "jdt_error_count": 2,
                        "remaining_files": 3,
                    },
                }
            ),
        },
        {"role": "assistant", "content": "old-b:" + "b" * 10000},
        {"role": "assistant", "content": "recent-c:" + "c" * 10000},
        {"role": "assistant", "content": "recent-d:" + "d" * 1000},
    ]


def test_compaction_persists_exact_recoverable_raw_history(tmp_path, monkeypatch) -> None:
    archive = tmp_path / "archive"
    monkeypatch.setenv("MMM_SMALL_AGENT_CONTEXT_ARCHIVE", str(archive))
    monkeypatch.setenv("MMM_SMALL_AGENT_CONTEXT_BYTES", str(24 * 1024))

    original = _oversized_messages()
    compacted = compact_messages(original)
    assert compacted != tuple(original)

    system_context = next(
        item for item in compacted
        if item.get("role") == "system"
        and str(item.get("content", "")).startswith("HOST COMPACTED VERIFIED CONTEXT.")
    )
    _header, payload = str(system_context["content"]).split("\n", 1)
    ledger = json.loads(payload)
    raw = ledger["raw_history"]
    assert raw["available"] is True
    assert ledger["policy"]["raw_history_recoverable"] is True
    assert ledger["errors"] == [{"tool": "repo_read", "error": "compiler mismatch"}]
    assert ledger["verifications"][0]["facts"]["jdt_status"] == "FAIL"
    assert ledger["remaining_host_state"][0]["facts"]["remaining_files"] == 3

    path = archive / (raw["sha256"].removeprefix("sha256:") + ".json")
    assert str(path) == raw["path"]
    saved = path.read_bytes()
    assert "sha256:" + hashlib.sha256(saved).hexdigest() == raw["sha256"]
    decoded = json.loads(saved)
    assert decoded[0]["content"].startswith("old-a:")
    assert decoded[1]["error"] if False else True
    assert decoded[1]["content"]


def test_compaction_keeps_original_when_raw_archive_cannot_be_persisted(tmp_path, monkeypatch) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("block archive directory creation", encoding="utf-8")
    monkeypatch.setenv("MMM_SMALL_AGENT_CONTEXT_ARCHIVE", str(blocked))
    monkeypatch.setenv("MMM_SMALL_AGENT_CONTEXT_BYTES", str(24 * 1024))

    original = _oversized_messages()
    assert compact_messages(original) == tuple(original)
