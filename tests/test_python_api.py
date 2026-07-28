"""Public Python API contract for conversational mod-building sessions.

The intended API is deliberately smaller than the internal Proposal/Pipeline
surface:

* ``ModAISession`` is importable from the package root.
* ``plan()`` starts and ``revise()`` continues the same natural-language brief;
* a plan reply exposes ``proposal``, ``buildable``, ``questions``, and
  user-facing ``message`` attributes;
* ``build(source_only=True)`` executes the latest buildable proposal in a
  fresh run directory.

These tests are written before the public session implementation.  They should
not be weakened by exposing approval hashes or requiring callers to manipulate
Proposal JSON.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from minecraft_mod_ai import ModAISession
from minecraft_mod_ai.planner import OpenAICompatiblePlanner


def test_natural_language_plan_and_revision_share_the_current_brief(
    tmp_path: Path,
) -> None:
    session = ModAISession(output_root=tmp_path / "outputs")

    first = session.plan("Create one frost item.")
    assert first.buildable is True
    assert first.questions == ()
    assert [content.kind.value for content in first.proposal.spec.contents] == [
        "item"
    ]
    assert first.proposal.spec.boss is None
    assert first.proposal.spec.arena is None

    revised = session.revise("Also add one block.")
    assert revised.buildable is True
    assert revised.questions == ()
    assert {content.kind.value for content in revised.proposal.spec.contents} == {
        "item",
        "block",
    }
    assert revised.proposal.spec.boss is None
    assert revised.proposal.spec.arena is None
    assert "Create one frost item." in revised.proposal.requested_prompt
    assert "Also add one block." in revised.proposal.requested_prompt
    assert isinstance(revised.message, str) and revised.message


def test_vague_skill_and_map_request_is_not_buildable_and_forces_no_boss(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "must-remain-empty"
    session = ModAISession(output_root=output_root)

    reply = session.plan(
        "Make a mod with a teleport skill and a large open-world map."
    )

    assert reply.buildable is False
    assert reply.proposal.spec.contents == ()
    assert reply.proposal.spec.boss is None
    assert reply.proposal.spec.arena is None
    assert {
        request.capability for request in reply.proposal.deferred_requests
    } >= {"skill_system", "field_map"}
    assert reply.questions or reply.proposal.deferred_requests

    with pytest.raises(ValueError, match="build|clarif|supported|ready|구현|계획"):
        session.build(source_only=True)
    assert not output_root.exists()


def test_explicit_supported_request_can_build_source_twice_in_unique_runs(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    session = ModAISession(output_root=output_root)
    reply = session.plan("Create one frost item and one frost block.")
    assert reply.buildable is True

    first = session.build(source_only=True)
    second = session.build(source_only=True)

    assert first.status == second.status == "SOURCE_READY"
    assert first.build_status == second.build_status == "NOT_RUN"
    assert first.gametest_status == second.gametest_status == "NOT_RUN"
    assert first.jar_path is second.jar_path is None
    assert Path(first.project_root).is_dir()
    assert Path(second.project_root).is_dir()
    assert Path(first.release_zip).is_file()
    assert Path(second.release_zip).is_file()
    assert Path(first.project_root) != Path(second.project_root)
    assert Path(first.release_zip) != Path(second.release_zip)
    assert output_root.resolve() in Path(first.project_root).resolve().parents
    assert output_root.resolve() in Path(second.project_root).resolve().parents


def test_unsupported_minecraft_version_fails_before_creating_output(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "unsupported-output"

    with pytest.raises(ValueError, match="1\\.20\\.1|1\\.21\\.4|unsupported|지원"):
        ModAISession(
            output_root=output_root,
            minecraft_version="1.21.4",
        )

    assert not output_root.exists()


class _MockHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_MockHTTPResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _limit: int = -1) -> bytes:
        return self._body


def test_openai_compatible_planner_filters_unrequested_content_and_never_leaks_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = "sk-test-DO-NOT-LEAK-7d4fdd9a"
    captured: dict[str, Any] = {}
    model_plan = {
        "mod_id": "external_block_plan",
        "mod_name": "External Block Plan",
        "package_name": "ai.minecraft.generated.external_block_plan",
        "summary": "One explicitly requested block.",
        "contents": [
            {
                "content_id": "invented_item",
                "kind": "item",
                "display_name_en": "Invented Item",
                "display_name_ko": "Invented Item",
                "color": "#ff0000",
                "recipe": True,
            },
            {
                "content_id": "requested_block",
                "kind": "block",
                "display_name_en": "Requested Block",
                "display_name_ko": "Requested Block",
                "color": "#00ff00",
                "recipe": True,
            },
        ],
        # The remote model is not allowed to turn an unrequested map into even
        # a deferred capability. Deferred work must originate in the user brief.
        "deferred_capabilities": ["custom_map"],
    }

    def fake_urlopen(
        request: urllib.request.Request,
        *,
        timeout: int,
    ) -> _MockHTTPResponse:
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["body"] = request.data
        captured["timeout"] = timeout
        return _MockHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(model_plan),
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    planner = OpenAICompatiblePlanner(
        base_url="https://planner.example.test/v1",
        model="strict-planner",
        api_key=api_key,
        timeout_seconds=15,
    )
    session = ModAISession(
        output_root=tmp_path / "external-output",
        planner=planner,
    )

    reply = session.plan("Create exactly one green block.")

    assert captured["url"] == "https://planner.example.test/v1/chat/completions"
    assert captured["authorization"] == f"Bearer {api_key}"
    assert captured["timeout"] == 15
    assert api_key.encode() not in captured["body"]
    assert api_key not in json.dumps(
        reply.proposal.to_dict(),
        ensure_ascii=False,
    )
    assert api_key not in reply.message
    assert api_key not in repr(session)
    assert api_key not in repr(reply)

    assert [
        (content.content_id, content.kind.value)
        for content in reply.proposal.spec.contents
    ] == [("requested_block", "block")]
    assert reply.proposal.spec.boss is None
    assert reply.proposal.spec.arena is None
    assert "custom_map" not in {
        request.capability for request in reply.proposal.deferred_requests
    }
    assert reply.buildable is True
