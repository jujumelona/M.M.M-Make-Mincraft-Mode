from pathlib import Path

from minecraft_mod_ai.api import CompleteChatReply
from minecraft_mod_ai.conversation import merge_design_brief
from minecraft_mod_ai.mcp_server import _complete_plan_response
from minecraft_mod_ai.mcp_tools import MMMToolService


def test_complete_brief_keeps_the_original_and_natural_revision() -> None:
    merged = merge_design_brief(
        "농사와 요리 모드를 만들어줘.",
        "전투는 빼고 계절 시스템을 추가해줘.",
    )

    assert merged.startswith("농사와 요리 모드를 만들어줘.")
    assert merged.endswith("전투는 빼고 계절 시스템을 추가해줘.")
    assert "approval" not in merged.lower()
    assert "sha256" not in merged.lower()


def test_complete_tool_revision_replans_the_merged_brief(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    service = MMMToolService(workspace_root=tmp_path)

    def fake_plan(
        prompt: str,
        media_paths=(),
        existing_input_sha256: str = "",
    ):
        captured["prompt"] = prompt
        captured["media_paths"] = tuple(media_paths)
        captured["existing_input_sha256"] = existing_input_sha256
        return {"message": "updated"}

    monkeypatch.setattr(service, "plan_complete_game", fake_plan)
    result = service.revise_complete_plan(
        "처음 요구",
        "맵은 넣지 마",
        ["reference.png"],
        "sha256:" + "a" * 64,
    )

    assert result == {"message": "updated"}
    assert captured["prompt"] == "처음 요구\n\nUser revision:\n맵은 넣지 마"
    assert captured["media_paths"] == ("reference.png",)


def test_frontdoor_returns_only_the_natural_plan_message() -> None:
    internal = {
        "message": "플레이 흐름을 이렇게 만들겠습니다.",
        "complete_proposal": {"secret": "execution state"},
        "approval_hash": "sha256:" + "a" * 64,
    }

    assert _complete_plan_response(
        internal,
        stage="frontdoor",
    ) == "플레이 흐름을 이렇게 만들겠습니다."
    assert _complete_plan_response(internal, stage="planning") is internal


def test_complete_chat_reply_repr_hides_execution_state() -> None:
    reply = CompleteChatReply(
        message="이 방향으로 만들까요?",
        approval_hash="sha256:" + "b" * 64,
        complete_proposal=object(),  # type: ignore[arg-type]
    )

    rendered = repr(reply)
    assert "이 방향으로 만들까요?" in rendered
    assert "sha256" not in rendered
    assert "complete_proposal" not in rendered
