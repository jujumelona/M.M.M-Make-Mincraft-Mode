from types import SimpleNamespace

from minecraft_mod_ai.cli import _build_parser, _render_complete_result


def test_execute_does_not_require_an_approval_hash() -> None:
    args = _build_parser().parse_args(["execute", "saved-plan.json"])

    assert args.approve is None
    assert args.json is False


def test_default_execution_result_is_a_short_human_summary() -> None:
    text = _render_complete_result(
        SimpleNamespace(
            status="SOURCE_READY",
            project_root="/content/mmm-output/my-mod",
            release_zip="/content/mmm-output/my-mod.zip",
            jar_path=None,
            unresolved_gates=(),
            run_resumed=True,
        )
    )

    assert "제작 상태: SOURCE_READY" in text
    assert "다운로드 ZIP:" in text
    assert "이어" in text
    assert "sha256" not in text.lower()
    assert '"status"' not in text
