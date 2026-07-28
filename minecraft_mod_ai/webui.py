from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .importer import ExistingProjectReport, inspect_existing_project_archive
from .pipeline import MinecraftModPipeline
from .planner import HeuristicPlanner, LocalTransformersPlanner
from .spec import Proposal


def _existing_input_markdown(report: ExistingProjectReport | None) -> str:
    if report is None:
        return "### 새 모드 만들기"
    source_note = (
        "소스가 포함된 ZIP입니다."
        if report.has_sources
        else "소스가 없는 ZIP입니다."
    )
    return (
        "### 기존 모드 수정\n\n"
        f"- 파일: `{report.archive_name}`\n"
        f"- 모드: `{report.mod_id or '확인 불가'}` / "
        f"`{report.mod_version or '버전 확인 불가'}`\n"
        f"- 파일 수: `{report.file_count}`\n\n"
        f"{source_note}"
    )


def create_demo(
    *,
    output_root: Path,
    local_model: bool = False,
    existing_input: str | Path | None = None,
) -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("UI extras are missing. Install with: pip install -e '.[ui]'") from exc

    output_root = output_root.resolve()
    existing_path = Path(existing_input).resolve() if existing_input is not None else None
    existing_report = (
        inspect_existing_project_archive(existing_path)
        if existing_path is not None
        else None
    )
    planner = LocalTransformersPlanner() if local_model else HeuristicPlanner()
    pipeline = MinecraftModPipeline(planner=planner)

    def make_plan(prompt: str) -> tuple[dict[str, object], str, str, str]:
        proposal = pipeline.plan(prompt, existing_input=existing_path)
        proposal_json = json.dumps(proposal.to_dict(), ensure_ascii=False)
        coverage = ["아이템·블록·조합법·번역"]
        if proposal.spec.boss:
            coverage.extend(
                [
                    "보스·bossbar·loot·spawn egg·GameTest",
                    "편집용 bbmodel/OBJ·엔티티 텍스처·런타임 biped renderer",
                ]
            )
        if proposal.spec.arena:
            coverage.append("결정론적 arena 함수·WorldDesignIR·경로 미리보기")
        coverage.append("실제 Gradle/JAR 검증")
        if existing_report is not None:
            coverage.append("기존 입력 inventory·snapshot 재검사")
        mode = "기존 모드 수정" if existing_report is not None else "새 모드"
        status = (
            f"**{mode} 계획을 만들었습니다.**\n\n"
            f"범위: {' · '.join(coverage)}\n\n"
            "내용을 확인하고 표시된 승인 해시를 아래 칸에 붙여 넣으세요."
        )
        return proposal.to_dict(), proposal.approval_hash, proposal_json, status

    def execute_plan(
        proposal_json: str,
        typed_hash: str,
        source_only: bool,
    ) -> tuple[str, str | None]:
        if not proposal_json:
            return "먼저 계획을 생성해 주세요.", None
        try:
            raw = json.loads(proposal_json)
            if not isinstance(raw, dict):
                raise ValueError("Proposal state must be a JSON object.")
            proposal = Proposal.from_dict(raw)
            result = pipeline.execute(
                proposal,
                approval_hash=typed_hash.strip(),
                output_root=output_root,
                build=not source_only,
                run_gametest=not source_only,
                existing_input=existing_path,
            )
            summary = (
                f"상태: {result.status}\n"
                f"결정론 검증: {result.validation_status}\n"
                f"Gradle 빌드: {result.build_status}\n"
                f"GameTest: {result.gametest_status}\n"
                f"설치용 JAR 발행: {'예' if result.release_ready else '아니오'}\n"
                f"기존 입력: {result.existing_input_kind or '없음'}\n"
                f"release ZIP: {result.release_zip}"
            )
            return summary, result.release_zip
        except Exception as exc:
            return f"실행 실패: {type(exc).__name__}: {exc}", None

    with gr.Blocks(title="M.M.M Make Mincraft Mode") as demo:
        gr.Markdown(
            """
# M.M.M Make Mincraft Mode — Fabric 1.20.1

원하는 모드를 적고 `계획 생성 → 승인 후 실행` 순서로 누르세요.
완료되면 아래에서 release ZIP을 내려받을 수 있습니다.
"""
        )
        gr.Markdown(_existing_input_markdown(existing_report))
        if existing_report is not None:
            gr.JSON(
                value=existing_report.to_dict(),
                label="업로드한 모드 정보",
            )

        state = gr.Textbox(visible=False)
        prompt = gr.Textbox(
            label="요청",
            value="서리 보스, 전투 아레나 맵, 3D 모델, 결정 아이템과 블록을 만들어줘",
            lines=4,
        )
        plan_button = gr.Button("1. 계획 생성", variant="primary")
        plan_view = gr.JSON(label="검토할 제안서")
        expected_hash = gr.Textbox(label="승인 대상 해시", interactive=False)
        plan_status = gr.Markdown()
        typed_hash = gr.Textbox(
            label="승인 해시 재입력",
            placeholder="sha256:...",
        )
        source_only = gr.Checkbox(
            label="소스만 생성 (빌드/GameTest 생략, 설치용 JAR 미발행)",
            value=False,
        )
        execute_button = gr.Button("2. 승인 후 실행")
        result_status = gr.Textbox(label="실행 결과", lines=8)
        release_file = gr.File(label="검증 증거를 포함한 release ZIP")

        plan_button.click(
            make_plan,
            inputs=[prompt],
            outputs=[plan_view, expected_hash, state, plan_status],
        )
        execute_button.click(
            execute_plan,
            inputs=[state, typed_hash, source_only],
            outputs=[result_status, release_file],
        )
    return demo


def launch(
    *,
    output_root: str | Path = "mmm-output",
    local_model: bool = False,
    share: bool = False,
    server_name: str = "127.0.0.1",
    existing_input: str | Path | None = None,
) -> Any:
    demo = create_demo(
        output_root=Path(output_root),
        local_model=local_model,
        existing_input=existing_input,
    )
    return demo.launch(share=share, server_name=server_name)
