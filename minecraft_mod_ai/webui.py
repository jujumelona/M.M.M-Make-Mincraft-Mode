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
        return (
            "### 입력 모드: 새 모드\n\n"
            "기존 프로젝트 업로드 없이 새 revision workspace를 생성합니다."
        )
    source_note = (
        "편집 가능한 source가 발견되었습니다."
        if report.has_sources
        else "편집 가능한 source가 없습니다. metadata/inventory 분석만 가능합니다."
    )
    return (
        "### 입력 모드: 기존 모드 수정 준비\n\n"
        f"- archive: `{report.archive_name}`\n"
        f"- kind: `{report.input_kind}`\n"
        f"- mod: `{report.mod_id or 'unknown'}` / `{report.mod_version or 'unknown'}`\n"
        f"- snapshot: `{report.source_snapshot_hash}`\n"
        f"- files: `{report.file_count}`\n\n"
        f"{source_note} ZIP 내용은 실행하거나 원본에 덮어쓰지 않습니다. "
        "현재 수직 슬라이스는 snapshot을 승인에 묶은 별도 candidate를 만들며, "
        "임의 기존 소스에 최소 diff를 자동 적용했다고 주장하지 않습니다."
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
        mode = (
            "기존 입력에 결합된 revision candidate"
            if existing_report is not None
            else "새 모드"
        )
        status = (
            f"**{mode} 계획만 생성했습니다. 아직 파일을 쓰거나 빌드하지 않았습니다.**\n\n"
            f"범위: {' · '.join(coverage)}\n\n"
            "아래 JSON과 제외·유예 항목을 검토한 뒤 표시된 해시를 그대로 입력해야 "
            "실행할 수 있습니다."
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

    with gr.Blocks(title="Minecraft Mod AI") as demo:
        gr.Markdown(
            """
# Minecraft Mod AI — Fabric 1.20.1

현재 검증된 수직 슬라이스는 아이템·블록과 제한된 보스·아레나 맵·3D 원본을
생성합니다. 계획은 읽기 전용이며, 표시된 SHA-256을 직접 재입력한 뒤에만
파일 생성과 빌드가 시작됩니다.
"""
        )
        gr.Markdown(_existing_input_markdown(existing_report))
        if existing_report is not None:
            gr.JSON(
                value=existing_report.to_dict(),
                label="기존 입력 inventory (읽기 전용)",
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
    output_root: str | Path = "minecraft-mod-ai-output",
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
