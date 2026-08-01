from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .mcp_tools import MMMToolService


def launch(
    *,
    output_root: Path,
    profile: str = "t4_local",
    share: bool = False,
    server_name: str = "127.0.0.1",
) -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("UI extras are missing. Install with: pip install -e '.[ui]'") from exc

    service = MMMToolService(workspace_root=output_root, profile=profile)

    def do_plan(
        message: str,
        files: list[str] | None,
        existing_zip: str | None,
        state: dict[str, Any] | None,
    ):
        existing_hash = ""
        if existing_zip:
            path = Path(existing_zip)
            existing_hash = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if state:
            original = str(
                state.get("complete_proposal", {}).get("requested_prompt", "")
            )
            result = service.revise_complete_plan(
                original,
                message,
                files or [],
                existing_hash,
            )
        else:
            result = service.plan_complete_game(
                message,
                files or [],
                existing_hash,
            )
        return result["message"], result

    def reset_plan():
        return "", "", None

    def do_generate(
        state: dict[str, Any] | None,
        run_name: str,
        source_only: bool,
        run_blockbench: bool,
        run_runtime: bool,
        run_client: bool,
        run_mineflayer: bool,
        run_visual: bool,
        server_launcher: str,
        accept_eula: bool,
        screenshots: list[str] | None,
        existing_zip: str | None,
    ):
        if not state:
            raise gr.Error("먼저 전체 제작 계획을 생성하세요.")
        options = {
            "source_only": source_only,
            "run_blockbench": run_blockbench,
            "run_runtime": run_runtime,
            "run_client": run_client,
            "run_mineflayer": run_mineflayer,
            "run_visual_review": run_visual,
            "server_launcher": server_launcher.strip() or None,
            "eula_accepted": accept_eula,
            "screenshot_paths": tuple(screenshots or []),
        }
        result = service.execute_complete_project(
            state["complete_proposal"],
            state["approval_hash"],
            run_name.strip() or "complete-run",
            options,
            existing_zip or None,
        )
        lines = [
            f"제작 상태: {result['status']}",
            f"프로젝트: {result['project_root']}",
        ]
        if result.get("release_zip"):
            lines.append(f"다운로드 ZIP: {result['release_zip']}")
        if result.get("jar_path"):
            lines.append(f"모드 JAR: {result['jar_path']}")
        unresolved = result.get("unresolved_gates") or []
        if unresolved:
            lines.append("아직 실행하지 않은 확인: " + ", ".join(unresolved))
        if result.get("run_resumed"):
            lines.append("이전 실행의 완료 작업을 이어서 사용했습니다.")
        return "\n".join(lines)

    with gr.Blocks(title="M.M.M Complete Minecraft Production") as demo:
        gr.Markdown(
            "# M.M.M Complete Minecraft Production\n"
            "원하는 규모에 맞춰 게임 기획부터 모드 소스, 자산, 빌드와 플레이 확인까지 이어서 제작합니다. "
            "계획에서 바꾸고 싶은 점은 평소 말하듯 입력하면 반영됩니다."
        )
        prompt = gr.Textbox(
            label="처음 만들 내용 또는 계획에서 바꿀 내용",
            lines=8,
        )
        media = gr.File(label="레퍼런스 이미지", file_count="multiple", type="filepath")
        existing_zip = gr.File(label="수정할 기존 Fabric 소스 ZIP(선택)", type="filepath")
        with gr.Row():
            plan_button = gr.Button("계획 만들기 / 수정 반영")
            reset_button = gr.Button("새 계획 시작")
        design = gr.Textbox(
            label="게임 기획",
            lines=22,
            interactive=False,
        )
        state = gr.State()
        plan_button.click(
            do_plan,
            [prompt, media, existing_zip, state],
            [design, state],
        )
        reset_button.click(
            reset_plan,
            outputs=[prompt, design, state],
        )

        run_name = gr.Textbox(label="실행 폴더 이름", value="complete-run")
        source_only = gr.Checkbox(label="소스만 생성", value=False)
        with gr.Row():
            run_blockbench = gr.Checkbox(label="Blockbench 검증", value=True)
            run_runtime = gr.Checkbox(label="Minecraft 실행 검증", value=True)
            run_client = gr.Checkbox(label="클라이언트 실행", value=True)
            run_mineflayer = gr.Checkbox(label="Mineflayer 플레이테스트", value=True)
            run_visual = gr.Checkbox(label="스크린샷 VLM 검사", value=True)
        server_launcher = gr.Textbox(label="Fabric server launcher 경로")
        accept_eula = gr.Checkbox(label="Minecraft EULA를 명시적으로 승인했습니다", value=False)
        screenshots = gr.File(label="런타임 검사용 스크린샷", file_count="multiple", type="filepath")
        generate_button = gr.Button("이 계획으로 만들기")
        result = gr.Textbox(label="제작 결과", lines=10, interactive=False)
        generate_button.click(
            do_generate,
            [
                state,
                run_name,
                source_only,
                run_blockbench,
                run_runtime,
                run_client,
                run_mineflayer,
                run_visual,
                server_launcher,
                accept_eula,
                screenshots,
                existing_zip,
            ],
            result,
        )
    return demo.launch(share=share, server_name=server_name)
