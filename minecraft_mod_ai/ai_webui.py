from __future__ import annotations

import hashlib
import json
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

    def do_plan(prompt: str, files: list[str] | None, existing_zip: str | None):
        existing_hash = ""
        if existing_zip:
            path = Path(existing_zip)
            existing_hash = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        result = service.plan_complete_game(prompt, files or [], existing_hash)
        return (
            json.dumps(result["game_design"], ensure_ascii=False, indent=2),
            json.dumps(result["complete_proposal"], ensure_ascii=False, indent=2),
            result["approval_hash"],
            result,
        )

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
        return json.dumps(result, ensure_ascii=False, indent=2)

    with gr.Blocks(title="M.M.M Complete Minecraft Production") as demo:
        gr.Markdown(
            "# M.M.M Complete Minecraft Production\n"
            "아이템 슬라이스가 아니라 퀘스트·직업·경제·GUI·엔티티·월드·오디오·빌드·실행·플레이테스트를 하나의 승인 그래프로 처리합니다. "
            "외부 프로그램이나 실행 증거가 없으면 성공으로 위장하지 않고 정확히 실패합니다."
        )
        prompt = gr.Textbox(label="전체 게임·모드 요구사항", lines=10)
        media = gr.File(label="레퍼런스 이미지", file_count="multiple", type="filepath")
        existing_zip = gr.File(label="수정할 기존 Fabric 소스 ZIP(선택)", type="filepath")
        plan_button = gr.Button("전체 제작 계획 생성")
        design = gr.Code(label="전체 게임 디자인", language="json")
        proposal = gr.Code(label="전체 불변 제작 제안서", language="json")
        approval_hash = gr.Textbox(label="불변 승인 해시", interactive=False)
        state = gr.State()
        plan_button.click(do_plan, [prompt, media, existing_zip], [design, proposal, approval_hash, state])

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
        generate_button = gr.Button("승인 해시로 전체 제작 실행")
        result = gr.Code(label="전체 제작 결과와 미해결 gate", language="json")
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
