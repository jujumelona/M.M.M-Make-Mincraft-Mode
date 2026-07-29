from __future__ import annotations

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

    def do_plan(prompt: str, files: list[str] | None):
        result = service.plan_game(prompt, files or [])
        return (
            json.dumps(result["game_design"], ensure_ascii=False, indent=2),
            json.dumps(result["proposal"], ensure_ascii=False, indent=2),
            result["approval_hash"],
            result,
        )

    def do_generate(state: dict[str, Any] | None, run_name: str):
        if not state:
            raise gr.Error("먼저 AI 계획을 생성하세요.")
        result = service.generate_fabric_project(
            state["proposal"], state["approval_hash"], run_name.strip() or "ui-run"
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    with gr.Blocks(title="M.M.M Minecraft Mod AI") as demo:
        gr.Markdown(
            "# M.M.M Minecraft Mod AI\n"
            "모델 실패는 숨기지 않으며, 현재 구현되지 않은 플러그인은 계획에 blocked로 표시됩니다."
        )
        prompt = gr.Textbox(label="게임·모드 요구사항", lines=8)
        media = gr.File(label="레퍼런스 이미지", file_count="multiple", type="filepath")
        plan_button = gr.Button("멀티모달 계획 생성")
        design = gr.Code(label="게임 디자인", language="json")
        proposal = gr.Code(label="현재 빌드 가능한 Fabric 슬라이스", language="json")
        approval_hash = gr.Textbox(label="불변 승인 해시", interactive=False)
        state = gr.State()
        plan_button.click(do_plan, [prompt, media], [design, proposal, approval_hash, state])
        run_name = gr.Textbox(label="실행 폴더 이름", value="ui-run")
        generate_button = gr.Button("승인하고 소스 생성")
        result = gr.Code(label="생성 결과", language="json")
        generate_button.click(do_generate, [state, run_name], result)
    return demo.launch(share=share, server_name=server_name)
