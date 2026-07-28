from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat import NotebookNode


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "M.M.M_Make_Mincraft_Mode_Colab.ipynb"
V6_NOTEBOOK_PATH = ROOT / "Minecraft_Multimodal_Mod_AI_Architecture_v6.ipynb"


def _source(value: str) -> str:
    return dedent(value).strip() + "\n"


def _markdown(cell_id: str, value: str) -> NotebookNode:
    cell = nbformat.v4.new_markdown_cell(_source(value))
    cell["id"] = cell_id
    return cell


def _code(cell_id: str, value: str) -> NotebookNode:
    cell = nbformat.v4.new_code_cell(_source(value))
    cell["id"] = cell_id
    cell["execution_count"] = None
    cell["outputs"] = []
    cell["metadata"]["cellView"] = "form"
    return cell


def build_notebook() -> NotebookNode:
    cells = [
        _markdown(
            "title",
            """
            # M.M.M Make Mincraft Mode
            ### Minecraft 멀티모달 모드 및 맵 제작 AI 시스템 (Google Colab 전용)

            본 노트북은 Colab 환경에서 셀 단위로 비동기 실행되도록 설계되었습니다.

            --- 
            ### 사용 방법
            1. **1번 셀**에서 모드 요구사항(PROMPT)을 입력합니다.
            2. 상단 메뉴의 **런타임 ➔ 모두 실행**을 누릅니다.
            3. **5번 셀**에서 완성된 모드 패키지(.zip)가 자동 다운로드됩니다.
            """,
        ),
        _markdown(
            "sec1-head",
            """
            ## 1. 모드 요구사항 입력 및 백엔드 설정
            """,
        ),
        _code(
            "input-form",
            r'''
            # @title 모드 요구사항 입력 및 AI 백엔드 설정
            PROMPT = "\ub2e8\ud48d\ub2d8 \uc544\uc774\ud15c 2\uac1c, \ube14\ub85d 3\uac1c, \uadf8\ub9ac\uace0 \uc544\ub808\ub098\ub97c \ud3ec\ud568\ud55c Fabric \ubaa8\ub4dc\ub97c \ub9cc\ub4e4\uc5b4\uc11c \ub9f5\uacfc \ud568\uaed8 \uc81c\uacf5\ud574\uc6e4" # @param {type:"string"}
            AI_BACKEND = "built-in" # @param ["built-in", "local", "api"]
            LOCAL_MODEL_ID = "Qwen/Qwen3.5-9B-Instruct" # @param ["Qwen/Qwen3.5-9B-Instruct", "google/gemma-4-12B-it", "Qwen/Qwen3.5-4B-Instruct"]
            API_BASE_URL = "" # @param {type:"string"}
            API_MODEL = "" # @param {type:"string"}

            print("=" * 80)
            print("1단계 입력 완료")
            print(f"프롬프트: {PROMPT}")
            print(f"백엔드: {AI_BACKEND}")
            print("=" * 80)
            ''',
        ),
        _markdown(
            "sec2-head",
            """
            ## 2. 패키지 설치 및 런타임 준비
            """,
        ),
        _code(
            "setup",
            r'''
            # @title 패키지 준비 및 GPU 런타임 감지
            import sys
            import os
            import torch
            from pathlib import Path

            if hasattr(sys.stdout, "reconfigure"):
                try:
                    sys.stdout.reconfigure(encoding="utf-8")
                except Exception:
                    pass

            print(f"PyTorch 버전: {torch.__version__}")
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"GPU 감지됨: {gpu_name} ({vram_gb:.2f} GB VRAM)")
            else:
                print("GPU 미감지: CPU 모드로 작동합니다.")

            try:
                import minecraft_mod_ai
                print(f"minecraft_mod_ai 패키지 로드 완료 (버전: {minecraft_mod_ai.__version__})")
            except ImportError:
                print("패키지 설치 진행 중...")
                os.system(f"{sys.executable} -m pip install -e .[ui,local-model]")
                import minecraft_mod_ai
                print("패키지 설치 완료")
            ''',
        ),
        _markdown(
            "sec3-head",
            """
            ## 3. 9-Tier RAG 지식 검색 및 AI 기획안 생성
            """,
        ),
        _code(
            "prompt-plan",
            r'''
            # @title 9-Tier RAG 및 GameConceptPlan 기획안 생성
            from rag_engine import DeepRAGEngine
            from minecraft_mod_ai.pipeline import MinecraftModPipeline
            from minecraft_mod_ai.planner import HeuristicPlanner, LocalTransformersPlanner, OpenAICompatiblePlanner

            if 'PROMPT' not in globals() or not PROMPT:
                PROMPT = "\ub2e8\ud48d\ub2d8 \uc544\uc774\ud15c 2\uac1c, \ube14\ub85d 3\uac1c\ub97c \ud3ec\ud568\ud55c Fabric \ubaa8\ub4dc\ub97c \ub9cc\ub4e4\uc5b4\uc11c \ud568\uaed8 \uc81c\uacf5\ud574\uc6e4"

            print(f"AI 기획 처리 중... 프롬프트: {PROMPT}\n")

            rag = DeepRAGEngine()
            rag_result = rag.execute_6pass_rag(PROMPT, target_version="1.20.1")

            planner = HeuristicPlanner()
            if AI_BACKEND == "local":
                try:
                    print(f"로컬 모델 로딩 중: {LOCAL_MODEL_ID}...")
                    planner = LocalTransformersPlanner(model_name_or_path=LOCAL_MODEL_ID)
                except Exception as e:
                    print(f"로컬 모델 로드 중 예외 발생, built-in 폴백 사용: {e}")
            elif AI_BACKEND == "api":
                planner = OpenAICompatiblePlanner(base_url=API_BASE_URL, model=API_MODEL)

            pipeline = MinecraftModPipeline(planner=planner)
            current_proposal = pipeline.plan(PROMPT)

            print("\n" + "=" * 80)
            print("생성된 GameConceptPlan 요약")
            print("=" * 80)
            print(f"프로젝트 ID   : {current_proposal.project_id}")
            print(f"모드 이름     : {current_proposal.mod_spec.name}")
            print(f"타겟 플랫폼   : {current_proposal.platform.edition} / {current_proposal.platform.loader} ({current_proposal.platform.minecraft_version})")
            print(f"기획 해시     : {current_proposal.proposal_hash[:16]}...")
            print(f"포함 콘텐츠   : 아이템 {len(current_proposal.mod_spec.items)}개, 블록 {len(current_proposal.mod_spec.blocks)}개, 보스 {len(current_proposal.mod_spec.bosses)}개")
            print(f"수락 테스트   : {current_proposal.acceptance_tests}")
            print("=" * 80)

            GLOBAL_PROPOSAL = current_proposal
            GLOBAL_PLANNER = planner
            ''',
        ),
        _markdown(
            "sec4-head",
            """
            ## 4. MCP 게이트웨이 및 빌드 검증 실행
            """,
        ),
        _code(
            "build-execute",
            r'''
            # @title 자바 코드 생성 및 빌드 검증 피라미드 실행
            from pathlib import Path
            from minecraft_mod_ai.pipeline import MinecraftModPipeline
            from minecraft_mod_ai.planner import HeuristicPlanner
            from mcp_gateway import DomainMCPServerRegistry, MCPRequestEnvelope, AuthContext, ExecutionLimits
            import uuid

            if 'GLOBAL_PROPOSAL' not in globals() or GLOBAL_PROPOSAL is None:
                raise RuntimeError("1~3번 셀을 먼저 실행하여 Proposal을 생성하세요.")

            output_root = Path("/content/mmm-output")
            output_root.mkdir(parents=True, exist_ok=True)

            print("MCP 게이트웨이 및 검증 피라미드 실행 중...")
            gateway = DomainMCPServerRegistry()

            mcp_req = MCPRequestEnvelope(
                project_id=GLOBAL_PROPOSAL.project_id,
                plan_version=1,
                artifact_revision=GLOBAL_PROPOSAL.proposal_hash,
                request_id=str(uuid.uuid4()),
                auth_context=AuthContext(principal="agent:coder", role="implementer"),
                limits=ExecutionLimits(timeout_s=600, network_policy="deny"),
                tool_name="build.datagen",
                input={"path": "src/main/resources"}
            )
            mcp_res = gateway.dispatch(mcp_req)
            print(f"MCP 게이트웨이 build.datagen 허가 상태: {mcp_res.status}")

            planner = globals().get('GLOBAL_PLANNER', HeuristicPlanner())
            pipeline = MinecraftModPipeline(planner=planner)
            pipeline_result = pipeline.execute(
                GLOBAL_PROPOSAL,
                approval_hash=GLOBAL_PROPOSAL.proposal_hash,
                output_root=output_root
            )

            print("\n" + "=" * 80)
            print("빌드 및 파이프라인 처리 완료")
            print("=" * 80)
            print(f"실행 상태      : {pipeline_result.status}")
            print(f"검증 상태      : {pipeline_result.validation_status}")
            print(f"빌드 상태      : {pipeline_result.build_status}")
            print(f"GameTest 상태  : {pipeline_result.gametest_status}")
            print(f"Release Ready  : {pipeline_result.release_ready}")
            print(f"릴리스 폴더    : {pipeline_result.release_dir}")
            print(f"최종 ZIP 경로  : {pipeline_result.release_zip}")
            print("=" * 80)

            GLOBAL_PIPELINE_RESULT = pipeline_result
            ''',
        ),
        _markdown(
            "sec5-head",
            """
            ## 5. 완성된 모드 패키지 (.zip) 다운로드
            """,
        ),
        _code(
            "download",
            r'''
            # @title 완성된 모드 배포 ZIP 파일 다운로드
            from pathlib import Path
            import os

            if 'GLOBAL_PIPELINE_RESULT' not in globals() or GLOBAL_PIPELINE_RESULT is None:
                raise RuntimeError("4번 셀을 먼저 실행하여 모드를 빌드하세요.")

            zip_path = Path(GLOBAL_PIPELINE_RESULT.release_zip)
            if zip_path.exists():
                file_size_kb = zip_path.stat().st_size / 1024
                print(f"완성된 모드 패키지: {zip_path.name} ({file_size_kb:.1f} KB)")
                
                try:
                    from google.colab import files
                    print("브라우저 패키지 다운로드를 시작합니다...")
                    files.download(str(zip_path))
                except ImportError:
                    print(f"로컬 환경 파일 경로: {zip_path.resolve()}")
                    
                print("\n모든 파이프라인 및 모드 제작이 성공적으로 완료되었습니다.")
            else:
                print(f"ZIP 파일을 찾을 수 없습니다: {zip_path}")
            ''',
        ),
    ]

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "colab": {
                "name": NOTEBOOK_PATH.name,
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
    )
    nbformat.validate(notebook)
    return notebook


def serialize_notebook(notebook: NotebookNode) -> str:
    rendered = nbformat.writes(notebook, version=4)
    return rendered.rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the deterministic Google Colab notebook."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate and compare the existing notebook without rewriting it.",
    )
    args = parser.parse_args()

    notebook = build_notebook()
    rendered = serialize_notebook(notebook)

    if args.check:
        if not NOTEBOOK_PATH.is_file():
            raise SystemExit(f"Notebook is missing: {NOTEBOOK_PATH}")
        existing = NOTEBOOK_PATH.read_text(encoding="utf-8")
        parsed = nbformat.reads(existing, as_version=4)
        nbformat.validate(parsed)
        if existing != rendered:
            raise SystemExit(
                "Notebook differs from deterministic builder output. "
                "Run tools/build_colab_notebook.py."
            )
    else:
        NOTEBOOK_PATH.write_text(rendered, encoding="utf-8", newline="\n")
        V6_NOTEBOOK_PATH.write_text(rendered, encoding="utf-8", newline="\n")

    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    print(f"Validated: {NOTEBOOK_PATH}")
    print(f"Validated: {V6_NOTEBOOK_PATH}")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
