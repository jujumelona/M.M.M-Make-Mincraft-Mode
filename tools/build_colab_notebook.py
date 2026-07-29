from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import nbformat
from nbformat import NotebookNode

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "M.M.M_Make_Mincraft_Mode_Colab.ipynb"
V6_NOTEBOOK_PATH = ROOT / "Minecraft_Multimodal_Mod_AI_Architecture_v6.ipynb"

CELL_SPECS = [
    (
        "markdown",
        "title",
        """# M.M.M Complete Minecraft Production
### 전체 기획 + Fabric 구현 + 월드 + 모델 + 오디오 + 수리 + 실행 검증 + 배포

이 노트북은 `main`만 사용하고 `config/model_registry.yaml`만 모델 설정 원본으로
사용합니다. 아이템·블록 슬라이스가 아니라 요청된 전체 기능을 하나의 불변 제안서로 승인하고 실행합니다. 실제 T4·Blockbench·Minecraft
런타임이 준비되지 않은 단계는 성공으로 위장하지 않고 실패 또는 미해결 gate로
기록합니다.
""",
    ),
    (
        "code",
        "configuration",
        """# @title 1. 전체 제작 요청과 실행 설정
PROMPT = "퀘스트, 직업, 경제, GUI, 애니메이션 몬스터와 마을이 있는 Fabric 1.20.1 모드를 만들어줘." # @param {type:"string"}
MODEL_PROFILE = "t4_local" # @param ["t4_local", "remote_quality"]
SOURCE_ONLY = True # @param {type:"boolean"}
APPROVE_PLAN = True # @param {type:"boolean"}
RUN_BLOCKBENCH = False # @param {type:"boolean"}
RUN_RUNTIME = False # @param {type:"boolean"}
RUN_CLIENT = False # @param {type:"boolean"}
RUN_MINEFLAYER = False # @param {type:"boolean"}
RUN_VISUAL_REVIEW = False # @param {type:"boolean"}
ACCEPT_EULA = False # @param {type:"boolean"}
SERVER_LAUNCHER = "" # @param {type:"string"}
OUTPUT_ROOT = "/content/mmm-output" # @param {type:"string"}
RUN_NAME = "complete-colab-run" # @param {type:"string"}
SCREENSHOTS = []  # runtime screenshot paths

if not PROMPT.strip():
    raise ValueError("PROMPT가 비어 있습니다.")
if RUN_RUNTIME and not ACCEPT_EULA:
    raise ValueError("RUN_RUNTIME=True이면 Minecraft EULA 승인을 명시해야 합니다.")
""",
    ),
    (
        "code",
        "setup",
        """# @title 2. main 저장소와 의존성 설치
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path("/content/M.M.M-Make-Mincraft-Mode")
if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)
subprocess.run(
    ["git", "clone", "--depth", "1", "--branch", "main",
     "https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode.git", str(REPO_DIR)],
    check=True,
)
os.chdir(REPO_DIR)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e",
     ".[ui,local-model,rag,image,speech,production-audio,training]"],
    check=True,
)

import torch
print("Python:", sys.version.split()[0])
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"VRAM free/total: {free_bytes/2**30:.2f}/{total_bytes/2**30:.2f} GiB")
""",
    ),
    (
        "code",
        "registry",
        """# @title 3. 실제 모델·MCP·전체 capability 확인
import json
from pathlib import Path
from minecraft_mod_ai import ModelRegistry
from minecraft_mod_ai.capabilities import capability_manifest
from minecraft_mod_ai.capability_plugins import plugin_manifest

registry = ModelRegistry().to_public_dict()
if MODEL_PROFILE not in registry["profiles"]:
    raise ValueError(f"알 수 없는 MODEL_PROFILE: {MODEL_PROFILE}")
print(json.dumps(registry["profiles"][MODEL_PROFILE], ensure_ascii=False, indent=2))
print(json.dumps(capability_manifest(), ensure_ascii=False, indent=2))
print(json.dumps(plugin_manifest(), ensure_ascii=False, indent=2))
print(Path(".mcp.json").read_text(encoding="utf-8"))
""",
    ),
    (
        "code",
        "plan",
        """# @title 4. 전체 불변 제작 제안서 생성
from minecraft_mod_ai import CompleteModAISession

session = CompleteModAISession(
    output_root=OUTPUT_ROOT,
    minecraft_version="1.20.1",
    model_profile=MODEL_PROFILE,
)
reply = session.plan(PROMPT)
print(reply.message)
print("approval_hash:", reply.approval_hash)
print(json.dumps(reply.complete_proposal.to_dict(), ensure_ascii=False, indent=2))
""",
    ),
    (
        "code",
        "build",
        """# @title 5. 승인된 전체 그래프 실행
from minecraft_mod_ai import CompleteExecutionOptions

BUILD_RESULT = None
if not APPROVE_PLAN:
    print("APPROVE_PLAN=False: 어떤 파일도 생성하지 않습니다.")
else:
    options = CompleteExecutionOptions(
        source_only=SOURCE_ONLY,
        run_blockbench=RUN_BLOCKBENCH,
        run_runtime=RUN_RUNTIME,
        run_client=RUN_CLIENT,
        run_mineflayer=RUN_MINEFLAYER,
        run_visual_review=RUN_VISUAL_REVIEW,
        eula_accepted=ACCEPT_EULA,
        server_launcher=SERVER_LAUNCHER or None,
        screenshot_paths=tuple(SCREENSHOTS),
    )
    BUILD_RESULT = session.build(
        reply,
        run_name=RUN_NAME,
        source_only=SOURCE_ONLY,
        options=options,
    )
    print(json.dumps(BUILD_RESULT.to_dict(), ensure_ascii=False, indent=2))
    if not SOURCE_ONLY and not BUILD_RESULT.release_ready:
        raise RuntimeError("외부 runtime/visual gate가 모두 통과하지 않았습니다.")
""",
    ),
    (
        "code",
        "download",
        """# @title 6. 결과 즉시 다운로드
if BUILD_RESULT is None or not BUILD_RESULT.release_zip:
    print("다운로드할 결과가 없습니다.")
else:
    release_zip = Path(BUILD_RESULT.release_zip)
    if not release_zip.is_file():
        raise FileNotFoundError(release_zip)
    print("release:", release_zip)
    print("size:", release_zip.stat().st_size, "bytes")
    try:
        from google.colab import files as colab_files
        colab_files.download(str(release_zip))
    except ImportError:
        print("로컬 경로:", release_zip.resolve())
""",
    ),
    (
        "markdown",
        "boundaries",
        """## 실행 증거 경계

`SOURCE_ONLY=True`는 전체 소스·리소스·월드·시스템을 만들지만 설치용 완성 판정은
하지 않습니다. 설치용 `VERIFIED` 판정에는 Gradle, GameTest, JAR 검사와 요청된
Blockbench, Minecraft 서버·클라이언트, Mineflayer, 스크린샷 VisualCritic gate가
실제로 통과한 증거가 필요합니다.
""",
    ),
]

METADATA = {
    "colab": {"name": NOTEBOOK_PATH.name, "provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "accelerator": "GPU",
}


def build_notebook() -> NotebookNode:
    notebook = nbformat.v4.new_notebook(metadata=METADATA)
    for kind, cell_id, source in CELL_SPECS:
        cell = nbformat.v4.new_markdown_cell(source) if kind == "markdown" else nbformat.v4.new_code_cell(source)
        cell["id"] = cell_id
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            cell["metadata"]["cellView"] = "form"
        notebook.cells.append(cell)
    nbformat.validate(notebook)
    return notebook


def serialize_notebook(notebook: NotebookNode) -> str:
    return nbformat.writes(notebook, version=4).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the complete M.M.M Colab notebooks.")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = serialize_notebook(build_notebook())
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    if args.check:
        for path in (NOTEBOOK_PATH, V6_NOTEBOOK_PATH):
            if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
                raise SystemExit(f"Notebook is stale: {path}")
        print(digest)
        return 0
    for path in (NOTEBOOK_PATH, V6_NOTEBOOK_PATH):
        path.write_text(rendered, encoding="utf-8")
        print(path)
    print("sha256:", digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
