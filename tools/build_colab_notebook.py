from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import nbformat
from nbformat import NotebookNode

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "M.M.M_Make_Mincraft_Mode_Colab.ipynb"
V6_NOTEBOOK_PATH = ROOT / "Minecraft_Multimodal_Mod_AI_Architecture_v6.ipynb"
CELL_SPECS = [('markdown', 'title', '# M.M.M Make Minecraft Mode\n### 역할 분리형 AI + 실제 MCP + Fabric 1.20.1 생성·빌드·검증\n\n이 노트북은 저장소의 `config/model_registry.yaml`만 모델 설정 원본으로\n사용합니다. 모델 로딩이나 생성에 실패하면 휴리스틱으로 몰래 전환하지\n않고 즉시 실패 원인을 표시합니다.\n'), ('code', 'configuration', '# @title 1. 제작 요청과 실행 설정\nPROMPT = "서리 테마 아이템 2개와 블록 2개, 41x41 아레나가 있는 Fabric 1.20.1 모드를 만들어줘. 보스는 넣지 마." # @param {type:"string"}\nMODEL_PROFILE = "t4_local" # @param ["t4_local", "remote_quality"]\nREPO_REF = "main" # @param {type:"string"}\nSOURCE_ONLY = False # @param {type:"boolean"}\nAPPROVE_PLAN = True # @param {type:"boolean"}\nOUTPUT_ROOT = "/content/mmm-output" # @param {type:"string"}\n\nif not PROMPT.strip():\n    raise ValueError("PROMPT가 비어 있습니다.")\nif not APPROVE_PLAN:\n    print("계획만 생성합니다. 빌드하려면 APPROVE_PLAN=True로 명시하세요.")\n'), ('code', 'setup', '# @title 2. 저장소와 의존성 설치\nimport os\nimport shutil\nimport subprocess\nimport sys\nfrom pathlib import Path\n\nREPO_DIR = Path("/content/M.M.M-Make-Mincraft-Mode")\nif REPO_DIR.exists():\n    shutil.rmtree(REPO_DIR)\n\nsubprocess.run(\n    [\n        "git", "clone", "--depth", "1", "--branch", REPO_REF,\n        "https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode.git",\n        str(REPO_DIR),\n    ],\n    check=True,\n)\nos.chdir(REPO_DIR)\nsubprocess.run(\n    [\n        sys.executable, "-m", "pip", "install", "-q", "-e",\n        ".[ui,local-model,rag,image,speech]",\n    ],\n    check=True,\n)\n\nimport torch\nprint("Python:", sys.version.split()[0])\nprint("CUDA:", torch.cuda.is_available())\nif torch.cuda.is_available():\n    free_bytes, total_bytes = torch.cuda.mem_get_info()\n    print("GPU:", torch.cuda.get_device_name(0))\n    print(f"VRAM free/total: {free_bytes/2**30:.2f}/{total_bytes/2**30:.2f} GiB")\n'), ('code', 'registry', '# @title 3. 역할별 모델·MCP 설정 확인\nimport json\nfrom minecraft_mod_ai import ModelRegistry\n\nregistry = ModelRegistry()\npublic_registry = registry.to_public_dict()\nif MODEL_PROFILE not in public_registry["profiles"]:\n    raise ValueError(\n        f"알 수 없는 MODEL_PROFILE={MODEL_PROFILE!r}: "\n        f"{sorted(public_registry[\'profiles\'])}"\n    )\nprint(json.dumps(\n    public_registry["profiles"][MODEL_PROFILE],\n    ensure_ascii=False,\n    indent=2,\n))\nprint("\\nMCP 설정:")\nprint(Path(".mcp.json").read_text(encoding="utf-8"))\n'), ('code', 'plan', '# @title 4. 멀티모달 GameDesignPlanner 계획 생성\nfrom minecraft_mod_ai import ModAISession\n\nsession = ModAISession.with_local_model(\n    output_root=OUTPUT_ROOT,\n    minecraft_version="1.20.1",\n    profile=MODEL_PROFILE,\n)\nreply = session.plan(PROMPT)\nprint(reply.message)\nprint("\\nready_to_build:", reply.ready_to_build)\nif reply.questions:\n    print("\\n추가로 확정해야 할 내용:")\n    for question in reply.questions:\n        print("-", question)\n'), ('code', 'build', '# @title 5. 승인된 계획으로 실제 생성·Gradle·GameTest 실행\nBUILD_RESULT = None\nif not APPROVE_PLAN:\n    print("APPROVE_PLAN=False: 파일을 생성하지 않았습니다.")\nelif not reply.ready_to_build:\n    raise RuntimeError(\n        "계획에 미확정 또는 아직 구현과 연결되지 않은 기능이 있습니다. "\n        "PROMPT를 수정한 뒤 다시 실행하세요."\n    )\nelse:\n    BUILD_RESULT = session.build(\n        reply,\n        source_only=SOURCE_ONLY,\n        output_root=OUTPUT_ROOT,\n    )\n    print(json.dumps(BUILD_RESULT.to_dict(), ensure_ascii=False, indent=2))\n    if not SOURCE_ONLY and not BUILD_RESULT.release_ready:\n        raise RuntimeError(\n            "Gradle/GameTest/JAR 검증이 모두 통과하지 않아 설치용 릴리스를 "\n            "완료하지 못했습니다. 생성된 로그와 report를 확인하세요."\n        )\n'), ('code', 'download', '# @title 6. 검증 결과 다운로드\nif BUILD_RESULT is None:\n    print("다운로드할 빌드 결과가 없습니다.")\nelse:\n    release_zip = Path(BUILD_RESULT.release_zip)\n    if not release_zip.is_file():\n        raise FileNotFoundError(release_zip)\n    print("release:", release_zip)\n    print("size:", release_zip.stat().st_size, "bytes")\n    try:\n        from google.colab import files as colab_files\n        colab_files.download(str(release_zip))\n    except ImportError:\n        print("로컬 경로:", release_zip.resolve())\n'), ('markdown', 'boundaries', '## 통합 기능 실행 방법\n\n기본 노트북은 승인된 Fabric 생성→정적 검증→Gradle→GameTest→JAR 검사 경로를\n실행합니다. 코드 RAG, JDT, Blockbench, GeckoLib, WorldDesignIR→NBT/Jigsaw,\ndisposable runtime, Mineflayer, 검증 trace 수집은 `mmm-mcp` 도구로 분리되어\n있으며 각 외부 실행 파일과 서버가 준비된 경우에만 성공합니다. 준비되지 않은\n도구를 성공한 것으로 표시하지 않습니다.\n')]
METADATA = {'colab': {'name': 'M.M.M_Make_Mincraft_Mode_Colab.ipynb', 'provenance': []}, 'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}, 'accelerator': 'GPU'}


def build_notebook() -> NotebookNode:
    notebook = nbformat.v4.new_notebook(metadata=METADATA)
    for kind, cell_id, source in CELL_SPECS:
        cell = (
            nbformat.v4.new_markdown_cell(source)
            if kind == "markdown"
            else nbformat.v4.new_code_cell(source)
        )
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
    parser = argparse.ArgumentParser(description="Build the M.M.M Colab notebooks.")
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
