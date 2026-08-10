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
        """# M.M.M Make Mincraft Mode

원하는 모드를 말하면 먼저 실제 게임 기획서처럼 정리하고, 다음 셀에서 제작합니다.
작은 모드부터 장기 제작이 필요한 대형 모드까지 같은 방식으로 실행하며,
중단되면 완료한 작업부터 이어서 진행합니다.
""",
    ),
    (
        "code",
        "configuration",
        """# @title 1. 만들 모드 입력
PROMPT = "계절마다 다른 작물을 재배하고 요리하는 모드를 만들어줘." # @param {type:"string"}
MODEL_PROFILE = "qwen36_35b_ud_q4" # @param ["qwen36_35b_ud_q4", "qwen36_27b_ud_q4", "qwen36_27b_q3_km", "qwen35_9b_ud_q4", "gemma4_26b_ud_iq4_nl", "gemma4_12b_ud_q4", "t4_local", "t4_quality", "remote_quality"] {type:"string"}
RESUME_MODE = False # @param {type:"boolean"}
REMOTE_BASE_URL = "" # @param {type:"string"}
REMOTE_TEXT_MODEL = "" # @param {type:"string"}
REMOTE_IMAGE_MODEL = "" # @param {type:"string"}
REMOTE_SPEECH_MODEL = "" # @param {type:"string"}
SOURCE_ONLY = True # @param {type:"boolean"}
PATCH_EXISTING = False # @param {type:"boolean"}
RUN_BLOCKBENCH = False # @param {type:"boolean"}
RUN_RUNTIME = False # @param {type:"boolean"}
RUN_CLIENT = False # @param {type:"boolean"}
RUN_MINEFLAYER = False # @param {type:"boolean"}
RUN_VISUAL_REVIEW = False # @param {type:"boolean"}
ACCEPT_EULA = False # @param {type:"boolean"}
SERVER_LAUNCHER = "" # @param {type:"string"}
SAVE_TO_GOOGLE_DRIVE = True # @param {type:"boolean"}
RUN_NAME = "complete-colab-run" # @param {type:"string"}
SCREENSHOTS = []

if not PROMPT.strip():
    raise ValueError("PROMPT를 입력해 주세요.")
if RUN_RUNTIME and not ACCEPT_EULA:
    raise ValueError("Minecraft 실행 검증에는 EULA 동의가 필요합니다.")
""",
    ),
    (
        "code",
        "setup",
        """# @title 2. GitHub 최신 main 설치
import importlib.util
import subprocess
import sys
from pathlib import Path

transformers_was_loaded = "transformers" in sys.modules
engine_was_loaded = any(
    name == "minecraft_mod_ai" or name.startswith("minecraft_mod_ai.")
    for name in sys.modules
)
loaded_engine_module = sys.modules.get("minecraft_mod_ai")
engine_module_file = getattr(loaded_engine_module, "__file__", "") or ""
REPO_DIR = Path("/content/M.M.M-Make-Mincraft-Mode")
EXPECTED_REPOSITORY = "https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode.git"
previous_commit = ""
if (REPO_DIR / ".git").is_dir():
    print("📥 Updating official M.M.M repository from GitHub main...", flush=True)
    origin_url = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "remote", "get-url", "origin"],
        text=True,
    ).strip()
    normalized_origin = origin_url.rstrip("/").removesuffix(".git")
    normalized_expected = EXPECTED_REPOSITORY.removesuffix(".git")
    if normalized_origin != normalized_expected:
        raise RuntimeError(
            "Existing Colab checkout is not the official M.M.M GitHub repository. "
            "Remove that checkout and rerun setup cell 2."
        )
    previous_commit = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    subprocess.run(
        [
            "git",
            "-C",
            str(REPO_DIR),
            "fetch",
            "origin",
            "+main:refs/remotes/origin/main",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "main"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(REPO_DIR),
            "merge",
            "--ff-only",
            "refs/remotes/origin/main",
        ],
        check=True,
    )
elif REPO_DIR.exists():
    raise RuntimeError(f"Git 저장소가 아닌 경로가 이미 있습니다: {REPO_DIR}")
else:
    print("📥 Cloning official M.M.M repository from GitHub main...", flush=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "main",
            EXPECTED_REPOSITORY,
            str(REPO_DIR),
        ],
        check=True,
    )

USED_COMMIT = subprocess.check_output(
    ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
    text=True,
).strip()
REMOTE_COMMIT = subprocess.check_output(
    ["git", "-C", str(REPO_DIR), "rev-parse", "refs/remotes/origin/main"],
    text=True,
).strip()
if USED_COMMIT != REMOTE_COMMIT:
    raise RuntimeError(
        "The checkout is not exactly GitHub origin/main. Remove the Colab "
        "checkout and rerun setup cell 2."
    )
tracked_changes = subprocess.check_output(
    ["git", "-C", str(REPO_DIR), "status", "--porcelain", "--untracked-files=no"],
    text=True,
).strip()
if tracked_changes:
    raise RuntimeError(
        "The Colab engine checkout contains tracked local changes. Remove the "
        "checkout and rerun setup cell 2 instead of executing mixed source."
    )
print("GitHub commit:", USED_COMMIT, flush=True)

# This bootstrap stays intentionally small. Every setup policy below comes from
# the just-pulled commit, so a stale open Colab tab cannot keep running an old
# dependency or CUDA preflight cell.
setup_script = REPO_DIR / "tools" / "colab_runtime_setup.py"
if not setup_script.is_file():
    raise FileNotFoundError(f"Pulled commit has no Colab setup script: {setup_script}")
setup_module_name = f"_mmm_colab_runtime_setup_{USED_COMMIT[:12]}"
setup_spec = importlib.util.spec_from_file_location(setup_module_name, setup_script)
if setup_spec is None or setup_spec.loader is None:
    raise RuntimeError(f"Cannot load Colab setup script: {setup_script}")
COLAB_SETUP_MODULE = importlib.util.module_from_spec(setup_spec)
sys.modules[setup_module_name] = COLAB_SETUP_MODULE
setup_spec.loader.exec_module(COLAB_SETUP_MODULE)

SETUP_STATE = COLAB_SETUP_MODULE.setup_colab_runtime(
    repo_dir=REPO_DIR,
    used_commit=USED_COMMIT,
    model_profile=MODEL_PROFILE,
    save_to_google_drive=SAVE_TO_GOOGLE_DRIVE,
    remote_base_url=REMOTE_BASE_URL,
    remote_text_model=REMOTE_TEXT_MODEL,
    remote_image_model=REMOTE_IMAGE_MODEL,
    remote_speech_model=REMOTE_SPEECH_MODEL,
    transformers_was_loaded=transformers_was_loaded,
    engine_was_loaded=engine_was_loaded,
    engine_module_file=engine_module_file,
    previous_commit=previous_commit,
)
REPO_DIR = Path(SETUP_STATE["repo_dir"])
OUTPUT_ROOT = SETUP_STATE["output_root"]
SETUP_RECEIPT = SETUP_STATE["receipt"]
SETUP_FINGERPRINT = SETUP_STATE["setup_fingerprint"]
""",
    ),
    (
        "code",
        "existing-input",
        """# @title 3. 기존 모드 수정 파일 (선택)
EXISTING_INPUT = None
if PATCH_EXISTING:
    from google.colab import files as colab_files

    print(
        "본인이 소유하거나 수정 권한이 있는 source/release ZIP 하나만 "
        "선택해 주세요."
    )
    uploaded = colab_files.upload()
    if len(uploaded) != 1:
        raise ValueError("기존 모드 수정에는 ZIP 파일을 정확히 하나만 선택해야 합니다.")
    uploaded_name, uploaded_bytes = next(iter(uploaded.items()))
    safe_name = Path(uploaded_name).name
    if (
        safe_name != uploaded_name
        or "/" in uploaded_name
        or "\\\\" in uploaded_name
        or Path(safe_name).suffix.lower() != ".zip"
    ):
        raise ValueError("기존 모드 입력은 경로가 없는 .zip 파일이어야 합니다.")
    existing_dir = Path("/content/mmm-existing-input")
    existing_dir.mkdir(parents=True, exist_ok=True)
    EXISTING_INPUT = existing_dir / safe_name
    EXISTING_INPUT.write_bytes(uploaded_bytes)
    from minecraft_mod_ai.importer import inspect_existing_project_archive

    existing_report = inspect_existing_project_archive(EXISTING_INPUT)
    if not existing_report.has_sources or not existing_report.has_gradle_project:
        raise ValueError(
            "수정에는 소스와 Gradle 프로젝트가 들어 있는 source/release ZIP이 필요합니다."
        )
    print(
        "기존 모드 수정 준비:",
        existing_report.mod_name or existing_report.mod_id or safe_name,
    )
else:
    print("새 모드: 업로드 없이 시작합니다.")
""",
    ),
    (
        "code",
        "registry",
        """# @title 4. 설치 확인
from minecraft_mod_ai import ModelRegistry

def assert_current_colab_setup():
    COLAB_SETUP_MODULE.assert_setup_state(
        SETUP_STATE,
        repo_dir=REPO_DIR,
        used_commit=USED_COMMIT,
        model_profile=MODEL_PROFILE,
        save_to_google_drive=SAVE_TO_GOOGLE_DRIVE,
        remote_base_url=REMOTE_BASE_URL,
        remote_text_model=REMOTE_TEXT_MODEL,
        remote_image_model=REMOTE_IMAGE_MODEL,
        remote_speech_model=REMOTE_SPEECH_MODEL,
    )


assert_current_colab_setup()
REGISTRY_PATH = REPO_DIR / "config/model_registry.yaml"
if not REGISTRY_PATH.is_file():
    raise FileNotFoundError(REGISTRY_PATH)
registry_manager = ModelRegistry()
registry = registry_manager.to_public_dict()
if MODEL_PROFILE not in registry["profiles"]:
    raise ValueError(f"지원하지 않는 모델 프로필: {MODEL_PROFILE}")
planner_config = registry_manager.role(MODEL_PROFILE, "planner")
print("모델 레지스트리:", REGISTRY_PATH)
print("모델 프로필:", MODEL_PROFILE)
print("기획 모델:", planner_config.model_id)
print("기획 백엔드:", planner_config.provider, "/", planner_config.adapter)
print("기획 양자화:", planner_config.quantization or "none")
print("기획 native context:", f"{planner_config.max_context:,} tokens")
print(
    "기획 page input:",
    f"{planner_config.max_input_tokens:,} tokens"
    if planner_config.max_input_tokens
    else "native-context bound (no separate page cap)",
)
print("기획 page output:", f"{planner_config.max_new_tokens:,} tokens")
print("설치 commit/fingerprint:", USED_COMMIT, "/", SETUP_FINGERPRINT)
print("결과 저장 위치:", OUTPUT_ROOT)
""",
    ),
    (
        "code",
        "plan",
        """# @title 5. 게임 기획 만들기
from minecraft_mod_ai import CompleteModAISession

assert_current_colab_setup()
session = CompleteModAISession(
    output_root=OUTPUT_ROOT,
    minecraft_version="1.20.1",
    model_profile=MODEL_PROFILE,
    existing_input=EXISTING_INPUT,
)
saved_plan_file = OUTPUT_ROOT / "proposal.json"
if RESUME_MODE and saved_plan_file.is_file():
    print(f"🔄 [Resume Mode] 이전 기획서({saved_plan_file})를 로드합니다...", flush=True)
    reply = session.load_plan(saved_plan_file)
else:
    if RESUME_MODE:
        print("⚠️ [Resume Mode] 저장된 이전 기획서가 없어 새로 생성합니다...", flush=True)
    reply = session.plan(PROMPT)
print(reply.message)
""",
    ),
    (
        "code",
        "revise",
        """# @title 6. 계획 수정 대화 (선택)
REVISION = "" # @param {type:"string"}
if REVISION.strip():
    assert_current_colab_setup()
    reply = session.revise(REVISION)
    print(reply.message)
else:
    print("수정할 내용이 없으면 이 셀은 건너뛰세요.")
""",
    ),
    (
        "code",
        "build",
        """# @title 7. 이 계획으로 만들기
from minecraft_mod_ai import CompleteExecutionOptions

assert_current_colab_setup()
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
    resume=True,
)
BUILD_RESULT = session.build(
    reply,
    run_name=RUN_NAME,
    source_only=SOURCE_ONLY,
    options=options,
)
print("제작 상태:", BUILD_RESULT.status)
print("프로젝트:", BUILD_RESULT.project_root)
print("결과 ZIP:", BUILD_RESULT.release_zip)
if BUILD_RESULT.run_resumed:
    print("이전 실행에서 끝낸 작업을 이어서 사용했습니다.")
if BUILD_RESULT.quality_report:
    print("품질 검증:", BUILD_RESULT.quality_report["overall_status"])
if BUILD_RESULT.unresolved_gates:
    print("아직 확인할 항목:", ", ".join(BUILD_RESULT.unresolved_gates))
""",
    ),
    (
        "code",
        "download",
        """# @title 8. 결과 다운로드
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
        """## 참고

`SOURCE_ONLY=True`는 소스 프로젝트와 리소스 ZIP을 만듭니다.
새 모드는 `PATCH_EXISTING=False` 그대로 실행하며 파일을 업로드하지 않습니다.
본인이 소유하거나 수정 권한이 있는 기존 source/release ZIP을 수정할 때만 `PATCH_EXISTING=True`로 바꾸세요.
실행까지 확인하려면 필요한 Fabric 서버 파일과 EULA 동의를 설정한 뒤 실행 검증 옵션을 켜세요.
Google Drive 저장을 사용하면 Colab 세션이 끊겨도 같은 실행 이름으로 이어서 만들 수 있습니다.
""",
    ),
]

METADATA = {
    "colab": {"name": NOTEBOOK_PATH.name, "provenance": []},
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3"},
    "accelerator": "GPU",
}


def build_notebook() -> NotebookNode:
    notebook = nbformat.v4.new_notebook(metadata=METADATA)
    for kind, cell_id, source in CELL_SPECS:
        if kind == "markdown":
            cell = nbformat.v4.new_markdown_cell(source)
        else:
            cell = nbformat.v4.new_code_cell(source)
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
    parser = argparse.ArgumentParser(
        description="Build the complete M.M.M Colab notebooks."
    )
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
