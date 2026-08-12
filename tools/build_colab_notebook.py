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

`모두 실행`을 기준으로 실행합니다.

- **Full**: 새 플랜 생성 → 사용자와 반복 수정/보완 → 확정 → 제작
- **Plan**: 새 플랜 생성 → 사용자와 반복 수정/보완 → 확정 → 플랜 저장
- **Revise**: 기존 source/release ZIP 업로드 → 수정 플랜 대화 → 확정 → 기존 모드 수정 제작
- **Execute**: 저장된 플랜 전체 확인 → 필요하면 추가 수정 → 사용자 승인 → 제작

플랜을 새로 만들거나 불러온 뒤에는 반드시 사용자 입력을 기다립니다. 수정 내용을 입력할 때마다 새 플랜 전체를 다시 보여주며, 사용자가 직접 확정하거나 제작을 승인하기 전에는 다음 단계로 넘어가지 않습니다.
""",
    ),
    (
        "code",
        "configuration",
        """# @title 1. 실행 모드 및 설정
import os

RUN_MODE = "Full" #@param ["Full", "Plan", "Revise", "Execute"]
PROMPT = "계절마다 다른 작물을 재배하고 요리하는 모드를 만들어줘." #@param {type:"string"}
PLAN_FILE = "" #@param {type:"string"}
MODEL_PROFILE = "Qwen3.5-9B_6GB" #@param ["Qwen3.5-9B_6GB", "Gemma4-12B_7GB", "Gemma4-26B_14GB", "Qwen3.6-35B_23GB", "Qwen3.6-27B_18GB", "Qwen3.6-27B_14GB", "mini_mod", "fast_test"]
KV_CACHE_QUANT = "q4_0" #@param ["q4_0", "q8_0", "f16"]
FAST_MODE = False #@param {type:"boolean"}
SAVE_TO_GOOGLE_DRIVE = True #@param {type:"boolean"}

REMOTE_BASE_URL, REMOTE_TEXT_MODEL, REMOTE_IMAGE_MODEL, REMOTE_SPEECH_MODEL, SOURCE_ONLY, RUN_BLOCKBENCH, RUN_RUNTIME, RUN_CLIENT, RUN_MINEFLAYER, RUN_VISUAL_REVIEW, ACCEPT_EULA, SERVER_LAUNCHER, RUN_NAME, SCREENSHOTS = "", "", "", "", True, False, False, False, False, False, False, "", "complete-colab-run", []

VALID_RUN_MODES = {
    "Full",
    "Plan",
    "Revise",
    "Execute",
}
VALID_KV_CACHE_QUANTS = {"q4_0", "q8_0", "f16"}
if RUN_MODE not in VALID_RUN_MODES:
    raise ValueError(f"지원하지 않는 실행 모드: {RUN_MODE}")
if KV_CACHE_QUANT not in VALID_KV_CACHE_QUANTS:
    raise ValueError(f"지원하지 않는 KV cache 양자화: {KV_CACHE_QUANT}")
os.environ["MMM_KV_CACHE_QUANT"] = KV_CACHE_QUANT
if RUN_MODE != "Execute" and not PROMPT.strip():
    raise ValueError("선택한 실행 모드에서는 PROMPT를 입력해야 합니다.")
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
    print("Updating M.M.M repository from GitHub main...", flush=True)
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
    print("Cloning M.M.M repository from GitHub main...", flush=True)
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
        """# @title 3. 기존 모드 입력
from minecraft_mod_ai.colab_run_modes import prepare_existing_mod_input

assert_setup_state = COLAB_SETUP_MODULE.assert_setup_state
EXISTING_INPUT = prepare_existing_mod_input(RUN_MODE)
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
    if os.environ.get("MMM_KV_CACHE_QUANT") != KV_CACHE_QUANT:
        raise RuntimeError(
            "KV cache 설정이 변경되었습니다. 1번 셀부터 다시 실행하세요."
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
print("KV cache:", KV_CACHE_QUANT)
print("결과 저장 위치:", OUTPUT_ROOT)

from minecraft_mod_ai.custom_module_generator import _extract_json
sample_json = _extract_json('{"operations": [], "runtime_tests": [], "complete": true, "next_cursor": ""}')
assert "operations" in sample_json, "Engine JSON parser self-check failed."
print("설치 확인: 완료")
""",
    ),
    (
        "code",
        "mtp-server",
        """# @title 4-1. [선택] 로컬 CUDA llama 서버 실행
from minecraft_mod_ai.colab_mtp_server import start_colab_mtp_server

assert_current_colab_setup()
LLAMA_SERVER_URL = start_colab_mtp_server(planner_config)
print("llama server:", LLAMA_SERVER_URL, "(structured=baseline host validation; verified free-text may use MTP)")
""",
    ),
    (
        "code",
        "plan",
        """# @title 5. 플랜 생성/불러오기 및 대화 확정
from minecraft_mod_ai import CompleteModAISession
from minecraft_mod_ai.colab_run_modes import resolve_plan_path, run_plan_dialog

assert_current_colab_setup()
session = CompleteModAISession(
    output_root=OUTPUT_ROOT,
    model_profile=MODEL_PROFILE,
    existing_input=EXISTING_INPUT,
    fast_mode=FAST_MODE,
    kv_cache_quant=KV_CACHE_QUANT,
)
PLAN_PATH = resolve_plan_path(
    run_mode=RUN_MODE,
    output_root=OUTPUT_ROOT,
    configured_path=PLAN_FILE,
)
PLAN_DIALOG = run_plan_dialog(
    session=session,
    run_mode=RUN_MODE,
    prompt=PROMPT,
    plan_path=PLAN_PATH,
)
reply = PLAN_DIALOG.reply
FINAL_PLAN_PATH = PLAN_DIALOG.plan_path
PLAN_APPROVED = PLAN_DIALOG.approved
print("플랜 확정:", FINAL_PLAN_PATH)
""",
    ),
    (
        "code",
        "build",
        """# @title 6. 확정된 플랜으로 제작
from minecraft_mod_ai import CompleteExecutionOptions
from minecraft_mod_ai.colab_run_modes import should_build

assert_current_colab_setup()
if not PLAN_APPROVED:
    raise RuntimeError("플랜이 사용자에게 확정되지 않았습니다.")

BUILD_RESULT = None
if not should_build(RUN_MODE):
    print("Plan: 제작 생략")
else:
    print("모드 생성: 시작", flush=True)
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
        print("재개 실행: 예")
    if BUILD_RESULT.quality_report:
        print("품질 검증:", BUILD_RESULT.quality_report["overall_status"])
    if BUILD_RESULT.unresolved_gates:
        print("미해결 항목:", ", ".join(BUILD_RESULT.unresolved_gates))
""",
    ),
    (
        "code",
        "download",
        """# @title 7. 결과 다운로드
if RUN_MODE == "Plan":
    plan_path = Path(FINAL_PLAN_PATH)
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    print("플랜 파일:", plan_path)
    try:
        from google.colab import files as colab_files
        colab_files.download(str(plan_path))
    except ImportError:
        print("로컬 경로:", plan_path.resolve())
elif BUILD_RESULT is None or not BUILD_RESULT.release_zip:
    print("다운로드할 제작 결과가 없습니다.")
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
        """## 실행 모드

기본값은 **Full**입니다.

**Full**과 **Plan**은 5번 셀에서 플랜을 만든 뒤 사용자 입력을 기다립니다. 수정/보완 내용을 입력하면 다시 기획하고 전체 플랜을 다시 표시하며, `확정`을 입력해야 다음 단계로 진행합니다.

**Revise**는 3번 셀에서 기존 source/release ZIP 업로드를 요구합니다. 이후 수정 요구를 기준으로 플랜을 만들고 같은 대화 확정 과정을 거친 뒤 기존 프로젝트를 수정합니다.

**Execute**는 `PLAN_FILE` 경로의 플랜을 사용합니다. 경로가 비어 있고 기본 `proposal.json`도 없으면 JSON 업로드를 요청합니다. 전체 플랜을 보여준 뒤 `제작`을 입력해야 제작하며, 그 전에 수정 내용을 입력하면 플랜을 다시 보완할 수 있습니다.
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
