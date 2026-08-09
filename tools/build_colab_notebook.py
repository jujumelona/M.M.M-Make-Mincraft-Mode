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
MODEL_PROFILE = "t4_quality" # @param ["t4_quality", "t4_local", "remote_quality"]
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
import os
import subprocess
import sys
from pathlib import Path

transformers_was_loaded = "transformers" in sys.modules
REPO_DIR = Path("/content/M.M.M-Make-Mincraft-Mode")
if (REPO_DIR / ".git").is_dir():
    subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", "main"], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "main"], check=True)
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", "main"],
        check=True,
    )
elif REPO_DIR.exists():
    raise RuntimeError(f"Git 저장소가 아닌 경로가 이미 있습니다: {REPO_DIR}")
else:
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "main",
            "https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode.git",
            str(REPO_DIR),
        ],
        check=True,
    )

os.chdir(REPO_DIR)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-e",
        ".[ui,local-model,rag,image,speech,production-audio,training]",
    ],
    check=True,
)

# Qwen3.5's optimized Transformers path needs both FLA's gated-delta
# kernels and causal-conv1d. Keep this Linux/CUDA-only dependency out of
# ordinary desktop installs, but require and verify it for local Colab runs.
if MODEL_PROFILE in {"t4_quality", "t4_local"}:
    import importlib
    from importlib.metadata import version as package_version

    import torch
    from packaging.version import Version

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{MODEL_PROFILE} requires a Colab GPU runtime; CUDA is unavailable."
        )
    if Version(torch.__version__.split("+", 1)[0]) < Version("2.7"):
        raise RuntimeError(
            "Qwen3.5 fast kernels require PyTorch >= 2.7. "
            f"The runtime provides {torch.__version__}; select a current Colab GPU runtime."
        )
    capability = torch.cuda.get_device_capability(0)
    if capability < (7, 5):
        raise RuntimeError(
            "Qwen3.5 fast kernels require an NVIDIA GPU with compute capability "
            f">= 7.5; this runtime reports {capability[0]}.{capability[1]}."
        )

    qwen_fastpath_requirement = (
        "flash-linear-attention[cuda,conv1d]>=0.5.2,<0.6"
    )
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                qwen_fastpath_requirement,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Qwen3.5 fast-kernel installation failed. The notebook will not "
            "silently continue with the much slower torch fallback."
        ) from exc

    importlib.invalidate_caches()
    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )
    except Exception as exc:
        raise RuntimeError(
            "Qwen3.5 fast-kernel import verification failed after installation."
        ) from exc

    fast_kernel_functions = {
        "causal_conv1d_fn": causal_conv1d_fn,
        "causal_conv1d_update": causal_conv1d_update,
        "chunk_gated_delta_rule": chunk_gated_delta_rule,
        "fused_recurrent_gated_delta_rule": fused_recurrent_gated_delta_rule,
    }
    missing_fast_kernels = [
        name for name, function in fast_kernel_functions.items() if not callable(function)
    ]
    if missing_fast_kernels:
        raise RuntimeError(
            "Qwen3.5 fast-kernel verification found non-callable functions: "
            + ", ".join(missing_fast_kernels)
        )

    # Verify the exact import-time switch used by Transformers.  If an older
    # failed run imported Transformers before the kernels were installed, its
    # optional-backend flags stay stale until the Colab runtime is restarted.
    from transformers.models.qwen3_5 import modeling_qwen3_5

    if not getattr(modeling_qwen3_5, "is_fast_path_available", False):
        restart_hint = (
            " Transformers was already loaded before setup; restart the Colab "
            "runtime and rerun from cell 1."
            if transformers_was_loaded
            else " Restart the Colab runtime and rerun from cell 1."
        )
        raise RuntimeError(
            "Transformers did not activate the Qwen3.5 fast path after kernel "
            "installation." + restart_hint
        )

    # Imports alone cannot detect a CUDA ABI, architecture, or Triton JIT
    # mismatch.  Exercise the same prefill and cached-decode primitives Qwen3.5
    # uses before any multi-gigabyte checkpoint is downloaded.
    try:
        with torch.inference_mode():
            dtype = torch.float16
            conv_x = torch.randn((1, 8, 16), device="cuda", dtype=dtype)
            conv_weight = torch.randn((8, 4), device="cuda", dtype=dtype)
            conv_out = causal_conv1d_fn(
                conv_x, conv_weight, activation="silu"
            )
            conv_state = torch.zeros((1, 8, 4), device="cuda", dtype=dtype)
            conv_step = causal_conv1d_update(
                conv_x[:, :, -1], conv_state, conv_weight, activation="silu"
            )

            q = torch.randn((1, 16, 1, 16), device="cuda", dtype=dtype)
            k = torch.nn.functional.normalize(
                torch.randn((1, 16, 1, 16), device="cuda", dtype=torch.float32),
                dim=-1,
            ).to(dtype)
            v = torch.randn((1, 16, 1, 16), device="cuda", dtype=dtype)
            g = torch.nn.functional.logsigmoid(
                torch.randn((1, 16, 1), device="cuda", dtype=torch.float32)
            )
            beta = torch.sigmoid(
                torch.randn((1, 16, 1), device="cuda", dtype=dtype)
            )
            chunk_out, _ = chunk_gated_delta_rule(
                q, k, v, g, beta, chunk_size=16
            )
            recurrent_out, _ = fused_recurrent_gated_delta_rule(
                q[:, :1],
                k[:, :1],
                v[:, :1],
                g=g[:, :1],
                beta=beta[:, :1],
            )
            smoke_outputs = (conv_out, conv_step, chunk_out, recurrent_out)
            if any(not torch.isfinite(output).all() for output in smoke_outputs):
                raise RuntimeError("a Qwen3.5 fast kernel returned non-finite values")
            torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(
            "Qwen3.5 CUDA fast-kernel smoke test failed. The notebook will not "
            "continue with an unverified slow fallback."
        ) from exc
    print(
        "Qwen3.5 fast kernels:",
        f"flash-linear-attention={package_version('flash-linear-attention')}",
        f"causal-conv1d={package_version('causal-conv1d')}",
    )

USED_COMMIT = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    text=True,
).strip()

if SAVE_TO_GOOGLE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    OUTPUT_ROOT = "/content/drive/MyDrive/M.M.M-output"
else:
    OUTPUT_ROOT = "/content/mmm-output"

os.environ["MMM_BLOCKBENCH_WORKSPACE_ROOT"] = "/content"
os.environ["MMM_ECOSYSTEM_DISCOVERY"] = "auto"
if MODEL_PROFILE == "remote_quality":
    from getpass import getpass

    if not REMOTE_BASE_URL.startswith("https://"):
        raise ValueError("remote_quality에는 HTTPS API 기본 주소가 필요합니다.")
    if not REMOTE_TEXT_MODEL.strip():
        raise ValueError("remote_quality에는 텍스트 모델 이름이 필요합니다.")
    remote_key = getpass("원격 모델 API 키: ").strip()
    if not remote_key:
        raise ValueError("원격 모델 API 키가 비어 있습니다.")
    for role in (
        "PLANNER",
        "RESEARCH",
        "CODER",
        "CODER_SAFE",
        "VISION",
    ):
        os.environ[f"MMM_{role}_BASE_URL"] = REMOTE_BASE_URL
        os.environ[f"MMM_{role}_MODEL"] = REMOTE_TEXT_MODEL
        os.environ[f"MMM_{role}_API_KEY"] = remote_key
    os.environ["MMM_IMAGE_BASE_URL"] = REMOTE_BASE_URL
    os.environ["MMM_IMAGE_MODEL"] = (
        REMOTE_IMAGE_MODEL.strip() or REMOTE_TEXT_MODEL
    )
    os.environ["MMM_IMAGE_API_KEY"] = remote_key
    os.environ["MMM_SPEECH_BASE_URL"] = REMOTE_BASE_URL
    os.environ["MMM_SPEECH_MODEL"] = (
        REMOTE_SPEECH_MODEL.strip() or REMOTE_TEXT_MODEL
    )
    os.environ["MMM_SPEECH_API_KEY"] = remote_key

import torch

print("GitHub commit:", USED_COMMIT)
print("Python:", sys.version.split()[0])
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"VRAM free/total: {free_bytes / 2**30:.2f}/{total_bytes / 2**30:.2f} GiB")
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

REGISTRY_PATH = REPO_DIR / "config/model_registry.yaml"
if not REGISTRY_PATH.is_file():
    raise FileNotFoundError(REGISTRY_PATH)
registry = ModelRegistry().to_public_dict()
if MODEL_PROFILE not in registry["profiles"]:
    raise ValueError(f"지원하지 않는 모델 프로필: {MODEL_PROFILE}")
print("모델 레지스트리:", REGISTRY_PATH)
print("모델 프로필:", MODEL_PROFILE)
print("결과 저장 위치:", OUTPUT_ROOT)
""",
    ),
    (
        "code",
        "plan",
        """# @title 5. 게임 기획 만들기
from minecraft_mod_ai import CompleteModAISession

session = CompleteModAISession(
    output_root=OUTPUT_ROOT,
    minecraft_version="1.20.1",
    model_profile=MODEL_PROFILE,
    existing_input=EXISTING_INPUT,
)
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
