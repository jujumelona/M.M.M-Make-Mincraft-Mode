from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat import NotebookNode


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "M.M.M_Make_Mincraft_Mode_Colab.ipynb"


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

            `런타임 → 모두 실행`을 누른 뒤 마지막에 열리는 화면에서 AI와 대화하세요.

            - 새 모드: 아무 파일도 올리지 않습니다.
            - 기존 모드 수정: 두 번째 칸의 체크박스만 켜고 ZIP 하나를 올립니다.
            """,
        ),
        _markdown(
            "setup-heading",
            """
            ## 1. 준비

            AI 방식을 고른 뒤 실행하세요.

            - `built-in`: 추가 모델 없이 바로 사용
            - `local`: Colab에서 로컬 AI 모델 사용
            - `api`: OpenAI 호환 API 사용. Colab의 열쇠 아이콘에서
              `MMM_API_KEY` Secret을 먼저 등록하고 노트북 접근을 허용하세요.

            API 주소와 모델 이름은 `api`를 고를 때만 입력합니다. API 키는 입력칸,
            노트북 파일, 출력에 저장하지 않습니다.
            """,
        ),
        _code(
            "setup",
            r'''
            # @title 최신 버전 준비 및 설치
            from pathlib import Path
            import importlib
            import os
            import subprocess
            import sys

            from IPython.display import clear_output

            AI_BACKEND = "built-in"  # @param ["built-in", "local", "api"]
            API_BASE_URL = ""  # @param {type:"string"}
            API_MODEL = ""  # @param {type:"string"}
            REPO_URL = "https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode.git"
            REPO_BRANCH = "main"
            PROJECT_ROOT = Path("/content/mmm-make-mincraft-mode")

            AI_BACKEND = AI_BACKEND.strip().lower()
            if AI_BACKEND not in {"built-in", "local", "api"}:
                raise ValueError("AI 방식은 built-in, local, api 중 하나여야 합니다.")
            if AI_BACKEND == "api" and (
                not API_BASE_URL.strip() or not API_MODEL.strip()
            ):
                raise ValueError("API 방식은 API 주소와 모델 이름을 입력해야 합니다.")


            def run_git(*args: str) -> str:
                completed = subprocess.run(
                    ["git", *args],
                    check=True,
                    text=True,
                    capture_output=True,
                )
                return completed.stdout.strip()


            def normalized_repo_url(value: str) -> str:
                return value.strip().rstrip("/").removesuffix(".git").lower()


            if PROJECT_ROOT.exists():
                if not (PROJECT_ROOT / ".git").is_dir():
                    raise RuntimeError("기존 작업 폴더가 Git 저장소가 아닙니다.")
                existing_origin = run_git(
                    "-C", str(PROJECT_ROOT), "config", "--get", "remote.origin.url"
                )
                if normalized_repo_url(existing_origin) != normalized_repo_url(REPO_URL):
                    raise RuntimeError("기존 작업 폴더가 다른 프로젝트입니다.")
                run_git("-C", str(PROJECT_ROOT), "fetch", "--prune", "origin", REPO_BRANCH)
                if run_git("-C", str(PROJECT_ROOT), "branch", "--show-current") != REPO_BRANCH:
                    run_git("-C", str(PROJECT_ROOT), "checkout", REPO_BRANCH)
                run_git("-C", str(PROJECT_ROOT), "pull", "--ff-only", "origin", REPO_BRANCH)
            else:
                run_git(
                    "clone",
                    "--branch",
                    REPO_BRANCH,
                    "--single-branch",
                    REPO_URL,
                    str(PROJECT_ROOT),
                )

            required = (
                PROJECT_ROOT / ".git",
                PROJECT_ROOT / "pyproject.toml",
                PROJECT_ROOT / "minecraft_mod_ai" / "__init__.py",
                PROJECT_ROOT / "colab_app.py",
            )
            if any(not path.exists() for path in required):
                raise RuntimeError("프로젝트 파일을 완전히 받지 못했습니다.")

            pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(
                encoding="utf-8"
            )
            if 'name = "mmm-make-mincraft-mode"' not in pyproject_text:
                raise RuntimeError("받은 프로젝트 이름이 올바르지 않습니다.")

            commit_sha = run_git("-C", str(PROJECT_ROOT), "rev-parse", "HEAD")
            remote_sha = run_git(
                "-C", str(PROJECT_ROOT), "rev-parse", f"origin/{REPO_BRANCH}"
            )
            if commit_sha != remote_sha:
                raise RuntimeError("최신 GitHub 버전과 현재 실행 버전이 다릅니다.")

            os.chdir(PROJECT_ROOT)
            install_target = (
                ".[ui,local-model]" if AI_BACKEND == "local" else ".[ui]"
            )
            installed = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", install_target],
                text=True,
                capture_output=True,
            )
            if installed.returncode != 0:
                diagnostics = (installed.stderr or installed.stdout).strip()[-3000:]
                raise RuntimeError(
                    "실행 화면 설치에 실패했습니다.\n" + diagnostics
                )

            importlib.invalidate_caches()
            stale_modules = [
                module_name
                for module_name in tuple(sys.modules)
                if module_name == "minecraft_mod_ai"
                or module_name.startswith("minecraft_mod_ai.")
                or module_name == "colab_app"
            ]
            if stale_modules:
                try:
                    import gradio

                    gradio.close_all()
                except Exception:
                    pass
                for module_name in sorted(
                    stale_modules,
                    key=lambda value: value.count("."),
                    reverse=True,
                ):
                    sys.modules.pop(module_name, None)
            importlib.import_module("minecraft_mod_ai")
            clear_output(wait=True)
            print("✅ 준비 완료")
            ''',
        ),
        _markdown(
            "existing-heading",
            """
            ## 2. 기존 모드 ZIP (선택)

            새 모드라면 기본값 그대로 실행합니다.
            """,
        ),
        _code(
            "existing-input",
            r'''
            # @title 기존 모드 수정일 때만 체크
            from pathlib import Path, PurePath
            import io
            import zipfile

            from IPython.display import clear_output

            PATCH_EXISTING = False  # @param {type:"boolean"}
            existing_input = None

            if PATCH_EXISTING:
                from google.colab import files
                from minecraft_mod_ai.importer import inspect_existing_project_archive

                uploaded = files.upload()
                if len(uploaded) != 1:
                    raise ValueError("기존 모드 ZIP 하나만 올려 주세요.")
                uploaded_name, uploaded_bytes = next(iter(uploaded.items()))
                safe_name = PurePath(uploaded_name.replace("\\", "/")).name
                if not safe_name.lower().endswith(".zip"):
                    raise ValueError("ZIP 파일만 올릴 수 있습니다.")
                if len(uploaded_bytes) > 512 * 1024 * 1024:
                    raise ValueError("ZIP은 512 MiB 이하여야 합니다.")
                if not zipfile.is_zipfile(io.BytesIO(uploaded_bytes)):
                    raise ValueError("올바른 ZIP 파일이 아닙니다.")

                input_root = Path("/content/mmm-existing-input")
                input_root.mkdir(parents=True, exist_ok=True)
                input_path = input_root / safe_name
                input_path.write_bytes(uploaded_bytes)
                inspect_existing_project_archive(input_path)
                existing_input = str(input_path)
                clear_output(wait=True)
                print(f"✅ 기존 모드 준비 완료: {safe_name}")
            else:
                clear_output(wait=True)
                print("✅ 새 모드 만들기")
            ''',
        ),
        _markdown(
            "launch-heading",
            """
            ## 3. AI와 대화하기

            아래 칸을 실행하면 모드 제작 화면이 열립니다.
            """,
        ),
        _code(
            "launch",
            """
            # @title 모드 제작 화면 열기
            from colab_app import launch
            import secrets

            api_base_url = API_BASE_URL.strip() if AI_BACKEND == "api" else None
            api_model = API_MODEL.strip() if AI_BACKEND == "api" else None
            runtime_api_key = None

            if AI_BACKEND == "api":
                from google.colab import userdata

                try:
                    runtime_api_key = userdata.get("MMM_API_KEY")
                except Exception:
                    raise RuntimeError(
                        "Colab Secret MMM_API_KEY를 등록하고 노트북 접근을 허용해 주세요."
                    ) from None
                if not runtime_api_key:
                    raise RuntimeError(
                        "Colab Secret MMM_API_KEY가 비어 있거나 접근할 수 없습니다."
                    )

            OUTPUT_ROOT = "/content/mmm-output"
            ui_username = "mmm"
            ui_password = secrets.token_urlsafe(12)
            print(f"공유 화면 로그인: {ui_username} / {ui_password}")
            try:
                demo = launch(
                    output_root=OUTPUT_ROOT,
                    local_model=AI_BACKEND == "local",
                    api_base_url=api_base_url,
                    api_model=api_model,
                    api_key=runtime_api_key,
                    share=True,
                    existing_input=existing_input,
                    auth=(ui_username, ui_password),
                )
            finally:
                runtime_api_key = None
                ui_password = None
            """,
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

    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    print(f"Validated: {NOTEBOOK_PATH}")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
