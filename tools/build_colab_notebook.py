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
    return cell


def build_notebook() -> NotebookNode:
    cells = [
        _markdown(
            "title-and-goal",
            """
            # M.M.M Make Mincraft Mode — Google Colab

            자연어로 Fabric 모드를 만들고 release ZIP으로 내려받습니다.

            - 새 모드: 업로드 없이 그대로 실행합니다.
            - 기존 모드 수정: `PATCH_EXISTING = True`로 바꾸고 ZIP 하나를 올립니다.
            """,
        ),
        _markdown(
            "sync-heading",
            """
            ## Setup

            ### 1. 최신 버전 준비

            GitHub `main`의 최신 버전을 자동으로 설치합니다.
            """,
        ),
        _code(
            "sync-repository",
            """
            from pathlib import Path
            import subprocess
            import tomllib

            REPO_URL = "https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode.git"
            REPO_BRANCH = "main"
            PROJECT_ROOT = Path("/content/mmm-make-mincraft-mode")


            def run_git(*args: str) -> str:
                completed = subprocess.run(
                    ["git", *args],
                    check=True,
                    text=True,
                    capture_output=True,
                )
                if completed.stderr.strip():
                    print(completed.stderr.strip())
                return completed.stdout.strip()


            def normalized_repo_url(value: str) -> str:
                return value.strip().rstrip("/").removesuffix(".git").lower()


            if PROJECT_ROOT.exists():
                if not (PROJECT_ROOT / ".git").is_dir():
                    raise RuntimeError(
                        f"{PROJECT_ROOT}가 존재하지만 Git 저장소가 아닙니다. "
                        "내용을 확인한 뒤 다른 런타임에서 다시 실행하세요."
                    )
                existing_origin = run_git(
                    "-C", str(PROJECT_ROOT), "config", "--get", "remote.origin.url"
                )
                if normalized_repo_url(existing_origin) != normalized_repo_url(REPO_URL):
                    raise RuntimeError(
                        "기존 디렉터리의 origin이 예상 저장소와 다릅니다: "
                        f"{existing_origin!r}"
                    )
                run_git(
                    "-C",
                    str(PROJECT_ROOT),
                    "fetch",
                    "--prune",
                    "origin",
                    REPO_BRANCH,
                )
                current_branch = run_git(
                    "-C", str(PROJECT_ROOT), "branch", "--show-current"
                )
                if current_branch != REPO_BRANCH:
                    run_git("-C", str(PROJECT_ROOT), "checkout", REPO_BRANCH)
                run_git(
                    "-C",
                    str(PROJECT_ROOT),
                    "pull",
                    "--ff-only",
                    "origin",
                    REPO_BRANCH,
                )
            else:
                run_git(
                    "clone",
                    "--branch",
                    REPO_BRANCH,
                    "--single-branch",
                    REPO_URL,
                    str(PROJECT_ROOT),
                )

            required_markers = [
                PROJECT_ROOT / ".git",
                PROJECT_ROOT / "pyproject.toml",
                PROJECT_ROOT / "minecraft_mod_ai" / "__init__.py",
                PROJECT_ROOT / "colab_app.py",
            ]
            missing_markers = [str(path) for path in required_markers if not path.exists()]
            if missing_markers:
                raise RuntimeError(
                    "필수 저장소 marker가 없습니다: " + ", ".join(missing_markers)
                )

            with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
                project_metadata = tomllib.load(pyproject_file)
            project_name = project_metadata.get("project", {}).get("name")
            if project_name != "mmm-make-mincraft-mode":
                raise RuntimeError(
                    f"예상하지 않은 pyproject project.name: {project_name!r}"
                )

            commit_sha = run_git(
                "-C", str(PROJECT_ROOT), "rev-parse", "--verify", "HEAD"
            )
            origin_main_sha = run_git(
                "-C",
                str(PROJECT_ROOT),
                "rev-parse",
                "--verify",
                f"origin/{REPO_BRANCH}",
            )
            if commit_sha != origin_main_sha:
                raise RuntimeError(
                    f"현재 HEAD({commit_sha})가 origin/{REPO_BRANCH}"
                    f"({origin_main_sha})와 다릅니다."
                )

            print(f"Repository: {REPO_URL}")
            print(f"Branch: {REPO_BRANCH}")
            print(f"Exact commit SHA: {commit_sha}")
            """,
        ),
        _markdown(
            "install-heading",
            """
            ### 2. 실행 화면 설치

            기본값 그대로 실행하면 됩니다.
            """,
        ),
        _code(
            "install-project",
            """
            import importlib
            import os
            import subprocess
            import sys

            INSTALL_LOCAL_MODEL = False
            os.chdir(PROJECT_ROOT)
            install_target = (
                ".[ui,local-model]" if INSTALL_LOCAL_MODEL else ".[ui]"
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", install_target],
                check=True,
            )

            minecraft_mod_ai = importlib.import_module("minecraft_mod_ai")
            print(f"Installed M.M.M Make Mincraft Mode {minecraft_mod_ai.__version__}")
            print(f"Running commit: {commit_sha}")
            """,
        ),
        _markdown(
            "existing-input-heading",
            """
            ### 3. 기존 모드 ZIP (선택)

            새 모드는 `PATCH_EXISTING = False`로 실행합니다. 기존 모드를 수정할 때만
            `True`로 바꾸고 source/release ZIP 하나를 올립니다.
            """,
        ),
        _code(
            "optional-existing-input",
            """
            from pathlib import Path, PurePath
            import io
            import zipfile

            PATCH_EXISTING = False
            existing_input = None

            if PATCH_EXISTING:
                from google.colab import files

                uploaded = files.upload()
                if len(uploaded) != 1:
                    raise ValueError(
                        f"기존 모드 ZIP 하나만 업로드하세요. 받은 파일 수: {len(uploaded)}"
                    )

                uploaded_name, uploaded_bytes = next(iter(uploaded.items()))
                safe_name = PurePath(uploaded_name.replace("\\\\", "/")).name
                if not safe_name.lower().endswith(".zip"):
                    raise ValueError("기존 모드 입력은 source/release ZIP이어야 합니다.")
                if len(uploaded_bytes) > 512 * 1024 * 1024:
                    raise ValueError("기존 모드 ZIP은 512 MiB 이하여야 합니다.")
                if not zipfile.is_zipfile(io.BytesIO(uploaded_bytes)):
                    raise ValueError("업로드한 파일이 유효한 ZIP이 아닙니다.")

                with zipfile.ZipFile(io.BytesIO(uploaded_bytes)) as archive:
                    file_infos = [item for item in archive.infolist() if not item.is_dir()]
                    if len(file_infos) > 10_000:
                        raise ValueError("ZIP 항목 수가 10,000개를 초과합니다.")
                    archive_names = [item.filename.replace("\\\\", "/") for item in file_infos]
                    has_editable_source = any(
                        name.lower().endswith((".java", ".kt"))
                        for name in archive_names
                    )
                    jar_count = sum(
                        name.lower().endswith(".jar") for name in archive_names
                    )

                input_root = Path("/content/mmm-existing-input")
                input_root.mkdir(parents=True, exist_ok=True)
                input_path = input_root / safe_name
                input_path.write_bytes(uploaded_bytes)
                existing_input = str(input_path)

                from minecraft_mod_ai.importer import inspect_existing_project_archive

                inspect_existing_project_archive(input_path)
                print(f"기존 모드 ZIP 준비 완료: {safe_name}")
                print(f"파일 수: {len(file_infos)}")
                if jar_count and not has_editable_source:
                    print("소스가 없는 JAR ZIP입니다.")
            else:
                print("PATCH_EXISTING=False — 새 모드 생성 모드입니다.")
            """,
        ),
        _markdown(
            "environment-heading",
            """
            ### 4. 실행 환경 확인
            """,
        ),
        _code(
            "environment-check",
            """
            import platform
            import shutil
            import subprocess

            print(f"Python: {platform.python_version()}")
            print(f"Project: {PROJECT_ROOT}")
            print(f"Commit: {commit_sha}")

            nvidia_smi = shutil.which("nvidia-smi")
            if nvidia_smi:
                subprocess.run(
                    [
                        nvidia_smi,
                        "--query-gpu=name,memory.total,memory.free",
                        "--format=csv,noheader",
                    ],
                    check=False,
                )
            else:
                print("GPU가 없어도 기본 모드로 실행할 수 있습니다.")
            """,
        ),
        _markdown(
            "launch-heading",
            """
            ### 5. 모드 만들기

            열린 화면에서 요청을 입력하고 `계획 생성 → 승인 후 실행`을 누릅니다.
            완료되면 **release ZIP**을 내려받습니다.
            """,
        ),
        _code(
            "launch-ui",
            """
            from colab_app import launch

            OUTPUT_ROOT = "/content/mmm-output"
            demo = launch(
                output_root=OUTPUT_ROOT,
                local_model=INSTALL_LOCAL_MODEL,
                share=True,
                existing_input=existing_input,
            )
            """,
        ),
        _markdown(
            "next-steps",
            """
            ## 완료

            화면 아래에서 release ZIP을 내려받으세요.
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
