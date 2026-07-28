from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat import NotebookNode


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "Minecraft_Multimodal_Mod_AI_Architecture_v6.ipynb"


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
            # Minecraft Mod AI — Google Colab

            ## Goal

            이 노트북은 매번 GitHub `main`의 최신 Minecraft Mod AI 엔진을 받아 설치하고,
            승인 기반 UI에서 새 Fabric 모드를 생성합니다.

            - 엔진 자체를 ZIP으로 업로드하지 않습니다.
            - 실제로 사용한 Git commit SHA를 출력해 실행 버전을 추적합니다.
            - 이미 만든 모드를 수정하려는 경우에만 별도의 **선택적 기존 모드 ZIP** 셀을 켭니다.
            - 생성 결과의 release ZIP은 UI의 다운로드 항목에서 받습니다.
            """,
        ),
        _markdown(
            "sync-heading",
            """
            ## Setup

            ### 1. GitHub `main` 동기화 및 저장소 확인

            `/content/minecraft-mod-ai`가 없으면 공식 프로젝트 URL에서 clone합니다. 이미 있으면
            같은 저장소인지 확인한 뒤 `fetch`와 `pull --ff-only`만 수행합니다. 로컬 변경이나
            분기 충돌이 있으면 덮어쓰지 않고 중단합니다.
            """,
        ),
        _code(
            "sync-repository",
            """
            from pathlib import Path
            import subprocess
            import tomllib

            REPO_URL = "https://github.com/jujumelona/minecraft-mod-ai.git"
            REPO_BRANCH = "main"
            PROJECT_ROOT = Path("/content/minecraft-mod-ai")


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
            if project_name != "minecraft-mod-ai":
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
            ### 2. UI 의존성 설치

            저장소 marker와 `pyproject.toml`의 프로젝트 이름을 확인한 뒤 현재 Python 커널에
            UI를 설치합니다. 기본값은 가벼운 결정론 planner이며, 선택적 로컬 모델 worker가
            필요할 때만 `INSTALL_LOCAL_MODEL = True`로 바꿉니다.
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
            print(f"Installed minecraft-mod-ai {minecraft_mod_ai.__version__}")
            print(f"Running commit: {commit_sha}")
            """,
        ),
        _markdown(
            "existing-input-heading",
            """
            ## Optional existing-mod revision input

            ### 3. 이미 만든 모드 또는 이전 release ZIP 가져오기

            새 모드를 만들 때는 `PATCH_EXISTING = False`를 그대로 둡니다. 기존 모드를
            분석하거나 수정 컨텍스트로 사용할 때만 `True`로 바꾸고 **source ZIP 또는
            release ZIP 정확히 하나**를 업로드합니다.

            이 셀은 ZIP을 저장한 뒤 프로젝트의 fail-closed importer로 경로·심볼릭 링크·
            중복·압축 해제 크기·자격증명 경로·Fabric metadata·파일별 SHA-256을 검사합니다.
            압축을 풀지 않고 Gradle wrapper·JAR·Python·셸·배치 등 내부 파일을 실행하지
            않습니다. Java/Kotlin 소스가 없고 JAR만 든 release ZIP은 편집 가능한 소스가
            아니므로 **metadata analysis 전용**으로 전달됩니다.
            """,
        ),
        _code(
            "optional-existing-input",
            """
            from pathlib import Path, PurePath
            import hashlib
            import io
            import json
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

                input_root = Path("/content/minecraft-mod-ai-existing-input")
                input_root.mkdir(parents=True, exist_ok=True)
                input_path = input_root / safe_name
                input_path.write_bytes(uploaded_bytes)
                existing_input = str(input_path)

                input_sha256 = hashlib.sha256(uploaded_bytes).hexdigest()
                from minecraft_mod_ai.importer import inspect_existing_project_archive

                import_report = inspect_existing_project_archive(input_path)
                print(f"Existing input: {existing_input}")
                print(f"SHA-256: {input_sha256}")
                print(f"ZIP files: {len(file_infos)}; JAR files: {jar_count}")
                print("Archive contents were inspected as data and were not executed.")
                print(json.dumps(import_report.to_dict(), ensure_ascii=False, indent=2))
                if jar_count and not has_editable_source:
                    print("JAR-only release detected: metadata analysis mode.")
            else:
                print("PATCH_EXISTING=False — 새 모드 생성 모드입니다.")
            """,
        ),
        _markdown(
            "environment-heading",
            """
            ## Checks

            ### 4. Colab 실행 환경 확인

            기본 deterministic planner는 GPU가 없어도 동작합니다. 아래 정보는 문제 재현을
            위한 환경 기록입니다.
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
                print("GPU not detected; deterministic planner remains available.")
            """,
        ),
        _markdown(
            "launch-heading",
            """
            ## Steps

            ### 5. 승인 기반 UI 실행

            공유 URL이 출력되면 UI에서 계획을 확인하고, 표시된 SHA-256 승인 해시를 직접
            입력한 뒤 실행합니다. 검증된 생성이 끝나면 UI의 **release ZIP** 파일 항목에서
            바로 다운로드할 수 있습니다. 선택적 업로드 경로는 `existing_input`으로만
            전달되며 노트북이 그 내용을 실행하지 않습니다.
            """,
        ),
        _code(
            "launch-ui",
            """
            from colab_app import launch

            OUTPUT_ROOT = "/content/minecraft-mod-ai-output"
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
            ## Next Steps

            UI의 release ZIP에는 생성 결과와 검증 증거가 함께 포함됩니다. 런타임을 다시
            시작하면 첫 번째 셀부터 실행해 그 시점의 GitHub `main`을 다시 동기화하고,
            출력된 exact commit SHA를 함께 기록하세요.
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
