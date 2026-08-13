from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("M.M.M_Make_Mincraft_Mode_Colab.ipynb")


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = {cell.get("id"): cell for cell in nb["cells"]}
    if "setup" not in cells or "existing-input" not in cells:
        raise SystemExit("canonical Colab cells are missing")

    setup = cells["setup"]["source"]
    if "import importlib\n" not in setup:
        setup = setup.replace(
            "import importlib.util\n",
            "import importlib\nimport importlib.util\n",
            1,
        )

    old_update = '''    subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "main"], check=True)
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
    )'''
    new_update = '''    subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "-f", "main"], check=True)
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "reset", "--hard", "refs/remotes/origin/main"],
        check=True,
    )
    subprocess.run(["git", "-C", str(REPO_DIR), "clean", "-fd"], check=True)'''
    if old_update in setup:
        setup = setup.replace(old_update, new_update, 1)
    elif new_update not in setup:
        raise SystemExit("setup checkout block not recognized")

    anchor = (
        'print("GitHub commit:", USED_COMMIT, flush=True)\n\n'
        'setup_script = REPO_DIR / "tools" / "colab_runtime_setup.py"\n'
    )
    purge = '''print("GitHub commit:", USED_COMMIT, flush=True)

# Failed or partial package imports from an older checkout must never survive a main sync.
for _name in list(sys.modules):
    if _name == "minecraft_mod_ai" or _name.startswith("minecraft_mod_ai."):
        sys.modules.pop(_name, None)
importlib.invalidate_caches()

setup_script = REPO_DIR / "tools" / "colab_runtime_setup.py"
'''
    if anchor in setup:
        setup = setup.replace(anchor, purge, 1)
    elif purge not in setup:
        raise SystemExit("setup import-purge anchor not recognized")
    cells["setup"]["source"] = setup

    cells["existing-input"]["source"] = '''# @title 3. 기존 모드 입력
# 2번 셀 뒤 GitHub main이 바뀌어도 구버전 엔진을 import하지 않는다.
import importlib
import importlib.util
import subprocess
import sys

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
LATEST_MAIN_COMMIT = subprocess.check_output(
    ["git", "-C", str(REPO_DIR), "rev-parse", "refs/remotes/origin/main"],
    text=True,
).strip()
CURRENT_ENGINE_COMMIT = subprocess.check_output(
    ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
    text=True,
).strip()

if CURRENT_ENGINE_COMMIT != LATEST_MAIN_COMMIT:
    print(
        "GitHub main changed after setup; refreshing engine:",
        CURRENT_ENGINE_COMMIT[:12],
        "->",
        LATEST_MAIN_COMMIT[:12],
        flush=True,
    )
    subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "-f", "main"], check=True)
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "reset", "--hard", "refs/remotes/origin/main"],
        check=True,
    )
    subprocess.run(["git", "-C", str(REPO_DIR), "clean", "-fd"], check=True)

    for _name in list(sys.modules):
        if _name == "minecraft_mod_ai" or _name.startswith("minecraft_mod_ai."):
            sys.modules.pop(_name, None)
    importlib.invalidate_caches()

    previous_commit = CURRENT_ENGINE_COMMIT
    USED_COMMIT = LATEST_MAIN_COMMIT
    REMOTE_COMMIT = LATEST_MAIN_COMMIT
    setup_script = REPO_DIR / "tools" / "colab_runtime_setup.py"
    setup_module_name = f"_mmm_colab_runtime_setup_{USED_COMMIT[:12]}"
    setup_spec = importlib.util.spec_from_file_location(setup_module_name, setup_script)
    if setup_spec is None or setup_spec.loader is None:
        raise RuntimeError(f"Cannot reload Colab setup script: {setup_script}")
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
        transformers_was_loaded="transformers" in sys.modules,
        engine_was_loaded=False,
        engine_module_file="",
        previous_commit=previous_commit,
    )
    REPO_DIR = Path(SETUP_STATE["repo_dir"])
    OUTPUT_ROOT = SETUP_STATE["output_root"]
    SETUP_RECEIPT = SETUP_STATE["receipt"]
    SETUP_FINGERPRINT = SETUP_STATE["setup_fingerprint"]

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

from minecraft_mod_ai.colab_run_modes import prepare_existing_mod_input

EXISTING_INPUT = prepare_existing_mod_input(RUN_MODE)
'''

    # Every code cell must parse before the notebook can be committed.
    code_cells = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        compile(cell.get("source", ""), f"<cell:{cell.get('id', '?')}>", "exec")
        code_cells += 1
    if code_cells < 7:
        raise SystemExit(f"unexpected code-cell count: {code_cells}")

    setup_source = cells["setup"]["source"]
    existing_source = cells["existing-input"]["source"]
    required_setup = (
        'reset", "--hard", "refs/remotes/origin/main',
        'clean", "-fd"',
        "importlib.invalidate_caches()",
    )
    for marker in required_setup:
        if marker not in setup_source:
            raise SystemExit(f"setup marker missing: {marker}")
    required_existing = (
        "LATEST_MAIN_COMMIT",
        "CURRENT_ENGINE_COMMIT",
        "COLAB_SETUP_MODULE.assert_setup_state",
        "from minecraft_mod_ai.colab_run_modes import prepare_existing_mod_input",
    )
    for marker in required_existing:
        if marker not in existing_source:
            raise SystemExit(f"cell 3 marker missing: {marker}")

    NOTEBOOK.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
