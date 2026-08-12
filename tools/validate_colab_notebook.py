from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "M.M.M_Make_Mincraft_Mode_Colab.ipynb"
OBSOLETE_NOTEBOOK_PATH = ROOT / "Minecraft_Multimodal_Mod_AI_Architecture_v6.ipynb"
EXPECTED_CELL_IDS = (
    "title",
    "configuration",
    "setup",
    "existing-input",
    "registry",
    "mtp-server",
    "plan",
    "build",
    "download",
    "boundaries",
)


def validate_notebook() -> str:
    if OBSOLETE_NOTEBOOK_PATH.exists():
        raise SystemExit(f"Obsolete duplicate notebook still exists: {OBSOLETE_NOTEBOOK_PATH}")
    if not NOTEBOOK_PATH.is_file():
        raise SystemExit(f"Missing canonical notebook: {NOTEBOOK_PATH}")

    raw = NOTEBOOK_PATH.read_text(encoding="utf-8")
    notebook = nbformat.reads(raw, as_version=4)
    nbformat.validate(notebook)

    cell_ids = tuple(cell.get("id", "") for cell in notebook.cells)
    if cell_ids != EXPECTED_CELL_IDS:
        raise SystemExit(
            "Unexpected Colab cell layout: "
            f"expected={EXPECTED_CELL_IDS}, actual={cell_ids}"
        )
    if len(cell_ids) != len(set(cell_ids)):
        raise SystemExit("Duplicate Colab cell id")

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise SystemExit(f"Checked-in notebook contains execution state: {cell.id}")
        compile(cell.source, f"<colab:{cell.id}>", "exec")

    colab_name = notebook.metadata.get("colab", {}).get("name")
    if colab_name != NOTEBOOK_PATH.name:
        raise SystemExit(
            f"Notebook metadata name mismatch: expected={NOTEBOOK_PATH.name}, actual={colab_name}"
        )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main() -> int:
    print(validate_notebook())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
