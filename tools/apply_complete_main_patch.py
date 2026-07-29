from __future__ import annotations

import base64
import hashlib
import json
import lzma
import shutil
import subprocess
from pathlib import Path

EXPECTED_B64_LENGTH = 86992
EXPECTED_B64_SHA256 = "066a13e968d6aca41344a21dbe3bb1f3338761d11f501ff87074740dda7db594"
EXPECTED_RAW_SHA256 = "9877b19f94b628bc29fb7fa873c3243bee42caed3cd12355c46f24c9b367a262"
EXPECTED_FILE_COUNT = 41
TEMP_MARKERS = (
    "docs/.tree-api-test.tmp",
    "docs/.direct-main-progress.txt",
    "docs/.integration-state.json",
    "docs/.do-not-delete-branches-before-verification.txt",
    "docs/.main-only-policy.txt",
)


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    root = Path.cwd().resolve()
    chunk_dir = root / "tools/.complete_patch_chunks"
    chunks = sorted(chunk_dir.glob("*.txt"))
    encoded = "".join(path.read_text(encoding="utf-8").strip() for path in chunks)
    if len(encoded) != EXPECTED_B64_LENGTH:
        raise RuntimeError(f"bundle length mismatch: {len(encoded)}")
    if hashlib.sha256(encoded.encode("ascii")).hexdigest() != EXPECTED_B64_SHA256:
        raise RuntimeError("bundle base64 SHA-256 mismatch")
    raw = lzma.decompress(base64.b64decode(encoded, validate=True))
    if hashlib.sha256(raw).hexdigest() != EXPECTED_RAW_SHA256:
        raise RuntimeError("bundle raw SHA-256 mismatch")
    files = json.loads(raw.decode("utf-8"))
    if not isinstance(files, dict) or len(files) != EXPECTED_FILE_COUNT:
        raise RuntimeError("bundle file-count mismatch")
    for relative, content in files.items():
        if not isinstance(relative, str) or not isinstance(content, str):
            raise RuntimeError("bundle contains non-text entry")
        target = (root / relative).resolve()
        target.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    for relative in TEMP_MARKERS:
        marker = root / relative
        if marker.exists():
            marker.unlink()
    shutil.rmtree(chunk_dir)
    Path(__file__).unlink()

    run("python", "-m", "compileall", "-q", "minecraft_mod_ai", "tools", "mcp_gateway.py", "download_resources.py")
    run("python", "tools/build_colab_notebook.py", "--check")
    run("pytest", "-q")

    run("git", "config", "user.name", "github-actions[bot]")
    run("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    run("git", "add", "-A")
    run("git", "commit", "-m", "Integrate complete Minecraft production lifecycle on main")
    run("git", "push", "origin", "HEAD:main")


if __name__ == "__main__":
    main()
