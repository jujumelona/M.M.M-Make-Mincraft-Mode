from __future__ import annotations

import base64
import hashlib
import json
import lzma
import shutil
import subprocess
from pathlib import Path

BRANCH = "feat/real-multimodal-mcp-router"
EXPECTED_B64_LENGTH = 64008
EXPECTED_B64_SHA256 = "7fd7296d7714b000021e35639b4a486e220a0e133db14d19034d2db15f76371c"
EXPECTED_PAYLOAD_SHA256 = "f53ed184fc08899f0367f48fb86314772aa4ce89fce406fa273fe4f1b844d080"
EXPECTED_FILE_COUNT = 87


def main() -> None:
    root = Path.cwd().resolve()
    chunk_dir = root / "tools" / ".patch_chunks"
    chunks = sorted(path for path in chunk_dir.glob("*.txt") if path.name != "READY")
    if not chunks:
        raise RuntimeError("No production patch chunks were found.")

    encoded = "".join(path.read_text(encoding="utf-8").strip() for path in chunks)
    if len(encoded) != EXPECTED_B64_LENGTH:
        raise RuntimeError(
            f"Patch Base64 length mismatch: {len(encoded)} != {EXPECTED_B64_LENGTH}"
        )
    encoded_sha = hashlib.sha256(encoded.encode("ascii")).hexdigest()
    if encoded_sha != EXPECTED_B64_SHA256:
        raise RuntimeError(
            f"Patch Base64 SHA-256 mismatch: {encoded_sha} != {EXPECTED_B64_SHA256}"
        )

    compressed = base64.b64decode(encoded, validate=True)
    raw = lzma.decompress(compressed)
    payload_sha = hashlib.sha256(raw).hexdigest()
    if payload_sha != EXPECTED_PAYLOAD_SHA256:
        raise RuntimeError(
            f"Patch payload SHA-256 mismatch: {payload_sha} != {EXPECTED_PAYLOAD_SHA256}"
        )
    files = json.loads(raw.decode("utf-8"))
    if not isinstance(files, dict) or len(files) != EXPECTED_FILE_COUNT:
        raise RuntimeError(
            f"Patch file count mismatch: {len(files) if isinstance(files, dict) else 'invalid'} "
            f"!= {EXPECTED_FILE_COUNT}"
        )

    for relative, content in files.items():
        if not isinstance(relative, str) or not isinstance(content, str):
            raise RuntimeError("Patch payload contains a non-text file entry.")
        target = (root / relative).resolve()
        target.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    shutil.rmtree(chunk_dir)
    for relative in (
        "tools/apply_production_patch_bundle.py",
        ".github/workflows/apply-production-patch.yml",
    ):
        path = root / relative
        if path.exists():
            path.unlink()

    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
    subprocess.run(
        [
            "git",
            "config",
            "user.email",
            "41898282+github-actions[bot]@users.noreply.github.com",
        ],
        check=True,
    )
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Apply complete production stack, MCP, skills and training patch",
        ],
        check=True,
    )
    subprocess.run(["git", "push", "origin", f"HEAD:{BRANCH}"], check=True)


if __name__ == "__main__":
    main()
