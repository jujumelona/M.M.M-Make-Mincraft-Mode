"""Byte identity of project inputs, separate from the text retrieval index."""
from __future__ import annotations

import hashlib
from pathlib import Path

_OUTPUT_ROOTS = frozenset({'.git', '.gradle', 'build', 'run', '.cache', 'node_modules',
                           '.minecraft_ai', '.mmm'})


def build_input_manifest_sha256(project_root):
    root = Path(project_root).resolve()
    digest = hashlib.sha256(b'mmm/build-inputs-v1\0')
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(part in _OUTPUT_ROOTS for part in relative.parts) or path.is_dir():
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError('BUILD_INPUT_NOT_REGULAR: ' + relative.as_posix())
        content = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                content.update(chunk)
        digest.update(relative.as_posix().encode('utf-8'))
        digest.update(b'\0')
        digest.update(content.digest())
    return 'sha256:' + digest.hexdigest()
