"""One source identity for every checkpoint delegating to construction."""
from __future__ import annotations

import ast
import hashlib
from functools import cache
from pathlib import Path


@cache
def construction_implementation_fingerprint():
    """Hash installed construction code once per process, not once per body.

    Production writes generated projects, not the installed Python package.
    A new deployment/process computes a new fingerprint, while independent
    members in the same run share the same implementation identity.
    """
    # Follow actual local imports, including imports inside functions. An
    # executor's own source hash cannot certify the implementations it calls.
    package = Path(__file__).resolve().parent
    pending = [package / 'construction_generation.py', package / 'construction_runtime.py',
               Path(__file__).resolve()]
    sources = {}
    while pending:
        path = pending.pop()
        if path in sources:
            continue
        raw = path.read_bytes()
        sources[path] = raw
        tree = ast.parse(raw, filename=str(path))
        relative = path.relative_to(package).parts[:-1]
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level > len(relative) + 1:
                        continue
                    base = (*relative[:len(relative) + 1 - node.level],
                            *(() if node.module is None else node.module.split('.')))
                elif node.module and node.module.startswith('minecraft_mod_ai.'):
                    base = tuple(node.module.split('.')[1:])
                else:
                    continue
                modules.append(base)
                modules.extend((*base, alias.name) for alias in node.names if alias.name != '*')
            elif isinstance(node, ast.Import):
                modules.extend(tuple(alias.name.split('.')[1:]) for alias in node.names
                               if alias.name.startswith('minecraft_mod_ai.'))
            for module in modules:
                location = package.joinpath(*module)
                candidate = location.with_suffix('.py')
                if candidate.is_file():
                    pending.append(candidate)
                elif (location / '__init__.py').is_file():
                    pending.append(location / '__init__.py')
    digest = hashlib.sha256(b'mmm/construction-implementation-v1\0')
    for path in sorted(sources):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(b'\0')
        digest.update(hashlib.sha256(sources[path]).digest())
    return 'sha256:' + digest.hexdigest()
