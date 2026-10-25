"""Bounded discovery of executable Java project-source dependencies."""
from __future__ import annotations

import re
from collections import deque
from pathlib import Path, PurePosixPath

MAX_CLOSURE_FILES = 80
MAX_CLOSURE_BYTES = 4_000_000
MAX_SOURCE_BYTES = 200_000
_IMPORT = re.compile(r'\bimport\s+(static\s+)?([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)(\.\*)?\s*;')
_PACKAGE = re.compile(r'(?m)^\s*package\s+([\w.]+)\s*;')


def _mask_literal(source, result, index, state):
    if source[index] == '\\':
        result[index:index + 2] = ' ' * min(2, len(source) - index)
        return index + 2, state
    if state == 'text_block' and source.startswith('"""', index):
        result[index:index + 3] = '   '
        return index + 3, 'code'
    if source[index] == state:
        state = 'code'
    if source[index] != '\n':
        result[index] = ' '
    return index + 1, state


def _mask_non_java_code(source: str) -> str:
    """Preserve executable tokens while blanking comments and string literals.

    Java import and type discovery is not permitted to trust text in comments,
    character literals or text blocks as actual source references.
    """
    result = list(source)
    state = 'code'
    index = 0
    while index < len(source):
        two = source[index:index + 2]
        three = source[index:index + 3]
        if state == 'code':
            if two == '//':
                result[index:index + 2] = '  '
                state = 'line'
                index += 2
                continue
            if two == '/*':
                result[index:index + 2] = '  '
                state = 'block'
                index += 2
                continue
            if three == '"""':
                result[index:index + 3] = '   '
                state = 'text_block'
                index += 3
                continue
            if source[index] in ('"', "'"):
                state = source[index]
                result[index] = ' '
        elif state == 'line':
            if source[index] == '\n':
                state = 'code'
            else:
                result[index] = ' '
        elif state == 'block':
            if two == '*/':
                result[index:index + 2] = '  '
                state = 'code'
                index += 2
                continue
            if source[index] != '\n':
                result[index] = ' '
        else:
            index, state = _mask_literal(source, result, index, state)
            continue
        index += 1
    return ''.join(result)


def _relative_java_path(value: str) -> Path:
    """Reject paths that can escape the source staging tree, on any host OS."""
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise ValueError('CONSTRUCTION_COMPILE_SOURCE_PATH_INVALID')
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or '..' in candidate.parts or not value.endswith('.java'):
        raise ValueError('CONSTRUCTION_COMPILE_SOURCE_PATH_INVALID')
    if ':' in value or not candidate.parts or candidate.parts[0] not in {'src', 'generated'}:
        raise ValueError('CONSTRUCTION_COMPILE_SOURCE_PATH_INVALID')
    return Path(*candidate.parts)


def _materialized_file(root: Path, relative: Path) -> Path | None:
    original = root / relative
    # Do not follow symlinks, including symlinked parent directories.
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return None
    resolved = original.resolve()
    if (not resolved.is_relative_to(root) or not resolved.is_file()
            or resolved.stat().st_size > MAX_SOURCE_BYTES):
        return None
    return resolved


def _java_roots(target_path: Path) -> tuple[Path, ...]:
    parts = target_path.parts
    if len(parts) >= 3 and parts[0] == 'src' and parts[2] == 'java':
        # A Fabric client compilation unit may depend on unbuilt main sources.
        if parts[1] == 'client':
            return (Path('src/client/java'), Path('src/main/java'))
        return (Path(*parts[:3]),)
    if len(parts) >= 2 and parts[0] == 'generated':
        return (Path('generated'),)
    return (Path('src/main/java'), Path('src/client/java'))


def _package_source_paths(package, referenced, *, root, roots):
    base = Path(*package.split('.'))
    for tree in roots:
        directory = root / tree / base
        if directory.is_dir() and not directory.is_symlink():
            for item in sorted(directory.glob('*.java')):
                if item.stem in referenced:
                    yield tree / base / item.name


def _referenced_source_paths(source, *, root, roots):
    """Discover existing files; javac still decides the actual Java identities."""
    visible = _mask_non_java_code(source)
    referenced = set(re.findall(r'\b[A-Za-z_$][\w$]*\b', visible))
    package = _PACKAGE.search(visible)
    if package:
        yield from _package_source_paths(package[1], referenced, root=root, roots=roots)
    for match in _IMPORT.finditer(visible):
        if match[3] and not match[1]:
            yield from _package_source_paths(match[2], referenced, root=root, roots=roots)
    # An import is a declared type dependency. Arbitrary dotted expressions
    # can instead be local variables/fields that merely shadow a package name.
    for match in _IMPORT.finditer(visible):
        if match[3] and not match[1]:
            continue
        names = match[2].split('.')
        for stop in range(len(names), 1, -1):
            qualified = Path(*names[:stop]).with_suffix('.java')
            for tree in roots:
                relative = tree / qualified
                if (root / relative).is_file():
                    yield relative


def _discover_closure(source: str, *, root: Path, target: Path, source_context) -> list[tuple[Path, Path]]:
    """Resolve bounded prerequisites without replacing the current candidate."""
    roots = _java_roots(target)
    queue, queued = deque(), {target}
    def enqueue(paths):
        for relative in paths:
            if relative not in queued:
                queue.append(relative)
                queued.add(relative)
    enqueue(_relative_java_path(record['path']) for record in source_context
            if isinstance(record, dict) and isinstance(record.get('path'), str)
            and record['path'].endswith('.java'))
    enqueue(_referenced_source_paths(source, root=root, roots=roots))
    visited: set[Path] = {target}
    result = []
    total_bytes = 0
    while queue:
        relative = queue.popleft()
        if relative in visited:
            continue
        visited.add(relative)
        if len(visited) > MAX_CLOSURE_FILES + 1:
            raise ValueError('CONSTRUCTION_COMPILE_SOURCE_CLOSURE_TOO_LARGE')
        original = _materialized_file(root, relative)
        if original is None:
            continue
        size = original.stat().st_size
        total_bytes += size
        if total_bytes > MAX_CLOSURE_BYTES:
            raise ValueError('CONSTRUCTION_COMPILE_SOURCE_CLOSURE_TOO_LARGE')
        result.append((relative, original))
        actual_source = original.read_text(encoding='utf-8')
        enqueue(_referenced_source_paths(actual_source, root=root, roots=roots))
    return result


