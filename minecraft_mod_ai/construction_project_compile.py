"""Target-JDK compilation of one generated Java unit and its real source closure.

This stage is validation, not a source publisher: no project files are modified.
It cannot certify gameplay; production Gradle and GameTest remain authoritative.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

MAX_CLOSURE_FILES = 80
MAX_CLOSURE_BYTES = 4_000_000
MAX_SOURCE_BYTES = 200_000
_IMPORT = re.compile(r'\bimport\s+(static\s+)?([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)(\.\*)?\s*;')
_PACKAGE = re.compile(r'(?m)^\s*package\s+([\w.]+)\s*;')


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
                state = 'line'; index += 2; continue
            if two == '/*':
                result[index:index + 2] = '  '
                state = 'block'; index += 2; continue
            if three == '"""':
                result[index:index + 3] = '   '
                state = 'text_block'; index += 3; continue
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
                state = 'code'; index += 2; continue
            if source[index] != '\n':
                result[index] = ' '
        elif state == 'text_block':
            if three == '"""':
                result[index:index + 3] = '   '
                state = 'code'; index += 3; continue
            if source[index] != '\n':
                result[index] = ' '
        else:
            if source[index] == '\\':
                result[index] = ' '
                if index + 1 < len(source):
                    result[index + 1] = ' '
                index += 2; continue
            if source[index] == state:
                state = 'code'
            result[index] = ' '
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


def _discover_closure(source: str, *, root: Path, target: Path, source_context) -> list[tuple[Path, Path]]:
    """Stage explicit prerequisites plus source files imported or used by owner.

    The selected output is never overwritten by an older on-disk source. Inspect
    local package siblings and direct imports, then follow their imports. The
    bounded closure is resolved by javac; missing files remain compiler errors.
    """
    initial = []
    for record in source_context:
        if not isinstance(record, dict) or not isinstance(record.get('path'), str):
            continue
        if record['path'].endswith('.java'):
            initial.append(_relative_java_path(record['path']))
    roots = _java_roots(target)
    queue: list[Path] = []
    visited: set[Path] = {target}
    def enqueue(p: Path):
        if p not in visited and p not in queue:
            queue.append(p)
    for candidate in initial:
        enqueue(candidate)
    def referenced_siblings(code: str):
        # Java package members require no import. Only inspect identifier
        # references in executable code, not class names in comments.
        visible = _mask_non_java_code(code)
        package = _PACKAGE.search(visible)
        if not package:
            return
        referenced = set(re.findall(r'\b[A-Za-z_$][A-Za-z0-9_$]*\b', visible))
        base = Path(*package.group(1).split('.'))
        for tree in roots:
            sibling = root / tree / base
            if sibling.is_dir() and not sibling.is_symlink():
                for item in sorted(sibling.glob('*.java')):
                    if item.stem in referenced:
                        enqueue(tree / base / item.name)

    def imported_paths(code: str):
        visible = _mask_non_java_code(code)
        referenced = set(re.findall(r'\b[A-Za-z_$][A-Za-z0-9_$]*\b', visible))
        for match in _IMPORT.finditer(visible):
            is_static = bool(match.group(1))
            dotted = match.group(2)
            is_wildcard = bool(match.group(3))
            names = dotted.split('.')
            # "import com.example.*" refers to a package, whereas a static
            # wildcard import refers to a class. Resolve package members by
            # referenced class names only, never include an entire package.
            if is_wildcard and not is_static:
                base = Path(*names)
                for tree in roots:
                    directory = root / tree / base
                    if directory.is_dir() and not directory.is_symlink():
                        for item in sorted(directory.glob('*.java')):
                            if item.stem in referenced:
                                enqueue(tree / base / item.name)
                continue
            # Static imports may end in a member name. Resolve the longest
            # existing class prefix rather than treating members as classes.
            for stop in range(len(names), 1, -1):
                qualified = Path(*names[:stop]).with_suffix('.java')
                for tree in roots:
                    relative = tree / qualified
                    if (root / relative).is_file():
                        enqueue(relative)
                        break
    imported_paths(source)
    referenced_siblings(source)
    result = []
    total_bytes = 0
    while queue:
        relative = queue.pop(0)
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
        imported_paths(actual_source)
        referenced_siblings(actual_source)
    return result


def verify_java_source_with_dependencies(source: str, *, context: dict,
                                         target_path: str, filename: str,
                                         source_context=()):
    paths = context.get('java_classpath')
    if not paths:
        raise ValueError('CONSTRUCTION_COMPILE_CLASSPATH_REQUIRED')
    target = context['target']
    java_version = str(target['java_version'])
    root = Path(context['project_root']).resolve()
    relative_target = _relative_java_path(target_path)
    if relative_target.name != filename:
        raise ValueError('CONSTRUCTION_COMPILE_TARGET_IDENTITY_MISMATCH')
    related = _discover_closure(source, root=root, target=relative_target,
                                source_context=source_context)
    from .java_lsp import _resolve_project_java_home
    jdk = Path(_resolve_project_java_home(int(java_version), require_compiler=True))
    compiler = jdk / 'bin' / ('javac.exe' if os.name == 'nt' else 'javac')
    if not compiler.is_file():
        raise ValueError('CONSTRUCTION_COMPILE_JAVAC_UNAVAILABLE: ' + java_version)
    with tempfile.TemporaryDirectory(prefix='mmm-full-source-') as scratch:
        temp = Path(scratch)
        root_src = temp / 'sources'
        generated = root_src / relative_target
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text(source, encoding='utf-8')
        inputs = [generated]
        for relative, original in related:
            staged = root_src / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(original.read_bytes())
            inputs.append(staged)
        output_dir = temp / 'compiled'
        output_dir.mkdir()
        cp = os.pathsep.join(str(Path(item).resolve()) for item in paths)
        source_paths = os.pathsep.join(str(root_src / tree) for tree in _java_roots(relative_target))
        cmd = [str(compiler), '-proc:none', '-encoding', 'UTF-8', '-Xdiags:verbose',
               '--release', java_version, '-classpath', cp, '-sourcepath', source_paths,
               '-d', str(output_dir), *map(str, inputs)]
        result = subprocess.run(cmd, cwd=temp, capture_output=True,
                                text=True, encoding='utf-8', errors='replace', timeout=120)
        if result.returncode:
            raise ValueError('JAVA_PROJECT_ANALYSIS_FAILED: ' +
                             (result.stderr or result.stdout)[-14000:])
    return []
