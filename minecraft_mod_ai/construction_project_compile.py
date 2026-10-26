"""Target-JDK compilation of one generated Java unit and its real source closure.

This stage is validation, not a source publisher: no project files are modified.
It cannot certify gameplay; production Gradle and GameTest remain authoritative.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from .java_source_closure import (
    _discover_closure as _discover_closure,
    _java_roots,
    _relative_java_path,
)


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
    from .java_source_snapshot import materialized_java_sources, direct_source_dependencies
    source_options = direct_source_dependencies(context)
    companions = materialized_java_sources(source, project_root=root,
        target_path=target_path, source_context=source_context, **source_options)
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
        for name, contents in companions.items():
            relative = _relative_java_path(name)
            staged = root_src / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_text(contents, encoding='utf-8')
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
    return companions
