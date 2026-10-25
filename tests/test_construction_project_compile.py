"""Compiler-aware regression: generated Java is checked with sibling sources."""
from __future__ import annotations
import importlib.util
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest


def load_compiler(monkeypatch):
    pkg = types.ModuleType('minecraft_mod_ai')
    pkg.__path__ = [str(Path(__file__).resolve().parents[1] / 'minecraft_mod_ai')]
    monkeypatch.setitem(sys.modules,'minecraft_mod_ai',pkg)
    java = types.ModuleType('minecraft_mod_ai.java_lsp')
    # Path validation is tested without launching a compiler. Real compile
    # cases below already require javac explicitly through skipif.
    compiler = Path(shutil.which('javac') or '/unavailable-test-jdk/bin/javac').resolve()
    java._resolve_project_java_home = lambda version, **kw: compiler.parent.parent
    monkeypatch.setitem(sys.modules,'minecraft_mod_ai.java_lsp',java)
    source = Path(__file__).resolve().parents[1]/'minecraft_mod_ai'/'construction_project_compile.py'
    spec = importlib.util.spec_from_file_location('minecraft_mod_ai.construction_project_compile',source)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,spec.name,mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(shutil.which('javac') is None,reason='requires javac')
def test_sibling_source_included_and_repository_unchanged(monkeypatch,tmp_path):
    version = subprocess.check_output(['javac','-version'],text=True).strip().split()[-1].split('.')[0]
    compiler = load_compiler(monkeypatch)
    path = tmp_path/'src/main/java/demo/Sibling.java'
    path.parent.mkdir(parents=True)
    path.write_text('package demo; public final class Sibling { public static int answer(){return 42;} }')
    before = path.read_bytes()
    code = 'package demo; public final class ModUnit { int value(){ return Sibling.answer(); } }'
    context = {'project_root':str(tmp_path),'java_classpath':[str(tmp_path)],
               'target':{'java_version':version}}
    compiler.verify_java_source_with_dependencies(code,context=context,
        target_path='src/main/java/demo/ModUnit.java',filename='ModUnit.java',
        source_context=[{'path':'src/main/java/demo/Sibling.java'}])
    assert path.read_bytes() == before
    assert not (tmp_path/'src/main/java/demo/ModUnit.java').exists()


@pytest.mark.skipif(shutil.which('javac') is None,reason='requires javac')
def test_real_compiler_diagnostic_returns_sibling_context(monkeypatch,tmp_path):
    version = subprocess.check_output(['javac','-version'],text=True).strip().split()[-1].split('.')[0]
    compiler = load_compiler(monkeypatch)
    path = tmp_path/'src/main/java/demo/Sibling.java'
    path.parent.mkdir(parents=True)
    path.write_text('package demo; public final class Sibling { public static int answer(){return 42;} }')
    bad = 'package demo; public class ModUnit { int value(){return Sibling.noSuchMethod();} }'
    context = {'project_root':str(tmp_path),'java_classpath':[str(tmp_path)],
               'target':{'java_version':version}}
    with pytest.raises(ValueError,match='JAVA_PROJECT_ANALYSIS_FAILED.*cannot find symbol'):
        compiler.verify_java_source_with_dependencies(bad,context=context,
           target_path='src/main/java/demo/ModUnit.java',filename='ModUnit.java',
           source_context=[{'path':'src/main/java/demo/Sibling.java'}])


@pytest.mark.skipif(shutil.which('javac') is None,reason='requires javac')
def test_same_package_siblings_discovered_without_source_context(monkeypatch,tmp_path):
    compiler = load_compiler(monkeypatch)
    version = subprocess.check_output(['javac','-version'],text=True).strip().split()[-1].split('.')[0]
    original = tmp_path/'src/main/java/demo/Sibling.java'
    original.parent.mkdir(parents=True)
    original.write_text('package demo; public final class Sibling { static int value(){return 7;} }')
    context = {'project_root':str(tmp_path),'java_classpath':[str(tmp_path)],
               'target':{'java_version':version}}
    code = 'package demo; public class ModUnit { int calc(){return Sibling.value();} }'
    compiler.verify_java_source_with_dependencies(code,context=context,
        target_path='src/main/java/demo/ModUnit.java', filename='ModUnit.java')


@pytest.mark.skipif(shutil.which('javac') is None,reason='requires javac')
def test_imported_java_dependency_discovered(monkeypatch,tmp_path):
    compiler = load_compiler(monkeypatch)
    version = subprocess.check_output(['javac','-version'],text=True).strip().split()[-1].split('.')[0]
    imported = tmp_path/'src/main/java/helpers/Utility.java'
    imported.parent.mkdir(parents=True)
    imported.write_text('package helpers; public final class Utility { public static int result(){return 4;} }')
    context = {'project_root':str(tmp_path),'java_classpath':[str(tmp_path)],
               'target':{'java_version':version}}
    code = ('package demo; import helpers.Utility; public class ModUnit {'
            ' int calc(){ return Utility.result(); } }')
    compiler.verify_java_source_with_dependencies(code,context=context,
        target_path='src/main/java/demo/ModUnit.java', filename='ModUnit.java')


def test_source_path_traversal_rejected(monkeypatch,tmp_path):
    compiler = load_compiler(monkeypatch)
    context = {'project_root':str(tmp_path),'java_classpath':[str(tmp_path)],
               'target':{'java_version':'21'}}
    for unsafe in ['/tmp/danger.java', 'src/main/java/../../secret.java',
                   'src\\main\\java\\demo.java', 'C:/src/main/java/Other.java']:
        with pytest.raises(ValueError,match='CONSTRUCTION_COMPILE_SOURCE_PATH_INVALID'):
            compiler.verify_java_source_with_dependencies('package demo; public class ModUnit {}',
                context=context,target_path=unsafe,filename='ModUnit.java')


@pytest.mark.skipif(shutil.which('javac') is None,reason='requires javac')
def test_old_generated_target_does_not_shadow_candidate(monkeypatch,tmp_path):
    compiler = load_compiler(monkeypatch)
    version = subprocess.check_output(['javac','-version'],text=True).strip().split()[-1].split('.')[0]
    target = tmp_path/'src/main/java/demo/ModUnit.java'
    target.parent.mkdir(parents=True)
    target.write_text('package demo; public class ModUnit { BROKEN_CODE }')
    old = target.read_bytes()
    context = {'project_root':str(tmp_path),'java_classpath':[str(tmp_path)],
               'target':{'java_version':version}}
    compiler.verify_java_source_with_dependencies('package demo; public class ModUnit {}',
        context=context,target_path='src/main/java/demo/ModUnit.java',filename='ModUnit.java')
    assert target.read_bytes() == old


@pytest.mark.skipif(shutil.which('javac') is None, reason='requires javac')
def test_unrelated_broken_sibling_not_compiled(monkeypatch,tmp_path):
    compiler = load_compiler(monkeypatch)
    version = subprocess.check_output(['javac','-version'],text=True).strip().split()[-1].split('.')[0]
    directory = tmp_path/'src/main/java/demo'
    directory.mkdir(parents=True)
    (directory/'Irrelevant.java').write_text('package demo; public class Irrelevant { BROKEN }')
    context = {'project_root':str(tmp_path),'java_classpath':[str(tmp_path)],
               'target':{'java_version':version}}
    compiler.verify_java_source_with_dependencies('package demo; public class ModUnit {}',
        context=context,target_path='src/main/java/demo/ModUnit.java',filename='ModUnit.java')

@pytest.mark.skipif(shutil.which('javac') is None, reason='requires javac')
def test_client_source_can_link_unbuilt_main_project_source(monkeypatch, tmp_path):
    """Client sources are in a distinct root but may depend on main classes."""
    compiler = load_compiler(monkeypatch)
    version = subprocess.check_output(['javac','-version'], text=True).strip().split()[-1].split('.')[0]
    main = tmp_path / 'src/main/java/common/CommonSupport.java'
    main.parent.mkdir(parents=True)
    main.write_text('package common; public class CommonSupport { public static int value(){return 5;} }')
    context = {'project_root': str(tmp_path), 'java_classpath': [str(tmp_path)],
               'target': {'java_version': version}}
    src = ('package client; import common.CommonSupport; '
           'public class ClientUnit { int answer(){ return CommonSupport.value(); } }')
    compiler.verify_java_source_with_dependencies(src, context=context,
        target_path='src/client/java/client/ClientUnit.java', filename='ClientUnit.java')


@pytest.mark.skipif(shutil.which('javac') is None, reason='requires javac')
def test_wildcard_package_import_resolves_real_java_source(monkeypatch, tmp_path):
    """Wildcard imports cannot be resolved by treating the package as a Java file."""
    compiler = load_compiler(monkeypatch)
    version = subprocess.check_output(['javac','-version'], text=True).strip().split()[-1].split('.')[0]
    helper = tmp_path / 'src/main/java/helpers/Util.java'
    helper.parent.mkdir(parents=True)
    helper.write_text('package helpers; public class Util { public static int value(){return 3;} }')
    context = {'project_root': str(tmp_path), 'java_classpath': [str(tmp_path)],
               'target': {'java_version': version}}
    src = ('package demo; import helpers.*; '
           'public class Unit { int answer(){ return Util.value(); } }')
    compiler.verify_java_source_with_dependencies(src, context=context,
        target_path='src/main/java/demo/Unit.java', filename='Unit.java')


@pytest.mark.skipif(shutil.which('javac') is None, reason='requires javac')
def test_comments_must_not_import_broken_java_sources(monkeypatch, tmp_path):
    """Only executable Java references may expand the compilation unit closure."""
    compiler = load_compiler(monkeypatch)
    version = subprocess.check_output(['javac','-version'], text=True).strip().split()[-1].split('.')[0]
    broken = tmp_path/'src/main/java/danger/Broken.java'
    broken.parent.mkdir(parents=True)
    broken.write_text('package danger; public class Broken { THIS_DOES_NOT_COMPILE }')
    context = {'project_root': str(tmp_path), 'java_classpath': [str(tmp_path)],
               'target': {'java_version': version}}
    src = ('package demo; /* import danger.Broken; */ public class Unit {}')
    compiler.verify_java_source_with_dependencies(src, context=context,
        target_path='src/main/java/demo/Unit.java', filename='Unit.java')
