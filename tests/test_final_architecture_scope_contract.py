from __future__ import annotations
from pathlib import Path
import minecraft_mod_ai.atomic_requirement_contract as atomic_requirement_contract
import minecraft_mod_ai.validation_execution_contract as validation_execution_contract
from minecraft_mod_ai.atomic_requirement_contract import compile_ir
from minecraft_mod_ai.validation_execution_contract import project_build_fingerprint

def test_conjunction_requirements_are_not_hidden_inside_one_atom() -> None:
    ir = compile_ir(_proposal('Add a frost sword and add a lunar portal.'))
    assert ir['atom_count'] == 2
    assert ir['atoms'][0]['text'] == 'Add a frost sword'
    assert 'lunar portal' in ir['atoms'][1]['text']
    assert ir['atoms'][1]['status'] != 'COVERED'
    assert getattr(atomic_requirement_contract._atom_ranges, '_mmm_conjunction_atomizer', False)

def test_evidence_metadata_does_not_invalidate_build_fingerprint(tmp_path: Path) -> None:
    root = tmp_path / 'project'
    (root / 'src/main/java/example').mkdir(parents=True)
    (root / 'src/main/java/example/Main.java').write_text('package example; final class Main {}\n', encoding='utf-8')
    (root / 'build.gradle').write_text('plugins {}\n', encoding='utf-8')
    first = project_build_fingerprint(root)
    metadata = root / '.minecraft_ai'
    metadata.mkdir()
    (metadata / 'quality-convergence.json').write_text('{"iteration":1}\n', encoding='utf-8')
    (metadata / 'work-ledger.sqlite3').write_bytes(b'evidence')
    assert project_build_fingerprint(root) == first
    (root / 'src/main/java/example/Main.java').write_text('package example; final class Main { int changed = 1; }\n', encoding='utf-8')
    assert project_build_fingerprint(root) != first
    assert getattr(validation_execution_contract._is_build_input, '_mmm_build_input_scope', False)
