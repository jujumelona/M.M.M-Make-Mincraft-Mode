from concurrent.futures import ThreadPoolExecutor

import pytest

from minecraft_mod_ai.source_patch import (
    SourcePatchError,
    TransactionalSourcePatcher,
    sha256_bytes,
)


def test_transaction_returns_revision_bound_changes_without_rereading(tmp_path, monkeypatch):
    from pathlib import Path

    target = tmp_path / 'Example.java'
    original_read = Path.read_bytes
    reads = []

    def read(path):
        if path == target:
            reads.append(path)
        return original_read(path)

    monkeypatch.setattr(Path, 'read_bytes', read)
    result = TransactionalSourcePatcher(tmp_path).apply([
        {'operation': 'create', 'path': 'Example.java', 'content': 'class Example {}'}
    ])
    assert 'change_set' in result
    change = result['change_set']
    assert change['base_revision'] == 0
    assert change['revision'] == 1
    assert change['model_revision'] == 0
    assert change['changes'] == result['operations']
    assert reads == []


def test_noop_and_failure_do_not_advance_revision(tmp_path):
    patcher = TransactionalSourcePatcher(tmp_path)
    first = patcher.apply([{'operation': 'create', 'path': 'A.java', 'content': 'class A {}'}])
    assert 'change_set' in first
    noop = patcher.apply([{'operation': 'replace', 'path': 'A.java',
                          'expected_sha256': sha256_bytes(b'class A {}'), 'content': 'class A {}'}])
    assert noop['change_set']['revision'] == 1
    assert noop['change_set']['changes'] == []
    with pytest.raises(SourcePatchError):
        patcher.apply([{'operation': 'create', 'path': 'A.java', 'content': 'invalid'}])
    changed = patcher.apply([{'operation': 'create', 'path': 'B.java', 'content': 'class B {}'}])
    assert changed['change_set']['base_revision'] == 1
    assert changed['change_set']['revision'] == 2


def test_model_revision_changes_only_for_build_inputs(tmp_path):
    patcher = TransactionalSourcePatcher(tmp_path)
    result = patcher.apply([{'operation': 'create', 'path': 'build.gradle', 'content': "plugins { id 'java' }"}])
    assert 'change_set' in result
    assert result['change_set']['model_revision'] == 1
    java = patcher.apply([{'operation': 'create', 'path': 'A.java', 'content': 'class A {}'}])
    assert java['change_set']['model_revision'] == 1


def test_disjoint_commits_have_unique_ordered_revisions(tmp_path):
    def create(i):
        return TransactionalSourcePatcher(tmp_path).apply([
            {'operation': 'create', 'path': f'A{i}.java', 'content': f'class A{i} {{}}'}
        ])

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(create, range(12)))
    assert all('change_set' in result for result in results)
    assert sorted(result['change_set']['revision'] for result in results) == list(range(1, 13))


def test_rollback_does_not_publish_successful_changes(tmp_path, monkeypatch):
    from minecraft_mod_ai import source_patch

    patcher = TransactionalSourcePatcher(tmp_path)
    first = patcher.apply([{'operation': 'create', 'path': 'A.java', 'content': 'class A {}'}])
    assert 'change_set' in first
    commit = source_patch._commit_staged_path

    def fail(path, content):
        if path.name == 'C.java':
            raise OSError('injected disk failure')
        commit(path, content)

    monkeypatch.setattr(source_patch, '_commit_staged_path', fail)
    with pytest.raises(SourcePatchError):
        patcher.apply([{'operation': 'create', 'path': name, 'content': 'class B {}'}
                       for name in ('B.java', 'C.java')])
    assert not (tmp_path / 'B.java').exists()
    result = patcher.apply([{'operation': 'create', 'path': 'D.java', 'content': 'class D {}'}])
    assert result['change_set']['revision'] == 2


def test_direct_json_patch_cannot_bypass_resource_validation(tmp_path):
    patcher = TransactionalSourcePatcher(tmp_path)
    with pytest.raises(SourcePatchError, match='Duplicate JSON key'):
        patcher.apply([{'operation': 'create', 'path': 'A.java', 'content': 'class A {}'},
                       {'operation': 'create', 'path': 'config/x.json', 'content': '{"x":1,"x":2}'}])
    assert not (tmp_path / 'A.java').exists()
