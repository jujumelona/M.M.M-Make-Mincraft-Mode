"""Generator dispatch must preserve the materializer's real patch provenance."""
from hashlib import sha256
from types import SimpleNamespace

from minecraft_mod_ai.artifact_job import ArtifactJob
from minecraft_mod_ai.integrity_dispatcher import execute_generator_job
from minecraft_mod_ai.task_template_catalog import load_template


def test_generator_dispatch_retains_patch_operation_and_before_hash(tmp_path, monkeypatch):
    leaf = 'minecraft/event/callback'
    manifest = load_template(leaf)
    spec_port = next(row['name'] for row in manifest['inputs'] if row['type'] == 'specification')
    source_port = next(row['name'] for row in manifest['outputs'] if row['type'] == 'code_fragment')
    relative = 'src/main/java/proof/Things.java'
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text('package proof; public class Things { /* MMM:members */ }')
    before = sha256(target.read_bytes()).hexdigest()
    spec = {'leaf_id': leaf, 'context_id': 'fixture', 'language': 'java', 'side': 'COMMON',
            'target_path': relative, 'operation': 'JAVA_PATCH', 'anchor': '/* MMM:members */',
            'java_prefix': 'package proof; public class Things {', 'java_suffix': '}', 'bindings': {}}
    job = ArtifactJob(job_id='patch', template_id='', owner_module='example', canonical_leaf=leaf,
                      implementation_id='fixture_generator',
                      deterministic_inputs={'_canonical_inputs': {spec_port: spec}})
    resolved = SimpleNamespace(context_id='fixture', to_dict=lambda: {'target': {'java_version': 25}})
    authority = SimpleNamespace(executors={'fixture_generator': lambda *a, **k: {
        source_port: 'public static int count = 1;',
    }})
    monkeypatch.setattr('minecraft_mod_ai.resolved_version_context.execution_context', lambda *a: resolved)
    monkeypatch.setattr('minecraft_mod_ai.integrity_dispatcher.verify_job_binding', lambda *a: authority)
    monkeypatch.setattr('minecraft_mod_ai.integrity_dispatcher.verify_candidate_content', lambda *a: None)
    monkeypatch.setattr('minecraft_mod_ai.integrity_dispatcher.require_registered_leaf_binding',
                        lambda *a: {'state': 'not_reviewed'})
    monkeypatch.setattr('minecraft_mod_ai.integrity_validators.validate_java_syntax',
                        lambda *a, **k: {'status': 'FIXTURE'})
    receipt = execute_generator_job(job, context={}, router=SimpleNamespace(), base_dir=tmp_path)
    materialized = receipt['materialization']
    assert materialized['operation'] == 'java_patch'
    assert materialized['before_sha256'] == before
    assert materialized['after_sha256'] == sha256(target.read_bytes()).hexdigest()
    assert 'public static int count = 1;' in target.read_text()
