from pathlib import Path

p = Path('minecraft_mod_ai/temporary_skill_contract.py')
text = p.read_text(encoding='utf-8')
old = '''from .remote_trajectory_store import (\n    flush_remote_outbox,\n    hydrate_remote_cache,\n    queue_remote_record,\n    remote_configured,\n)\nfrom .trajectory_memory import (\n'''
new = '''from .remote_trajectory_store import (\n    flush_remote_outbox,\n    hydrate_remote_cache,\n    queue_remote_record,\n    remote_configured,\n)\nfrom . import trajectory_memory as _trajectory_memory\nfrom .trajectory_memory import (\n'''
if text.count(old) != 1:
    raise SystemExit('temporary_skill_contract import anchor mismatch')
text = text.replace(old, new, 1)
old = '    relevant_trajectories,\n'
if text.count(old) != 1:
    raise SystemExit('temporary_skill_contract direct alias count mismatch')
text = text.replace(old, '', 1)
old = '    records = relevant_trajectories(\n'
new = '    records = _trajectory_memory.relevant_trajectories(\n'
if text.count(old) != 1:
    raise SystemExit('temporary_skill_contract call anchor mismatch')
text = text.replace(old, new, 1)
p.write_text(text, encoding='utf-8')

p = Path('tests/test_runtime_json_gap_regression.py')
text = p.read_text(encoding='utf-8')
insert = '''\n\ndef test_runtime_trajectory_retrieval_keeps_execution_context_contract():\n    import inspect\n\n    from minecraft_mod_ai import temporary_skill_contract, trajectory_memory\n\n    signature = inspect.signature(trajectory_memory.relevant_trajectories)\n    assert "current_context" in signature.parameters\n    assert temporary_skill_contract._trajectory_memory is trajectory_memory\n    assert "relevant_trajectories" not in temporary_skill_contract.__dict__\n'''
if 'test_runtime_trajectory_retrieval_keeps_execution_context_contract' not in text:
    text += insert
p.write_text(text, encoding='utf-8')
