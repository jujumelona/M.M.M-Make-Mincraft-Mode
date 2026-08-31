from pathlib import Path

path = Path("tests/test_worker12_shared_core.py")
text = path.read_text(encoding="utf-8")
if "import inspect\n" not in text:
    text = text.replace("import os\n", "import inspect\nimport os\n", 1)
old = '''    from minecraft_mod_ai import performance_final_contract\n\n    monkeypatch.setattr(\n        performance_final_contract,\n        "_clone_source_snapshot",\n        lambda _root: stage,\n    )\n    calls = 0\n'''
new = '''    from minecraft_mod_ai import performance_final_contract, source_patch\n\n    monkeypatch.setenv("MMM_REPAIR_CANDIDATE_JDT", "on")\n    monkeypatch.setattr(\n        performance_final_contract,\n        "_clone_source_snapshot",\n        lambda _root: stage,\n    )\n\n    class NoopPatcher:\n        def __init__(self, _stage):\n            pass\n\n        def apply(self, _operations):\n            return None\n\n    monkeypatch.setattr(source_patch, "TransactionalSourcePatcher", NoopPatcher)\n    calls = 0\n'''
if old in text:
    text = text.replace(old, new, 1)
old_call = '''    _score, verifier = agentic_optimization_contract._verify_repair_candidate(\n        engine, root, [], {}\n    )\n    assert calls == 1\n'''
new_call = '''    verifier_impl = inspect.unwrap(agentic_optimization_contract._verify_repair_candidate)\n    _score, verifier = verifier_impl(engine, root, [], {})\n    assert calls == 1, verifier.get("verifier_error")\n'''
if old_call in text:
    text = text.replace(old_call, new_call, 1)
else:
    old_call = '''    _score, verifier = agentic_optimization_contract._verify_repair_candidate(\n        engine, root, [], {}\n    )\n    assert calls == 1, verifier\n'''
    if old_call not in text:
        raise RuntimeError("worker12 verifier call anchor missing")
    text = text.replace(old_call, new_call, 1)
path.write_text(text, encoding="utf-8")
print("worker12 verifier fixture now targets canonical implementation")
