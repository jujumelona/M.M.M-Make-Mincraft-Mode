from pathlib import Path

path = Path("tests/test_worker12_shared_core.py")
text = path.read_text(encoding="utf-8")
old = '''    from minecraft_mod_ai import performance_final_contract\n\n    monkeypatch.setattr(\n        performance_final_contract,\n        "_clone_source_snapshot",\n        lambda _root: stage,\n    )\n    calls = 0\n'''
new = '''    from minecraft_mod_ai import performance_final_contract, source_patch\n\n    monkeypatch.setattr(\n        performance_final_contract,\n        "_clone_source_snapshot",\n        lambda _root: stage,\n    )\n    monkeypatch.setattr(\n        source_patch.TransactionalSourcePatcher,\n        "apply",\n        lambda _self, _operations: None,\n    )\n    calls = 0\n'''
if text.count(old) != 1:
    raise RuntimeError(f"expected one verifier fixture anchor, found {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("worker12 verifier fixture isolated")
