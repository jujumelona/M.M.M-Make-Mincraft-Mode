from pathlib import Path

path = Path("minecraft_mod_ai/generator.py")
text = path.read_text(encoding="utf-8")
old = '''        adapter = adapter_for_lock_values(spec.platform)\n        root = root.resolve()\n'''
new = '''        adapter = adapter_for_lock_values(spec.platform)\n        if not adapter.deterministic_module_kinds:\n            raise GenerationError(\n                f"Target {adapter.minecraft_version} has no reviewed deterministic module templates."\n            )\n        root = root.resolve()\n'''
if text.count(old) == 0 and text.count(new) == 1:
    raise SystemExit(0)
if text.count(old) != 1:
    raise RuntimeError(f"generator admission patch target count={text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("generator fail-closed admission patched")
