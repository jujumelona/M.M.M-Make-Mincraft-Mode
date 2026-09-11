from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    old_count = text.count(old)
    if old_count == 0 and text.count(new) == 1:
        return
    if old_count != 1:
        raise RuntimeError(f"{path}: admission patch target count={old_count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "minecraft_mod_ai/generator.py",
    '''        adapter = adapter_for_lock_values(spec.platform)\n        root = root.resolve()\n''',
    '''        adapter = adapter_for_lock_values(spec.platform)\n        if not adapter.deterministic_module_kinds:\n            raise GenerationError(\n                f"Target {adapter.minecraft_version} has no reviewed deterministic module templates."\n            )\n        root = root.resolve()\n''',
)

replace_once(
    "minecraft_mod_ai/platform_generation_contract.py",
    '''    def generate(self: Any, spec: Any, root: Path):\n        adapter = adapter_for_lock_values(spec.platform)\n        result = original_generate(self, spec, root)\n''',
    '''    def generate(self: Any, spec: Any, root: Path):\n        adapter = adapter_for_lock_values(spec.platform)\n        if not adapter.deterministic_module_kinds:\n            raise generator_module.GenerationError(\n                f"Target {adapter.minecraft_version} has no reviewed deterministic module templates."\n            )\n        result = original_generate(self, spec, root)\n''',
)

print("generator admission boundary patched at source and runtime wrapper")
