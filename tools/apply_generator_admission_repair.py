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


receipt_guard = '''        reviewed_kinds = tuple(getattr(spec.platform, "deterministic_module_kinds", ()) or ())
        if not reviewed_kinds:
            raise GenerationError(
                f"Target {spec.platform.minecraft_version} has no reviewed deterministic module templates."
            )
'''

replace_once(
    "minecraft_mod_ai/generator.py",
    '''        spec.validate()\n        adapter = adapter_for_lock_values(spec.platform)\n        root = root.resolve()\n''',
    '''        spec.validate()\n''' + receipt_guard + '''        adapter = adapter_for_lock_values(spec.platform)\n        root = root.resolve()\n''',
)

replace_once(
    "minecraft_mod_ai/generator.py",
    '''    def _write_contract(self, root: Path, spec: ModSpec) -> None:\n        package_path = Path(*spec.package_name.split("."))\n''',
    '''    def _write_contract(self, root: Path, spec: ModSpec) -> None:\n''' + receipt_guard + '''        package_path = Path(*spec.package_name.split("."))\n''',
)

replace_once(
    "minecraft_mod_ai/platform_generation_contract.py",
    '''    def generate(self: Any, spec: Any, root: Path):\n        adapter = adapter_for_lock_values(spec.platform)\n        result = original_generate(self, spec, root)\n''',
    '''    def generate(self: Any, spec: Any, root: Path):\n        reviewed_kinds = tuple(getattr(spec.platform, "deterministic_module_kinds", ()) or ())\n        if not reviewed_kinds:\n            raise generator_module.GenerationError(\n                f"Target {spec.platform.minecraft_version} has no reviewed deterministic module templates."\n            )\n        adapter = adapter_for_lock_values(spec.platform)\n        result = original_generate(self, spec, root)\n''',
)

print("generator admission boundary patched from immutable platform receipt")
