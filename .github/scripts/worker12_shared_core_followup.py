from __future__ import annotations

from pathlib import Path


def replace(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    found = text.count(old)
    if found != count:
        raise SystemExit(f"{path}: expected {count} exact matches, found {found}")
    target.write_text(text.replace(old, new, count), encoding="utf-8")


replace(
    "minecraft_mod_ai/validation_checkpoint_policy.py",
    '''        from . import (\n            java_lsp,\n            java_lsp_process_safety_contract,\n            orchestrator_jdt_gate_contract,\n        )\n\n        common.extend(\n            (\n                java_lsp,\n                java_lsp_process_safety_contract,\n                orchestrator_jdt_gate_contract,\n            )\n        )\n''',
    '''        from . import (\n            java_lsp,\n            java_lsp_process_safety_contract,\n            validation_diagnostic_contract,\n        )\n\n        common.extend(\n            (\n                java_lsp,\n                java_lsp_process_safety_contract,\n                validation_diagnostic_contract,\n            )\n        )\n''',
)

test_path = Path("tests/test_worker12_shared_core.py")
text = test_path.read_text(encoding="utf-8")
text = text.replace(
    "from minecraft_mod_ai import complete_orchestrator, validation_execution_contract\n",
    "from minecraft_mod_ai import (\n    complete_orchestrator,\n    validation_checkpoint_policy,\n    validation_diagnostic_contract,\n    validation_execution_contract,\n)\n",
    1,
)
text += '''\n\ndef test_jdt_cache_fingerprint_tracks_canonical_diagnostic_policy() -> None:\n    modules = validation_checkpoint_policy._validation_modules("validate-jdt")\n    assert validation_diagnostic_contract in modules\n    assert all(module.__name__ != "minecraft_mod_ai.orchestrator_jdt_gate_contract" for module in modules)\n'''
test_path.write_text(text, encoding="utf-8")
