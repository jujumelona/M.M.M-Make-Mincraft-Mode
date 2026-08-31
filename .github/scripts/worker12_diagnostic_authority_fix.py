from pathlib import Path

path = Path("minecraft_mod_ai/agentic_optimization_contract.py")
text = path.read_text(encoding="utf-8")
old = '''        from .performance_final_contract import _clone_source_snapshot\n        from .repair_diagnostics_contract import diagnostic_errors\n        from .source_patch import TransactionalSourcePatcher\n        from .validation_diagnostic_contract import run_diagnostics\n'''
new = '''        from .performance_final_contract import _clone_source_snapshot\n        from .source_patch import TransactionalSourcePatcher\n        from .validation_diagnostic_contract import diagnostic_errors, run_diagnostics\n'''
if old not in text:
    raise RuntimeError("worker12 diagnostic authority anchor missing")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("worker12 diagnostic interpretation authority unified")
