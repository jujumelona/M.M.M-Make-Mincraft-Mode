from __future__ import annotations

from pathlib import Path

path = Path("minecraft_mod_ai/reuse_proof_executor.py")
text = path.read_text(encoding="utf-8")
old = '''        scaffold_minimal_ephemeral_workspace(sandbox_path, target_context)\n\n        exact_dependency_receipts = tuple(\n            receipt for receipt in resolved_dependencies if receipt.is_resolved\n        )\n        try:\n            _render_proof_build_model(\n                sandbox_path,\n                target_context,\n                exact_dependency_receipts,\n            )\n        except (OSError, RuntimeError, ValueError) as inj_err:\n            dependency_injection_failed = True\n            reason = f"BUILD_MODEL_RENDER_FAILED: {inj_err}"\n            unresolved_symbols.append(reason)\n            unresolved_mandatory_deps.append(reason)\n'''
new = '''        exact_dependency_receipts = tuple(\n            receipt for receipt in resolved_dependencies if receipt.is_resolved\n        )\n        if authoritative_compile_execution:\n            scaffold_minimal_ephemeral_workspace(sandbox_path, target_context)\n            try:\n                _render_proof_build_model(\n                    sandbox_path,\n                    target_context,\n                    exact_dependency_receipts,\n                )\n            except (OSError, RuntimeError, ValueError, ImportError) as inj_err:\n                dependency_injection_failed = True\n                reason = f"BUILD_MODEL_RENDER_FAILED: {inj_err}"\n                unresolved_symbols.append(reason)\n                unresolved_mandatory_deps.append(reason)\n'''
count = text.count(old)
if count != 1:
    raise SystemExit(f"expected exactly one build-model boundary, found {count}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
