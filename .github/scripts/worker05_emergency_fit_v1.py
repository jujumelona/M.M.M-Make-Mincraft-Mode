from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "minecraft_mod_ai/llama_context_safety_contract.py"
text = PATH.read_text(encoding="utf-8")
old = '''            # This path is entered after the backend has already proved context pressure.\n            # If the byte estimator believed the unchanged payload fit, force a smaller\n            # protocol-safe window rather than retrying the identical request.\n            target = exact_budget\n            if original_size <= exact_budget:\n                target = max(1, min(exact_budget, int(original_size * 0.75)))\n            return _protocol_safe_minimal_fit(\n                context_module,\n                fitted if fitted_size < original_size else original,\n                budget=target,\n            )\n'''
new = '''            # Backend context pressure requires a strictly smaller retry. Preserve as much\n            # history as possible: the retry target is only one byte below the previous\n            # payload unless the caller's real hard budget is already tighter. Protocol-safe\n            # packing then drops only enough optional history to satisfy that monotonic bound.\n            source = fitted if fitted_size < original_size else original\n            target = min(exact_budget, max(1, original_size - 1))\n            candidate = _protocol_safe_minimal_fit(\n                context_module,\n                source,\n                budget=target,\n            )\n            candidate_size = context_module._canonical_size(candidate)\n            if candidate_size >= original_size:\n                raise ContextPackingError(\n                    "backend context pressure could not be reduced without violating the "\n                    "mandatory task/tool protocol; refusing to retry an identical payload. "\n                    f"original_bytes={original_size} candidate_bytes={candidate_size} "\n                    f"budget_bytes={exact_budget}"\n                )\n            return candidate\n'''
count = text.count(old)
if count != 1:
    raise SystemExit(
        "llama_context_safety_contract.py: expected one emergency shrink policy, "
        f"found {count}"
    )
PATH.write_text(text.replace(old, new, 1), encoding="utf-8")
Path(__file__).unlink(missing_ok=True)
