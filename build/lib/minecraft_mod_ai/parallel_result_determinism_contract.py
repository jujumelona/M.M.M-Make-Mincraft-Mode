from __future__ import annotations

import json
from functools import wraps
from typing import Any


def _canonical_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

def _sort_receipts(items: Any) -> Any:
    return sorted(items, key=_canonical_key) if isinstance(items, list) else items

def _canonicalize_generation_result(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    for key in ("module_receipts", "blockbench_receipts"):
        if key in result:
            result[key] = _sort_receipts(result[key])
    unresolved = result.get("unresolved")
    if isinstance(unresolved, list):
        result["unresolved"] = sorted(str(item) for item in unresolved)
    receipt = result.get("asset_receipt")
    if isinstance(receipt, dict) and isinstance(receipt.get("shards"), list):
        receipt["shards"] = _sort_receipts(receipt["shards"])
    return result

def install(*, orchestrator_module: Any) -> None:
    current_execute = orchestrator_module.CompleteProductionOrchestrator._execute_generation_work
    if getattr(current_execute, "_mmm_parallel_result_determinism", False):
        return
    @wraps(current_execute)
    def deterministic_execute(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _canonicalize_generation_result(current_execute(self, *args, **kwargs))
    deterministic_execute._mmm_parallel_result_determinism = True
    orchestrator_module.CompleteProductionOrchestrator._execute_generation_work = deterministic_execute
