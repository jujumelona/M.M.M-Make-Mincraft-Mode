from __future__ import annotations

import json
from functools import wraps
from typing import Any, Iterable


def _canonical_key(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _normalize_synthesized_batches(
    batches: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten parallel audio batches into one stable sound-id ordered batch."""

    synthesized: list[dict[str, Any]] = []
    for batch in batches:
        if not isinstance(batch, dict):
            continue
        for item in batch.get("synthesized", []):
            if isinstance(item, dict):
                synthesized.append(item)
    synthesized.sort(
        key=lambda item: (
            str(item.get("sound_id", "")),
            _canonical_key(item),
        )
    )
    return [{"synthesized": synthesized}] if synthesized else []


def _sort_receipts(items: Any) -> Any:
    if not isinstance(items, list):
        return items
    return sorted(items, key=_canonical_key)


def _canonicalize_generation_result(result: Any) -> Any:
    if not isinstance(result, dict):
        return result

    for key in ("module_receipts", "blockbench_receipts"):
        if key in result:
            result[key] = _sort_receipts(result[key])

    unresolved = result.get("unresolved")
    if isinstance(unresolved, list):
        result["unresolved"] = sorted(str(item) for item in unresolved)

    for key in ("asset_receipt", "audio_receipt"):
        receipt = result.get(key)
        if isinstance(receipt, dict) and isinstance(receipt.get("shards"), list):
            receipt["shards"] = _sort_receipts(receipt["shards"])

    return result


def install(
    *,
    audio_generator_module: Any,
    orchestrator_module: Any,
) -> None:
    current_finalizer = audio_generator_module.finalize_audio_registry
    if not getattr(current_finalizer, "_mmm_parallel_result_determinism", False):

        @wraps(current_finalizer)
        def deterministic_finalizer(*args: Any, **kwargs: Any) -> dict[str, Any]:
            if "synthesized_batches" in kwargs:
                kwargs["synthesized_batches"] = _normalize_synthesized_batches(
                    kwargs["synthesized_batches"]
                )
            receipt = current_finalizer(*args, **kwargs)
            if isinstance(receipt, dict) and isinstance(receipt.get("sounds"), list):
                receipt["sounds"] = sorted(
                    receipt["sounds"],
                    key=lambda item: (
                        str(item.get("sound_id", ""))
                        if isinstance(item, dict)
                        else "",
                        _canonical_key(item),
                    ),
                )
            return receipt

        deterministic_finalizer._mmm_parallel_result_determinism = True
        audio_generator_module.finalize_audio_registry = deterministic_finalizer
        # complete_orchestrator imports the function into its module namespace, so
        # patch that bound reference as well.
        orchestrator_module.finalize_audio_registry = deterministic_finalizer

    current_execute = orchestrator_module.CompleteProductionOrchestrator._execute_generation_work
    if not getattr(current_execute, "_mmm_parallel_result_determinism", False):

        @wraps(current_execute)
        def deterministic_execute(self: Any, *args: Any, **kwargs: Any) -> Any:
            return _canonicalize_generation_result(
                current_execute(self, *args, **kwargs)
            )

        deterministic_execute._mmm_parallel_result_determinism = True
        orchestrator_module.CompleteProductionOrchestrator._execute_generation_work = (
            deterministic_execute
        )
