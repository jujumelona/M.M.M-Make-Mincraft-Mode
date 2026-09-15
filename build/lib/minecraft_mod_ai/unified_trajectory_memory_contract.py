from __future__ import annotations

"""Unify repair experience with the verifier-qualified trajectory v3 store.

The repair search predates the cross-repository trajectory store and historically
persisted a second ``repair-experience.jsonl`` file. Keep the repair search and its
candidate-width logic, but source reusable experience from the already authoritative
trajectory corpus instead. New successful repairs continue to be recorded by
``temporary_skill_contract`` and synced through ``remote_trajectory_store``.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_INSTALLED = False


def _tokens(value: str) -> set[str]:
    from .trajectory_memory import _tokens as trajectory_tokens

    return trajectory_tokens(value)


def _repair_memory_rows(
    signature: str,
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Adapt strong v3 repair trajectories to the legacy repair-search view."""

    from .procedure_trace import sequence_actions
    from .trajectory_record_integrity import record_strong_skill_eligible

    target = _tokens(signature)
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for record in records:
        if str(record.get("task_class", "")) != "repair":
            continue
        if not record_strong_skill_eligible(record):
            continue
        procedure = record.get("procedure")
        actions = sequence_actions(
            procedure if isinstance(procedure, Mapping) else None
        )[:16]
        facts = (
            dict(record.get("verified_facts", {}))
            if isinstance(record.get("verified_facts"), Mapping)
            else {}
        )
        error_signature = str(record.get("error_signature", "")).strip()
        searchable = " ".join(
            (
                error_signature,
                json.dumps(facts, ensure_ascii=False, sort_keys=True),
                " ".join(actions),
            )
        )
        values = _tokens(searchable)
        similarity = (
            len(target & values) / max(1, len(target)) if target else 0.0
        )
        if similarity <= 0.0:
            continue
        identity = str(record.get("trajectory_id", ""))
        ranked.append(
            (
                similarity,
                identity,
                {
                    "similarity": round(similarity, 6),
                    "signature_sha256": identity,
                    "evidence": facts,
                    "repair_pattern": [
                        {"operation": action} for action in actions[:12] if action
                    ],
                    "source": "mmm/verified-trajectory-v3",
                },
            )
        )
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [row for _score, _identity, row in ranked[: max(1, int(limit))]]


def _install_repair_memory_adapter() -> None:
    from . import (
        agentic_optimization_contract,
        temporary_skill_contract,
        trajectory_memory,
    )

    current_read = agentic_optimization_contract._read_memory
    if getattr(current_read, "_mmm_v3_repair_memory", False):
        return

    def read_v3(root: Path, signature: str, *, limit: int = 4) -> list[dict[str, Any]]:
        context = temporary_skill_contract._host_execution_context()
        records = trajectory_memory.relevant_trajectories(
            root,
            signature,
            task_class="repair",
            router=None,
            limit=max(8, int(limit) * 3),
            current_context=context,
        )
        return _repair_memory_rows(signature, records, limit=limit)

    read_v3._mmm_v3_repair_memory = True  # type: ignore[attr-defined]
    read_v3.__wrapped__ = current_read  # type: ignore[attr-defined]
    agentic_optimization_contract._read_memory = read_v3

    current_write = agentic_optimization_contract._write_memory
    if not getattr(current_write, "_mmm_v3_repair_memory", False):

        def write_via_v3(root: Path, trace: Mapping[str, Any]) -> None:
            # RepairEngine.repair is already wrapped by temporary_skill_contract,
            # which records the final verifier-qualified result and queues remote sync.
            return None

        write_via_v3._mmm_v3_repair_memory = True  # type: ignore[attr-defined]
        write_via_v3.__wrapped__ = current_write  # type: ignore[attr-defined]
        agentic_optimization_contract._write_memory = write_via_v3


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_repair_memory_adapter()
    _INSTALLED = True


__all__ = ["install"]
