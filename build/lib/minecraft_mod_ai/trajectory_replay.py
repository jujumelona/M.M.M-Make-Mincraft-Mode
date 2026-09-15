from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .procedural_retrieval import decompose_task_procedure
from .procedure_trace import sequence_actions
from .trajectory_memory import (
    execution_context_from_values,
    relevant_trajectories,
    synthesize_temporary_skill,
)
from .trajectory_record_integrity import derive_levels, record_strong_skill_eligible


@dataclass(frozen=True)
class ReplayDecision:
    mode: str
    source_trajectory_id: str
    replay_prefix_actions: tuple[str, ...]
    branch_after_index: int
    branch_after_action: str
    avoid_action: str
    verification_level: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source_trajectory_id": self.source_trajectory_id,
            "replay_prefix_actions": list(self.replay_prefix_actions),
            "branch_after_index": self.branch_after_index,
            "branch_after_action": self.branch_after_action,
            "avoid_action": self.avoid_action,
            "verification_level": self.verification_level,
        }


def build_generation_replay_context(
    base: str | Path,
    query: str,
    *,
    router: Any | None,
    target: Mapping[str, Any],
    mode: str = "reuse",
    limit: int = 6,
) -> dict[str, Any] | None:
    """Build source-free replay guidance from verifier-qualified trajectory memory.

    This never replays source bodies or tool payloads.  It reuses only ordered
    procedure, structural verifier facts and failure boundaries whose execution
    context is compatible with the current Minecraft target.
    """
    if mode == "fresh":
        return None
    query = str(query or "").strip()
    if not query:
        return None
    context = execution_context_from_values(
        {
            "minecraft_version": target.get("minecraft_version"),
            "loader": target.get("loader"),
            "mappings": target.get("mappings"),
            "java_version": target.get("java") or target.get("java_version"),
        }
    )
    records = relevant_trajectories(
        base,
        query,
        task_class="generation",
        router=router,
        limit=max(1, min(8, int(limit))),
        current_context=context,
    )
    if not records:
        return None
    skill = synthesize_temporary_skill(query, records, task_class="generation")
    decisions = replay_decisions(records, mode=mode)
    summaries = [_trajectory_summary(row) for row in records[:6]]
    current_plan = decompose_task_procedure(query)
    body = {
        "schema_version": "mmm/generation-trajectory-replay-v1",
        "source_free": True,
        "mode": mode,
        "execution_context": context,
        "current_procedure_plan": current_plan.to_dict(),
        "replay_decisions": [item.to_dict() for item in decisions],
        "trajectory_summaries": summaries,
        "temporary_skill": skill,
        "authority": [
            "current exact repository evidence",
            "current platform lock and admitted dependency evidence",
            "current JDT/compiler diagnostics",
            "current Gradle/GameTest/runtime verification",
            "verified trajectory memory",
        ],
        "rules": [
            "Replay only the verified procedural prefix; do not replay source text or stale file contents.",
            "Branch at the recorded verifier boundary or first current-evidence divergence.",
            "Verified failures are negative evidence and must not be repeated as successful procedure.",
            "If current exact evidence conflicts with memory, follow current evidence and abandon replay.",
        ],
    }
    body["replay_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return body


def replay_decisions(
    records: Sequence[Mapping[str, Any]],
    *,
    mode: str,
) -> tuple[ReplayDecision, ...]:
    successes = [row for row in records if record_strong_skill_eligible(row)]
    failures = [row for row in records if _verified_failure(row)]
    selected: list[Mapping[str, Any]] = []
    if mode == "replay":
        selected.extend(successes[:2])
    elif mode == "counterfactual":
        selected.extend(failures[:2])
        if not selected:
            selected.extend(successes[1:3] or successes[:1])
    else:
        selected.extend(successes[:2])
        selected.extend(failures[:2])
    result: list[ReplayDecision] = []
    for row in selected[:4]:
        decision = _decision_from_row(row, mode=mode)
        if decision is not None:
            result.append(decision)
    return tuple(result)


def _decision_from_row(row: Mapping[str, Any], *, mode: str) -> ReplayDecision | None:
    procedure = row.get("procedure")
    raw_steps = procedure.get("steps") if isinstance(procedure, Mapping) else None
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes, bytearray)):
        return None
    actions = sequence_actions(procedure if isinstance(procedure, Mapping) else None)
    if not actions:
        return None

    failure_index: int | None = None
    verifier_index: int | None = None
    avoid_action = ""
    for index, raw in enumerate(raw_steps[: len(actions)]):
        if not isinstance(raw, Mapping):
            continue
        kind = str(raw.get("kind", ""))
        status = str(raw.get("status", "")).upper()
        if verifier_index is None and kind == "verifier":
            verifier_index = index
        if status == "FAIL":
            failure_index = index
            avoid_action = str(raw.get("action", ""))[:160]
            break
    boundary = failure_index if failure_index is not None else verifier_index
    if boundary is None:
        boundary = len(actions)
    branch_after = max(-1, boundary - 1)
    prefix = actions[: max(0, boundary)]
    verification = row.get("verification")
    level = str(verification.get("level", "L0")) if isinstance(verification, Mapping) else "L0"
    return ReplayDecision(
        mode=mode,
        source_trajectory_id=str(row.get("trajectory_id", "")),
        replay_prefix_actions=tuple(prefix),
        branch_after_index=branch_after,
        branch_after_action=actions[branch_after] if branch_after >= 0 else "",
        avoid_action=avoid_action,
        verification_level=level,
    )


def _trajectory_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    procedure = row.get("procedure")
    verification = row.get("verification")
    verification = verification if isinstance(verification, Mapping) else {}
    outcome = str(row.get("outcome", ""))
    failure = str(row.get("error_signature", "")).strip() if outcome == "FAIL" else ""
    # Compact hypothesis/progress/failure representation.  It intentionally omits
    # source bodies, prompts, arbitrary tool payloads and completion prose.
    return {
        "trajectory_id": str(row.get("trajectory_id", "")),
        "hypothesis": {
            "task_shape": dict(row.get("task_shape", {})) if isinstance(row.get("task_shape"), Mapping) else {},
            "procedure_actions": list(sequence_actions(procedure if isinstance(procedure, Mapping) else None)),
        },
        "progress": {
            "outcome": outcome,
            "verification_level": str(verification.get("level", "L0")),
            "verification_confidence": verification.get("confidence"),
            "verified_facts": dict(row.get("verified_facts", {})) if isinstance(row.get("verified_facts"), Mapping) else {},
        },
        "failure": {
            "verified": _verified_failure(row),
            "signature": " ".join(failure.split())[:360],
        },
    }


def _verified_failure(row: Mapping[str, Any]) -> bool:
    derived = derive_levels(row)
    return bool(derived and derived.get("verified_failure") is True)


__all__ = [
    "ReplayDecision",
    "build_generation_replay_context",
    "replay_decisions",
]
