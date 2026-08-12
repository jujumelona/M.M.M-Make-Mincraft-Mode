from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class PreferenceTraceError(ValueError):
    pass


_LOCK = threading.RLock()


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha(value: Any) -> str:
    payload = value if isinstance(value, str) else _canonical(value)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PreferenceCandidate:
    candidate_id: str
    response: Any
    score: float
    verifier: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "response": self.response,
            "score": float(self.score),
            "verifier": dict(self.verifier),
        }


class PreferenceTraceStore:
    """Append-only verified preference data produced by MMM test-time search.

    The store intentionally records only candidates that were evaluated by a host
    verifier.  It is not an SFT success store: rejected candidates are retained so
    later DPO/ranker training can learn the same choice the runtime verifier made.
    """

    schema_version = "mmm/preference-trace-v1"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        task: str,
        prompt: Any,
        candidates: Sequence[PreferenceCandidate | Mapping[str, Any]],
        winner_index: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = tuple(_candidate(value, index) for index, value in enumerate(candidates))
        if len(normalized) < 2:
            raise PreferenceTraceError("Preference traces require at least two candidates.")
        if not 0 <= winner_index < len(normalized):
            raise PreferenceTraceError("winner_index is outside the candidate list.")
        if any(not isinstance(candidate.score, (int, float)) for candidate in normalized):
            raise PreferenceTraceError("Every candidate requires a numeric verifier score.")

        winner = normalized[winner_index]
        best_score = max(candidate.score for candidate in normalized)
        if winner.score != best_score:
            raise PreferenceTraceError(
                "winner_index must identify a highest-scoring verified candidate."
            )
        body = {
            "schema_version": self.schema_version,
            "task": str(task),
            "prompt": prompt,
            "prompt_sha256": _sha(prompt),
            "candidates": [candidate.to_dict() for candidate in normalized],
            "winner_index": winner_index,
            "winner_id": winner.candidate_id,
            "metadata": dict(metadata or {}),
        }
        body["trace_id"] = _sha(body)
        line = _canonical(body)
        with _LOCK:
            existing_ids = set()
            if self.path.is_file():
                with self.path.open("r", encoding="utf-8") as handle:
                    for raw in handle:
                        try:
                            value = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(value, dict) and isinstance(value.get("trace_id"), str):
                            existing_ids.add(value["trace_id"])
            if body["trace_id"] not in existing_ids:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
        return {
            "schema_version": "mmm/preference-trace-record-v1",
            "trace_id": body["trace_id"],
            "winner_id": winner.candidate_id,
            "candidate_count": len(normalized),
            "path": str(self.path),
        }

    def iter_records(self) -> Iterable[dict[str, Any]]:
        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise PreferenceTraceError(
                        f"Invalid preference JSONL at line {line_number}."
                    ) from exc
                if not isinstance(value, dict) or value.get("schema_version") != self.schema_version:
                    raise PreferenceTraceError(
                        f"Invalid preference record at line {line_number}."
                    )
                yield value

    def export_dpo(self, output_path: str | Path) -> dict[str, Any]:
        """Export every verified winner/rejected pair as generic DPO JSONL."""

        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        pairs = 0
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            for record in self.iter_records():
                candidates = record.get("candidates", [])
                winner_index = int(record.get("winner_index", -1))
                if not isinstance(candidates, list) or not 0 <= winner_index < len(candidates):
                    raise PreferenceTraceError("Preference record has an invalid winner.")
                chosen = candidates[winner_index]
                if not isinstance(chosen, dict):
                    raise PreferenceTraceError("Chosen preference candidate is invalid.")
                for index, rejected in enumerate(candidates):
                    if index == winner_index or not isinstance(rejected, dict):
                        continue
                    handle.write(
                        json.dumps(
                            {
                                "prompt": record.get("prompt"),
                                "chosen": chosen.get("response"),
                                "rejected": rejected.get("response"),
                                "metadata": {
                                    "trace_id": record.get("trace_id"),
                                    "task": record.get("task"),
                                    "chosen_score": chosen.get("score"),
                                    "rejected_score": rejected.get("score"),
                                    **(
                                        record.get("metadata")
                                        if isinstance(record.get("metadata"), dict)
                                        else {}
                                    ),
                                },
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    pairs += 1
        return {
            "schema_version": "mmm/preference-dpo-export-v1",
            "output_path": str(output),
            "pairs": pairs,
            "sha256": "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest(),
        }


def _candidate(value: PreferenceCandidate | Mapping[str, Any], index: int) -> PreferenceCandidate:
    if isinstance(value, PreferenceCandidate):
        return value
    if not isinstance(value, Mapping):
        raise PreferenceTraceError("Preference candidate must be an object.")
    verifier = value.get("verifier", {})
    if not isinstance(verifier, Mapping):
        raise PreferenceTraceError("Preference candidate verifier must be an object.")
    response = value.get("response")
    candidate_id = str(value.get("candidate_id") or _sha({"index": index, "response": response}))
    return PreferenceCandidate(
        candidate_id=candidate_id,
        response=response,
        score=float(value.get("score", 0.0)),
        verifier=dict(verifier),
    )


__all__ = [
    "PreferenceCandidate",
    "PreferenceTraceError",
    "PreferenceTraceStore",
]
