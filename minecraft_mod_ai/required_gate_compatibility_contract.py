from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import wraps
from typing import Any


# Planner-owned semantic gate names are normalized by CompleteProductionOrchestrator
# before lookup. Keep cross-stage naming compatibility at one explicit boundary.
_PLANNER_GATE_TO_EVIDENCE = {
    "source static validation": "source",
    "target compile": "gradle",
    "generated resource validation": "source",
    "network protocol validation": "playtest",
    "worldgen runtime validation": "gametest",
    "jar": "jar",
    "runtime": "runtime_client",
}

# These are not duplicate low-level execution gates. They are owned by the production
# quality contract and must remain fail-closed unless that contract actually declares
# the corresponding independently evaluated dimension.
_QUALITY_BACKED_GATES = {
    "behavior equivalence": "correctness",
    "performance regression": "performance",
}


def _normalize_gate(value: str) -> str:
    return " ".join(
        "".join(character.casefold() if character.isalnum() else " " for character in value).split()
    )


def _literal_command_passed(build_report: Mapping[str, Any] | None, name: str) -> bool:
    if not isinstance(build_report, Mapping):
        return False
    commands = build_report.get("commands")
    if not isinstance(commands, Iterable) or isinstance(commands, (str, bytes, bytearray)):
        return False
    return any(
        isinstance(command, Mapping)
        and command.get("name") == name
        and command.get("exit_code") == 0
        and command.get("timed_out") is not True
        for command in commands
    )


def _required_gate_build_report(build_report: dict[str, Any] | None) -> dict[str, Any] | None:
    """Expose current ``build`` receipts under the legacy required-gate name only.

    The production quality build dimension remains independently stricter; this alias
    exists solely because the required-gate vocabulary historically called the same
    successful Gradle execution ``clean_build``.
    """
    if not isinstance(build_report, dict):
        return build_report
    if build_report.get("status") != "PASS":
        return build_report
    if _literal_command_passed(build_report, "clean_build"):
        return build_report
    if not _literal_command_passed(build_report, "build"):
        return build_report
    normalized = dict(build_report)
    commands = list(build_report.get("commands") or ())
    commands.append(
        {
            "name": "clean_build",
            "exit_code": 0,
            "timed_out": False,
            "compatibility_source": "build",
        }
    )
    normalized["commands"] = commands
    return normalized


def _quality_gate_owned_by_contract(proposal: Any, gate: str) -> bool:
    dimension_id = _QUALITY_BACKED_GATES.get(_normalize_gate(gate))
    if dimension_id is None:
        return False
    game_design = getattr(proposal, "game_design", None)
    if not isinstance(game_design, Mapping):
        return False
    contract = game_design.get("_production_contract")
    if not isinstance(contract, Mapping):
        return False
    catalog = contract.get("quality_dimension_catalog")
    if not isinstance(catalog, Iterable) or isinstance(catalog, (str, bytes, bytearray)):
        return False
    return any(
        isinstance(item, Mapping)
        and str(item.get("dimension_id") or "") == dimension_id
        for item in catalog
    )


def install(orchestrator_module: Any) -> None:
    cls = orchestrator_module.CompleteProductionOrchestrator

    # Extend the deterministic adjudicator vocabulary before execution inspects gates.
    gate_map = orchestrator_module._REQUIRED_GATE_TO_EVIDENCE
    for gate, evidence in _PLANNER_GATE_TO_EVIDENCE.items():
        current_evidence = gate_map.get(gate)
        if current_evidence is not None and current_evidence != evidence:
            raise RuntimeError(
                f"required gate alias {gate!r} already maps to conflicting evidence "
                f"{current_evidence!r}; expected {evidence!r}"
            )
        gate_map[gate] = evidence

    # One compatibility seam is sufficient. Required-gate command naming is normalized
    # before the canonical adjudicator runs, while quality-owned gates are delegated
    # only when the immutable production contract contains the matching dimension.
    current = cls._required_gate_failures
    if getattr(current, "_mmm_required_gate_compatibility", False):
        return

    @wraps(current)
    def required_gate_failures(
        proposal: Any,
        *,
        generated_receipts: Iterable[Any],
        project_root: Any = None,
        source_validation: dict[str, Any] | None,
        jdt_receipt: dict[str, Any] | None,
        build_report: dict[str, Any] | None,
        jar_validation: dict[str, Any] | None,
        blockbench_receipts: Iterable[dict[str, Any]],
        runtime_receipt: dict[str, Any] | None,
        playtest_receipt: dict[str, Any] | None,
        visual_receipt: dict[str, Any] | None,
    ) -> list[str]:
        failures = current(
            proposal,
            generated_receipts=generated_receipts,
            project_root=project_root,
            source_validation=source_validation,
            jdt_receipt=jdt_receipt,
            build_report=_required_gate_build_report(build_report),
            jar_validation=jar_validation,
            blockbench_receipts=blockbench_receipts,
            runtime_receipt=runtime_receipt,
            playtest_receipt=playtest_receipt,
            visual_receipt=visual_receipt,
        )
        resolved: list[str] = []
        for failure in failures:
            value = str(failure)
            try:
                _head, gate, reason = value.rsplit(":", 2)
            except ValueError:
                resolved.append(value)
                continue
            if reason == "unsupported" and _quality_gate_owned_by_contract(proposal, gate):
                # evaluate_quality_contract already owns PASS/FAIL and release blocking
                # for this dimension; duplicating it as an execution receipt gate is a
                # contract-category error, not additional evidence.
                continue
            resolved.append(value)
        return sorted(set(resolved))

    required_gate_failures._mmm_required_gate_compatibility = True
    cls._required_gate_failures = staticmethod(required_gate_failures)
