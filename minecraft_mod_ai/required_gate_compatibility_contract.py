from __future__ import annotations

from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
from collections.abc import Iterable, Mapping
from typing import Any


# Planner-owned semantic gate names are normalized by CompleteProductionOrchestrator
# before lookup. Keep every cross-stage alias here so planning and adjudication share
# one runtime compatibility boundary instead of silently drifting.
_PLANNER_GATE_TO_EVIDENCE = {
    "source static validation": "source",
    "target compile": "gradle",
    "generated resource validation": "source",
    "network protocol validation": "playtest",
    "worldgen runtime validation": "gametest",
    "jar": "jar",
    "runtime": "runtime_client",
}

# These gates require quality evidence rather than being weakened to a generic build
# or runtime receipt.
_QUALITY_BACKED_GATES = {
    "behavior equivalence": "correctness",
    "performance regression": "performance",
}

_PENDING_GATE_FAILURES: ContextVar[tuple[str, tuple[str, ...]] | None] = ContextVar(
    "mmm_required_gate_failures",
    default=None,
)


def _normalize_gate(value: str) -> str:
    return " ".join(
        "".join(character.casefold() if character.isalnum() else " " for character in value).split()
    )


def _literal_command_passed(build_report: dict[str, Any] | None, name: str) -> bool:
    if not isinstance(build_report, dict):
        return False
    return any(
        isinstance(command, dict)
        and command.get("name") == name
        and command.get("exit_code") == 0
        and command.get("timed_out") is not True
        for command in build_report.get("commands", [])
    )


def _quality_dimension_passed(
    quality_report: Mapping[str, Any] | None,
    dimension_id: str,
) -> bool:
    if not isinstance(quality_report, Mapping):
        return False
    dimensions = quality_report.get("dimensions")
    if not isinstance(dimensions, Iterable) or isinstance(dimensions, (str, bytes, bytearray)):
        return False
    return any(
        isinstance(item, Mapping)
        and str(item.get("dimension_id") or "") == dimension_id
        and item.get("status") == "PASS"
        for item in dimensions
    )


def _resolve_quality_backed_failures(
    failures: Iterable[str],
    quality_report: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    resolved: list[str] = []
    for failure in failures:
        value = str(failure)
        try:
            head, gate, reason = value.rsplit(":", 2)
        except ValueError:
            resolved.append(value)
            continue
        dimension_id = _QUALITY_BACKED_GATES.get(_normalize_gate(gate))
        if dimension_id is None or reason != "unsupported":
            resolved.append(value)
            continue
        if _quality_dimension_passed(quality_report, dimension_id):
            continue
        resolved.append(f"{head}:{gate}:missing-quality-{dimension_id}")
    return tuple(sorted(set(resolved)))


def _jdt_receipt(module_receipts: Iterable[Any]) -> dict[str, Any] | None:
    found: dict[str, Any] | None = None
    for receipt in module_receipts:
        if (
            isinstance(receipt, dict)
            and receipt.get("schema_version") == "mmm/jdt-gate-v1"
        ):
            found = receipt
    return found


def install(orchestrator_module: Any) -> None:
    cls = orchestrator_module.CompleteProductionOrchestrator

    # Extend the adjudicator vocabulary before any execution can inspect a plan gate.
    gate_map = orchestrator_module._REQUIRED_GATE_TO_EVIDENCE
    for gate, evidence in _PLANNER_GATE_TO_EVIDENCE.items():
        current = gate_map.get(gate)
        if current is not None and current != evidence:
            raise RuntimeError(
                f"required gate alias {gate!r} already maps to conflicting evidence "
                f"{current!r}; expected {evidence!r}"
            )
        gate_map[gate] = evidence

    current = cls._command_receipt_passed
    if not getattr(current, "_mmm_legacy_clean_build_compat", False):

        @wraps(current)
        def command_receipt_passed(
            build_report: dict[str, Any] | None,
            name: str,
        ) -> bool:
            if name == "clean_build":
                # Current runtime calls this command "build"; old evidence receipts used
                # "clean_build". Both are valid for required-gate compatibility. The
                # quality build dimension is separately stricter and requires clean-room.
                return (
                    current(build_report, name)
                    or current(build_report, "build")
                    or _literal_command_passed(build_report, "clean_build")
                )
            return current(build_report, name)

        command_receipt_passed._mmm_legacy_clean_build_compat = True
        cls._command_receipt_passed = staticmethod(command_receipt_passed)

    evaluate = cls._evaluate_quality
    if not getattr(evaluate, "_mmm_required_gate_adjudication", False):

        @wraps(evaluate)
        def evaluate_with_required_gates(
            self: Any,
            *,
            approved: Any,
            run_root: Any,
            project_root: Any,
            source_validation: dict[str, Any] | None,
            build_report: dict[str, Any] | None,
            jar_validation: dict[str, Any] | None,
            module_receipts: Iterable[dict[str, Any]],
            asset_receipt: dict[str, Any] | None,
            blockbench_receipts: Iterable[dict[str, Any]],
            runtime_receipt: dict[str, Any] | None,
            playtest_receipt: dict[str, Any] | None,
            visual_receipt: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            module_values = tuple(module_receipts)
            blockbench_values = tuple(blockbench_receipts)
            report = evaluate(
                self,
                approved=approved,
                run_root=run_root,
                project_root=project_root,
                source_validation=source_validation,
                build_report=build_report,
                jar_validation=jar_validation,
                module_receipts=module_values,
                asset_receipt=asset_receipt,
                blockbench_receipts=blockbench_values,
                runtime_receipt=runtime_receipt,
                playtest_receipt=playtest_receipt,
                visual_receipt=visual_receipt,
            )
            failures = self._required_gate_failures(
                approved,
                generated_receipts=module_values,
                project_root=project_root,
                source_validation=source_validation,
                jdt_receipt=_jdt_receipt(module_values),
                build_report=build_report,
                jar_validation=jar_validation,
                blockbench_receipts=blockbench_values,
                runtime_receipt=runtime_receipt,
                playtest_receipt=playtest_receipt,
                visual_receipt=visual_receipt,
            )
            failures = _resolve_quality_backed_failures(failures, report)
            _PENDING_GATE_FAILURES.set(
                (str(approved.calculate_hash()), failures)
            )
            return report

        evaluate_with_required_gates._mmm_required_gate_adjudication = True
        cls._evaluate_quality = evaluate_with_required_gates

    coverage = orchestrator_module.build_requirement_coverage_receipt
    if not getattr(coverage, "_mmm_required_gate_coverage", False):

        @wraps(coverage)
        def coverage_with_required_gates(*args: Any, **kwargs: Any) -> dict[str, Any]:
            proposal_hash = str(kwargs.get("proposal_hash") or "")
            unresolved = list(kwargs.get("unresolved_gates") or ())
            pending = _PENDING_GATE_FAILURES.get()
            if pending is not None and pending[0] == proposal_hash:
                unresolved.extend(pending[1])
            kwargs["unresolved_gates"] = tuple(sorted(set(str(item) for item in unresolved)))
            return coverage(*args, **kwargs)

        coverage_with_required_gates._mmm_required_gate_coverage = True
        orchestrator_module.build_requirement_coverage_receipt = coverage_with_required_gates

    execute = cls.execute
    if not getattr(execute, "_mmm_required_gate_result", False):

        @wraps(execute)
        def execute_with_required_gate_result(self: Any, *args: Any, **kwargs: Any) -> Any:
            token = _PENDING_GATE_FAILURES.set(None)
            try:
                result = execute(self, *args, **kwargs)
                pending = _PENDING_GATE_FAILURES.get()
                if pending is None or not hasattr(result, "unresolved_gates"):
                    return result
                merged = tuple(
                    sorted(
                        set(str(item) for item in result.unresolved_gates)
                        | set(pending[1])
                    )
                )
                if merged == tuple(result.unresolved_gates):
                    return result
                return replace(result, unresolved_gates=merged)
            finally:
                _PENDING_GATE_FAILURES.reset(token)

        execute_with_required_gate_result._mmm_required_gate_result = True
        cls.execute = execute_with_required_gate_result
