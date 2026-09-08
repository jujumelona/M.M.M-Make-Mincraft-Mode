from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from dataclasses import replace
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

_PENDING_GATE_FAILURES: ContextVar[tuple[str, ...]] = ContextVar(
    "mmm_required_gate_failures",
    default=(),
)


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

    current_failures = cls._required_gate_failures
    if not getattr(current_failures, "_mmm_required_gate_compatibility", False):

        @wraps(current_failures)
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
            failures = current_failures(
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
                    continue
                resolved.append(value)
            return sorted(set(resolved))

        required_gate_failures._mmm_required_gate_compatibility = True
        cls._required_gate_failures = staticmethod(required_gate_failures)

    current_evaluate = cls._evaluate_quality
    if not getattr(current_evaluate, "_mmm_required_gate_execution_bridge", False):

        @wraps(current_evaluate)
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
            report = current_evaluate(
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
            generated_receipts: tuple[Any, ...] = (
                *module_values,
                *((asset_receipt,) if isinstance(asset_receipt, dict) else ()),
                *blockbench_values,
            )
            failures = cls._required_gate_failures(
                approved,
                generated_receipts=generated_receipts,
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
            _PENDING_GATE_FAILURES.set(tuple(sorted(set(failures))))
            return report

        evaluate_with_required_gates._mmm_required_gate_execution_bridge = True
        cls._evaluate_quality = evaluate_with_required_gates

    current_coverage = orchestrator_module.build_requirement_coverage_receipt
    if not getattr(current_coverage, "_mmm_required_gate_coverage_bridge", False):

        @wraps(current_coverage)
        def coverage_with_required_gates(*args: Any, **kwargs: Any) -> dict[str, Any]:
            failures = _PENDING_GATE_FAILURES.get()
            if failures:
                unresolved = tuple(kwargs.get("unresolved_gates") or ())
                kwargs["unresolved_gates"] = tuple(
                    sorted({*(str(item) for item in unresolved), *failures})
                )
            return current_coverage(*args, **kwargs)

        coverage_with_required_gates._mmm_required_gate_coverage_bridge = True
        orchestrator_module.build_requirement_coverage_receipt = coverage_with_required_gates

    current_execute = cls.execute
    if not getattr(current_execute, "_mmm_required_gate_result_bridge", False):

        @wraps(current_execute)
        def execute_with_required_gates(self: Any, *args: Any, **kwargs: Any) -> Any:
            token = _PENDING_GATE_FAILURES.set(())
            try:
                result = current_execute(self, *args, **kwargs)
                failures = _PENDING_GATE_FAILURES.get()
                if not failures or not hasattr(result, "unresolved_gates"):
                    return result
                merged = tuple(
                    sorted(
                        {
                            *(str(item) for item in getattr(result, "unresolved_gates", ())),
                            *failures,
                        }
                    )
                )
                changes: dict[str, Any] = {"unresolved_gates": merged}
                if getattr(result, "release_ready", False):
                    changes["release_ready"] = False
                if getattr(result, "status", "") == "VERIFIED":
                    changes["status"] = "BUILT_WITH_UNRESOLVED_GATES"
                return replace(result, **changes)
            finally:
                _PENDING_GATE_FAILURES.reset(token)

        execute_with_required_gates._mmm_required_gate_result_bridge = True
        cls.execute = execute_with_required_gates
