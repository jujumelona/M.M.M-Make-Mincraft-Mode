from __future__ import annotations

"""Preflight policy for approved complete-production execution."""

from pathlib import Path
from typing import Any

from .complete_orchestrator_support import CompleteProductionError

REQUIRED_GATE_TO_EVIDENCE = {
    "registry": "source",
    "resource": "source",
    "recipe": "source",
    "source static validation": "source",
    "generated resource validation": "source",
    "jdt": "jdt",
    "jdt diagnostics": "jdt",
    "gradle": "gradle",
    "gradle clean build": "gradle",
    "target compile": "gradle",
    "gametest": "gametest",
    "gametest spawn and attributes": "gametest",
    "worldgen runtime validation": "gametest",
    "jar": "jar",
    "jar validation": "jar",
    "runtime": "runtime_client",
    "minecraft server client runtime": "runtime_client",
    "network protocol validation": "playtest",
    "mineflayer playtest": "playtest",
    "runtime interaction tests": "playtest",
    "runtime animation review": "runtime_visual",
    "blockbench uv and bone hierarchy review": "blockbench",
    "blockbench uv render review": "blockbench",
    "visual review": "visual",
    "client gui and validated network action test": "playtest_visual",
    "research ledger integrity": "research_ledger",
}


def normalize_required_gate(value: str) -> str:
    return " ".join(
        "".join(
            character.casefold() if character.isalnum() else " "
            for character in value
        ).split()
    )


def validate_required_gate_contract(proposal: Any) -> None:
    unsupported: list[str] = []
    for module in getattr(proposal, "modules", ()):
        module_id = str(getattr(module, "module_id", "") or "")
        for gate in getattr(module, "required_gates", ()):
            rendered = str(gate).strip()
            if not rendered:
                continue
            if normalize_required_gate(rendered) not in REQUIRED_GATE_TO_EVIDENCE:
                unsupported.append(f"{module_id}:{rendered}")
    if unsupported:
        raise CompleteProductionError(
            "Approved proposal contains unsupported required gates: "
            + ", ".join(sorted(unsupported))
        )


def validate_external_execution_preflight(
    proposal: Any,
    options: Any,
) -> None:
    if bool(getattr(options, "source_only", False)):
        return

    required = bool(getattr(proposal, "external_runtime_required", False))
    if required:
        disabled = [
            name
            for name, enabled in (
                ("runtime", getattr(options, "run_runtime", False)),
                ("client", getattr(options, "run_client", False)),
                ("mineflayer", getattr(options, "run_mineflayer", False)),
                ("visual-review", getattr(options, "run_visual_review", False)),
            )
            if not enabled
        ]
        if disabled:
            raise CompleteProductionError(
                "Approved proposal requires external runtime verification, but "
                "these checks are disabled: " + ", ".join(disabled)
            )

    if bool(getattr(options, "run_client", False)) and not bool(
        getattr(options, "run_runtime", False)
    ):
        raise CompleteProductionError(
            "Client verification requires runtime verification."
        )

    has_entity_review = any(
        getattr(module, "kind", "") in {"entity", "boss", "npc"}
        for module in getattr(proposal, "modules", ())
    )
    if (
        has_entity_review
        and not bool(getattr(options, "source_only", False))
        and not bool(getattr(options, "run_blockbench", False))
    ):
        raise CompleteProductionError(
            "Entity production requires Blockbench UV/render verification."
        )

    if bool(getattr(options, "run_runtime", False)):
        if not bool(getattr(options, "eula_accepted", False)):
            raise CompleteProductionError(
                "Runtime verification was requested without explicit "
                "Minecraft EULA acceptance."
            )
        raw_launcher = getattr(options, "server_launcher", None)
        if not isinstance(raw_launcher, str) or not raw_launcher.strip():
            raise CompleteProductionError(
                "Runtime verification requires server_launcher before "
                "generation starts."
            )
        launcher = Path(raw_launcher).expanduser().resolve()
        if not launcher.is_file() or launcher.is_symlink():
            raise CompleteProductionError(
                "server_launcher must be an existing regular file before "
                "generation starts."
            )

    if bool(getattr(options, "run_mineflayer", False)):
        if not bool(getattr(options, "run_runtime", False)):
            raise CompleteProductionError(
                "Mineflayer verification requires runtime verification."
            )
        actions = getattr(options, "playtest_actions", ())
        if not isinstance(actions, (list, tuple)) or not actions:
            raise CompleteProductionError(
                "Mineflayer verification requires explicit playtest_actions "
                "before generation starts."
            )
        expected_tests = tuple(
            str(value)
            for value in getattr(proposal, "acceptance_tests", ())
        )
        if expected_tests:
            expected_set = set(expected_tests)
            covered: set[str] = set()
            unknown: set[str] = set()
            for action in actions:
                if not isinstance(action, dict):
                    continue
                if str(action.get("action") or "") != "wait_for":
                    continue
                raw_test = action.get("acceptance_test")
                if raw_test is None:
                    continue
                test = str(raw_test)
                if test in expected_set:
                    covered.add(test)
                else:
                    unknown.add(test)
            missing = [test for test in expected_tests if test not in covered]
            if missing or unknown:
                details: list[str] = []
                if missing:
                    details.append("missing=" + ", ".join(missing))
                if unknown:
                    details.append("unknown=" + ", ".join(sorted(unknown)))
                raise CompleteProductionError(
                    "Mineflayer playtest actions do not match the approved "
                    "acceptance tests: " + "; ".join(details)
                )

    if bool(getattr(options, "run_visual_review", False)) and not bool(
        getattr(options, "run_runtime", False)
    ):
        raise CompleteProductionError(
            "Visual verification requires the disposable runtime."
        )


__all__ = [
    "REQUIRED_GATE_TO_EVIDENCE",
    "normalize_required_gate",
    "validate_external_execution_preflight",
    "validate_required_gate_contract",
]
