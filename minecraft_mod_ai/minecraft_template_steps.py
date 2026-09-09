from __future__ import annotations

"""Exhaustive public execution surface for Minecraft implementation template families.

There is no runtime template chooser here. Every host-owned family is evaluated on every
capability in a stable order. Applicability changes only the recorded result:
``required`` materializes that family's bounded implementation micro-DAG;
``not_applicable`` records a completed no-op execution for that family.

The large builder implementations live in ``minecraft_template_family_builders``. This
module is the only public planning surface and intentionally exposes exhaustive accounting.
"""

from dataclasses import dataclass
from typing import Any

from .minecraft_template_catalog import MinecraftTemplateProfile
from . import minecraft_template_family_builders as _families

ROOT_PROVIDE = _families.ROOT_PROVIDE
TemplateStep = _families.TemplateStep

FAMILY_LEDGER_SCHEMA = "mmm/exhaustive-template-family-ledger-v1"
TERMINAL_FAMILY_STATUSES = frozenset({"required", "not_applicable"})


@dataclass(frozen=True)
class TemplateFamilyExecution:
    family_id: str
    capability: str
    status: str
    steps: tuple[TemplateStep, ...]
    reason: str

    @property
    def executed(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "capability": self.capability,
            "status": self.status,
            "executed": True,
            "step_count": len(self.steps),
            "step_names": [step.name for step in self.steps],
            "reason": self.reason,
        }


def family_ids() -> tuple[str, ...]:
    """Return every fixed family in its host-owned execution order."""

    families = tuple(_families._BUILDERS)
    if not families or len(families) != len(set(families)):
        raise ValueError("template family catalog must be non-empty and unique")
    return families


def execute_all_template_families(
    profile: MinecraftTemplateProfile,
) -> tuple[TemplateFamilyExecution, ...]:
    """Execute applicability for every family and materialize only required work.

    ``profile.template_id`` is treated solely as a deterministic host-owned applicability
    fact produced by the capability catalog. It is never used to decide whether a family
    executes: the loop below visits every family unconditionally.
    """

    expected = family_ids()
    applicability = str(profile.template_id or "").strip()
    if applicability not in expected:
        raise ValueError(
            f"capability applicability does not match a fixed family: {applicability!r}"
        )

    executions: list[TemplateFamilyExecution] = []
    required_count = 0
    for family_id in expected:
        applicable = family_id == applicability
        if applicable:
            required_count += 1
            builder = _families._BUILDERS[family_id]
            steps = tuple(builder(profile.capability, profile))
            if not steps or profile.capability not in steps[-1].provides:
                raise ValueError(
                    f"required family {family_id!r} does not terminate in "
                    f"{profile.capability!r}"
                )
            executions.append(
                TemplateFamilyExecution(
                    family_id=family_id,
                    capability=profile.capability,
                    status="required",
                    steps=steps,
                    reason="host capability applicability contract requires this family",
                )
            )
        else:
            executions.append(
                TemplateFamilyExecution(
                    family_id=family_id,
                    capability=profile.capability,
                    status="not_applicable",
                    steps=(),
                    reason="family executed; applicability contract resolved to not_applicable",
                )
            )

    if required_count != 1:
        raise ValueError(
            f"exactly one implementation family must be required, observed {required_count}"
        )
    if tuple(item.family_id for item in executions) != expected:
        raise ValueError("template family execution order diverged from fixed catalog")
    if any(item.status not in TERMINAL_FAMILY_STATUSES for item in executions):
        raise ValueError("template family ledger contains a non-terminal execution")
    return tuple(executions)


def family_execution_ledger(
    profile: MinecraftTemplateProfile,
) -> dict[str, Any]:
    """Return a complete immutable-style ledger view for one capability."""

    executions = execute_all_template_families(profile)
    required = sum(item.status == "required" for item in executions)
    not_applicable = sum(item.status == "not_applicable" for item in executions)
    return {
        "schema": FAMILY_LEDGER_SCHEMA,
        "capability": profile.capability,
        "total": len(executions),
        "executed": len(executions),
        "required": required,
        "not_applicable": not_applicable,
        "failed": 0,
        "unexecuted": 0,
        "complete": (
            len(executions) == len(family_ids())
            and required == 1
            and not_applicable == len(executions) - 1
        ),
        "executions": [item.as_dict() for item in executions],
    }


def steps_for_profile(profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    """Compatibility API returning work steps after exhaustive family accounting."""

    executions = execute_all_template_families(profile)
    steps = tuple(
        step
        for execution in executions
        if execution.status == "required"
        for step in execution.steps
    )
    if not steps:
        raise ValueError(
            f"exhaustive family execution produced no required steps for {profile.capability!r}"
        )
    return steps


__all__ = [
    "FAMILY_LEDGER_SCHEMA",
    "ROOT_PROVIDE",
    "TERMINAL_FAMILY_STATUSES",
    "TemplateFamilyExecution",
    "TemplateStep",
    "execute_all_template_families",
    "family_execution_ledger",
    "family_ids",
    "steps_for_profile",
]
