from __future__ import annotations

"""Compile canonical Minecraft artifact responsibilities into narrow execution steps."""

from collections.abc import Iterable
from dataclasses import dataclass

from .minecraft_template_catalog import validate_artifact_kinds
from .task_template_catalog import load_template

ROOT_PROVIDE = "translation:artifact_dependency_graph"


@dataclass(frozen=True)
class TemplateStep:
    name: str
    outcome: str
    consumes: tuple[str, ...]
    provides: tuple[str, ...]
    anchor_kinds: tuple[str, ...]
    template_id: str = ""
    branch_features: tuple[str, ...] = ()


def _string_contract(identifier: str, field: str, raw) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"TEMPLATE_RESPONSIBILITY: {identifier} {field} must be a list")
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"TEMPLATE_RESPONSIBILITY: {identifier} {field} contains an invalid value"
            )
        value = item.strip()
        if value in values:
            raise ValueError(
                f"TEMPLATE_RESPONSIBILITY: {identifier} {field} repeats {value!r}"
            )
        values.append(value)
    return tuple(values)


def responsibility_ids_for_artifact(artifact_kind: str) -> tuple[str, ...]:
    """Return one artifact's statically declared responsibility templates."""
    (artifact_kind,) = validate_artifact_kinds((artifact_kind,))
    manifest = load_template(f"minecraft/{artifact_kind}")
    if manifest.get("execution") != "sequence":
        raise ValueError(f"TEMPLATE_ARTIFACT: minecraft/{artifact_kind} must execute as a sequence")
    raw_steps = manifest.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"TEMPLATE_ARTIFACT: minecraft/{artifact_kind} has no responsibility steps")
    prefix = f"minecraft/{artifact_kind}/"
    identifiers: list[str] = []
    for identifier in raw_steps:
        if not isinstance(identifier, str) or not identifier.startswith(prefix):
            raise ValueError(
                f"TEMPLATE_ARTIFACT: {artifact_kind} contains non-local responsibility {identifier!r}"
            )
        identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"TEMPLATE_ARTIFACT: {artifact_kind} repeats a responsibility")
    return tuple(identifiers)


def steps_for_artifact(artifact_kind: str) -> tuple[TemplateStep, ...]:
    """Compile one artifact while preserving each leaf template's declared data contract."""
    previous = ROOT_PROVIDE
    compiled: list[TemplateStep] = []
    for identifier in responsibility_ids_for_artifact(artifact_kind):
        record = load_template(identifier)
        task = record.get("task")
        rules = record.get("rules", ())
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"TEMPLATE_RESPONSIBILITY: {identifier} is missing task")
        if not isinstance(rules, list):
            raise ValueError(f"TEMPLATE_RESPONSIBILITY: {identifier} rules must be a list")

        declared_consumes = _string_contract(identifier, "consumes", record.get("consumes"))
        declared_provides = _string_contract(identifier, "provides", record.get("provides"))
        anchors = _string_contract(identifier, "anchor_kinds", record.get("anchor_kinds"))
        if not declared_provides:
            raise ValueError(f"TEMPLATE_RESPONSIBILITY: {identifier} provides must not be empty")
        if not anchors:
            raise ValueError(f"TEMPLATE_RESPONSIBILITY: {identifier} anchor_kinds must not be empty")

        responsibility = identifier.rsplit("/", 1)[-1]
        completion = f"{identifier}:complete"
        consumes = tuple(dict.fromkeys((previous, *declared_consumes)))
        provides = tuple(dict.fromkeys((completion, *declared_provides)))
        compiled.append(
            TemplateStep(
                name=f"{artifact_kind}_{responsibility}",
                template_id=identifier,
                outcome=" ".join(
                    [task.strip(), *[str(rule).strip() for rule in rules if str(rule).strip()]]
                ),
                consumes=consumes,
                provides=provides,
                anchor_kinds=anchors,
            )
        )
        previous = completion
    return tuple(compiled)


def steps_for_artifacts(artifact_kinds: Iterable[str]) -> tuple[TemplateStep, ...]:
    """Compile validated artifacts in canonical order without semantic routing."""
    compiled: list[TemplateStep] = []
    previous = ROOT_PROVIDE
    for artifact_kind in validate_artifact_kinds(artifact_kinds):
        artifact_steps = list(steps_for_artifact(artifact_kind))
        if artifact_steps and previous != ROOT_PROVIDE:
            first = artifact_steps[0]
            artifact_steps[0] = TemplateStep(
                name=first.name,
                template_id=first.template_id,
                outcome=first.outcome,
                consumes=tuple(dict.fromkeys((previous, *first.consumes))),
                provides=first.provides,
                anchor_kinds=first.anchor_kinds,
                branch_features=first.branch_features,
            )
        if artifact_steps:
            previous = artifact_steps[-1].provides[0]
            compiled.extend(artifact_steps)
    return tuple(compiled)


__all__ = [
    "ROOT_PROVIDE",
    "TemplateStep",
    "responsibility_ids_for_artifact",
    "steps_for_artifact",
    "steps_for_artifacts",
]
