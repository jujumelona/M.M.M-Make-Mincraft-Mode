from __future__ import annotations

"""Compile canonical Minecraft artifact responsibilities into narrow execution steps."""

from dataclasses import dataclass, replace
from collections.abc import Iterable
from typing import Any

from .minecraft_template_catalog import validate_artifact_kinds
from .task_template_catalog import load_template

ROOT_PROVIDE = "translation:artifact_dependency_graph"


@dataclass(frozen=True)
class ContextProjection:
    max_bytes: int = 4096
    include: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepairContract:
    scope: tuple[str, ...] = ()
    writable_anchors: tuple[str, ...] = ()


@dataclass(frozen=True)
class TemplateStep:
    name: str
    outcome: str
    consumes: tuple[str, ...]
    provides: tuple[str, ...]
    anchor_kinds: tuple[str, ...]
    template_id: str = ""
    branch_features: tuple[str, ...] = ()
    execution_mode: str = "model"
    side: tuple[str, ...] = ("common",)
    inputs: tuple[Any, ...] = ()
    outputs: tuple[Any, ...] = ()
    host_requirements: tuple[Any, ...] = ()
    implementation_key: str = ""
    validators: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    context_projection: ContextProjection = ContextProjection()
    repair: RepairContract = RepairContract()

    @property
    def execution(self) -> str:
        return self.execution_mode

    @property
    def implementation(self) -> str:
        return self.implementation_key


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

        if "execution_mode" not in record:
            raise ValueError(f"TEMPLATE_RESPONSIBILITY_MISSING_CONTRACT: {identifier} missing execution_mode")
        raw_exec_mode = record["execution_mode"]
        if raw_exec_mode not in {"model", "deterministic", "composite"}:
            raise ValueError(
                f"TEMPLATE_RESPONSIBILITY: {identifier} invalid execution_mode {raw_exec_mode!r}"
            )

        if "side" not in record:
            raise ValueError(f"TEMPLATE_RESPONSIBILITY_MISSING_CONTRACT: {identifier} missing side")
        raw_side = record["side"]
        if isinstance(raw_side, str):
            raw_side = [raw_side]
        if not isinstance(raw_side, (list, tuple)) or not raw_side:
            raise ValueError(f"TEMPLATE_RESPONSIBILITY: {identifier} side must be a non-empty list")
        side_values = []
        for s in raw_side:
            if s not in {"client", "server", "common"}:
                raise ValueError(f"TEMPLATE_RESPONSIBILITY: {identifier} invalid side {s!r}")
            side_values.append(s)
        side = tuple(side_values)

        if "inputs" not in record:
            raise ValueError(f"TEMPLATE_RESPONSIBILITY_MISSING_CONTRACT: {identifier} missing inputs")
        raw_inputs = record["inputs"]
        if not isinstance(raw_inputs, (list, tuple)) or not raw_inputs:
            raise ValueError(f"TEMPLATE_RESPONSIBILITY: {identifier} inputs must be a non-empty list")
        for item in raw_inputs:
            if not isinstance(item, dict) or not item.get("name") or not item.get("type"):
                raise ValueError(f"TEMPLATE_RESPONSIBILITY: {identifier} input {item!r} missing name/type")
        inputs = tuple(raw_inputs)

        if "outputs" not in record:
            raise ValueError(f"TEMPLATE_RESPONSIBILITY_MISSING_CONTRACT: {identifier} missing outputs")
        raw_outputs = record["outputs"]
        if not isinstance(raw_outputs, (list, tuple)) or not raw_outputs:
            raise ValueError(f"TEMPLATE_RESPONSIBILITY: {identifier} outputs must be a non-empty list")
        for item in raw_outputs:
            if not isinstance(item, dict) or not item.get("name") or not item.get("type"):
                raise ValueError(f"TEMPLATE_RESPONSIBILITY: {identifier} output {item!r} missing name/type")
        outputs = tuple(raw_outputs)

        if "host_requirements" not in record:
            raise ValueError(f"TEMPLATE_RESPONSIBILITY_MISSING_CONTRACT: {identifier} missing host_requirements")
        raw_host_req = record["host_requirements"]
        if not isinstance(raw_host_req, dict) or not {"capabilities", "symbols", "schemas"}.issubset(raw_host_req.keys()):
            raise ValueError(f"TEMPLATE_RESPONSIBILITY: {identifier} host_requirements must be a dict with capabilities, symbols, schemas")
        host_requirements = tuple(sorted(raw_host_req.items()))

        if "implementation_key" not in record or not str(record["implementation_key"]).strip():
            raise ValueError(f"TEMPLATE_RESPONSIBILITY_MISSING_CONTRACT: {identifier} missing implementation_key")
        implementation_key = str(record["implementation_key"]).strip()

        if "validators" not in record:
            raise ValueError(f"TEMPLATE_RESPONSIBILITY_MISSING_CONTRACT: {identifier} missing validators")
        raw_validators = record["validators"]
        if not isinstance(raw_validators, (list, tuple)) or not raw_validators:
            raise ValueError(f"TEMPLATE_RESPONSIBILITY: {identifier} validators must be a non-empty list")
        validators = tuple(str(v) for v in raw_validators)

        if "postconditions" not in record:
            raise ValueError(f"TEMPLATE_RESPONSIBILITY_MISSING_CONTRACT: {identifier} missing postconditions")
        raw_postconditions = record["postconditions"]
        if not isinstance(raw_postconditions, (list, tuple)) or not raw_postconditions:
            raise ValueError(f"TEMPLATE_RESPONSIBILITY: {identifier} postconditions must be a non-empty list")
        postconditions = tuple(str(p) for p in raw_postconditions)

        if "context_projection" not in record:
            raise ValueError(f"TEMPLATE_RESPONSIBILITY_MISSING_CONTRACT: {identifier} missing context_projection")
        raw_proj = record["context_projection"]
        if not isinstance(raw_proj, dict):
            raise ValueError(f"TEMPLATE_RESPONSIBILITY: {identifier} context_projection must be a dict")
        max_bytes = raw_proj.get("max_bytes", 4096)
        if not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError(f"TEMPLATE_RESPONSIBILITY: {identifier} max_bytes must be a positive int")
        if max_bytes > 4096:
            raise ValueError(
                f"TEMPLATE_RESPONSIBILITY: {identifier} context_projection max_bytes {max_bytes} > 4096"
            )
        include = tuple(raw_proj.get("include", ()))
        if raw_exec_mode == "model" and not include:
            raise ValueError(f"TEMPLATE_RESPONSIBILITY: {identifier} model leaf must declare context_projection.include")
        context_projection = ContextProjection(max_bytes=max_bytes, include=include)

        if "repair" not in record:
            raise ValueError(f"TEMPLATE_RESPONSIBILITY_MISSING_CONTRACT: {identifier} missing repair")
        raw_repair = record["repair"]
        if not isinstance(raw_repair, dict) or not raw_repair.get("scope") or not raw_repair.get("writable_anchors"):
            raise ValueError(f"TEMPLATE_RESPONSIBILITY: {identifier} repair must declare scope and writable_anchors")
        repair = RepairContract(
            scope=tuple(raw_repair["scope"]),
            writable_anchors=tuple(raw_repair["writable_anchors"]),
        )

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
                execution_mode=raw_exec_mode,
                side=side,
                inputs=inputs,
                outputs=outputs,
                host_requirements=host_requirements,
                implementation_key=implementation_key,
                validators=validators,
                postconditions=postconditions,
                context_projection=context_projection,
                repair=repair,
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
            artifact_steps[0] = replace(
                first,
                consumes=tuple(dict.fromkeys((previous, *first.consumes))),
            )
        if artifact_steps:
            previous = artifact_steps[-1].provides[0]
            compiled.extend(artifact_steps)
    return tuple(compiled)


__all__ = [
    "ContextProjection",
    "ROOT_PROVIDE",
    "RepairContract",
    "TemplateStep",
    "responsibility_ids_for_artifact",
    "steps_for_artifact",
    "steps_for_artifacts",
]
