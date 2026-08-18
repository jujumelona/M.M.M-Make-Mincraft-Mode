from __future__ import annotations

"""Fail-closed composition kernel for runtime monkey-patch contracts.

MMM intentionally keeps individual runtime policies small, but many of those policies
wrap the same production callables. The dangerous case is not a clean first install;
it is a *partial* install followed by re-entry after a later stage failed. Replaying
already-mutated functions can duplicate wrappers, change ownership order, and surface
as an unrelated TypeError/AttributeError much later.

This module gives the two approved composition owners (runtime bootstrap and the
native llama tuning pipeline) one shared execution contract:

* completed stages are receipted immediately and are not replayed;
* a stage that raises poisons the composition owner for the rest of the process;
* one owner may use only one composition version and one declared stage/boundary graph
  for the process lifetime;
* stage graph identity includes installer code plus immutable/callable dependencies,
  while deliberately excluding mutable runtime contents from the fingerprint;
* recursive/re-entrant composition is rejected with the exact owner/stage;
* watched callables may be wrapped, but may not disappear, become non-callable, or
  stop accepting explicitly declared production call shapes;
* state is stored on the composition owner, making failures inspectable without a
  second global registry.

There is deliberately no rollback or in-process upgrade. Python monkey-patching is
not transactionally reversible once an installer has performed arbitrary side
effects. A clean process restart is the only safe boundary for another composition
version or graph.
"""

import hashlib
import inspect
import marshal
from dataclasses import dataclass
from functools import partial
from threading import RLock
from typing import Any, Callable, Iterable


_COMPOSITION_LOCK = RLock()
_STATE_ATTR = "_mmm_contract_composition_state"


class ContractCompositionError(RuntimeError):
    """Raised when a runtime contract composition cannot safely continue."""


@dataclass(frozen=True)
class CallShape:
    """A signature-only production call that a wrapped callable must still accept."""

    positional: int = 0
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class CallableBoundary:
    """A production binding whose callable/API shape must survive composition."""

    label: str
    owner: Any
    attribute: str
    call_shapes: tuple[CallShape, ...] = ()

    def value(self) -> Any:
        return getattr(self.owner, self.attribute, None)


@dataclass(frozen=True)
class ContractStage:
    """One ordered, independently receipted composition stage."""

    name: str
    install: Callable[[], None]


@dataclass(frozen=True)
class StageReceipt:
    name: str
    callable_lineage: tuple[tuple[str, str], ...]


def call_shape(positional: int = 0, *keywords: str) -> CallShape:
    if int(positional) < 0:
        raise ValueError("positional call-shape count must be non-negative")
    normalized = tuple(str(name).strip() for name in keywords)
    if any(not name for name in normalized):
        raise ValueError("call-shape keyword names must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError("call-shape keyword names must be unique")
    return CallShape(positional=int(positional), keywords=normalized)


def _callable_identity(value: Any) -> str:
    if not callable(value):
        return "<non-callable>"
    module = str(getattr(value, "__module__", ""))
    qualname = str(
        getattr(value, "__qualname__", "")
        or getattr(value, "__name__", "")
        or type(value).__qualname__
    )
    marker_names = sorted(
        name
        for name in dir(value)
        if name.startswith("_mmm_") and bool(getattr(value, name, False))
    )
    marker_suffix = ",".join(marker_names)
    return f"{module}:{qualname}" + (f"[{marker_suffix}]" if marker_suffix else "")


def _code_digest(value: Any) -> str:
    target = getattr(value, "__func__", value)
    code = getattr(target, "__code__", None)
    if code is None:
        return "<no-code>"
    try:
        encoded = marshal.dumps(code)
    except (TypeError, ValueError):
        encoded = code.co_code
    return hashlib.sha256(encoded).hexdigest()


def _dependency_identity(value: Any) -> Any:
    """Return a stable implementation identity, never mutable execution contents."""

    if inspect.ismodule(value):
        return ("module", str(getattr(value, "__name__", "")))
    if isinstance(value, partial):
        return (
            "partial",
            _dependency_identity(value.func),
            tuple(_dependency_identity(item) for item in value.args),
            tuple(
                sorted(
                    (str(key), _dependency_identity(item))
                    for key, item in (value.keywords or {}).items()
                )
            ),
        )
    if inspect.ismethod(value):
        owner = getattr(value, "__self__", None)
        return (
            "method",
            _callable_identity(value.__func__),
            _code_digest(value.__func__),
            (type(owner).__module__, type(owner).__qualname__)
            if owner is not None
            else None,
        )
    if inspect.isfunction(value) or inspect.isbuiltin(value):
        return ("callable", _callable_identity(value), _code_digest(value))
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return ("literal", repr(value))
    if isinstance(value, tuple) and len(value) <= 32:
        return ("tuple", tuple(_dependency_identity(item) for item in value))
    if isinstance(value, frozenset) and len(value) <= 32:
        return (
            "frozenset",
            tuple(sorted(repr(_dependency_identity(item)) for item in value)),
        )
    # Lists/dicts/sets and arbitrary stateful objects are intentionally represented
    # only by type. Their contents can legitimately change while an installer runs;
    # graph identity must describe implementation wiring, not execution state.
    return ("object-type", type(value).__module__, type(value).__qualname__)


def _installer_identity(value: Callable[[], None]) -> tuple[Any, ...]:
    partial_identity = _dependency_identity(value) if isinstance(value, partial) else None
    target: Any = value.func if isinstance(value, partial) else value
    bound_owner = getattr(target, "__self__", None)
    target = getattr(target, "__func__", target)
    closure = getattr(target, "__closure__", None) or ()
    freevars = tuple(getattr(getattr(target, "__code__", None), "co_freevars", ()))
    captured: list[tuple[str, Any]] = []
    for name, cell in zip(freevars, closure):
        try:
            item = cell.cell_contents
        except ValueError:
            captured.append((name, "<empty-cell>"))
            continue
        captured.append((name, _dependency_identity(item)))

    return (
        _callable_identity(target),
        _code_digest(target),
        tuple(captured),
        _dependency_identity(bound_owner) if bound_owner is not None else None,
        partial_identity,
    )


def _state_map(state_owner: Any) -> dict[str, dict[str, Any]]:
    current = getattr(state_owner, _STATE_ATTR, None)
    if isinstance(current, dict):
        return current
    current = {}
    setattr(state_owner, _STATE_ATTR, current)
    return current


def _graph_signature(
    stages: tuple[ContractStage, ...],
    boundaries: tuple[CallableBoundary, ...],
) -> tuple[Any, ...]:
    return (
        tuple((stage.name, _installer_identity(stage.install)) for stage in stages),
        tuple(
            (
                boundary.label,
                boundary.attribute,
                tuple(
                    (shape.positional, tuple(shape.keywords))
                    for shape in boundary.call_shapes
                ),
            )
            for boundary in boundaries
        ),
    )


def _fresh_state(version: int, graph_signature: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "version": int(version),
        "graph_signature": graph_signature,
        "completed": [],
        "receipts": [],
        "active": None,
        "failed": None,
        "installed": False,
    }


def composition_state(state_owner: Any, owner_name: str) -> dict[str, Any] | None:
    """Return a diagnostic copy of one composition owner's state."""

    state = getattr(state_owner, _STATE_ATTR, None)
    if not isinstance(state, dict):
        return None
    value = state.get(owner_name)
    if not isinstance(value, dict):
        return None
    return {
        "version": value.get("version"),
        "graph_signature": value.get("graph_signature"),
        "completed": tuple(value.get("completed", ())),
        "receipts": tuple(value.get("receipts", ())),
        "active": value.get("active"),
        "failed": value.get("failed"),
        "installed": bool(value.get("installed", False)),
    }


def callable_boundary(
    label: str,
    owner: Any,
    attribute: str,
    *,
    call_shapes: Iterable[CallShape] = (),
) -> CallableBoundary:
    return CallableBoundary(
        label=label,
        owner=owner,
        attribute=attribute,
        call_shapes=tuple(call_shapes),
    )


def _snapshot_boundaries(
    boundaries: Iterable[CallableBoundary],
) -> tuple[tuple[CallableBoundary, bool, str], ...]:
    rows: list[tuple[CallableBoundary, bool, str]] = []
    for boundary in boundaries:
        value = boundary.value()
        rows.append((boundary, callable(value), _callable_identity(value)))
    return tuple(rows)


def _verify_call_shapes(
    *,
    owner_name: str,
    stage_name: str,
    boundary: CallableBoundary,
    value: Any,
) -> None:
    if not boundary.call_shapes:
        return
    try:
        signature = inspect.signature(value, follow_wrapped=False)
    except (TypeError, ValueError) as exc:
        raise ContractCompositionError(
            f"contract composition {owner_name!r} stage {stage_name!r} made "
            f"boundary {boundary.label!r} uninspectable: {exc}"
        ) from exc

    token = object()
    for shape in boundary.call_shapes:
        args = (token,) * shape.positional
        kwargs = {name: token for name in shape.keywords}
        try:
            signature.bind(*args, **kwargs)
        except TypeError as exc:
            rendered = f"{shape.positional} positional"
            if shape.keywords:
                rendered += f" + keywords {shape.keywords!r}"
            raise ContractCompositionError(
                f"contract composition {owner_name!r} stage {stage_name!r} changed "
                f"call signature for boundary {boundary.label!r}; it no longer "
                f"accepts {rendered}: {exc}"
            ) from exc


def _verify_boundaries(
    *,
    owner_name: str,
    stage_name: str,
    before: tuple[tuple[CallableBoundary, bool, str], ...],
) -> tuple[tuple[str, str], ...]:
    lineage: list[tuple[str, str]] = []
    for boundary, was_callable, _before_identity in before:
        value = boundary.value()
        is_callable = callable(value)
        if was_callable and not is_callable:
            raise ContractCompositionError(
                f"contract composition {owner_name!r} stage {stage_name!r} destroyed "
                f"callable boundary {boundary.label!r} "
                f"({boundary.attribute!r})"
            )
        if is_callable:
            _verify_call_shapes(
                owner_name=owner_name,
                stage_name=stage_name,
                boundary=boundary,
                value=value,
            )
            lineage.append((boundary.label, _callable_identity(value)))
    return tuple(lineage)


def compose_contract_stages(
    *,
    owner_name: str,
    version: int,
    state_owner: Any,
    stages: Iterable[ContractStage],
    boundaries: Iterable[CallableBoundary] = (),
) -> tuple[StageReceipt, ...]:
    """Install ordered contract stages once behind a process-lifetime graph pin."""

    if not owner_name.strip():
        raise ValueError("owner_name must be non-empty")
    if int(version) <= 0:
        raise ValueError("composition version must be positive")

    stage_values = tuple(stages)
    names = tuple(stage.name for stage in stage_values)
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ContractCompositionError(
            f"contract composition {owner_name!r} has duplicate stages: {duplicates}"
        )

    boundary_values = tuple(boundaries)
    requested_graph = _graph_signature(stage_values, boundary_values)
    with _COMPOSITION_LOCK:
        states = _state_map(state_owner)
        prior_state = states.get(owner_name)
        if isinstance(prior_state, dict):
            prior_version = int(prior_state.get("version", 0) or 0)
            if prior_version != int(version):
                raise ContractCompositionError(
                    f"contract composition {owner_name!r} already owns version "
                    f"{prior_version}; process restart is required before requesting "
                    f"version {int(version)}"
                )
            if prior_state.get("graph_signature") != requested_graph:
                raise ContractCompositionError(
                    f"contract composition {owner_name!r} graph changed while still "
                    f"declaring version {int(version)}; bump the composition version "
                    "and restart the process before installing the new graph"
                )
            if prior_state.get("failed"):
                failure = prior_state["failed"]
                raise ContractCompositionError(
                    f"contract composition {owner_name!r} is poisoned by prior failure "
                    f"in version {prior_version} at stage {failure.get('stage')!r}: "
                    f"{failure.get('error')}; process restart is required"
                )
            state = prior_state
        else:
            state = _fresh_state(int(version), requested_graph)
            states[owner_name] = state

        if state.get("active"):
            raise ContractCompositionError(
                f"contract composition {owner_name!r} re-entered while stage "
                f"{state['active']!r} is active"
            )
        if state.get("installed"):
            return tuple(state.get("receipts", ()))

        completed = set(state.get("completed", ()))
        for stage in stage_values:
            if stage.name in completed:
                continue
            state["active"] = stage.name
            before = _snapshot_boundaries(boundary_values)
            try:
                stage.install()
                lineage = _verify_boundaries(
                    owner_name=owner_name,
                    stage_name=stage.name,
                    before=before,
                )
            except Exception as exc:
                state["failed"] = {
                    "stage": stage.name,
                    "type": type(exc).__name__,
                    "error": str(exc),
                }
                state["active"] = None
                if isinstance(exc, ContractCompositionError):
                    raise
                raise ContractCompositionError(
                    f"contract composition {owner_name!r} failed at stage "
                    f"{stage.name!r}: {type(exc).__name__}: {exc}"
                ) from exc

            receipt = StageReceipt(stage.name, lineage)
            state["completed"].append(stage.name)
            state["receipts"].append(receipt)
            completed.add(stage.name)
            state["active"] = None

        state["installed"] = True
        return tuple(state["receipts"])


def stage(name: str, install: Callable[..., None], /, *args: Any, **kwargs: Any) -> ContractStage:
    return ContractStage(name=name, install=partial(install, *args, **kwargs))


__all__ = [
    "CallShape",
    "CallableBoundary",
    "ContractCompositionError",
    "ContractStage",
    "StageReceipt",
    "call_shape",
    "callable_boundary",
    "compose_contract_stages",
    "composition_state",
    "stage",
]
