from __future__ import annotations

"""Fail-closed composition kernel for runtime monkey-patch contracts.

MMM intentionally keeps individual runtime policies small, but many of those policies
wrap the same production callables.  The dangerous case is not a clean first install;
it is a *partial* install followed by re-entry after a later stage failed.  Replaying
already-mutated functions can duplicate wrappers, change ownership order, and surface
as an unrelated TypeError/AttributeError much later.

This module gives the two approved composition owners (runtime bootstrap and the
native llama tuning pipeline) one shared execution contract:

* completed stages are receipted immediately and are not replayed for the same
  composition version;
* a stage that raises poisons that composition version, so partially-applied code is
  never invoked a second time in the same process;
* recursive/re-entrant composition is rejected with the exact owner/stage;
* callables that existed at a watched boundary may be wrapped, but may not disappear
  or become non-callable;
* state is stored on the composition owner, making failures inspectable without a
  second global registry.

There is deliberately no rollback. Python monkey-patching is not transactionally
reversible once an installer has performed arbitrary side effects. Fail-closed state
is therefore safer than pretending a failed stage can be retried in-place.
"""

from dataclasses import dataclass
from functools import partial
from threading import RLock
from typing import Any, Callable, Iterable


_COMPOSITION_LOCK = RLock()
_STATE_ATTR = "_mmm_contract_composition_state"


class ContractCompositionError(RuntimeError):
    """Raised when a runtime contract composition cannot safely continue."""


@dataclass(frozen=True)
class CallableBoundary:
    """A production binding that must remain callable once it is callable."""

    label: str
    owner: Any
    attribute: str

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


def _state_map(state_owner: Any) -> dict[str, dict[str, Any]]:
    current = getattr(state_owner, _STATE_ATTR, None)
    if isinstance(current, dict):
        return current
    current = {}
    setattr(state_owner, _STATE_ATTR, current)
    return current


def _fresh_state(version: int) -> dict[str, Any]:
    return {
        "version": int(version),
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
        "completed": tuple(value.get("completed", ())),
        "receipts": tuple(value.get("receipts", ())),
        "active": value.get("active"),
        "failed": value.get("failed"),
        "installed": bool(value.get("installed", False)),
    }


def callable_boundary(label: str, owner: Any, attribute: str) -> CallableBoundary:
    return CallableBoundary(label=label, owner=owner, attribute=attribute)


def _snapshot_boundaries(
    boundaries: Iterable[CallableBoundary],
) -> tuple[tuple[CallableBoundary, bool, str], ...]:
    rows: list[tuple[CallableBoundary, bool, str]] = []
    for boundary in boundaries:
        value = boundary.value()
        rows.append((boundary, callable(value), _callable_identity(value)))
    return tuple(rows)


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
    """Install ordered contract stages once, retaining safe progress on failure.

    A failed stage is intentionally *not* retried. Callers must restart the process
    after fixing the code/configuration because arbitrary installer side effects may
    already have happened before the exception was raised.
    """

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
    with _COMPOSITION_LOCK:
        states = _state_map(state_owner)
        state = states.get(owner_name)
        if not isinstance(state, dict) or int(state.get("version", 0) or 0) != int(version):
            state = _fresh_state(int(version))
            states[owner_name] = state

        if state.get("failed"):
            failure = state["failed"]
            raise ContractCompositionError(
                f"contract composition {owner_name!r} is poisoned by prior failure "
                f"at stage {failure.get('stage')!r}: {failure.get('error')}"
            )
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
    """Compact helper for turning a callable plus arguments into a stage."""

    return ContractStage(name=name, install=partial(install, *args, **kwargs))


__all__ = [
    "CallableBoundary",
    "ContractCompositionError",
    "ContractStage",
    "StageReceipt",
    "callable_boundary",
    "compose_contract_stages",
    "composition_state",
    "stage",
]
