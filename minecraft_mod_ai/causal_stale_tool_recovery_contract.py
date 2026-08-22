from __future__ import annotations

"""Recover stale-but-authorized model actions without ever executing them.

A tool can remain parseable because it belongs to the loop's frozen authorized surface
while no longer being executable on the current causal frontier. The host must not feed
that stale action into the execution loop just to manufacture a failure observation.
Instead, retry the same semantic turn against one host-selected current action while
keeping the frozen surface available only for parser validation. Repeated stale output
is discarded and re-synchronized a bounded number of times before failing closed.

The recovery hook is installed on the canonical adapter class *in place*. Late runtime
contracts import that class before finalization, so rebinding one module-level class
name would leave those pre-bound aliases on the unrecovered implementation.
"""

from dataclasses import replace
from functools import wraps
from typing import Any, Mapping, Sequence

from . import causal_frontier_adapter as causal_frontier_adapter_module
from .causal_frontier_adapter import (
    _with_capability_context,
    current_frontier_names,
)

_MARKER = "_mmm_stale_tool_recovery_v1"
_MAX_RESYNC_ATTEMPTS = 3


def _name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    return str(function.get("name", "")).strip() if isinstance(function, Mapping) else ""


def _stale_names(
    turn: Any,
    *,
    authorized_names: frozenset[str],
    visible_names: frozenset[str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(call.name)
                for call in getattr(turn, "tool_calls", ())
                if str(call.name) in authorized_names
                and str(call.name) not in visible_names
            }
        )
    )


def _install_generate_turn(base: type[Any]) -> None:
    current = base.generate_turn
    if bool(getattr(current, _MARKER, False)):
        setattr(base, _MARKER, True)
        return

    @wraps(current)
    def generate_turn(self: Any, request: Any) -> Any:
        from .model_adapters import ModelConfigurationError

        turn = current(self, request)
        visible = tuple(current_frontier_names() or ())
        candidates: Sequence[Mapping[str, Any]] = (
            self.authorized_surface or tuple(request.tools)
        )
        authorized = frozenset(
            name for schema in candidates if (name := _name(schema))
        )
        visible_set = frozenset(visible)
        stale = _stale_names(
            turn,
            authorized_names=authorized,
            visible_names=visible_set,
        )
        if not stale:
            return turn

        # The canonical adapter recorded the stale fingerprint. Consume that result
        # here before the core tool loop can execute it, then own the bounded recovery
        # budget so the next outer turn cannot trip the base repeated-stale guard.
        self._reset_stale_guard()
        by_name = {_name(schema): schema for schema in candidates if _name(schema)}
        selected = tuple(by_name[name] for name in visible if name in by_name)
        if not selected:
            raise ModelConfigurationError(
                "Model emitted a stale authorized tool call after the causal frontier "
                "became empty: stale=" + ",".join(stale)
            )

        rejected = stale
        for attempt in range(_MAX_RESYNC_ATTEMPTS):
            forced_schema = selected[attempt % len(selected)]
            forced_name = _name(forced_schema)
            forced_tools = (forced_schema,)
            feedback = {
                "role": "system",
                "content": (
                    "The previous stale tool action was discarded without execution because "
                    "it is not the host-selected action for the current causal state. "
                    f"Rejected: {', '.join(rejected)}. Call exactly {forced_name!r} "
                    "now with schema-valid arguments; do not emit any other tool name."
                ),
            }
            retry_messages = _with_capability_context(
                (*tuple(request.messages), feedback),
                stage=self.stage,
                role=self.role,
                tools=forced_tools,
            )
            retry_request = replace(
                request,
                messages=retry_messages,
                # Expose exactly one current legal action. The frozen authorized
                # surface remains validation-only so a repeated stale name can still
                # be parsed and discarded without granting it execution authority.
                tools=forced_tools,
                tool_validation_schemas=tuple(candidates),
                tool_choice={
                    "type": "function",
                    "function": {"name": forced_name},
                },
                parallel_tool_calls=False,
            )
            self._publish_frontier((forced_name,))
            print(
                "causal stale-tool resync:",
                f"attempt={attempt + 1}/{_MAX_RESYNC_ATTEMPTS}",
                f"rejected={','.join(rejected)}",
                f"forced={forced_name}",
                flush=True,
            )
            corrected = self.inner.generate_turn(retry_request)
            calls = tuple(getattr(corrected, "tool_calls", ()))
            names = tuple(str(call.name) for call in calls)
            if calls and all(name == forced_name for name in names):
                self._reset_stale_guard()
                return corrected

            rejected = tuple(sorted(set(names))) or ("<missing-tool-call>",)

        repeated_stale = tuple(
            name for name in rejected if name in authorized and name not in visible_set
        )
        detail = (
            " stale=" + ",".join(repeated_stale)
            if repeated_stale
            else " rejected=" + ",".join(rejected)
        )
        raise ModelConfigurationError(
            "Model failed bounded causal-frontier re-synchronization after "
            f"{_MAX_RESYNC_ATTEMPTS} discarded attempts"
            + detail
            + " visible="
            + ",".join(visible)
        )

    setattr(generate_turn, _MARKER, True)
    generate_turn.__wrapped__ = current  # type: ignore[attr-defined]
    base.generate_turn = generate_turn
    setattr(base, _MARKER, True)


def install(causal_frontier_contract_module: Any) -> None:
    # Mutate the canonical class object instead of swapping a module binding. The
    # coder route-integrity layer imports this class before runtime finalization; an
    # in-place method install therefore reaches both that pre-bound alias and all
    # later dynamic lookups.
    canonical = causal_frontier_adapter_module.CausalFrontierAdapter
    _install_generate_turn(canonical)
    causal_frontier_contract_module.CausalFrontierAdapter = canonical


__all__ = ["install"]
