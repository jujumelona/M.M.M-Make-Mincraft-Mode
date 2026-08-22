from __future__ import annotations

"""Recover stale-but-authorized model actions without ever executing them.

A tool can remain parseable because it belongs to the loop's frozen authorized surface
while no longer being executable on the current causal frontier. The host must not feed
that stale action into the execution loop just to manufacture a failure observation.
Instead, retry the same semantic turn against one host-selected current action while
keeping the frozen surface available only for parser validation. Repeated stale output
is discarded and re-synchronized a bounded number of times before failing closed.
"""

from dataclasses import replace
from typing import Any, Mapping, Sequence

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


def install(causal_frontier_contract_module: Any) -> None:
    base = causal_frontier_contract_module.CausalFrontierAdapter
    if bool(getattr(base, _MARKER, False)):
        return

    class RecoveringCausalFrontierAdapter(base):
        """Bounded host correction for stale authorized tool emission."""

        def generate_turn(self, request: Any) -> Any:
            from .model_adapters import ModelConfigurationError

            turn = super().generate_turn(request)
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

            # The base adapter recorded this stale fingerprint so that an uncorrected
            # caller cannot spin forever. Recovery consumes the stale result here,
            # before the core loop can execute it, and owns a bounded retry budget.
            self._reset_stale_guard()
            by_name = {_name(schema): schema for schema in candidates if _name(schema)}
            selected = tuple(by_name[name] for name in visible if name in by_name)
            if not selected:
                raise ModelConfigurationError(
                    "Model emitted a stale authorized tool call after the causal frontier "
                    "became empty: stale=" + ",".join(stale)
                )

            forced_name = _name(selected[0])
            forced_tools = (selected[0],)
            retry_messages = tuple(request.messages)
            rejected = stale

            for attempt in range(1, _MAX_RESYNC_ATTEMPTS + 1):
                feedback = {
                    "role": "system",
                    "content": (
                        "The previous tool action was discarded without execution because "
                        "it is not the host-selected action for the current causal state. "
                        f"Rejected: {', '.join(rejected)}. Call exactly {forced_name!r} "
                        "now; do not emit any other tool name."
                    ),
                }
                retry_messages = _with_capability_context(
                    (*retry_messages, feedback),
                    stage=self.stage,
                    role=self.role,
                    tools=forced_tools,
                )
                retry_request = replace(
                    request,
                    messages=retry_messages,
                    # Expose exactly one current legal action and force that exact name.
                    # The broader frozen surface remains validation-only so a model that
                    # still repeats a stale name can be parsed, classified and discarded
                    # by the host without granting it execution authority.
                    tools=forced_tools,
                    tool_validation_schemas=tuple(candidates),
                    tool_choice={
                        "type": "function",
                        "function": {"name": forced_name},
                    },
                    parallel_tool_calls=False,
                )
                corrected = self.inner.generate_turn(retry_request)
                calls = tuple(getattr(corrected, "tool_calls", ()))
                names = tuple(str(call.name) for call in calls)
                if calls and all(name == forced_name for name in names):
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
                f"{_MAX_RESYNC_ATTEMPTS} discarded attempts; expected={forced_name!r}"
                + detail
                + " visible="
                + ",".join(visible)
            )

    RecoveringCausalFrontierAdapter.__name__ = base.__name__
    RecoveringCausalFrontierAdapter.__qualname__ = base.__qualname__
    setattr(RecoveringCausalFrontierAdapter, _MARKER, True)
    causal_frontier_contract_module.CausalFrontierAdapter = RecoveringCausalFrontierAdapter


__all__ = ["install"]
