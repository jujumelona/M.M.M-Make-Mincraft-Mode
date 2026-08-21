from __future__ import annotations

"""Recover one stale-but-authorized model action without executing it.

A tool can remain parseable because it belongs to the loop's frozen authorized surface
while no longer being executable on the current causal frontier. The host must not feed
that stale action into the execution loop just to manufacture a failure observation.
Instead, retry the same semantic turn once with the current frontier only and force the
highest-ranked legal action. If the model still emits an out-of-frontier action, fail
closed because the transport/model is no longer following the host contract.
"""

from dataclasses import replace
from typing import Any, Mapping, Sequence

from .causal_frontier_adapter import (
    _with_capability_context,
    current_frontier_names,
)

_MARKER = "_mmm_stale_tool_recovery_v1"


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
        """One-shot host correction for stale authorized tool emission."""

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
            # caller cannot spin forever. We are consuming it here before the core loop
            # sees or executes the invalid action, therefore begin one fresh correction
            # attempt rather than carrying that guard into the retry.
            self._reset_stale_guard()
            by_name = {_name(schema): schema for schema in candidates if _name(schema)}
            selected = tuple(by_name[name] for name in visible if name in by_name)
            if not selected:
                raise ModelConfigurationError(
                    "Model emitted a stale authorized tool call after the causal frontier "
                    "became empty: stale=" + ",".join(stale)
                )

            forced_name = _name(selected[0])
            feedback = {
                "role": "system",
                "content": (
                    "The previous tool action is stale for the current causal state and "
                    "was not executed. Do not repeat it. Perform the host-selected legal "
                    f"next action {forced_name!r} using only the currently visible "
                    "frontier, then continue from its observation."
                ),
            }
            retry_messages = _with_capability_context(
                (*request.messages, feedback),
                stage=self.stage,
                role=self.role,
                tools=selected,
            )
            retry_request = replace(
                request,
                messages=retry_messages,
                tools=selected,
                tool_validation_schemas=tuple(candidates),
                tool_choice={
                    "type": "function",
                    "function": {"name": forced_name},
                },
                parallel_tool_calls=False,
            )
            corrected = self.inner.generate_turn(retry_request)
            corrected_stale = _stale_names(
                corrected,
                authorized_names=authorized,
                visible_names=visible_set,
            )
            if corrected_stale:
                raise ModelConfigurationError(
                    "Model ignored a host-forced causal frontier correction: stale="
                    + ",".join(corrected_stale)
                    + " visible="
                    + ",".join(visible)
                )
            calls = tuple(getattr(corrected, "tool_calls", ()))
            if not calls or any(str(call.name) not in visible_set for call in calls):
                raise ModelConfigurationError(
                    "Model did not produce the host-forced legal causal action "
                    f"{forced_name!r} after stale-tool recovery."
                )
            return corrected

    RecoveringCausalFrontierAdapter.__name__ = base.__name__
    RecoveringCausalFrontierAdapter.__qualname__ = base.__qualname__
    setattr(RecoveringCausalFrontierAdapter, _MARKER, True)
    causal_frontier_contract_module.CausalFrontierAdapter = RecoveringCausalFrontierAdapter


__all__ = ["install"]
