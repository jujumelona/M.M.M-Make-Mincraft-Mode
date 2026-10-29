"""Progress checks for model-driven menu navigation and expression recursion.

Navigation can legitimately return to an earlier page; only repeating the same
(state, choice) transition without accepting a member is a loop. This preserves
pagination, backwards inspection, and changed search queries.
"""
from __future__ import annotations


MAX_UNPRODUCTIVE_TRANSITION_REPEATS = 2


def record_navigation_transition(history, state, action, *, error):
    key = (state, action)
    history[key] = history.get(key, 0) + 1
    if history[key] > MAX_UNPRODUCTIVE_TRANSITION_REPEATS:
        raise RuntimeError(f"{error}: state={state!r} action={action!r}")


def require_expression_depth(stack, slot, limit):
    if len(stack) >= limit:
        raise RuntimeError(
            "JAVA_VALUE_EXPRESSION_NESTING_EXHAUSTED: "
            f"slot={slot}; author intermediate results as sequential Java operations"
        )
