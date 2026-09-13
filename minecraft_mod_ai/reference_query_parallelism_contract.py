from __future__ import annotations

"""Validate the canonical reference-research concurrency owner.

Reference retrieval is implemented by ``reference_source_research`` itself.  This runtime
contract deliberately does not replace that function: late monkey-patching previously
changed retrieval semantics, duplicated provider orchestration, reduced evidence recall,
and made the source implementation differ from the function that actually ran after
package bootstrap.

The contract now has one job only: fail closed if the canonical host-owned implementation
is missing, then mark that boundary as validated for runtime integrity checks.
"""

from typing import Any


_REQUIRED_CALLABLES = (
    "_reference_search_plan",
    "_retrieve_provider",
    "retrieve_reference_grounded_evidence",
)


def install(reference_module: Any) -> None:
    if getattr(reference_module, "_mmm_query_parallelism_contract", False):
        return

    missing = [
        name
        for name in _REQUIRED_CALLABLES
        if not callable(getattr(reference_module, name, None))
    ]
    if missing:
        raise RuntimeError(
            "reference research canonical concurrency owner is incomplete: "
            + ", ".join(missing)
        )

    reference_module._mmm_query_parallelism_contract = True


__all__ = ["install"]
