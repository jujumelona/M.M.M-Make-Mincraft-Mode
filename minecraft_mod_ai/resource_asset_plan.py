from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class ResourceAssetPlanError(RuntimeError):
    """Raised when an approved resource plan cannot satisfy an execution shard."""


def select_plan_rows(
    plan: Mapping[str, Any],
    requests: Sequence[Any],
) -> list[dict[str, Any]]:
    """Select the approved rows for a semantic asset shard in request order.

    The approved plan is authoritative. A shard may execute a strict subset of it, but
    it may not invent asset IDs, silently drop duplicate plan rows, or reorder requests.
    """
    raw_rows = plan.get("assets")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes, bytearray)):
        raise ResourceAssetPlanError("Resource asset plan has no asset rows.")

    by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ResourceAssetPlanError("Resource asset plan contains a non-object row.")
        asset_id = str(raw.get("asset_id") or "").strip()
        if not asset_id:
            raise ResourceAssetPlanError("Resource asset plan contains an empty asset ID.")
        if asset_id in by_id:
            raise ResourceAssetPlanError(
                f"Resource asset plan contains duplicate asset ID {asset_id!r}."
            )
        by_id[asset_id] = raw

    selected: list[dict[str, Any]] = []
    seen_requests: set[str] = set()
    for request in requests:
        asset_id = str(getattr(request, "asset_id", "") or "").strip()
        if not asset_id:
            raise ResourceAssetPlanError("Resource asset shard contains an empty asset ID.")
        if asset_id in seen_requests:
            raise ResourceAssetPlanError(
                f"Resource asset shard contains duplicate asset ID {asset_id!r}."
            )
        seen_requests.add(asset_id)
        row = by_id.get(asset_id)
        if row is None:
            raise ResourceAssetPlanError(
                f"Approved resource asset plan has no row for requested asset {asset_id!r}."
            )
        selected.append(dict(row))
    return selected


# Private compatibility name used by the migration regression test. The implementation
# lives in this focused plan module rather than the image producer.
_select_plan_rows = select_plan_rows


__all__ = ["ResourceAssetPlanError", "select_plan_rows"]
