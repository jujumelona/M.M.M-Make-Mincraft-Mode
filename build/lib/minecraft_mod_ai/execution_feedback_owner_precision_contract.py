from __future__ import annotations

"""Keep batched generation receipt ownership exact when member/receipt order differs."""

from collections.abc import Mapping, Sequence
from typing import Any


def install(feedback_module: Any) -> None:
    current = feedback_module._receipt_owner_ids
    if getattr(current, "_mmm_receipt_first_ownership", False):
        return

    def receipt_owner_ids(module: Any, receipt: Mapping[str, Any]) -> list[str]:
        owners: set[str] = set()
        # Receipt-declared ownership is authoritative.  A generation node can emit a
        # different number/order of receipts than members (for example one aggregate
        # deterministic receipt for many modules), so positional zip ownership must
        # never be mixed in when the receipt names its real owner(s).
        for key in ("module_id", "entity_id"):
            raw = receipt.get(key)
            if isinstance(raw, str) and raw.strip():
                owners.add(raw.strip())
        raw_modules = receipt.get("modules")
        if isinstance(raw_modules, Sequence) and not isinstance(
            raw_modules, (str, bytes, bytearray)
        ):
            for raw in raw_modules:
                if isinstance(raw, str) and raw.strip():
                    owners.add(raw.strip())
                elif isinstance(raw, Mapping):
                    raw_id = str(raw.get("module_id") or "").strip()
                    if raw_id:
                        owners.add(raw_id)

        # pack_id identifies a generated pack, not necessarily a ProductionModule, so
        # it is useful provenance but not semantic module ownership.
        if not owners:
            module_id = str(getattr(module, "module_id", "") or "").strip()
            if module_id:
                owners.add(module_id)
        return sorted(owners)

    receipt_owner_ids._mmm_receipt_first_ownership = True
    receipt_owner_ids.__wrapped__ = current
    feedback_module._receipt_owner_ids = receipt_owner_ids


__all__ = ["install"]
