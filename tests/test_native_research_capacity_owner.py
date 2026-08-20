from __future__ import annotations

from minecraft_mod_ai import central_intelligence_amplifier as central
from minecraft_mod_ai import research_bottleneck_runtime as bottleneck


def test_late_research_bridge_does_not_replace_native_capacity_owner() -> None:
    current = central._research_domain_worker_count
    bottleneck.install()
    assert central._research_domain_worker_count is current
    assert not getattr(current, "_mmm_receipt_capacity_bridge_v1", False)
