from __future__ import annotations

import inspect

from minecraft_mod_ai import central_intelligence_amplifier


def test_central_intelligence_has_no_raw_executor_waits() -> None:
    source = inspect.getsource(central_intelligence_amplifier)
    assert "ThreadPoolExecutor" not in source
    assert "as_completed" not in source
    assert ".result()" not in source
    assert "iter_completed_with_deadlines" in source


def test_every_central_fanout_has_a_named_deadline_stage() -> None:
    source = inspect.getsource(central_intelligence_amplifier)
    stages = (
        "central-research-providers",
        "central-research-domain-wave-",
        "central-research-domain-retry-",
        "central-design-sections",
        "central-intelligence-overlap",
        "central-critical-gap-correction",
        "central-council-specialists",
        "central-council-synthesis",
        "central-reviews",
    )
    for stage in stages:
        assert stage in source
