from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_live_ecosystem_network_in_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unit tests opt in explicitly when exercising the live-search path."""

    monkeypatch.setenv("MMM_ECOSYSTEM_DISCOVERY", "off")
