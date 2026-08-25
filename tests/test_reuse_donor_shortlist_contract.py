from __future__ import annotations

from threading import Lock

from minecraft_mod_ai import reuse_planner
from minecraft_mod_ai.source_transplant import DonorSlice


def test_deep_donor_inspection_has_no_fixed_first_six_cap(monkeypatch):
    repositories = tuple(f"example/repo-{index}" for index in range(9))
    seen: set[str] = set()
    seen_lock = Lock()

    def fake_inspect(*, repository, capability, adapter, discovery_client):
        del adapter, discovery_client
        with seen_lock:
            seen.add(repository)
        exact = repository == repositories[-1]
        return DonorSlice(
            capability=capability,
            repository=repository,
            commit_sha="a" * 40,
            license_id="MIT",
            source_url=f"https://github.com/{repository}",
            target_compatibility="exact" if exact else "adapt",
            files=(),
            seed_files=(),
            source_symbols=(),
            required_dependencies=(),
            donor_tests=(),
            confidence=0.99 if exact else 0.60,
        )

    monkeypatch.setattr(reuse_planner, "inspect_repository_slice", fake_inspect)
    monkeypatch.setattr(reuse_planner, "_workers", lambda: 4)

    donor = reuse_planner._discover_best_donor(
        "economy.trade",
        adapter=object(),
        discovery_client=object(),
        repositories=repositories,
    )

    assert seen == set(repositories)
    assert donor is not None
    assert donor.repository == repositories[-1]
    assert donor.exact_target is True
