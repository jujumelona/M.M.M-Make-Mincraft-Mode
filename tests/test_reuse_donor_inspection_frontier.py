from __future__ import annotations

from minecraft_mod_ai import reuse_planner, source_transplant


def test_source_transplant_owns_donor_inspection_without_planner_monkeypatch() -> None:
    assert callable(source_transplant.inspect_repository_slice)
    assert "inspect_repository_slice" not in reuse_planner.__dict__
    assert "_discover_donor_candidates" not in reuse_planner.__dict__
    assert "_discover_best_donor" not in reuse_planner.__dict__
