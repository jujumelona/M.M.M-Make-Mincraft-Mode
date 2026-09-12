from __future__ import annotations


def test_postbuild_validation_uses_shared_deadline_executor(monkeypatch):
    from minecraft_mod_ai import complete_orchestrator as orchestrator
    from minecraft_mod_ai import deadline_executor

    seen: dict[str, object] = {}

    def fake_iter(units, worker, *, max_workers, stage, sort_key=None, **_kwargs):
        ordered = list(units)
        seen["stage"] = stage
        seen["max_workers"] = max_workers
        seen["names"] = [item[0] for item in ordered]
        for item in ordered:
            yield item, worker(item)

    monkeypatch.setattr(deadline_executor, "iter_completed_with_deadlines", fake_iter)

    calls: list[str] = []

    def validate_source():
        calls.append("source")
        return {"status": "PASS", "kind": "source"}

    def validate_jdt():
        calls.append("jdt")
        return {"status": "PASS", "kind": "jdt"}

    source, jdt, refreshed = orchestrator._refresh_validation_after_build(
        prebuild_manifest="before",
        final_manifest="after",
        source_report={"status": "STALE"},
        jdt_receipt=None,
        validate_source=validate_source,
        validate_jdt=validate_jdt,
    )

    assert refreshed is True
    assert source == {"status": "PASS", "kind": "source"}
    assert jdt == {"status": "PASS", "kind": "jdt"}
    assert calls == ["source", "jdt"]
    assert seen == {
        "stage": "complete_post_build_validation",
        "max_workers": 2,
        "names": ["source", "jdt"],
    }


def test_postbuild_validation_skips_executor_when_tree_is_unchanged(monkeypatch):
    from minecraft_mod_ai import complete_orchestrator as orchestrator
    from minecraft_mod_ai import deadline_executor

    def forbidden(*_args, **_kwargs):
        raise AssertionError("deadline executor must not run for an unchanged tree")

    monkeypatch.setattr(deadline_executor, "iter_completed_with_deadlines", forbidden)
    source_report = {"status": "PASS", "kind": "existing"}
    jdt_receipt = {"status": "PASS", "kind": "existing-jdt"}

    source, jdt, refreshed = orchestrator._refresh_validation_after_build(
        prebuild_manifest="same",
        final_manifest="same",
        source_report=source_report,
        jdt_receipt=jdt_receipt,
        validate_source=lambda: (_ for _ in ()).throw(AssertionError("unexpected source validation")),
        validate_jdt=lambda: (_ for _ in ()).throw(AssertionError("unexpected jdt validation")),
    )

    assert refreshed is False
    assert source is source_report
    assert jdt is jdt_receipt
