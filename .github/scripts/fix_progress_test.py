from pathlib import Path

path = Path("tests/test_planner_incremental_repair_contract.py")
text = path.read_text(encoding="utf-8")
old = '''def test_identical_bad_patch_output_is_not_sent_or_consumed_forever(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    bad = _batch("broken_scope", scope="")
    repeated = _field_patch(bad, scope="")
    router = _Router(_outline(bad), repeated, repeated)

    with pytest.raises(SpecValidationError, match="repeated identical model output"):
        _run(router, stage="repeated batch patch output")

    # One outline call + exactly two distinct repair modes. No third repair call exists.
    assert len(router.calls) == 3
'''
new = '''def test_identical_bad_patch_state_is_not_sent_or_consumed_forever(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    bad = _batch("broken_scope", scope="")
    repeated = _field_patch(bad, scope="")
    router = _Router(_outline(bad), repeated)

    with pytest.raises(SpecValidationError, match="repeated an identical semantic state"):
        _run(router, stage="repeated batch patch state")

    # Outline + one unchanged field patch is enough to prove exact no-progress.
    assert len(router.calls) == 2
'''
count = text.count(old)
if count != 1:
    raise SystemExit(f"no-progress regression shape changed: {count}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("semantic no-progress regression aligned")
