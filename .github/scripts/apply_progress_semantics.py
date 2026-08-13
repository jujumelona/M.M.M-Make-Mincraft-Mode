from __future__ import annotations

from pathlib import Path


def once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, got {count}")
    return text.replace(old, new, 1)


# Object batches remain field-patchable for as many changing validator states
# as necessary. Whole-object replacement is only required for non-object input.
path = Path("minecraft_mod_ai/planner_incremental_resume_contract.py")
text = path.read_text(encoding="utf-8")
text = once(
    text,
    "def _install_bounded_batch_repair(incremental_module: Any) -> None:\n",
    "def _install_progressive_batch_repair(incremental_module: Any) -> None:\n",
    "repair installer name",
)
text = once(
    text,
    '    if getattr(current, "_mmm_bounded_semantic_batch_repair", False):\n',
    '    if getattr(current, "_mmm_progressive_semantic_batch_repair", False):\n',
    "repair marker read",
)
text = once(
    text,
    '        """Field-patch once, then regenerate this one batch once; never retry a mode."""\n',
    '        """Repair until validation succeeds; object batches remain field-patchable."""\n',
    "repair docstring",
)
text = once(
    text,
    "            replacement = not isinstance(current_value, dict) or attempt > 1\n",
    "            replacement = not isinstance(current_value, dict)\n",
    "forced replacement policy",
)
text = once(
    text,
    "    patch_one_invalid_batch._mmm_bounded_semantic_batch_repair = True  # type: ignore[attr-defined]\n",
    "    patch_one_invalid_batch._mmm_progressive_semantic_batch_repair = True  # type: ignore[attr-defined]\n",
    "repair marker write",
)
text = once(
    text,
    "    _install_bounded_batch_repair(incremental_module)\n",
    "    _install_progressive_batch_repair(incremental_module)\n",
    "repair install call",
)
text = text.replace(
    '    """Install resume plus terminating outline/batch repair semantics."""',
    '    """Install resume plus progress-driven outline/batch repair semantics."""',
)
if "attempt > 1" in text or "_install_bounded_batch_repair" in text:
    raise SystemExit("forced repair-mode policy still present")
path.write_text(text, encoding="utf-8")


# A model cursor is transport metadata. A page that produced new globally unique
# modules made semantic progress even if the model reused or omitted its cursor.
path = Path("minecraft_mod_ai/complete_planner.py")
text = path.read_text(encoding="utf-8")
old = '''            complete = page["complete"]
            next_cursor = page["next_cursor"]
            if type(complete) is not bool or not isinstance(next_cursor, str):
                raise SpecValidationError("Module batch pagination contract is invalid.")
            if complete:
                if next_cursor:
                    raise SpecValidationError("Complete module page may not have next_cursor.")
            elif not next_cursor or next_cursor in seen_cursors:
                raise SpecValidationError("Module batch pagination did not advance.")
            for module in page_modules:
                module_catalog.add(module.module_id)
            generated.extend(page_modules)
            if complete:
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor
'''
new = '''            complete = page["complete"]
            next_cursor = page["next_cursor"]
            if type(complete) is not bool or not isinstance(next_cursor, str):
                raise SpecValidationError("Module batch pagination contract is invalid.")
            if complete and next_cursor:
                raise SpecValidationError("Complete module page may not have next_cursor.")

            for module in page_modules:
                module_catalog.add(module.module_id)
            generated.extend(page_modules)
            if complete:
                break

            # New unique module IDs are semantic progress. Cursor text is only
            # transport metadata, so derive a host cursor when the model repeats
            # or omits it instead of truncating the plan.
            if not next_cursor or next_cursor in seen_cursors:
                next_cursor = "host_resume_" + _canonical_json_sha256(
                    {
                        "batch_id": batch_id,
                        "catalog": module_catalog.receipt(),
                    }
                )[:20]
                if next_cursor in seen_cursors:
                    raise SpecValidationError(
                        "Module batch pagination repeated both cursor and catalog state."
                    )
            seen_cursors.add(next_cursor)
            cursor = next_cursor
'''
text = once(text, old, new, "legacy cursor progress")
if "elif not next_cursor or next_cursor in seen_cursors" in text:
    raise SystemExit("cursor-only truncation still present")
path.write_text(text, encoding="utf-8")


# Regression: two changing invalid fields may be repaired in two field patches.
path = Path("tests/test_planner_incremental_repair_contract.py")
text = path.read_text(encoding="utf-8")
old = '''def test_invalid_field_patch_escalates_to_one_complete_batch_regeneration(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    bad = _batch("broken_scope", scope="")
    repaired = _batch("broken_scope", scope="Implement the corrected scope.")
    router = _Router(
        _outline(bad),
        _field_patch(bad, scope=""),
        _replacement_patch(bad, repaired),
    )

    page = _run(router, stage="field patch then regenerate")

    assert page["complete"] is True
    assert page["production_batches"] == [repaired]
    assert len(router.calls) == 3
    assert "field-level JSON patcher" in str(router.calls[1]["messages"][0]["content"])
    assert "regenerate exactly ONE invalid production batch" in str(
        router.calls[2]["messages"][0]["content"]
    )
    second_request = json.loads(str(router.calls[2]["messages"][1]["content"]))
    assert second_request["repair_mode"] == "replacement"
'''
new = '''def test_invalid_batch_keeps_field_patching_while_validation_state_changes(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    bad = _batch("broken_scope", scope="", deliverables=[])
    repaired = _batch(
        "broken_scope",
        scope="Implement the corrected scope.",
        deliverables=["broken_scope_done"],
    )
    router = _Router(
        _outline(bad),
        _field_patch(bad, scope=repaired["scope"]),
        _field_patch(bad, deliverables=repaired["deliverables"]),
    )

    page = _run(router, stage="progressive field patch")

    assert page["complete"] is True
    assert page["production_batches"] == [repaired]
    assert len(router.calls) == 3
    assert "field-level JSON patcher" in str(router.calls[1]["messages"][0]["content"])
    assert "field-level JSON patcher" in str(router.calls[2]["messages"][0]["content"])
    second_request = json.loads(str(router.calls[2]["messages"][1]["content"]))
    assert second_request["repair_mode"] == "field_patch"
    assert second_request["current_batch"]["scope"] == repaired["scope"]
'''
text = once(text, old, new, "progressive field-patch test")
text = text.replace(
    "    # One outline call + exactly two distinct repair modes. No third repair call exists.\n",
    "    # One outline call + two identical field-patch outputs prove no progress.\n",
)
path.write_text(text, encoding="utf-8")


# Regression: repeated model cursor is tolerated while unique modules keep arriving.
path = Path("tests/test_complete_planner_scaling.py")
text = path.read_text(encoding="utf-8")
anchor = "def test_module_pagination_rejects_duplicate_ids_within_page() -> None:\n"
test = '''def test_module_pagination_reused_cursor_does_not_cap_semantic_progress() -> None:
    router = _ResponseRouter(
        [
            {
                "modules": [_module("page_one")],
                "complete": False,
                "next_cursor": "model_cursor",
            },
            {
                "modules": [_module("page_two")],
                "complete": False,
                "next_cursor": "model_cursor",
            },
            {
                "modules": [_module("page_three")],
                "complete": True,
                "next_cursor": "",
            },
        ]
    )

    modules = CompleteGameDesignPlanner(router)._expand_batches(
        prompt="Keep planning while new modules are produced.",
        game_design={"title": "Progress-driven cursor"},
        batches=[
            {
                "batch_id": "core",
                "scope": "Produce all three modules.",
                "depends_on_batches": [],
            }
        ],
        media_paths=(),
    )

    assert [module.module_id for module in modules] == [
        "page_one",
        "page_two",
        "page_three",
    ]
    assert router.requests[0]["cursor"] == ""
    assert router.requests[1]["cursor"] == "model_cursor"
    assert str(router.requests[2]["cursor"]).startswith("host_resume_")


'''
if "test_module_pagination_reused_cursor_does_not_cap_semantic_progress" not in text:
    if anchor not in text:
        raise SystemExit("pagination test anchor missing")
    text = text.replace(anchor, test + anchor, 1)
path.write_text(text, encoding="utf-8")

print("remaining progress semantics patched")
