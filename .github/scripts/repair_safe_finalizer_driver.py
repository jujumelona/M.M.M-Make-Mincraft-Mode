from pathlib import Path

path = Path(".github/scripts/finalize_safe_execution_ownership.py")
text = path.read_text(encoding="utf-8")
start = text.find("def patch_scheduler() -> None:\n")
end = text.find("\ndef patch_lane_tests() -> None:\n", start)
if start < 0 or end < 0:
    raise SystemExit("finalizer patch_scheduler function not found")
replacement = '''def patch_scheduler() -> None:
    text = SCHED.read_text(encoding="utf-8")

    # Remove obsolete per-stage locks. Shared source/registry mutation is owned by the
    # single project commit lane instead, so cross-stage races cannot bypass each other.
    start = text.find("_STAGE_WRITE_LOCKS = {")
    if start >= 0:
        end = text.find("}_INDEX_COMMIT_LOCK", start)
        if end < 0:
            end = text.find("}\\n_INDEX_COMMIT_LOCK", start)
            suffix = "\\n_INDEX_COMMIT_LOCK"
        else:
            suffix = "_INDEX_COMMIT_LOCK"
        if end < 0:
            raise SystemExit("stage lock constant end missing")
        text = text[:start] + "_INDEX_COMMIT_LOCK" + text[end + len(suffix) :]

    text = remove_function(text, "_stage_write_lock")

    marker = "            stage_lock = _stage_write_lock(node)\\n"
    if marker in text:
        block_start = text.index(marker)
        block_end_marker = "            if not isinstance(receipt, dict):\\n"
        block_end = text.find(block_end_marker, block_start)
        if block_end < 0:
            raise SystemExit("run_work_node receipt boundary missing")
        replacement_block = '''            if (
                node.resource_class == "commit"
                and shared_index is not None
                and hasattr(shared_index, "root")
            ):
                with project_write_lock(shared_index.root):
                    receipt = action()
            else:
                receipt = action()
'''
        text = text[:block_start] + replacement_block + text[block_end:]

    if "stage_lock = _stage_write_lock(node)" in text:
        raise SystemExit("stage-local write lock still owns generation")
    text = text.replace('    "_stage_write_lock",\\n', "")
    compile(text, str(SCHED), "exec")
    SCHED.write_text(text, encoding="utf-8")
'''
path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
