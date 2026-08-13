from pathlib import Path

path = Path(".github/scripts/optimize_work_graph_core.py")
text = path.read_text(encoding="utf-8")
start = text.find("def patch_tests() -> None:\n")
end = text.find("\ndef main() -> None:\n", start)
if start < 0 or end < 0:
    raise SystemExit("optimizer patch_tests function not found")
replacement = '''def patch_tests() -> None:
    # Only migrate the execution-efficiency ownership assertion here. Generation-lane
    # tests are normalized by finalize_safe_execution_ownership.py against the current
    # repository state, avoiding brittle historical-text replacements.
    text = EFF_TEST.read_text(encoding="utf-8")
    text = text.replace(
        "from minecraft_mod_ai import complete_planner, execution_efficiency_contract, work_graph\\n",
        "from minecraft_mod_ai import complete_planner, work_graph\\n",
        1,
    )
    text = text.replace(
        "source = inspect.getsource(execution_efficiency_contract._dependency_wave_shards)",
        "source = inspect.getsource(work_graph._module_shards)",
        1,
    )
    EFF_TEST.write_text(text, encoding="utf-8")
'''
path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
