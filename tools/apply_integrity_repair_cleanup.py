from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, value: str) -> None:
    (ROOT / path).write_text(value, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    value = read(path)
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one patch target, found {count}")
    write(path, value.replace(old, new, 1))


# Item visuals need both a palette and a silhouette/shape. The host-owned
# content-property protocol only asks required fields, so shape must be explicit here.
replace_once(
    "minecraft_mod_ai/content_design_graph.py",
    '            FactType.ITEM_EXISTS: {"display_name", "main_color"},\n',
    '            FactType.ITEM_EXISTS: {"display_name", "main_color", "shape"},\n',
)

# Test fakes must be able to answer the newly explicit item shape slot when a fixture
# did not author one itself.
for path in ("tests/test_atomic_design_pipeline.py", "tests/test_content_design_graph_expansion.py"):
    value = read(path)
    marker = '                    "main_color": "#808080",\n'
    if value.count(marker) != 1:
        raise RuntimeError(f"{path}: fake default palette target missing")
    value = value.replace(
        marker,
        marker + '                    "shape": "faceted chunk",\n',
        1,
    )
    write(path, value)

# source_id/target_id are no longer model output. The host enumerates only known
# entity pairs, which makes a dangling relation unrepresentable at this boundary.
replace_once(
    "tests/test_atomic_design_pipeline.py",
    '''def test_graph_rejects_dangling_relation():\n    with pytest.raises(SlotFillError, match="DANGLING"):\n        compile_atomic_design(\n            "materials",\n            GraphRouter(\n                edges=[\n                    {\n                        "relation_type": "drops",\n                        "source_id": "missing",\n                        "target_id": "raw_material",\n                    }\n                ]\n            ),\n        )\n''',
    '''def test_relation_pairs_are_host_scoped_and_cannot_be_dangling():\n    design = compile_atomic_design(\n        "materials",\n        GraphRouter(\n            edges=[\n                {\n                    "relation_type": "drops",\n                    "source_id": "missing",\n                    "target_id": "raw_material",\n                }\n            ]\n        ),\n    )\n    assert design["_content_relations"] == []\n''',
)

# This integration test verifies graph/materializer reachability, not production HOST
# admission. Admission fail-closed behavior is covered separately, so isolate the
# materialization layer by replacing only the test's execution-context resolver.
path = "tests/test_atomic_design_pipeline.py"
value = read(path)
old_sig = "def test_recipe_content_graph_reaches_artifact_files(tmp_path):\n"
new_sig = "def test_recipe_content_graph_reaches_artifact_files(tmp_path, monkeypatch):\n"
if value.count(old_sig) != 1:
    raise RuntimeError("recipe integration signature target missing")
value = value.replace(old_sig, new_sig, 1)
old_exec = "    execute_artifact_graph(jobs, base_dir=tmp_path)\n"
new_exec = '''    import minecraft_mod_ai.resolved_version_context as resolved_version_context\n\n    monkeypatch.setattr(\n        resolved_version_context,\n        "execution_context",\n        lambda context, job: None,\n    )\n    execute_artifact_graph(jobs, base_dir=tmp_path)\n'''
if value.count(old_exec) != 1:
    raise RuntimeError("recipe integration execution target missing")
write(path, value.replace(old_exec, new_exec, 1))

print("repair cleanup applied")
