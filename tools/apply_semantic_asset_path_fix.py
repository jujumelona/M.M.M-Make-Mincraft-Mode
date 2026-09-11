from pathlib import Path

pipeline = Path("minecraft_mod_ai/planning_pipeline.py")
text = pipeline.read_text(encoding="utf-8")
old = '''            assets = []
            for asset in design.get("_atomic_assets", ()):
                if not isinstance(asset, AssetRequest):
                    raise PlanningStageError(PlanningStage.DESIGN, "atomic asset must be typed")
                parts = asset.target_path.split("/")
                if len(parts) < 3 or parts[0] != "assets":
                    raise PlanningStageError(PlanningStage.DESIGN, "atomic asset has invalid namespace path")
                parts[1] = proposal.spec.mod_id
                assets.append(replace(asset, target_path="/".join(parts)))
            design = {**design, "assets":assets, "_atomic_assets":assets}
'''
new = '''            assets = []
            for asset in design.get("_atomic_assets", ()):
                if not isinstance(asset, AssetRequest):
                    raise PlanningStageError(PlanningStage.DESIGN, "atomic asset must be typed")
                assets.append(asset)
            design = {**design, "assets": assets, "_atomic_assets": assets}
'''
if text.count(old) != 1:
    raise SystemExit(f"planning_pipeline semantic asset block expected once, found {text.count(old)}")
pipeline.write_text(text.replace(old, new, 1), encoding="utf-8")

test = Path("tests/test_atomic_design_pipeline.py")
text = test.read_text(encoding="utf-8")
old = '''def test_normal_planning_projection_keeps_graph_content_and_asset_namespace(
    monkeypatch,
):
'''
new = '''def test_normal_planning_projection_keeps_graph_content_and_semantic_assets(
    monkeypatch,
):
'''
if text.count(old) != 1:
    raise SystemExit("atomic design projection test name did not match once")
text = text.replace(old, new, 1)
old = '''    assert all(
        a.target_path.startswith(f"assets/{proposal.spec.mod_id}/")
        for a in design["_atomic_assets"]
    )
'''
new = '''    assert design["_atomic_assets"]
    assert all(a.container in {"mod", "resource_pack"} for a in design["_atomic_assets"])
    assert all(a.render_kind for a in design["_atomic_assets"])
    assert all(a.subject_id for a in design["_atomic_assets"])
    assert all(not hasattr(a, "target_path") for a in design["_atomic_assets"])
'''
if text.count(old) != 1:
    raise SystemExit("atomic design target_path assertion did not match once")
test.write_text(text.replace(old, new, 1), encoding="utf-8")
