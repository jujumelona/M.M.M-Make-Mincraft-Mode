from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_production import _compile_new_authored_modules


def test_saved_sections_do_not_reserve_java_files_before_implementation_ir():
    text = "# behavior_contract\nTrade ore.\n# state_model\nPlayerCredits, Ship, ShipPart, persistence.\n# verification\nTest transactions."
    modules, manifest = _compile_new_authored_modules(
        AuthoredPlan("space economy", text), mod_id="authored_test",
        package_name="example", target={"minecraft_version": "1.21.1", "loader": "fabric", "mappings": "none"},
    )
    assert manifest["policy"] == "host_implementation_graph_before_source"
    assert manifest["unit_count"] == 0
    assert manifest["units"] == []
    assert manifest["entrypoint"]["feature_symbols"] == []
    assert len(modules) == 1
    assert modules[0].config["implementation_graph_request"]["text"] == text
    assert "AuthoredFeature" not in str(modules[0].config)
