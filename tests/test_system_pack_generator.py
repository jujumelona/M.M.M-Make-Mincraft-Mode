from pathlib import Path

from minecraft_mod_ai.system_pack_generator import generate_system_pack


def test_system_pack_generates_java17_foundation_and_contract(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    result = generate_system_pack(
        project_root=project,
        pack_id="quest-system",
        mod_id="testmod",
        package_name="ai.minecraft.testmod",
        config={"quests": [{"id": "start"}]},
    )
    assert result["status"] == "fabric_binding_and_runtime_tests_required"
    paths = [Path(path) for path in result["files"]]
    assert all(path.is_file() for path in paths)
    assert any(path.name == "QuestSystemContract.java" for path in paths)
