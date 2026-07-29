from pathlib import Path

from minecraft_mod_ai.rag_index import ProjectRAGIndex


def _metadata() -> dict:
    return {
        "minecraft_version": "1.20.1",
        "loader": "fabric",
        "mapping_namespace": "yarn",
        "java_version": 17,
        "license": "Apache-2.0",
        "source_commit": "abc123",
    }


def test_lexical_project_rag_is_version_and_license_aware(tmp_path: Path) -> None:
    source = tmp_path / "project" / "src" / "Example.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        "Registry.register(Registries.ITEM, new Identifier(MOD_ID, \"crystal\"), item);",
        encoding="utf-8",
    )
    index_path = tmp_path / "index.json"
    result = ProjectRAGIndex(index_path).build(
        [source.parent.parent],
        metadata=_metadata(),
    )
    assert result["files_indexed"] == 1
    hits = ProjectRAGIndex(index_path).search(
        "Fabric Registry.register item crystal",
        required_metadata={"minecraft_version": "1.20.1", "loader": "fabric"},
    )
    assert hits
    assert hits[0].source_path.endswith("Example.java")
    assert hits[0].metadata["license"] == "Apache-2.0"
