from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import project_manifest_hash_efficiency_contract as contract


class _FakeOrchestrator:
    def __init__(self) -> None:
        self.calls = 0

    def _project_manifest_hash(self, project_root: Path) -> str:
        self.calls += 1
        payload = []
        root = Path(project_root)
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if any(part == ".minecraft_ai" for part in relative.parts):
                continue
            if path.suffix.lower() not in {".java", ".json", ".gradle", ".properties"} and path.name not in {
                "build.gradle",
                "settings.gradle",
                "gradle.properties",
                "fabric.mod.json",
            }:
                continue
            payload.append((relative.as_posix(), path.read_bytes()))
        return repr(payload)


def _modules():
    orchestrator_module = SimpleNamespace(CompleteProductionOrchestrator=_FakeOrchestrator)
    project_index_module = SimpleNamespace(
        _IGNORED_PARTS={".minecraft_ai", ".git", ".gradle", "build", "run", ".cache", "node_modules"},
        _TEXT_SUFFIXES={".java", ".json", ".gradle", ".properties"},
    )
    return orchestrator_module, project_index_module


def test_unchanged_tree_reuses_full_manifest_hash(tmp_path: Path) -> None:
    contract._CACHE.clear()
    orchestrator_module, project_index_module = _modules()
    contract.install(orchestrator_module, project_index_module)

    source = tmp_path / "src/main/java/example/Test.java"
    source.parent.mkdir(parents=True)
    source.write_text("class Test {}\n", encoding="utf-8")

    orchestrator = orchestrator_module.CompleteProductionOrchestrator()
    first = orchestrator._project_manifest_hash(tmp_path)
    second = orchestrator._project_manifest_hash(tmp_path)

    assert second == first
    assert orchestrator.calls == 1


def test_source_change_invalidates_metadata_cache(tmp_path: Path) -> None:
    contract._CACHE.clear()
    orchestrator_module, project_index_module = _modules()
    contract.install(orchestrator_module, project_index_module)

    source = tmp_path / "src/main/java/example/Test.java"
    source.parent.mkdir(parents=True)
    source.write_text("class Test {}\n", encoding="utf-8")

    orchestrator = orchestrator_module.CompleteProductionOrchestrator()
    first = orchestrator._project_manifest_hash(tmp_path)
    source.write_text("class Test { int value = 1; }\n", encoding="utf-8")
    second = orchestrator._project_manifest_hash(tmp_path)

    assert second != first
    assert orchestrator.calls == 2


def test_audit_metadata_does_not_invalidate_source_manifest_cache(tmp_path: Path) -> None:
    contract._CACHE.clear()
    orchestrator_module, project_index_module = _modules()
    contract.install(orchestrator_module, project_index_module)

    source = tmp_path / "src/main/java/example/Test.java"
    source.parent.mkdir(parents=True)
    source.write_text("class Test {}\n", encoding="utf-8")

    orchestrator = orchestrator_module.CompleteProductionOrchestrator()
    first = orchestrator._project_manifest_hash(tmp_path)

    audit = tmp_path / ".minecraft_ai/project-index.json"
    audit.parent.mkdir(parents=True)
    audit.write_text('{"status":"updated"}\n', encoding="utf-8")
    second = orchestrator._project_manifest_hash(tmp_path)

    assert second == first
    assert orchestrator.calls == 1
