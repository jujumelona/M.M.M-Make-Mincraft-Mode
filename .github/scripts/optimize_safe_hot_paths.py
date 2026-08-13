from __future__ import annotations

from pathlib import Path

PROD = Path("minecraft_mod_ai/production_tools.py")
CUSTOM = Path("minecraft_mod_ai/custom_module_generator.py")
TEST = Path("tests/test_safe_hot_path_optimization.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise SystemExit(f"missing anchor: {label}")
    return text.replace(old, new, 1)


def patch_production_tools() -> None:
    text = PROD.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from dataclasses import asdict\nfrom pathlib import Path\n",
        "from dataclasses import asdict\nfrom functools import cached_property\nfrom pathlib import Path\n",
        "cached_property import",
    )
    text = replace_once(
        text,
        "        self.profile = profile\n        self.runtime = MinecraftRuntimeManager(self.workspace_root)\n        self.mineflayer = MineflayerBridge()\n\n",
        "        self.profile = profile\n\n"
        "    @cached_property\n"
        "    def runtime(self) -> MinecraftRuntimeManager:\n"
        "        return MinecraftRuntimeManager(self.workspace_root)\n\n"
        "    @cached_property\n"
        "    def mineflayer(self) -> MineflayerBridge:\n"
        "        return MineflayerBridge()\n\n",
        "lazy runtime services",
    )
    compile(text, str(PROD), "exec")
    PROD.write_text(text, encoding="utf-8")


def patch_custom_generator() -> None:
    text = CUSTOM.read_text(encoding="utf-8")
    old = '''        live_manifest = ProjectIndex(root, policy=self.policy).manifest_receipt()
        ProductionToolService(
'''
    new = '''        # Reuse the already-built whole-project index for the RAG source receipt.
        # Constructing a second ProjectIndex here rescanned the entire project before
        # every custom-module decode. If another lane changed the source after the
        # cached index was created, refresh once and keep the cache coherent.
        try:
            live_manifest = index.manifest_receipt()
        except ValueError as exc:
            if not _is_stale_project_index_error(exc):
                raise
            index = ProjectIndex(root, policy=self.policy)
            self._cached_index = index
            self._cached_root = root
            live_manifest = index.manifest_receipt()

        ProductionToolService(
'''
    text = replace_once(text, old, new, "reuse project index manifest")
    compile(text, str(CUSTOM), "exec")
    CUSTOM.write_text(text, encoding="utf-8")


def write_tests() -> None:
    TEST.write_text(
        '''from __future__ import annotations

import inspect

from minecraft_mod_ai import custom_module_generator, production_tools


def test_production_tool_service_lazily_constructs_runtime_bridges(monkeypatch, tmp_path) -> None:
    calls = {"runtime": 0, "mineflayer": 0}

    class Runtime:
        def __init__(self, root) -> None:
            calls["runtime"] += 1
            self.root = root

    class Mineflayer:
        def __init__(self) -> None:
            calls["mineflayer"] += 1

    monkeypatch.setattr(production_tools, "MinecraftRuntimeManager", Runtime)
    monkeypatch.setattr(production_tools, "MineflayerBridge", Mineflayer)

    service = production_tools.ProductionToolService(
        workspace_root=tmp_path,
        profile="test",
    )
    assert calls == {"runtime": 0, "mineflayer": 0}

    assert service.runtime.root == tmp_path.resolve()
    assert service.runtime is service.runtime
    assert calls == {"runtime": 1, "mineflayer": 0}

    assert service.mineflayer is service.mineflayer
    assert calls == {"runtime": 1, "mineflayer": 1}


def test_custom_generation_reuses_existing_project_index_manifest() -> None:
    source = inspect.getsource(custom_module_generator.CustomModuleGenerator.generate)
    assert "ProjectIndex(root, policy=self.policy).manifest_receipt()" not in source
    assert "live_manifest = index.manifest_receipt()" in source
    assert "self._cached_index = index" in source
''',
        encoding="utf-8",
    )


def main() -> None:
    patch_production_tools()
    patch_custom_generator()
    write_tests()
    for path in (PROD, CUSTOM, TEST):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


if __name__ == "__main__":
    main()
