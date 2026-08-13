from __future__ import annotations

import re
from pathlib import Path

RAG = Path("minecraft_mod_ai/rag_index.py")
PROD = Path("minecraft_mod_ai/production_tools.py")
CUSTOM = Path("minecraft_mod_ai/custom_module_generator.py")
REPAIR = Path("minecraft_mod_ai/repair_engine.py")
BOOT = Path("minecraft_mod_ai/runtime_bootstrap.py")
WRAPPER = Path("minecraft_mod_ai/production_tool_parallel_contract.py")
OLD_TEST = Path("tests/test_production_tool_parallel_contract.py")
NEW_TEST = Path("tests/test_production_rag_coordination.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise SystemExit(f"missing anchor: {label}")
    return text.replace(old, new, 1)


def replace_method(text: str, name: str, replacement: str) -> str:
    marker = f"    def {name}("
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"missing method: {name}")
    next_method = re.search(r"\n    def [A-Za-z_]", text[start + 1 :])
    if next_method is None:
        raise SystemExit(f"cannot find end of method: {name}")
    end = start + 1 + next_method.start() + 1
    return text[:start] + replacement.rstrip() + "\n\n" + text[end:]


def patch_rag_index() -> None:
    text = RAG.read_text(encoding="utf-8")
    anchor = "    def search(\n"
    method = '''    def matches_build(\n        self,\n        *,\n        metadata: dict[str, Any],\n        semantic: bool,\n    ) -> bool:\n        \"\"\"Return whether the durable index already represents this exact build.\n\n        This is intentionally metadata-driven: callers supply a source_commit derived\n        from the current project manifest, so unchanged source can reuse the index while\n        changed source forces an atomic rebuild.\n        \"\"\"\n\n        _validate_metadata(metadata)\n        if not self.index_path.is_file():\n            return False\n        if _is_sqlite(self.index_path):\n            connection = sqlite3.connect(str(self.index_path))\n            try:\n                index_meta = _read_index_meta(connection)\n                if index_meta.get(\"schema_version\") != self.schema_version:\n                    return False\n                stored = json.loads(index_meta.get(\"metadata\", \"{}\"))\n                return (\n                    _canonical_json(stored) == _canonical_json(metadata)\n                    and (index_meta.get(\"semantic_embeddings\") == \"1\")\n                    is bool(semantic)\n                )\n            except (ValueError, json.JSONDecodeError, sqlite3.DatabaseError):\n                return False\n            finally:\n                connection.close()\n\n        try:\n            raw = json.loads(self.index_path.read_text(encoding=\"utf-8\"))\n        except (OSError, json.JSONDecodeError):\n            return False\n        if raw.get(\"schema_version\") != _LEGACY_SCHEMA_VERSION:\n            return False\n        chunks = raw.get(\"chunks\", [])\n        if not isinstance(chunks, list) or not chunks:\n            return False\n        first = chunks[0]\n        if not isinstance(first, dict):\n            return False\n        stored = first.get(\"metadata\")\n        if not isinstance(stored, dict):\n            return False\n        semantic_present = any(\n            isinstance(item, dict) and bool(item.get(\"embedding\"))\n            for item in chunks\n        )\n        return (\n            _canonical_json(stored) == _canonical_json(metadata)\n            and semantic_present is bool(semantic)\n        )\n\n'''
    if "    def matches_build(" not in text:
        text = text.replace(anchor, method + anchor, 1)
    compile(text, str(RAG), "exec")
    RAG.write_text(text, encoding="utf-8")


def patch_production_tools() -> None:
    text = PROD.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from __future__ import annotations\n\nfrom dataclasses import asdict\nfrom pathlib import Path\nfrom typing import Any, Sequence\n",
        "from __future__ import annotations\n\nimport hashlib\nimport threading\nfrom dataclasses import asdict\nfrom functools import cached_property\nfrom pathlib import Path\nfrom typing import Any, Sequence\n",
        "production imports",
    )
    imports_end = "from .training import TrainingTraceStore\n\n\n"
    ownership = '''from .training import TrainingTraceStore\n\n\n_RAG_BUILD_LOCKS = tuple(threading.RLock() for _ in range(64))\n_RAG_BUILD_EPOCHS_LOCK = threading.RLock()\n_RAG_BUILD_EPOCHS: dict[Path, int] = {}\n\n\ndef _rag_build_lock(path: Path) -> threading.RLock:\n    digest = hashlib.sha256(str(path).encode(\"utf-8\")).digest()\n    slot = int.from_bytes(digest[:2], \"big\") % len(_RAG_BUILD_LOCKS)\n    return _RAG_BUILD_LOCKS[slot]\n\n\ndef _rag_build_epoch(path: Path) -> int:\n    with _RAG_BUILD_EPOCHS_LOCK:\n        return _RAG_BUILD_EPOCHS.get(path, 0)\n\n\ndef _advance_rag_build_epoch(path: Path) -> None:\n    with _RAG_BUILD_EPOCHS_LOCK:\n        _RAG_BUILD_EPOCHS[path] = _RAG_BUILD_EPOCHS.get(path, 0) + 1\n\n\n'''
    text = replace_once(text, imports_end, ownership, "RAG coordination owner")
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
        "lazy runtime ownership",
    )
    replacement = '''    def index_project_rag(\n        self,\n        roots: Sequence[str],\n        *,\n        index_path: str = \"rag/project-index.json\",\n        metadata: dict[str, Any],\n        semantic: bool = False,\n    ) -> dict[str, Any]:\n        \"\"\"Force one atomic rebuild while deduplicating only concurrent callers.\"\"\"\n\n        resolved = [str(self._existing_path(root)) for root in roots]\n        target = self._replaceable_file(index_path)\n        observed_epoch = _rag_build_epoch(target)\n        with _rag_build_lock(target):\n            if _rag_build_epoch(target) != observed_epoch and target.exists():\n                raise FileExistsError(\n                    f\"RAG index was refreshed by a concurrent builder: {target}\"\n                )\n            result = self._build_project_rag(\n                resolved,\n                target=target,\n                metadata=metadata,\n                semantic=semantic,\n            )\n            _advance_rag_build_epoch(target)\n            return result\n\n    def ensure_project_rag(\n        self,\n        roots: Sequence[str],\n        *,\n        index_path: str = \"rag/project-index.json\",\n        metadata: dict[str, Any],\n        semantic: bool = False,\n    ) -> dict[str, Any]:\n        \"\"\"Reuse an exact live index or rebuild it once when source metadata changed.\"\"\"\n\n        resolved = [str(self._existing_path(root)) for root in roots]\n        target = self._replaceable_file(index_path)\n        with _rag_build_lock(target):\n            index = ProjectRAGIndex(target)\n            if index.matches_build(metadata=metadata, semantic=semantic):\n                return {\n                    \"schema_version\": \"mmm/rag-ensure-result-v1\",\n                    \"status\": \"CURRENT\",\n                    \"index_path\": str(target),\n                    \"source_commit\": str(metadata[\"source_commit\"]),\n                    \"semantic_embeddings\": semantic,\n                }\n            result = self._build_project_rag(\n                resolved,\n                target=target,\n                metadata=metadata,\n                semantic=semantic,\n            )\n            _advance_rag_build_epoch(target)\n            return {**result, \"status\": \"REBUILT\"}\n\n    def _build_project_rag(\n        self,\n        resolved: Sequence[str],\n        *,\n        target: Path,\n        metadata: dict[str, Any],\n        semantic: bool,\n    ) -> dict[str, Any]:\n        router = ModelRouter(profile=self.profile) if semantic else None\n        return ProjectRAGIndex(target).build(\n            resolved,\n            metadata=metadata,\n            router=router,\n            semantic=semantic,\n        )\n'''
    text = replace_method(text, "index_project_rag", replacement)
    compile(text, str(PROD), "exec")
    PROD.write_text(text, encoding="utf-8")


def patch_custom_generator() -> None:
    text = CUSTOM.read_text(encoding="utf-8")
    old = '''        self.router.bind_agent_workspace(root.parent, require_fresh_evidence=True)\n        from .production_tools import ProductionToolService\n\n        live_manifest = ProjectIndex(root, policy=self.policy).manifest_receipt()\n        ProductionToolService(\n            workspace_root=root.parent,\n            profile=self.router.profile,\n        ).index_project_rag(\n            [root.name],\n            metadata={\n                \"minecraft_version\": minecraft_version,\n                \"loader\": loader,\n                \"mapping_namespace\": _mapping_namespace(mappings),\n                \"java_version\": \"17\",\n                \"license\": \"project-local\",\n                \"source_commit\": str(live_manifest[\"sha256\"]),\n            },\n            semantic=False,\n        )\n'''
    new = '''        bind_workspace = getattr(self.router, \"bind_agent_workspace\", None)\n        if callable(bind_workspace):\n            bind_workspace(root.parent, require_fresh_evidence=True)\n        from .production_tools import ProductionToolService\n\n        try:\n            live_manifest = index.manifest_receipt()\n        except ValueError as exc:\n            if not _is_stale_project_index_error(exc):\n                raise\n            index = ProjectIndex(root, policy=self.policy)\n            self._cached_index = index\n            self._cached_root = root\n            live_manifest = index.manifest_receipt()\n\n        ProductionToolService(\n            workspace_root=root.parent,\n            profile=str(getattr(self.router, \"profile\", \"t4_local\")),\n        ).ensure_project_rag(\n            [root.name],\n            metadata={\n                \"minecraft_version\": minecraft_version,\n                \"loader\": loader,\n                \"mapping_namespace\": _mapping_namespace(mappings),\n                \"java_version\": \"17\",\n                \"license\": \"project-local\",\n                \"source_commit\": str(live_manifest[\"sha256\"]),\n            },\n            semantic=False,\n        )\n'''
    text = replace_once(text, old, new, "custom generator live RAG")
    compile(text, str(CUSTOM), "exec")
    CUSTOM.write_text(text, encoding="utf-8")


def patch_repair_engine() -> None:
    text = REPAIR.read_text(encoding="utf-8")
    old = '''        root, project_index = active\n        self.router.bind_agent_workspace(root.parent, require_fresh_evidence=True)\n        from .production_tools import ProductionToolService\n\n        manifest = project_index.manifest_receipt()\n        ProductionToolService(\n            workspace_root=root.parent,\n            profile=self.router.profile,\n        ).index_project_rag(\n            [root.name],\n            metadata=_repair_rag_metadata(manifest),\n            semantic=False,\n        )\n'''
    new = '''        root, project_index = active\n        bind_workspace = getattr(self.router, \"bind_agent_workspace\", None)\n        if callable(bind_workspace):\n            bind_workspace(root.parent, require_fresh_evidence=True)\n        from .production_tools import ProductionToolService\n\n        manifest = project_index.manifest_receipt()\n        ProductionToolService(\n            workspace_root=root.parent,\n            profile=str(getattr(self.router, \"profile\", \"t4_local\")),\n        ).ensure_project_rag(\n            [root.name],\n            metadata=_repair_rag_metadata(manifest),\n            semantic=False,\n        )\n'''
    text = replace_once(text, old, new, "repair live RAG")
    compile(text, str(REPAIR), "exec")
    REPAIR.write_text(text, encoding="utf-8")


def patch_bootstrap() -> None:
    text = BOOT.read_text(encoding="utf-8")
    text = text.replace("        production_tools,\n", "", 1)
    text = text.replace(
        "    from .production_tool_parallel_contract import install as install_production_tool_parallel_safety\n",
        "",
        1,
    )
    text = text.replace("    install_production_tool_parallel_safety(production_tools)\n", "", 1)
    if "production_tool_parallel_contract" in text or "install_production_tool_parallel_safety" in text:
        raise SystemExit("production tool wrapper still referenced by runtime bootstrap")
    compile(text, str(BOOT), "exec")
    BOOT.write_text(text, encoding="utf-8")


def write_tests() -> None:
    NEW_TEST.write_text(
        '''from __future__ import annotations\n\nimport threading\nimport time\nfrom pathlib import Path\n\nimport pytest\n\nfrom minecraft_mod_ai import production_tools\nfrom minecraft_mod_ai.production_tools import ProductionToolService\nfrom minecraft_mod_ai.rag_index import ProjectRAGIndex\n\n\nMETA = {\n    \"minecraft_version\": \"1.20.1\",\n    \"loader\": \"fabric\",\n    \"mapping_namespace\": \"yarn\",\n    \"java_version\": \"17\",\n    \"license\": \"project-local\",\n    \"source_commit\": \"one\",\n}\n\n\ndef _project(tmp_path: Path) -> Path:\n    root = tmp_path / \"project\"\n    source = root / \"src/main/java/example/X.java\"\n    source.parent.mkdir(parents=True)\n    source.write_text(\"class X {}\", encoding=\"utf-8\")\n    return root\n\n\ndef test_rag_only_service_does_not_eagerly_construct_runtime_bridges(monkeypatch, tmp_path: Path) -> None:\n    calls = {\"runtime\": 0, \"mineflayer\": 0}\n\n    class Runtime:\n        def __init__(self, root):\n            calls[\"runtime\"] += 1\n\n    class Mineflayer:\n        def __init__(self):\n            calls[\"mineflayer\"] += 1\n\n    monkeypatch.setattr(production_tools, \"MinecraftRuntimeManager\", Runtime)\n    monkeypatch.setattr(production_tools, \"MineflayerBridge\", Mineflayer)\n    service = ProductionToolService(workspace_root=tmp_path, profile=\"test\")\n    assert calls == {\"runtime\": 0, \"mineflayer\": 0}\n    _ = service.runtime\n    assert calls == {\"runtime\": 1, \"mineflayer\": 0}\n    _ = service.mineflayer\n    assert calls == {\"runtime\": 1, \"mineflayer\": 1}\n\n\ndef test_ensure_project_rag_skips_same_manifest_and_rebuilds_changed_manifest(tmp_path: Path) -> None:\n    _project(tmp_path)\n    service = ProductionToolService(workspace_root=tmp_path, profile=\"test\")\n    first = service.ensure_project_rag([\"project\"], metadata=dict(META))\n    second = service.ensure_project_rag([\"project\"], metadata=dict(META))\n    changed = dict(META, source_commit=\"two\")\n    third = service.ensure_project_rag([\"project\"], metadata=changed)\n    assert first[\"status\"] == \"REBUILT\"\n    assert second[\"status\"] == \"CURRENT\"\n    assert third[\"status\"] == \"REBUILT\"\n    index = ProjectRAGIndex(tmp_path / \"rag/project-index.json\")\n    assert index.matches_build(metadata=changed, semantic=False)\n\n\ndef test_sequential_forced_refreshes_are_allowed(tmp_path: Path) -> None:\n    _project(tmp_path)\n    service = ProductionToolService(workspace_root=tmp_path, profile=\"test\")\n    first = service.index_project_rag([\"project\"], metadata=dict(META))\n    second = service.index_project_rag(\n        [\"project\"],\n        metadata=dict(META, source_commit=\"two\"),\n    )\n    assert first[\"index_path\"] == second[\"index_path\"]\n\n\ndef test_concurrent_same_target_forced_refresh_has_one_builder(monkeypatch, tmp_path: Path) -> None:\n    _project(tmp_path)\n    service = ProductionToolService(workspace_root=tmp_path, profile=\"test\")\n    original = ProjectRAGIndex.build\n    active = 0\n    max_active = 0\n    guard = threading.Lock()\n\n    def slow_build(self, *args, **kwargs):\n        nonlocal active, max_active\n        with guard:\n            active += 1\n            max_active = max(max_active, active)\n        try:\n            time.sleep(0.08)\n            return original(self, *args, **kwargs)\n        finally:\n            with guard:\n                active -= 1\n\n    monkeypatch.setattr(ProjectRAGIndex, \"build\", slow_build)\n    barrier = threading.Barrier(3)\n    outcomes = []\n\n    def worker() -> None:\n        barrier.wait()\n        try:\n            outcomes.append(service.index_project_rag([\"project\"], metadata=dict(META)))\n        except BaseException as exc:\n            outcomes.append(exc)\n\n    threads = [threading.Thread(target=worker) for _ in range(2)]\n    for thread in threads:\n        thread.start()\n    barrier.wait()\n    for thread in threads:\n        thread.join(timeout=5)\n        assert not thread.is_alive()\n    assert max_active == 1\n    assert sum(isinstance(item, dict) for item in outcomes) == 1\n    assert sum(isinstance(item, FileExistsError) for item in outcomes) == 1\n\n\ndef test_parallel_contract_module_was_removed() -> None:\n    assert not Path(\"minecraft_mod_ai/production_tool_parallel_contract.py\").exists()\n    bootstrap = Path(\"minecraft_mod_ai/runtime_bootstrap.py\").read_text(encoding=\"utf-8\")\n    assert \"production_tool_parallel_contract\" not in bootstrap\n''',
        encoding="utf-8",
    )


def main() -> None:
    patch_rag_index()
    patch_production_tools()
    patch_custom_generator()
    patch_repair_engine()
    patch_bootstrap()
    WRAPPER.unlink()
    OLD_TEST.unlink()
    write_tests()
    for path in (RAG, PROD, CUSTOM, REPAIR, BOOT, NEW_TEST):
        if path.suffix == ".py":
            compile(path.read_text(encoding="utf-8"), str(path), "exec")


if __name__ == "__main__":
    main()
