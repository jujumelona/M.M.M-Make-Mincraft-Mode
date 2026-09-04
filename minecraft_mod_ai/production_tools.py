from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from functools import cached_property
from pathlib import Path
from typing import Any

from .blockbench_client import BlockbenchMCPClient, allowed_blockbench_operations
from .geckolib_generator import generate_geckolib_entity_assets
from .java_lsp_trace import TracedJavaLanguageService
from .model_router import ModelRouter
from .model_smoke import run_model_smoke
from .rag_index import ProjectRAGIndex
from .spec import Proposal, ProposalStatus, SpecValidationError
from .system_pack_generator import generate_system_pack, supported_system_packs
from .training import TrainingTraceStore


class ProductionToolService:
    """Additional production tools separated from the core proposal pipeline."""

    def __init__(self, *, workspace_root: str | Path='mmm-output', profile: str='t4_local') -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.profile = profile

    @cached_property
    def runtime(self):
        """Create the Minecraft runtime manager only when a runtime tool is used."""
        from .runtime_manager import MinecraftRuntimeManager
        return MinecraftRuntimeManager(self.workspace_root)

    @cached_property
    def mineflayer(self):
        """Start Mineflayer bridge ownership only when a bot tool is requested."""
        from .mineflayer_bridge import MineflayerBridge
        return MineflayerBridge()

    @cached_property
    def java(self) -> TracedJavaLanguageService:
        """Keep one initialized, root-cause-visible JDT LS owner per service."""
        return TracedJavaLanguageService()

    @cached_property
    def model_router(self) -> ModelRouter:
        """Reuse one lazy model owner across semantic/reranked RAG calls."""
        return ModelRouter(profile=self.profile)

    def close(self) -> None:
        """Release lazily-created persistent subprocess owners without creating new ones."""
        java = self.__dict__.get("java")
        if java is not None:
            java.close()
            self.__dict__.pop("java", None)

    def index_project_rag(self, roots: Sequence[str], *, index_path: str='rag/project-index.json', metadata: dict[str, Any], semantic: bool=False) -> dict[str, Any]:
        resolved = [str(self._existing_path(root)) for root in roots]
        target = self._replaceable_file(index_path)
        repair_like = (
            bool(metadata.get('source_commit'))
            and str(metadata.get('license', '')) == 'project-local'
        )
        global_cpu_dense = os.environ.get('MMM_RAG_ENABLE_CPU_DENSE', '').strip() == '1'
        eager_repair_semantic = os.environ.get('MMM_RAG_EAGER_REPAIR_SEMANTIC', '').strip() == '1'
        effective_semantic = bool(
            semantic
            and global_cpu_dense
            and (not repair_like or eager_repair_semantic)
        )
        router = self.model_router if effective_semantic else None
        return ProjectRAGIndex(target).build(
            resolved,
            metadata=metadata,
            router=router,
            semantic=effective_semantic,
        )

    def search_code_rag(self, query: str, *, index_path: str='rag/project-index.json', limit: int=8, semantic: bool=False, rerank: bool=False, required_metadata: dict[str, Any] | None=None) -> dict[str, Any]:
        target = self._resolve(index_path, allow_root=True)
        if target.is_dir():
            canonical = self._resolve('rag/project-index.json', allow_root=True)
            target = canonical if canonical.is_file() else target
        elif not target.exists():
            canonical = self._resolve('rag/project-index.json', allow_root=True)
            if canonical.is_file():
                target = canonical
        router = self.model_router if semantic or rerank else None
        result = ProjectRAGIndex(target).search_with_receipt(query, limit=limit, router=router, semantic=semantic, rerank=rerank, required_metadata=required_metadata)
        return {'schema_version': 'mmm/code-rag-result-v1', 'query': query, 'hits': [asdict(hit) for hit in result.hits], 'receipt': asdict(result.receipt)}

    def read_reuse_source(
        self,
        project_root: str,
        path: str,
        *,
        offset_bytes: int = 0,
        limit_bytes: int = 16 * 1024,
    ) -> dict[str, Any]:
        """Read one host-materialized, manifest-authorized donor source slice."""
        if type(offset_bytes) is not int or offset_bytes < 0:
            raise SpecValidationError("offset_bytes must be a non-negative integer.")
        if type(limit_bytes) is not int or not 1 <= limit_bytes <= 32 * 1024:
            raise SpecValidationError("limit_bytes must be between 1 and 32768.")
        project = self._existing_dir(project_root)
        donor_root = (project / ".minecraft_ai" / "reuse" / "donors").resolve()
        candidate = Path(path).expanduser()
        target = candidate.resolve() if candidate.is_absolute() else (project / candidate).resolve()
        try:
            relative = target.relative_to(donor_root)
        except ValueError as exc:
            raise SpecValidationError("Reuse source path escaped the approved donor root.") from exc
        if len(relative.parts) < 2 or not target.is_file() or target.is_symlink():
            raise FileNotFoundError(target)
        donor_dir = donor_root / relative.parts[0]
        manifest_path = donor_dir / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(files, list):
            raise SpecValidationError("Reuse donor manifest has no authorized file list.")
        authorized = None
        for item in files:
            if not isinstance(item, dict):
                continue
            manifest_file = Path(str(item.get("path") or "")).expanduser()
            manifest_target = manifest_file.resolve() if manifest_file.is_absolute() else (donor_dir / manifest_file).resolve()
            if manifest_target == target:
                authorized = item
                break
        if authorized is None:
            raise SpecValidationError("Reuse source file is not authorized by the donor manifest.")
        raw = target.read_bytes()
        actual_sha256 = "sha256:" + hashlib.sha256(raw).hexdigest()
        expected_sha256 = str(authorized.get("sha256") or "")
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise SpecValidationError("Reuse source file no longer matches its pinned manifest hash.")
        start = min(offset_bytes, len(raw))
        end = min(len(raw), start + limit_bytes)
        chunk = raw[start:end]
        return {
            "schema_version": "mmm/reuse-source-read-v1",
            "repository": manifest.get("repository"),
            "commit_sha": manifest.get("commit_sha"),
            "license_id": manifest.get("license_id"),
            "capability": manifest.get("capability"),
            "path": str(target),
            "sha256": actual_sha256,
            "size_bytes": len(raw),
            "offset_bytes": start,
            "next_offset_bytes": end if end < len(raw) else None,
            "eof": end >= len(raw),
            "content": chunk.decode("utf-8", errors="replace"),
        }

    def java_diagnostics(self, project_root: str, relative_files: list[str] | None=None, timeout_seconds: int=60) -> dict[str, Any]:
        root = self._existing_dir(project_root)
        return self.java.diagnostics(root, relative_files=relative_files, timeout_seconds=timeout_seconds)

    def java_workspace_symbols(self, project_root: str, query: str, timeout_seconds: int=60) -> dict[str, Any]:
        return self.java.workspace_symbols(self._existing_dir(project_root), query, timeout_seconds=timeout_seconds)

    def blockbench_list_tools(self, timeout_seconds: int=60) -> dict[str, Any]:
        client = BlockbenchMCPClient(timeout_seconds=timeout_seconds)
        try:
            return {'schema_version': 'mmm/blockbench-tools-v1', 'reviewed_allowlist': list(allowed_blockbench_operations()), 'available': client.list_tools()}
        finally:
            client.close()

    def blockbench_execute(self, operation: str, arguments: dict[str, Any], timeout_seconds: int=60) -> dict[str, Any]:
        client = BlockbenchMCPClient(timeout_seconds=timeout_seconds)
        try:
            return client.call(operation, arguments)
        finally:
            client.close()

    def generate_geckolib_entity(self, *, project_root: str, mod_id: str, package_name: str, entity_id: str, proposal: dict[str, Any], approval_hash: str, geckolib_version: str='4.8.2') -> dict[str, Any]:
        self._approved(proposal, approval_hash)
        return generate_geckolib_entity_assets(project_root=self._existing_dir(project_root), mod_id=mod_id, package_name=package_name, entity_id=entity_id, geckolib_version=geckolib_version)

    def generate_system_plugin(self, *, project_root: str, pack_id: str, mod_id: str, package_name: str, config: dict[str, Any], proposal: dict[str, Any], approval_hash: str) -> dict[str, Any]:
        self._approved(proposal, approval_hash)
        return generate_system_pack(project_root=self._existing_dir(project_root), pack_id=pack_id, mod_id=mod_id, package_name=package_name, config=config)

    def runtime_prepare_instance(self, *, instance_name: str, mod_jar: str, server_launcher: str, eula_accepted: bool, proposal: dict[str, Any], approval_hash: str) -> dict[str, Any]:
        self._approved(proposal, approval_hash)
        return self.runtime.prepare_instance(instance_name, mod_jar=mod_jar, server_launcher=server_launcher, eula_accepted=eula_accepted)

    def runtime_start_server(self, timeout_seconds: int=180) -> dict[str, Any]:
        return self.runtime.start_server(timeout_seconds=timeout_seconds)

    def runtime_start_client(self) -> dict[str, Any]:
        return self.runtime.start_client()

    def runtime_send_command(self, command: str) -> dict[str, Any]:
        return self.runtime.send_server_command(command)

    def runtime_logs(self, lines: int=120) -> dict[str, Any]:
        return self.runtime.tail_logs(lines)

    def runtime_register_screenshot(self, screenshot_path: str) -> dict[str, Any]:
        return self.runtime.register_screenshot(screenshot_path)

    def runtime_status(self) -> dict[str, Any]:
        return self.runtime.status()

    def runtime_stop(self, cleanup: bool=False) -> dict[str, Any]:
        if cleanup:
            return self.runtime.cleanup()
        self.runtime.stop_client()
        return self.runtime.stop_server()

    def mineflayer_connect(self, host: str='127.0.0.1', port: int=25565, username: str='MMMTestBot', timeout_seconds: float=180.0) -> dict[str, Any]:
        return self.mineflayer.call('connect', timeout_seconds=timeout_seconds, host=host, port=port, username=username)

    def mineflayer_status(self, timeout_seconds: float=180.0) -> dict[str, Any]:
        return self.mineflayer.call('status', timeout_seconds=timeout_seconds)

    def mineflayer_walk_to(self, x: float, y: float, z: float, range: int=1, timeout_seconds: float=180.0) -> dict[str, Any]:
        return self.mineflayer.call('walk_to', timeout_seconds=timeout_seconds, x=x, y=y, z=z, range=range)

    def mineflayer_interact_block(self, x: int, y: int, z: int, timeout_seconds: float=180.0) -> dict[str, Any]:
        return self.mineflayer.call('interact_block', timeout_seconds=timeout_seconds, x=x, y=y, z=z)

    def mineflayer_inventory(self, timeout_seconds: float=180.0) -> dict[str, Any]:
        return self.mineflayer.call('inventory', timeout_seconds=timeout_seconds)

    def mineflayer_disconnect(self, timeout_seconds: float=30.0) -> dict[str, Any]:
        return self.mineflayer.call('disconnect', timeout_seconds=timeout_seconds)

    def run_model_smoke(self, role: str, output_dir: str='model-smoke', media_path: str | None=None) -> dict[str, Any]:
        target = self._resolve(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        return run_model_smoke(role=role, profile=self.profile, output_dir=target, media_path=self._existing_file(media_path) if media_path else None)

    def record_training_trace(self, trace: dict[str, Any], store_path: str='training/traces') -> dict[str, Any]:
        return TrainingTraceStore(self._resolve(store_path)).record(trace)

    def export_training_dataset(self, store_path: str='training/traces', output_path: str='training/mmm-fabric-coder-1201.jsonl') -> dict[str, Any]:
        return TrainingTraceStore(self._resolve(store_path)).export_sft(self._resolve(output_path))

    @staticmethod
    def system_plugin_ids() -> dict[str, Any]:
        return {'schema_version': 'mmm/system-plugin-list-v1', 'plugins': list(supported_system_packs())}

    @staticmethod
    def _approved(proposal: dict[str, Any], approval_hash: str) -> Proposal:
        parsed = Proposal.from_dict(proposal)
        approved = parsed.approve(approval_hash)
        if approved.status is not ProposalStatus.APPROVED:
            raise SpecValidationError('Proposal approval failed.')
        return approved

    def _existing_path(self, value: str) -> Path:
        path = self._resolve(value)
        if not path.exists() or path.is_symlink():
            raise FileNotFoundError(path)
        return path

    def _existing_file(self, value: str | Path) -> Path:
        path = self._resolve(str(value))
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        return path

    def _existing_dir(self, value: str) -> Path:
        path = self._resolve(value, allow_root=True)
        if not path.is_dir() or path.is_symlink():
            raise FileNotFoundError(path)
        return path

    def _replaceable_file(self, value: str) -> Path:
        path = self._resolve(value)
        if path.exists() and (path.is_symlink() or not path.is_file()):
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _new_file(self, value: str) -> Path:
        path = self._resolve(value)
        if path.exists():
            raise FileExistsError(path)
        return path

    def _new_path(self, value: str) -> Path:
        path = self._resolve(value)
        if path.exists():
            raise FileExistsError(path)
        return path

    def _resolve(self, value: str, *, allow_root: bool = False) -> Path:
        candidate = Path(value).expanduser()
        path = candidate.resolve() if candidate.is_absolute() else (self.workspace_root / candidate).resolve()
        try:
            path.relative_to(self.workspace_root)
        except ValueError as exc:
            raise SpecValidationError('Tool path escaped the configured workspace.') from exc
        if not allow_root and path == self.workspace_root:
            raise SpecValidationError('Tools may not target the workspace root itself.')
        return path