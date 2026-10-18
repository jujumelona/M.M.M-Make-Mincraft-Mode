from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import asdict
from functools import cached_property
from pathlib import Path
from typing import Any

from .blockbench_client import BlockbenchMCPClient, allowed_blockbench_operations
from .geckolib_generator import generate_geckolib_entity_assets
from .java_lsp_trace import TracedJavaLanguageService as JavaLanguageService
from .model_router import ModelRouter
from .project_java_diagnostics import run_project_java_diagnostics
from .model_smoke import run_model_smoke
from .rag_index import ProjectRAGIndex
from .retrieval_cpu_budget_contract import _dense_opted_in
from .spec import Proposal, ProposalStatus, SpecValidationError
from .system_pack_generator import generate_system_pack, supported_system_packs


def _code_rag_target_or_workspace_result(
    service: Any, target: Path, query: str, limit: int, required_metadata: dict[str, Any] | None
) -> tuple[Path, dict[str, Any] | None]:
    canonical = service._resolve("rag/project-index.json", allow_root=True)
    if target == canonical:
        # Even if an old JSON index exists, prefer the current SQLite build.
        sqlite_index = service._resolve(".minecraft_ai/project-index.db", allow_root=True)
        if sqlite_index.is_file():
            return sqlite_index, None
        if not target.exists():
            from .agent_tool_runtime import _looks_like_bound_project
            from .workspace_code_search import search_workspace_source
            if _looks_like_bound_project(service.workspace_root):
                return target, search_workspace_source(
                    service.workspace_root, query, limit=limit, required_metadata=required_metadata
                )
    # Directory index discovery belongs to ProjectRAGIndex: it checks that
    # directory's SQLite index before any historical .json path.
    return target, None


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
    def java(self) -> JavaLanguageService:
        """Keep one initialized, root-cause-visible JDT LS owner per service."""
        return JavaLanguageService()

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
        global_cpu_dense = _dense_opted_in()
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
        target, workspace_result = _code_rag_target_or_workspace_result(
            self, target, query, limit, required_metadata
        )
        if workspace_result is not None:
            return workspace_result
        # An explicitly requested missing index must not silently search an
        # unrelated canonical path and claim its results as relevant.
        dense = _dense_opted_in()
        effective_semantic = bool(semantic and dense)
        effective_rerank = bool(rerank and dense)
        router = self.model_router if effective_semantic or effective_rerank else None
        result = ProjectRAGIndex(target).search_with_receipt(
            query,
            limit=limit,
            router=router,
            semantic=effective_semantic,
            rerank=effective_rerank,
            required_metadata=required_metadata,
        )
        return {'schema_version': 'mmm/code-rag-result-v1', 'query': query, 'hits': [asdict(hit) for hit in result.hits], 'receipt': asdict(result.receipt)}

    def java_diagnostics(self, project_root: str, relative_files: list[str] | None=None, timeout_seconds: int=60) -> dict[str, Any]:
        return run_project_java_diagnostics(
            self,
            project_root,
            relative_files=relative_files,
            timeout_seconds=timeout_seconds,
        )

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
