from __future__ import annotations

import inspect
import json
import os
from functools import wraps
from typing import Any, Callable

from .platform_catalog import adapter_for_lock_values, adapter_for_target


def install(mcp_tools_module: Any, production_tools_module: Any) -> None:
    """Install the target-bound MCP surface once."""
    _install_core_tools(mcp_tools_module)
    _install_production_tools(production_tools_module)


def _call_supported(
    callable_obj: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
):
    """Forward only kwargs declared by a concrete adapter/test double."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return callable_obj(*args, **kwargs)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return callable_obj(*args, **kwargs)
    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return callable_obj(*args, **accepted)


def _required_adapter(module: Any, minecraft_version: str, loader: str = "fabric"):
    version = str(minecraft_version or "").strip()
    selected_loader = str(loader or "fabric").strip().lower()
    if not version:
        raise module.SpecValidationError(
            "This standalone research tool requires a Minecraft target. "
            "Run/resolve a plan first or configure MMM_MCP_MINECRAFT_VERSION."
        )
    try:
        return adapter_for_target(version, selected_loader)
    except ValueError as exc:
        raise module.SpecValidationError(str(exc)) from exc


def _service_adapter(
    self: Any,
    module: Any,
    minecraft_version: str = "",
    loader: str = "fabric",
):
    active = getattr(self, "_mmm_last_platform_adapter", None)
    if active is not None:
        return active
    configured = os.environ.get("MMM_MCP_MINECRAFT_VERSION", "").strip()
    configured_loader = os.environ.get("MMM_MCP_LOADER", loader).strip() or loader
    if configured:
        return _required_adapter(module, configured, configured_loader)
    return _required_adapter(module, minecraft_version, loader)


def _install_core_tools(module: Any) -> None:
    cls = module.MMMToolService

    def discover_ecosystem_resources(
        self: Any,
        provider: str,
        query: str,
        cursor: str = "",
        limit: int = 20,
        target_profile: str = "minecraft_mod",
        minecraft_version: str = "",
        loader: str = "fabric",
    ) -> dict[str, Any]:
        adapter = _service_adapter(self, module, minecraft_version, loader)
        client = self.discovery_client_factory()
        return _call_supported(
            client.search,
            provider,
            query,
            cursor=cursor,
            limit=limit,
            minecraft_version=adapter.minecraft_version,
            loader=adapter.loader,
            target_profile=target_profile,
        )

    def inspect_modrinth_project(
        self: Any,
        project_id: str,
        minecraft_version: str = "",
        loader: str = "fabric",
    ) -> dict[str, Any]:
        adapter = _service_adapter(self, module, minecraft_version, loader)
        client = self.discovery_client_factory()
        return _call_supported(
            client.inspect_modrinth_project,
            project_id,
            minecraft_version=adapter.minecraft_version,
            loader=adapter.loader,
        )

    def build_technology_radar(
        self: Any,
        prompt: str,
        research_brief: dict[str, Any] | None = None,
        cursor: str = "",
        page_size: int = 50,
        minecraft_version: str = "",
        loader: str = "fabric",
    ) -> dict[str, Any]:
        adapter = _service_adapter(self, module, minecraft_version, loader)
        return module.create_technology_radar(
            prompt,
            research_brief,
            target={
                "edition": adapter.edition,
                "minecraft_version": adapter.minecraft_version,
                "loader": adapter.loader,
                "mappings": adapter.yarn_mappings,
                "java_version": adapter.java_version,
                "fabric_loader": adapter.fabric_loader,
                "fabric_api": adapter.fabric_api,
            },
            cursor=cursor,
            page_size=page_size,
        )

    def search_project_rag(
        self: Any,
        query: str,
        minecraft_version: str = "",
        limit: int = 6,
        loader: str = "fabric",
    ) -> dict[str, Any]:
        adapter = _service_adapter(self, module, minecraft_version, loader)
        if type(limit) is not int or limit < 1:
            raise module.SpecValidationError("limit must be a positive integer.")
        sources = module.AuthoritativeEvidenceRetriever().search(
            query,
            minecraft_version=adapter.minecraft_version,
            limit=limit,
        )
        return {
            "schema_version": "mmm/rag-result-v2",
            "query": query,
            "target": {
                "minecraft_version": adapter.minecraft_version,
                "loader": adapter.loader,
                "mappings": adapter.yarn_mappings,
            },
            "sources": [source.__dict__ for source in sources],
        }

    discover_ecosystem_resources._mmm_platform_bound = True
    discover_ecosystem_resources._mmm_signature_compatible_target_forwarding = True
    inspect_modrinth_project._mmm_platform_bound = True
    inspect_modrinth_project._mmm_signature_compatible_target_forwarding = True
    build_technology_radar._mmm_platform_bound = True
    search_project_rag._mmm_platform_bound = True
    cls.discover_ecosystem_resources = discover_ecosystem_resources
    cls.inspect_modrinth_project = inspect_modrinth_project
    cls.build_technology_radar = build_technology_radar
    cls.search_project_rag = search_project_rag

    original_plan_complete = cls.plan_complete_game

    @wraps(original_plan_complete)
    def plan_complete_game(
        self: Any,
        prompt: str,
        media_paths=(),
        existing_input_sha256: str = "",
        minecraft_version: str = "",
        loader: str = "fabric",
        existing_minecraft_version: str = "",
        existing_loader: str = "",
    ) -> dict[str, Any]:
        router = self.router_factory()
        if minecraft_version:
            adapter = _required_adapter(module, minecraft_version, loader)
            router._mmm_requested_minecraft_version = adapter.minecraft_version
            router._mmm_requested_loader = adapter.loader
        if existing_minecraft_version:
            adapter = _required_adapter(
                module,
                existing_minecraft_version,
                existing_loader or loader,
            )
            router._mmm_existing_minecraft_version = adapter.minecraft_version
            router._mmm_existing_loader = adapter.loader
        proposal = module.CompleteGameDesignPlanner(router).plan(
            prompt,
            media_paths=self._scoped_media_paths(media_paths),
            existing_input_sha256=existing_input_sha256,
        )
        adapter = adapter_for_lock_values(proposal.base_proposal.spec.platform)
        self._mmm_last_platform_adapter = adapter
        os.environ["MMM_MCP_MINECRAFT_VERSION"] = adapter.minecraft_version
        os.environ["MMM_MCP_LOADER"] = adapter.loader
        proposal_ref = self._store_complete_proposal(proposal)
        return {
            "schema_version": "mmm/complete-plan-result-v4",
            "profile": self.profile,
            "message": module.render_complete_plan(
                requested_prompt=proposal.requested_prompt,
                game_design=proposal.game_design,
                modules=proposal.modules,
                acceptance_tests=proposal.acceptance_tests,
            ),
            "proposal_ref": proposal_ref,
            "approval_hash": proposal.calculate_hash(),
            "counts": self._complete_proposal_counts(proposal),
            "detail_tool": "read_complete_plan_section",
            "platform": proposal.base_proposal.spec.platform.__dict__,
        }

    plan_complete_game._mmm_platform_bound = True
    cls.plan_complete_game = plan_complete_game

    original_revise = cls.revise_complete_plan

    @wraps(original_revise)
    def revise_complete_plan(
        self: Any,
        original_prompt: str,
        revision: str,
        media_paths=(),
        existing_input_sha256: str = "",
        minecraft_version: str = "",
        loader: str = "fabric",
        existing_minecraft_version: str = "",
        existing_loader: str = "",
    ) -> dict[str, Any]:
        try:
            merged = module.merge_design_brief(original_prompt, revision)
        except ValueError as exc:
            raise module.SpecValidationError("revision must not be empty.") from exc
        active = getattr(self, "_mmm_last_platform_adapter", None)
        return _call_supported(
            self.plan_complete_game,
            merged,
            media_paths=media_paths,
            existing_input_sha256=existing_input_sha256,
            minecraft_version=(
                minecraft_version
                or (active.minecraft_version if active is not None else "")
            ),
            loader=(loader or (active.loader if active is not None else "fabric")),
            existing_minecraft_version=existing_minecraft_version,
            existing_loader=existing_loader,
        )

    revise_complete_plan._mmm_platform_bound = True
    revise_complete_plan._mmm_signature_compatible_target_forwarding = True
    cls.revise_complete_plan = revise_complete_plan


def _install_production_tools(module: Any) -> None:
    cls = module.ProductionToolService
    original_prepare = cls.runtime_prepare_instance
    if getattr(original_prepare, "_mmm_platform_bound", False):
        return

    @wraps(original_prepare)
    def runtime_prepare_instance(
        self: Any,
        *,
        instance_name: str,
        mod_jar: str,
        server_launcher: str,
        eula_accepted: bool,
        proposal: dict[str, Any],
        approval_hash: str,
    ) -> dict[str, Any]:
        approved = self._approved(proposal, approval_hash)
        adapter = adapter_for_lock_values(approved.spec.platform)
        config_path = self.workspace_root / ".minecraft_ai" / "mcp-runtime-profile.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": "mmm/runtime-profiles-v1",
                    "profiles": {
                        "fabric_target_disposable": {
                            "minecraft_version": adapter.minecraft_version,
                            "loader": adapter.loader,
                            "platform_adapter": adapter.adapter_id,
                            "java_project_version": int(adapter.java_version),
                            "server_java_command": os.environ.get(
                                f"MMM_JAVA_{adapter.java_version}_CMD", "java"
                            ),
                            "server_memory_mb": 3072,
                            "server_launcher_relative": "runtime/fabric-server-launch.jar",
                            "client_command_env": "MMM_MINECRAFT_CLIENT_COMMAND_JSON",
                            "allowed_server_commands": [
                                "^list$",
                                "^stop$",
                                "^gametest runall$",
                                "^say [A-Za-z0-9 _.,!?-]{1,120}$",
                                "^tp testplayer -?[0-9]{1,7} -?[0-9]{1,7} -?[0-9]{1,7}$",
                                "^give testplayer [a-z0-9_.-]+:[a-z0-9_./-]+( [1-9][0-9]{0,3})?$",
                            ],
                            "startup_ready_patterns": [
                                "Done \\([0-9.]+s\\)! For help, type",
                                "For help, type \\\"help\\\"",
                            ],
                            "disposable_only": True,
                            "eula_must_be_explicitly_accepted": True,
                        }
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.runtime = module.MinecraftRuntimeManager(
            self.workspace_root,
            profile_name="fabric_target_disposable",
            config_path=config_path,
        )
        self._mmm_runtime_adapter = adapter
        os.environ["MMM_MINEFLAYER_MC_VERSION"] = adapter.minecraft_version
        self.mineflayer = module.MineflayerBridge()
        result = self.runtime.prepare_instance(
            instance_name,
            mod_jar=mod_jar,
            server_launcher=server_launcher,
            eula_accepted=eula_accepted,
        )
        return {
            **result,
            "platform_adapter": adapter.adapter_id,
            "minecraft_version": adapter.minecraft_version,
        }

    runtime_prepare_instance._mmm_platform_bound = True
    cls.runtime_prepare_instance = runtime_prepare_instance

    original_connect = cls.mineflayer_connect

    @wraps(original_connect)
    def mineflayer_connect(self: Any, *args: Any, **kwargs: Any):
        adapter = getattr(self, "_mmm_runtime_adapter", None)
        if adapter is None:
            raise module.SpecValidationError(
                "Prepare an approved target-bound runtime instance before Mineflayer connects."
            )
        os.environ["MMM_MINEFLAYER_MC_VERSION"] = adapter.minecraft_version
        result = original_connect(self, *args, **kwargs)
        if str(result.get("version", "")) != adapter.minecraft_version:
            raise module.SpecValidationError(
                f"Mineflayer connected to {result.get('version')!r}, expected "
                f"{adapter.minecraft_version!r}."
            )
        return result

    mineflayer_connect._mmm_platform_bound = True
    cls.mineflayer_connect = mineflayer_connect
