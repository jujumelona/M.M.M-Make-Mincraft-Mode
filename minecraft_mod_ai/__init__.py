"""M.M.M Make Mincraft Mode: scalable multimodal Minecraft mod production tools.

Import package submodules without eagerly initializing the full production
orchestrator, model router, discovery stack or optional runtime integrations.
Public symbols remain available through lazy attribute resolution.
"""

from importlib import import_module


__all__ = [
    "AssetRequest", "AuthoredPlan", "BossSpec", "CompleteChatReply",
    "CompleteExecutionOptions", "CompleteGameDesignPlanner", "CompleteModAISession",
    "CompletePipelineResult", "CompleteProductionError", "CompleteProductionOrchestrator",
    "CompleteProposal", "ContentSpec", "EcosystemDiscoveryClient", "ExistingProjectImportError",
    "ExistingProjectReport", "ExternalMCPRegistry", "ModDevelopmentMethod", "ModSpec",
    "ModelBackendError", "ModelConfigurationError", "ModelRegistry", "ModelRouter",
    "PlatformLock", "ProductionContractCompilation", "ProductionModule",
    "ProductionToolService", "ProjectIndex", "ProjectRAGIndex", "Proposal",
    "ScalableFabricProjectGenerator", "ScalePolicy", "ScalePolicyError",
    "assess_technology_compatibility", "build_technology_radar",
    "compile_production_contract", "evaluate_quality_contract", "inspect_existing_project_archive",
    "mod_development_method_catalog", "quality_contract_summary", "quality_unresolved",
    "resolve_mod_development_methods", "supported_minecraft_versions", "technology_research_routes",
]


__version__ = "0.8.0"

_PUBLIC_MODULES = {
    "assess_technology_compatibility": ".technology_radar",
    "AssetRequest": ".complete_spec",
    "AuthoredPlan": ".authored_plan",
    "BossSpec": ".spec",
    "build_technology_radar": ".technology_radar",
    "compile_production_contract": ".production_contract",
    "CompleteChatReply": ".api",
    "CompleteExecutionOptions": ".complete_orchestrator",
    "CompleteGameDesignPlanner": ".complete_planner",
    "CompleteModAISession": ".api",
    "CompletePipelineResult": ".complete_orchestrator",
    "CompleteProductionError": ".complete_orchestrator_support",
    "CompleteProductionOrchestrator": ".complete_orchestrator",
    "CompleteProposal": ".complete_spec",
    "ContentSpec": ".spec",
    "EcosystemDiscoveryClient": ".ecosystem_discovery",
    "evaluate_quality_contract": ".production_contract",
    "ExistingProjectImportError": ".importer",
    "ExistingProjectReport": ".importer",
    "ExternalMCPRegistry": ".external_mcp",
    "inspect_existing_project_archive": ".importer",
    "mod_development_method_catalog": ".mod_development_methods",
    "ModDevelopmentMethod": ".mod_development_methods",
    "ModelBackendError": ".model_adapters",
    "ModelConfigurationError": ".model_adapters",
    "ModelRegistry": ".model_registry",
    "ModelRouter": ".model_router",
    "ModSpec": ".spec",
    "PlatformLock": ".spec",
    "ProductionContractCompilation": ".production_contract",
    "ProductionModule": ".complete_spec",
    "ProductionToolService": ".production_tools",
    "ProjectIndex": ".project_index",
    "ProjectRAGIndex": ".rag_index",
    "Proposal": ".spec",
    "quality_contract_summary": ".production_contract",
    "quality_unresolved": ".production_contract",
    "resolve_mod_development_methods": ".mod_development_methods",
    "ScalableFabricProjectGenerator": ".scalable_generator",
    "ScalePolicy": ".scale_policy",
    "ScalePolicyError": ".scale_policy",
    "supported_minecraft_versions": ".api",
    "technology_research_routes": ".technology_radar",
}


def __getattr__(name: str):
    module = _PUBLIC_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
