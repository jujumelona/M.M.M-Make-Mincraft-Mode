from __future__ import annotations
'Single planning owner for platform selection and live-target module lowering.'
from dataclasses import replace
from functools import wraps
from typing import Any, Mapping
from .complete_spec import CompleteProposal, ProductionModule
from .platform_catalog import PlatformAdapter
from .platform_resolver import resolve_platform, retarget_proposal
_LIVE_NON_SOURCE_KINDS = frozenset({'integration'})

def install(*, game_design_module: Any, complete_planner_module: Any) -> None:
    """Install the sole platform-selection wrapper and the execution lowering layer."""
    _install_central_target_choice(game_design_module)
    _install_live_module_lowering(complete_planner_module)

def _target_research_callback(research_brief: Mapping[str, Any]):
    """Create one host-owned target-scoped evidence callback for optimizer hypotheses."""

    def retrieve(adapter: PlatformAdapter) -> Mapping[str, Any]:
        from . import central_research, retrieval
        from .agentic_research_fusion import retrieve_target_agentic_evidence
        brief = {**dict(research_brief), '_mmm_platform_target': {'minecraft_version': adapter.minecraft_version, 'loader': adapter.loader, 'mappings': adapter.yarn_mappings}}
        return retrieve_target_agentic_evidence(brief, central_module=central_research, retrieve=retrieval.retrieve_official_evidence, minecraft_version=adapter.minecraft_version, loader=adapter.loader, mappings=adapter.yarn_mappings)
    return retrieve

def _research_failure(adapter: PlatformAdapter, exc: Exception) -> dict[str, Any]:
    return {'schema_version': 'mmm/central-evidence-graph-v1', 'target': {'minecraft_version': adapter.minecraft_version, 'loader': adapter.loader, 'mappings': adapter.yarn_mappings}, 'domains': [], 'unresolved_official_domains': ['target_research_unavailable'], 'authorization': 'none', 'retrieval_is_authority': False, 'status': 'unavailable', 'errors': [{'type': type(exc).__name__, 'message': str(exc)}]}

def _install_central_target_choice(module: Any) -> None:
    cls = module.GameDesignPlanner
    original = cls.plan
    if getattr(original, '_mmm_platform_selection_owner', False):
        return

    @wraps(original)
    def plan(self: Any, prompt: str, *, media_paths=()):
        design, proposal = original(self, prompt, media_paths=media_paths)
        router = self.router
        existing_version = getattr(router, '_mmm_existing_minecraft_version', None)
        existing_loader = getattr(router, '_mmm_existing_loader', None)
        requested_version = getattr(router, '_mmm_requested_minecraft_version', None)
        requested_loader = getattr(router, '_mmm_requested_loader', None)
        effective_prompt = str(prompt)
        if requested_version and str(requested_version) not in effective_prompt:
            effective_prompt += f'\n[HOST_TARGET_CONSTRAINT Minecraft {requested_version}]'
        if requested_loader and str(requested_loader).casefold() not in effective_prompt.casefold():
            effective_prompt += f'\n[HOST_LOADER_CONSTRAINT {requested_loader}]'
        research_brief = design.get('_research_brief')
        if not isinstance(research_brief, dict):
            research_brief = module.normalize_research_brief(prompt, design)
        target_research = _target_research_callback(research_brief)
        selection = resolve_platform(effective_prompt, design=design, existing_version=existing_version, existing_loader=existing_loader, target_research_fn=target_research)
        proposal = retarget_proposal(proposal, selection)
        proposal.validate()
        selection_dict = selection.to_dict()
        if selection.migration_requested and existing_version:
            selection_dict['migration_from'] = {'minecraft_version': str(existing_version), 'loader': str(existing_loader or 'unknown').strip().casefold()}
        target = dict(selection_dict['target'])
        bound_brief = {**research_brief, '_mmm_platform_target': target}
        platform_evidence: Mapping[str, Any] | None = None
        if selection.optimization is not None:
            deep = selection.optimization.evidence.deep_research
            if isinstance(deep, Mapping):
                platform_evidence = dict(deep)
        if platform_evidence is None:
            try:
                platform_evidence = dict(target_research(selection.adapter))
            except Exception as exc:
                platform_evidence = _research_failure(selection.adapter, exc)
        pre_design = design.get('_pre_design_research')
        if isinstance(pre_design, dict):
            deterministic = pre_design.get('deterministic')
            if isinstance(deterministic, dict):
                deterministic = {**deterministic, 'official_rag': dict(platform_evidence)}
            else:
                deterministic = {'official_rag': dict(platform_evidence)}
            pre_design = {**pre_design, 'research_brief': bound_brief, 'deterministic': deterministic}
        design = {**design, '_platform_selection': selection_dict, '_platform_evidence': dict(platform_evidence), '_research_brief': bound_brief}
        if isinstance(pre_design, dict):
            design['_pre_design_research'] = pre_design
        return (design, proposal)
    plan._mmm_platform_selection_owner = True
    cls.plan = plan

def _bootstrap_content_payload(result: CompleteProposal) -> list[dict[str, Any]]:
    return [{'content_id': content.content_id, 'kind': content.kind.value, 'display_name_en': content.display_name_en, 'display_name_ko': content.display_name_ko, 'color': content.color, 'recipe': content.recipe} for content in result.base_proposal.spec.contents]

def _bootstrap_boss_payload(result: CompleteProposal) -> dict[str, Any] | None:
    boss = result.base_proposal.spec.boss
    if boss is None:
        return None
    return {'entity_id': boss.entity_id, 'display_name_en': boss.display_name_en, 'display_name_ko': boss.display_name_ko, 'max_health': boss.max_health, 'attack_damage': boss.attack_damage, 'movement_speed': boss.movement_speed, 'scale': boss.scale, 'primary_color': boss.primary_color, 'secondary_color': boss.secondary_color, 'model_kind': boss.model_kind}

def _input_acceptance_tests(result: CompleteProposal) -> tuple[str, ...]:
    contract = result.game_design.get('_production_contract')
    if isinstance(contract, dict):
        catalog = contract.get('acceptance_catalog')
        if isinstance(catalog, list):
            values = tuple((str(item.get('statement', '')).strip() for item in catalog if isinstance(item, dict) and item.get('origin') == 'input' and str(item.get('statement', '')).strip()))
            if values:
                return values
    return tuple(result.acceptance_tests)

def _recompile_live_contract(module: Any, result: CompleteProposal, *, game_design: dict[str, Any], lowered: tuple[ProductionModule, ...]) -> tuple[dict[str, Any], tuple[str, ...]]:
    contract_design = {key: value for key, value in game_design.items() if not str(key).startswith('_')}
    research_brief = game_design.get('_research_brief')
    evidence_plan = game_design.get('_evidence_first_plan')
    compiled = module.compile_production_contract(requested_prompt=result.requested_prompt, game_design=contract_design, research_brief=research_brief if isinstance(research_brief, dict) else None, modules=lowered, assets=result.assets, acceptance_tests=_input_acceptance_tests(result), evidence_plan=evidence_plan if isinstance(evidence_plan, Mapping) else None)
    return ({**game_design, '_production_contract': compiled.contract}, tuple(compiled.acceptance_tests))

def _carrier_index(modules: list[ProductionModule]) -> int | None:
    custom = next((index for index, item in enumerate(modules) if item.kind == 'custom_java'), None)
    if custom is not None:
        return custom
    return None

def _as_custom_carrier(item: ProductionModule, *, extra_config: dict[str, Any]) -> ProductionModule:
    config = {**item.config, 'implementation': 'custom', 'requested_kind': item.config.get('requested_kind', item.kind), 'platform_generation': 'central_ai_live_target', **extra_config}
    return ProductionModule(module_id=item.module_id, kind='custom_java', config=config, depends_on=item.depends_on, required_gates=item.required_gates)

def _validated_retain_only(result: CompleteProposal) -> bool:
    if result.modules:
        return False
    plan = result.game_design.get('_evidence_first_plan')
    if not isinstance(plan, Mapping):
        return False
    from .evidence_first_planning import validate_evidence_first_plan
    validate_evidence_first_plan(plan, prompt=result.requested_prompt)
    return bool(plan.get('verified_provides')) and not plan.get('gap_catalog') and not plan.get('tasks')

def _install_live_module_lowering(module: Any) -> None:
    cls = module.CompleteGameDesignPlanner
    original = cls._plan_in_session
    if getattr(original, '_mmm_live_ai_module_lowering', False):
        return

    @wraps(original)
    def plan_in_session(self: Any, prompt: str, *, media_paths=(), existing_input_sha256=''):
        result = original(self, prompt, media_paths=media_paths, existing_input_sha256=existing_input_sha256)
        selection = result.game_design.get('_platform_selection', {})
        target = selection.get('target', {}) if isinstance(selection, dict) else {}
        if not isinstance(target, dict) or target.get('source_api_family') != 'fabric_live_ai':
            return result
        migration_requested = bool(isinstance(selection, dict) and selection.get('migration_requested'))
        migration_from = selection.get('migration_from') if isinstance(selection, dict) else None
        if not migration_requested and _validated_retain_only(result):
            # The validated existing project already supplies every requirement.
            # Base ModSpec content is descriptive input here, not permission to
            # invent an otherwise unnecessary source-generation carrier.
            return result
        bootstrap_contents = _bootstrap_content_payload(result)
        bootstrap_boss = _bootstrap_boss_payload(result)
        lowered: list[ProductionModule] = []
        changed = False
        bootstrap_bound = False
        for item in result.modules:
            uses_base_content = item.kind == 'integration' and isinstance(item.config.get('uses_base_content'), list)
            if uses_base_content:
                lowered.append(ProductionModule(module_id=item.module_id, kind='custom_java', config={**item.config, 'implementation': 'custom', 'requested_kind': 'bootstrap_content', 'platform_generation': 'central_ai_live_target', 'bootstrap_contents': bootstrap_contents, 'bootstrap_boss': bootstrap_boss, 'require_exact_base_spec': True}, depends_on=item.depends_on, required_gates=item.required_gates))
                bootstrap_bound = True
                changed = True
                continue
            if item.kind in _LIVE_NON_SOURCE_KINDS or item.kind == 'custom_java':
                lowered.append(item)
                continue
            lowered.append(_as_custom_carrier(item, extra_config={}))
            changed = True
        if (bootstrap_contents or bootstrap_boss) and (not bootstrap_bound):
            target_index = _carrier_index(lowered)
            if target_index is None:
                raise module.SpecValidationError('Live target has base ModSpec content but no production module that can carry it.')
            lowered[target_index] = _as_custom_carrier(lowered[target_index], extra_config={'bootstrap_contents': bootstrap_contents, 'bootstrap_boss': bootstrap_boss, 'require_exact_base_spec': True})
            changed = True
        if migration_requested:
            target_index = _carrier_index(lowered)
            if target_index is None:
                raise module.SpecValidationError('Version migration requires at least one source-generation module.')
            lowered[target_index] = _as_custom_carrier(lowered[target_index], extra_config={'platform_migration': {'from': dict(migration_from) if isinstance(migration_from, dict) else {'minecraft_version': 'existing-project', 'loader': 'unknown'}, 'to': dict(target), 'requirements': ['migrate build and loader metadata to the approved target', 'port API usage using target-scoped official evidence', 'preserve requested behavior and existing project content', 'finish only after language, build and game tests pass']}})
            changed = True
        if not changed:
            return result
        lowered_tuple = tuple(lowered)
        game_design = {**result.game_design, '_platform_execution': {'mode': 'central_ai_compile_repair', 'source_api_family': 'fabric_live_ai', 'semantic_kinds_preserved_in': 'module.config.requested_kind', 'base_modspec_bound_to_live_generation': bool(bootstrap_contents or bootstrap_boss), 'migration_bound_to_live_generation': migration_requested, 'production_contract_rebound_after_lowering': True}}
        game_design, acceptance_tests = _recompile_live_contract(module, result, game_design=game_design, lowered=lowered_tuple)
        updated: CompleteProposal = replace(result, game_design=game_design, modules=lowered_tuple, acceptance_tests=acceptance_tests, approval_hash='').with_hash()
        updated.validate(policy=getattr(self, 'policy', None))
        return updated
    plan_in_session._mmm_live_ai_module_lowering = True
    cls._plan_in_session = plan_in_session
