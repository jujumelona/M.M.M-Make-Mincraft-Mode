from __future__ import annotations

'Minecraft pixel-resource planning and real binary production.\n\nQwen plans *which* resources change and emits multiple final prompts per asset while it\nis resident.  Production later runs only those persisted jobs through the fixed\nFLUX.2 Klein 9B Q4 + PixelArt Redmond LoRA backend, then performs deterministic\nMinecraft PNG/path/reference/pack validation.  No placeholder counts as success.\n'
import hashlib
import json
import math
import os
import re
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

from .complete_spec import AssetRequest, CompleteProposal, ProductionModule
from .model_adapters.base import ModelConfigurationError
from .spec import SpecValidationError

FLUX_MODEL_ID = 'black-forest-labs/FLUX.2-klein-9B'
PIXEL_LORA_ID = 'artificialguybr/PIXELART-REDMOND-FLUXKLEIN9B'
PIXEL_LORA_WEIGHT = '[FLUX.2.Klein]PixelArt_Redmond.safetensors'
PIXEL_LORA_TRIGGER = 'Pixel Art, PixArFK'
QUANTIZATION = 'bnb_4bit_nf4'
PLAN_SCHEMA = 'mmm/resource-asset-generation-plan-v1'
_PROMPTS_PER_ASSET = 4
_PROMPT_BATCH = 12
_LEGACY_GENERATE_ASSETS: Any | None = None

class AssetProductionError(RuntimeError):
    pass

def attach_generation_plan(router: Any, proposal: CompleteProposal) -> CompleteProposal:
    """Generate and persist multiple executable image prompts for every planned asset."""
    if not proposal.assets:
        return proposal
    existing = proposal.game_design.get('_asset_generation_plan')
    if _valid_plan(existing, proposal.assets):
        return proposal
    planned: list[dict[str, Any]] = []
    assets = tuple(proposal.assets)
    for offset in range(0, len(assets), _PROMPT_BATCH):
        batch = assets[offset:offset + _PROMPT_BATCH]
        schema = _prompt_response_schema(batch)
        request = {'task': 'Write the actual image-generation prompts that will be executed later. Do not describe a future plan. Do not add resources that the user did not request.', 'original_request': proposal.requested_prompt, 'target': _target_receipt(proposal.game_design), 'fixed_image_backend': {'model_id': FLUX_MODEL_ID, 'quantization': QUANTIZATION, 'lora_model_id': PIXEL_LORA_ID, 'lora_trigger': PIXEL_LORA_TRIGGER}, 'assets': [{'asset_id': asset.asset_id, 'kind': asset.kind, 'current_prompt': asset.prompt, 'target_path': asset.target_path.replace('\\', '/'), 'width': asset.width, 'height': asset.height} for asset in batch], 'rules': [f'Return exactly {_PROMPTS_PER_ASSET} distinct prompt strings for every asset_id.', f'Every prompt must begin with {PIXEL_LORA_TRIGGER!r}.', 'Prompts are low-resolution Minecraft resource assets, not screenshots or voxel scenes.', 'Preserve transparent-background intent for isolated items/icons when appropriate.', 'Do not change target paths, dimensions, asset IDs, or the fixed model/LoRA.', 'Each prompt must be independently executable and specify the requested replacement appearance.']}
        text = router.generate_text('planner', [{'role': 'system', 'content': 'Return only the JSON object required by the schema. Produce concrete, diverse image prompts for the exact listed Minecraft resources.'}, {'role': 'user', 'content': json.dumps(request, ensure_ascii=False, sort_keys=True)}], response_format='json', response_schema=schema, tool_stage='planning')
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SpecValidationError('Asset prompt expansion returned invalid JSON.') from exc
        planned.extend(_validate_prompt_page(value, batch))
    plan = {'schema_version': PLAN_SCHEMA, 'model_id': FLUX_MODEL_ID, 'quantization': QUANTIZATION, 'lora_model_id': PIXEL_LORA_ID, 'lora_weight_name': PIXEL_LORA_WEIGHT, 'lora_trigger': PIXEL_LORA_TRIGGER, 'prompt_candidates_per_asset': _PROMPTS_PER_ASSET, 'assets': planned}
    game_design = {**proposal.game_design, '_asset_generation_plan': plan}
    updated = replace(proposal, game_design=game_design, approval_hash='').with_hash()
    updated.validate()
    return updated

def bind_reuse_plan(proposal: CompleteProposal) -> CompleteProposal:
    """Bind the approved plan once and assign each capability to one production owner."""
    if '_evidence_first_plan' in proposal.game_design:
        evidence_plan = proposal.game_design.get('_evidence_first_plan')
        if not isinstance(evidence_plan, Mapping):
            raise SpecValidationError('Evidence-first reuse binding requires an object plan.')
        return _bind_evidence_reuse_plan(proposal, evidence_plan)
    selection = proposal.game_design.get('_platform_selection')
    reuse_plan = selection.get('reuse_plan') if isinstance(selection, Mapping) else None
    if not isinstance(reuse_plan, Mapping):
        return proposal
    raw_decisions = reuse_plan.get('capabilities')
    decisions = [dict(item) for item in raw_decisions if isinstance(item, Mapping)] if isinstance(raw_decisions, Sequence) and (not isinstance(raw_decisions, (str, bytes))) else []
    owners = _assign_capability_owners(proposal.modules, decisions)
    game_design = {**proposal.game_design, '_reuse_plan': dict(reuse_plan)}
    modules: list[ProductionModule] = []
    for index, module in enumerate(proposal.modules):
        owned = owners.get(index, ())
        if not owned:
            modules.append(module)
            continue
        owned_plan = {**dict(reuse_plan), 'capabilities': [dict(item) for item in owned]}
        config = {**module.config, '_approved_reuse_plan': dict(reuse_plan), '_owned_reuse_plan': owned_plan, '_owned_capabilities': [str(item.get('capability') or '') for item in owned]}
        modules.append(replace(module, config=config))
    updated = replace(proposal, game_design=game_design, modules=tuple(modules), approval_hash='').with_hash()
    updated.validate()
    return updated


def _bind_evidence_reuse_plan(
    proposal: CompleteProposal,
    evidence_plan: Mapping[str, Any],
) -> CompleteProposal:
    """Bind reuse through the validated semantic DAG, without similarity routing.

    Evidence-mode ownership is derived only from exact host-owned identifiers.  The
    final task that provides a requirement's canonical capability owns its reuse
    decision.  Intermediate tasks retain their exact task-level reference contract,
    but never acquire a capability through token overlap or a first-module fallback.
    """
    try:
        from .evidence_first_planning import validate_evidence_first_plan

        validate_evidence_first_plan(evidence_plan, prompt=proposal.requested_prompt)
    except (ImportError, ValueError, TypeError, RecursionError) as exc:
        raise SpecValidationError(
            f'Evidence-first reuse binding rejected an invalid plan: {exc}'
        ) from exc

    plan_sha256 = str(evidence_plan.get('plan_sha256') or '')
    request_catalog = evidence_plan.get('request_catalog')
    raw_requirements = (
        request_catalog.get('requirements')
        if isinstance(request_catalog, Mapping)
        else None
    )
    raw_decisions = evidence_plan.get('reuse_decisions')
    raw_tasks = evidence_plan.get('tasks')
    raw_components = evidence_plan.get('component_catalog')
    if not all(
        isinstance(value, list)
        for value in (raw_requirements, raw_decisions, raw_tasks, raw_components)
    ):
        raise SpecValidationError(
            'Evidence-first reuse binding requires list catalogs for requirements, decisions, tasks, and components.'
        )

    requirements = {
        str(item.get('requirement_id') or ''): item
        for item in raw_requirements
        if isinstance(item, Mapping)
    }
    decisions = {
        str(item.get('requirement_ref') or ''): item
        for item in raw_decisions
        if isinstance(item, Mapping)
    }
    tasks = {
        str(item.get('task_id') or ''): item
        for item in raw_tasks
        if isinstance(item, Mapping)
    }
    components = {
        str(item.get('component_id') or ''): item
        for item in raw_components
        if isinstance(item, Mapping)
    }
    if len(requirements) != len(raw_requirements):
        raise SpecValidationError('Evidence-first requirement catalog contains an invalid or duplicate reference.')
    if len(decisions) != len(raw_decisions):
        raise SpecValidationError('Evidence-first reuse decisions contain an invalid or duplicate requirement reference.')
    if len(tasks) != len(raw_tasks):
        raise SpecValidationError('Evidence-first task catalog contains an invalid or duplicate task reference.')
    if len(components) != len(raw_components):
        raise SpecValidationError('Evidence-first component catalog contains an invalid or duplicate component reference.')
    if set(decisions) != set(requirements):
        raise SpecValidationError('Evidence-first reuse decisions do not exactly cover the requirement catalog.')

    modules = {module.module_id: module for module in proposal.modules}
    if len(modules) != len(proposal.modules) or set(modules) != set(tasks):
        raise SpecValidationError(
            'Evidence-first production modules must map one-to-one to semantic task IDs.'
        )

    for task_id, task in tasks.items():
        module = modules[task_id]
        _validate_evidence_module_binding(
            module=module,
            task=task,
            evidence_plan_sha256=plan_sha256,
            request_catalog=request_catalog,
            requirements=requirements,
            decisions=decisions,
            components=components,
        )

    owner_decisions: dict[str, list[Mapping[str, Any]]] = {}
    for requirement_ref, decision in decisions.items():
        action = str(decision.get('action') or '')
        if action == 'retain':
            continue
        requirement = requirements[requirement_ref]
        required_provides = set(
            _strict_string_refs(
                requirement.get('provides'),
                f'requirement {requirement_ref} provides',
            )
        )
        if not required_provides:
            raise SpecValidationError(
                f'Evidence-first requirement {requirement_ref} has no capability provide.'
            )
        exact_owners = [
            task_id
            for task_id, task in tasks.items()
            if requirement_ref
            in _strict_string_refs(
                task.get('requirement_refs'),
                f'task {task_id} requirement_refs',
            )
            and required_provides
            <= set(
                _strict_string_refs(
                    task.get('provides'),
                    f'task {task_id} provides',
                )
            )
        ]
        if len(exact_owners) != 1:
            raise SpecValidationError(
                f'Evidence-first requirement {requirement_ref} must have exactly one task '
                f'providing {sorted(required_provides)}; found {sorted(exact_owners)}.'
            )
        owner_decisions.setdefault(exact_owners[0], []).append(decision)

    target_decision = evidence_plan.get('target_decision')
    target = (
        dict(target_decision.get('coordinates') or {})
        if isinstance(target_decision, Mapping)
        else {}
    )
    projected = [
        _project_evidence_reuse_decision(decision)
        for decision in raw_decisions
        if isinstance(decision, Mapping) and decision.get('action') != 'retain'
    ]
    approved_plan: dict[str, Any] = {
        'schema_version': 'mmm/evidence-bound-reuse-plan-v1',
        'target': target,
        'evidence_plan_sha256': plan_sha256,
        'capabilities': projected,
        'evidence_decisions': [dict(item) for item in raw_decisions],
    }
    approved_receipt = {
        'schema_version': approved_plan['schema_version'],
        'target': target,
        'evidence_plan_sha256': plan_sha256,
    }

    rebound_modules: list[ProductionModule] = []
    for module in proposal.modules:
        owned_evidence = owner_decisions.get(module.module_id, [])
        if not owned_evidence:
            rebound_modules.append(module)
            continue
        owned_projected = [
            _project_evidence_reuse_decision(decision)
            for decision in owned_evidence
        ]
        owned_plan = {
            **approved_plan,
            'capabilities': owned_projected,
            'evidence_decisions': [dict(item) for item in owned_evidence],
        }
        owned_capabilities = [str(item.get('capability') or '') for item in owned_evidence]
        owned_component_refs = list(
            dict.fromkeys(
                reference
                for item in owned_evidence
                for reference in _strict_string_refs(
                    item.get('component_refs'),
                    f"reuse decision {item.get('decision_id')} component_refs",
                )
            )
        )
        config = {
            **module.config,
            '_approved_reuse_plan': approved_receipt,
            '_owned_reuse_plan': owned_plan,
            '_owned_capabilities': owned_capabilities,
            '_owned_component_refs': owned_component_refs,
        }
        rebound_modules.append(replace(module, config=config))

    game_design = {**proposal.game_design, '_reuse_plan': approved_plan}
    game_design, acceptance_tests = _refresh_evidence_production_contract(
        proposal=proposal,
        game_design=game_design,
        modules=tuple(rebound_modules),
        evidence_plan=evidence_plan,
    )
    updated = replace(
        proposal,
        game_design=game_design,
        modules=tuple(rebound_modules),
        acceptance_tests=acceptance_tests,
        approval_hash='',
    ).with_hash()
    updated.validate()
    return updated


def _refresh_evidence_production_contract(
    *,
    proposal: CompleteProposal,
    game_design: Mapping[str, Any],
    modules: tuple[ProductionModule, ...],
    evidence_plan: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Rebind a v2 contract after host-only reuse metadata changes modules."""
    current = game_design.get('_production_contract')
    if current is None:
        return dict(game_design), proposal.acceptance_tests
    if not isinstance(current, Mapping):
        raise SpecValidationError('Evidence-first production contract must be an object.')
    raw_acceptance = current.get('acceptance_catalog')
    if not isinstance(raw_acceptance, list):
        raise SpecValidationError('Evidence-first production contract has no acceptance catalog.')
    # The authority boundary may deduplicate an input acceptance when it is
    # byte-identical to the requirement-owned canonical public acceptance. Rebinding must
    # therefore preserve the complete current public acceptance surface, not require a
    # surviving origin=input row. Feeding these statements back through compilation is
    # lossless: the authority boundary deterministically removes duplicate input rows again.
    recompilation_acceptance = tuple(
        str(item.get('statement') or '')
        for item in raw_acceptance
        if isinstance(item, Mapping)
        and item.get('visibility') == 'public'
        and item.get('origin') in {'input', 'requirement'}
    )
    if not recompilation_acceptance or any(not item for item in recompilation_acceptance):
        raise SpecValidationError(
            'Evidence-first production contract has no exact public acceptance receipts.'
        )
    contract_design = {
        key: value
        for key, value in game_design.items()
        if not str(key).startswith('_') and key != 'production_outline'
    }
    research_brief = game_design.get('_research_brief')
    try:
        from .production_contract import compile_production_contract

        compiled = compile_production_contract(
            requested_prompt=proposal.requested_prompt,
            game_design=contract_design,
            research_brief=(research_brief if isinstance(research_brief, Mapping) else None),
            modules=modules,
            assets=proposal.assets,
            acceptance_tests=recompilation_acceptance,
            evidence_plan=evidence_plan,
        )
    except (ImportError, ValueError, TypeError, RecursionError) as exc:
        raise SpecValidationError(
            f'Evidence-first production contract could not bind exact reuse ownership: {exc}'
        ) from exc
    return (
        {**dict(game_design), '_production_contract': compiled.contract},
        tuple(compiled.acceptance_tests),
    )


def _validate_evidence_module_binding(
    *,
    module: ProductionModule,
    task: Mapping[str, Any],
    evidence_plan_sha256: str,
    request_catalog: Mapping[str, Any],
    requirements: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
    components: Mapping[str, Mapping[str, Any]],
) -> None:
    task_id = str(task.get('task_id') or '')
    config = module.config
    if config.get('evidence_plan_sha256') != evidence_plan_sha256:
        raise SpecValidationError(
            f'Evidence task {task_id} is bound to a stale plan hash.'
        )
    embedded = config.get('evidence_task')
    if not isinstance(embedded, Mapping):
        raise SpecValidationError(f'Evidence task {task_id} has no host-owned task receipt.')
    extras = set(embedded) - set(task)
    if extras - {'request_context'}:
        raise SpecValidationError(
            f'Evidence task {task_id} contains unrecognized receipt fields: {sorted(extras)}.'
        )
    for key, value in task.items():
        if embedded.get(key) != value:
            raise SpecValidationError(
                f'Evidence task {task_id} changed host-owned field {key!r}.'
            )
    if 'request_context' in embedded:
        requirement_refs = _strict_string_refs(
            task.get('requirement_refs'),
            f'task {task_id} requirement_refs',
        )
        expected_context = {
            'prompt_sha256': request_catalog.get('prompt_sha256'),
            'requirements': [dict(requirements[reference]) for reference in requirement_refs],
        }
        if embedded.get('request_context') != expected_context:
            raise SpecValidationError(
                f'Evidence task {task_id} request context is stale or references another requirement.'
            )

    exact_fields = (
        'requirement_refs',
        'gap_refs',
        'reuse_refs',
        'owned_anchors',
        'consumes',
        'provides',
        'acceptance',
        'impact_probes',
    )
    for key in exact_fields:
        if config.get(key) != task.get(key):
            raise SpecValidationError(
                f'Evidence task {task_id} module binding changed {key!r}.'
            )
    if config.get('batch_id') != task_id:
        raise SpecValidationError(f'Evidence task {task_id} module batch ID does not match.')
    if tuple(module.depends_on) != tuple(
        _strict_string_refs(task.get('depends_on'), f'task {task_id} depends_on')
    ):
        raise SpecValidationError(f'Evidence task {task_id} module dependencies changed.')
    if tuple(module.required_gates) != tuple(
        _strict_string_refs(task.get('required_gates'), f'task {task_id} required_gates')
    ):
        raise SpecValidationError(f'Evidence task {task_id} module gates changed.')

    requirement_refs = _strict_string_refs(
        task.get('requirement_refs'),
        f'task {task_id} requirement_refs',
    )
    expected_reuse_refs: list[str] = []
    for requirement_ref in requirement_refs:
        if requirement_ref not in requirements or requirement_ref not in decisions:
            raise SpecValidationError(
                f'Evidence task {task_id} references unknown requirement {requirement_ref!r}.'
            )
        decision = decisions[requirement_ref]
        capability = str(decision.get('capability') or '')
        if _canonical_evidence_capability(capability) != _canonical_evidence_capability(
            requirements[requirement_ref].get('capability')
        ):
            raise SpecValidationError(
                f'Evidence task {task_id} decision capability does not exactly match its requirement.'
            )
        component_refs = _strict_string_refs(
            decision.get('component_refs'),
            f"reuse decision {decision.get('decision_id')} component_refs",
        )
        if any(reference not in components for reference in component_refs):
            raise SpecValidationError(
                f'Evidence task {task_id} reuse decision references an unknown component.'
            )
        expected_reuse_refs.extend(component_refs)
        expected_reuse_refs.extend(
            _strict_string_refs(
                decision.get('source_refs'),
                f"reuse decision {decision.get('decision_id')} source_refs",
            )
        )
    expected_reuse_refs = list(dict.fromkeys(expected_reuse_refs))
    actual_reuse_refs = list(
        _strict_string_refs(task.get('reuse_refs'), f'task {task_id} reuse_refs')
    )
    if actual_reuse_refs != expected_reuse_refs:
        raise SpecValidationError(
            f'Evidence task {task_id} reuse_refs do not exactly match its hashed decisions.'
        )


def _project_evidence_reuse_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    capability = str(decision.get('capability') or '').strip()
    action = str(decision.get('action') or '')
    if action == 'fresh':
        return {
            'capability': capability,
            'mode': 'fresh',
            'source_id': '',
            'rationale': str(decision.get('residual_work') or ''),
        }
    if action == 'adapt':
        receipt = decision.get('external_receipt')
        if not isinstance(receipt, Mapping):
            raise SpecValidationError(
                f'Evidence reuse decision {decision.get("decision_id")} has no external receipt.'
            )
        projected = dict(receipt)
        if _canonical_evidence_capability(projected.get('capability')) != _canonical_evidence_capability(capability):
            raise SpecValidationError(
                f'Evidence reuse decision {decision.get("decision_id")} external capability changed.'
            )
        return projected
    raise SpecValidationError(
        f'Evidence reuse decision {decision.get("decision_id")} cannot be assigned to a generation task.'
    )


def _strict_string_refs(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise SpecValidationError(f'{label} must be a list of exact string references.')
    refs = tuple(item for item in value if isinstance(item, str) and item.strip())
    if len(refs) != len(value) or len(set(refs)) != len(refs):
        raise SpecValidationError(f'{label} contains an invalid or duplicate reference.')
    return refs


def _canonical_evidence_capability(value: Any) -> str:
    capability = str(value or '').strip().casefold()
    capability = capability.removeprefix('capability:')
    if not capability:
        raise SpecValidationError('Evidence reuse binding encountered an empty capability ID.')
    return f'capability:{capability}'

def _assign_capability_owners(modules: Sequence[ProductionModule], decisions: Sequence[Mapping[str, Any]]) -> dict[int, tuple[Mapping[str, Any], ...]]:
    candidates = [index for index, module in enumerate(modules) if module.kind != 'audio']
    if not candidates or not decisions:
        return {}
    preferred = next((index for index in candidates if modules[index].kind == 'custom_java'), candidates[0])
    module_tokens = {index: _module_semantic_tokens(modules[index]) for index in candidates}
    assigned: dict[int, list[Mapping[str, Any]]] = {index: [] for index in candidates}
    for decision in decisions:
        capability = str(decision.get('capability') or '').strip().casefold()
        cap_tokens = _semantic_words(capability)
        scored = []
        for index in candidates:
            overlap = len(cap_tokens & module_tokens[index])
            prefix = sum(token and any(word.startswith(token) or token.startswith(word) for word in module_tokens[index]) for token in cap_tokens)
            scored.append((overlap * 4 + prefix, -index, index))
        score, _tie, owner = max(scored)
        if score <= 0:
            owner = preferred
        assigned[owner].append(decision)
    return {index: tuple(values) for index, values in assigned.items() if values}

def _module_semantic_tokens(module: ProductionModule) -> set[str]:
    values = [module.module_id, module.kind]
    config = module.config if isinstance(module.config, Mapping) else {}
    for key in ('requested_kind', 'name', 'feature', 'system', 'capability', 'description'):
        value = config.get(key)
        if isinstance(value, str):
            values.append(value)
    return _semantic_words(' '.join(values))

def _semantic_words(value: str) -> set[str]:
    return {token.casefold() for token in re.findall('[A-Za-z_][A-Za-z0-9_]{1,127}', value.replace('.', ' ').replace('-', ' ').replace('_', ' ')) if len(token) > 2}

def install_prebootstrap_asset_runtime() -> None:
    """Install the raw producer/backend before existing GPU handoff wraps it."""
    global _LEGACY_GENERATE_ASSETS
    from . import complete_orchestrator_services, model_runtime_performance
    from .model_adapters import base as base_module
    from .model_adapters.image_diffusion import ImageDiffusionAdapter
    current = complete_orchestrator_services.generate_assets
    if _LEGACY_GENERATE_ASSETS is None and current is not generate_assets:
        _LEGACY_GENERATE_ASSETS = current
    complete_orchestrator_services.generate_assets = generate_assets
    install_flux2_q4_image_adapter(ImageDiffusionAdapter, model_runtime_performance, base_module)

def generate_assets(router: Any, proposal: CompleteProposal, project_root: Path, run_root: Path) -> dict[str, Any]:
    """Materialize requested assets from persisted prompt candidates and validate them."""
    if not hasattr(proposal, 'game_design'):
        legacy = _LEGACY_GENERATE_ASSETS
        if legacy is None:
            raise AssetProductionError('Legacy asset producer is unavailable.')
        return legacy(router, proposal, project_root, run_root)
    if not proposal.assets:
        return {'status': 'TEXTURE_PRODUCTION_PASS', 'assets': [], 'count': 0, 'backend': _backend_receipt()}
    plan = proposal.game_design.get('_asset_generation_plan')
    if not _valid_plan(plan, proposal.assets):
        raise AssetProductionError('Asset generation requires the persisted multi-prompt plan; refusing placeholder/fallback output.')
    prompt_map = {str(item['asset_id']): tuple(str(value) for value in item['prompts']) for item in plan['assets']}
    candidate_root = run_root / '.minecraft_ai' / 'asset-candidates'
    candidate_root.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    for request in proposal.assets:
        prompts = prompt_map.get(request.asset_id, ())
        if len(prompts) != _PROMPTS_PER_ASSET:
            raise AssetProductionError(f'Asset {request.asset_id} must have exactly {_PROMPTS_PER_ASSET} persisted prompts.')
        source_width, source_height = _source_dimensions(request.width, request.height)
        raw_candidates: list[tuple[int, str, Path]] = []
        for index, prompt in enumerate(prompts):
            output = candidate_root / request.asset_id / f'candidate-{index:02d}.png'
            output.parent.mkdir(parents=True, exist_ok=True)
            router.generate_image('image_generator', prompt=_ensure_trigger(prompt), output_path=output, width=source_width, height=source_height, seed=_candidate_seed(request.asset_id, index))
            if not output.is_file() or output.is_symlink():
                raise AssetProductionError(f'Image backend produced no PNG for {request.asset_id} candidate {index}.')
            raw_candidates.append((index, prompt, output))
        workers = min(_asset_workers(), len(raw_candidates))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='mmm-png-score') as pool:
                scored = list(pool.map(lambda item: _prepare_candidate(request, *item), raw_candidates))
        else:
            scored = [_prepare_candidate(request, *item) for item in raw_candidates]
        winner = sorted(scored, key=lambda item: (-item['score'], item['index']))[0]
        target = _safe_target(project_root, request.target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(winner['normalized_path'], target)
        final = _validate_png(target, request.width, request.height)
        receipts.append({'asset_id': request.asset_id, 'kind': request.kind, 'target_path': request.target_path.replace('\\', '/'), 'resolved_path': str(target), 'target': str(target), 'width': request.width, 'height': request.height, 'sha256': final['sha256'], 'mode': final['mode'], 'selected_prompt_index': winner['index'], 'selected_prompt': winner['prompt'], 'candidate_count': len(scored), 'candidate_scores': [{'index': item['index'], 'score': item['score']} for item in sorted(scored, key=lambda value: value['index'])], 'png_decode': True, 'placeholder': False})
    target = _target_receipt(proposal.game_design)
    reference_report = _validate_reference_closure(project_root)
    resource_pack_zip = ''
    standalone_assets = [item for item in proposal.assets if item.target_path.replace('\\', '/').startswith('assets/')]
    if standalone_assets:
        pack_format = target.get('resource_pack_format')
        if type(pack_format) is not int or pack_format < 1:
            raise AssetProductionError('Selected platform did not supply resource_pack_format.')
        resource_pack_zip = str(_package_resource_pack(project_root, run_root, pack_format=pack_format))
    return {'status': 'TEXTURE_PRODUCTION_PASS', 'schema_version': 'mmm/texture-production-receipt-v1', 'assets': receipts, 'count': len(receipts), 'backend': _backend_receipt(), 'reference_validation': reference_report, 'resource_pack_zip': resource_pack_zip, 'checks': {'binary_png_produced': True, 'png_decodes': True, 'dimensions_match': True, 'resource_paths_safe': True, 'references_resolve': reference_report['ok'], 'no_placeholder': True, 'pack_format_from_platform': bool(not standalone_assets or resource_pack_zip)}}

def install_flux2_q4_image_adapter(image_adapter_cls: Any, model_runtime_module: Any, base_module: Any) -> None:
    """Replace the legacy FP16 loader with fixed FLUX2 9B NF4 + fixed PixelArt LoRA."""
    current = image_adapter_cls.generate_image
    if getattr(current, '_mmm_flux2_klein9b_q4_pixelart', False):
        return

    def generate_image(self: Any, *, prompt: str, output_path: Path, width: int, height: int, seed: int) -> Path:
        cfg = self.config
        try:
            _require_fixed_backend_config(cfg)
            base_module.require_package('torch')
            base_module.require_package('diffusers', minimum='0.39.0')
            base_module.require_package('transformers', minimum='4.56.0')
            base_module.require_package('accelerate', minimum='1.0.0')
            base_module.require_package('bitsandbytes', minimum='0.45.0')
            base_module.require_package('peft', minimum='0.17.0')
            if not str(prompt).strip():
                raise ModelConfigurationError('Image prompt is empty.')
            if width % 16 or height % 16 or (not (256 <= width <= 1024 and 256 <= height <= 1024)):
                raise ModelConfigurationError('FLUX.2 image dimensions must be 256-1024 and divisible by 16.')
            base_module.preflight_cuda(cfg)
            import torch
            from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
            from diffusers import Flux2KleinPipeline
            from diffusers.quantizers import PipelineQuantizationConfig
            from transformers import (
                BitsAndBytesConfig as TransformersBitsAndBytesConfig,
            )
            if not torch.cuda.is_available():
                raise ModelConfigurationError('FLUX.2 Klein 9B Q4 asset production requires CUDA.')
            key = (FLUX_MODEL_ID, QUANTIZATION, PIXEL_LORA_ID, PIXEL_LORA_WEIGHT, 'float16', 'q4_auto_offload')
            with model_runtime_module._IMAGE_LOCK:
                pipeline = model_runtime_module._IMAGE_PIPELINE
                pipeline_key = model_runtime_module._IMAGE_PIPELINE_KEY
                if pipeline is None or pipeline_key != key:
                    previous = model_runtime_module._IMAGE_PIPELINE
                    model_runtime_module._IMAGE_PIPELINE = None
                    model_runtime_module._IMAGE_PIPELINE_KEY = None
                    model_runtime_module._IMAGE_PIPELINE_ON_GPU = False
                    if previous is not None:
                        del previous
                        base_module._release_cuda()
                    compute_dtype = torch.float16
                    quantization_config = PipelineQuantizationConfig(quant_mapping={'transformer': DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True), 'text_encoder': TransformersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)})
                    max_memory = _accelerate_memory_budget(torch)
                    pipeline = Flux2KleinPipeline.from_pretrained(FLUX_MODEL_ID, torch_dtype=compute_dtype, quantization_config=quantization_config, device_map='auto', max_memory=max_memory, low_cpu_mem_usage=True, trust_remote_code=False)
                    load_lora = getattr(pipeline, 'load_lora_weights', None)
                    if not callable(load_lora):
                        raise ModelConfigurationError('Installed Diffusers pipeline lacks Flux2 LoRA loading.')
                    load_lora(PIXEL_LORA_ID, weight_name=PIXEL_LORA_WEIGHT, adapter_name='mmm_pixelart_redmond')
                    set_adapters = getattr(pipeline, 'set_adapters', None)
                    if callable(set_adapters):
                        set_adapters('mmm_pixelart_redmond', adapter_weights=1.0)
                    progress = getattr(pipeline, 'set_progress_bar_config', None)
                    if callable(progress):
                        progress(disable=True)
                    model_runtime_module._IMAGE_PIPELINE = pipeline
                    model_runtime_module._IMAGE_PIPELINE_KEY = key
                    model_runtime_module._IMAGE_PIPELINE_ON_GPU = False
                generator = torch.Generator(device='cpu').manual_seed(int(seed))
                with torch.inference_mode():
                    result = pipeline(prompt=_ensure_trigger(prompt), width=int(width), height=int(height), num_inference_steps=4, guidance_scale=1.0, generator=generator)
                images = getattr(result, 'images', None)
                if not images:
                    raise ModelConfigurationError('FLUX.2 Klein 9B Q4 returned no image.')
                output_path = Path(output_path).expanduser().resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                images[0].convert('RGBA').save(output_path, format='PNG', optimize=False)
                return output_path
        except base_module.ModelBackendError:
            raise
        except Exception as exc:
            with model_runtime_module._IMAGE_LOCK:
                model_runtime_module._IMAGE_PIPELINE = None
                model_runtime_module._IMAGE_PIPELINE_KEY = None
                model_runtime_module._IMAGE_PIPELINE_ON_GPU = False
            base_module._release_cuda()
            raise base_module.ModelBackendError(role=cfg.role, model_id=cfg.model_id, cause=exc) from exc
    generate_image._mmm_flux2_klein9b_q4_pixelart = True
    generate_image._mmm_adaptive_image_residency = True
    image_adapter_cls.generate_image = generate_image

def _require_fixed_backend_config(cfg: Any) -> None:
    extra = getattr(cfg, 'extra', {}) or {}
    errors = []
    if getattr(cfg, 'model_id', '') != FLUX_MODEL_ID:
        errors.append(f'model_id must be {FLUX_MODEL_ID}')
    if getattr(cfg, 'quantization', None) != QUANTIZATION:
        errors.append(f'quantization must be {QUANTIZATION}')
    if str(extra.get('lora_model_id') or '') != PIXEL_LORA_ID:
        errors.append(f'lora_model_id must be {PIXEL_LORA_ID}')
    if str(extra.get('lora_weight_name') or '') != PIXEL_LORA_WEIGHT:
        errors.append(f'lora_weight_name must be {PIXEL_LORA_WEIGHT}')
    if str(extra.get('lora_trigger') or '') != PIXEL_LORA_TRIGGER:
        errors.append(f'lora_trigger must be {PIXEL_LORA_TRIGGER}')
    if float(extra.get('lora_scale', 0.0) or 0.0) != 1.0:
        errors.append('lora_scale must be 1.0')
    if errors:
        raise ModelConfigurationError('Fixed Minecraft asset backend misconfigured: ' + '; '.join(errors))

def _accelerate_memory_budget(torch_module: Any) -> dict[Any, str]:
    total = int(torch_module.cuda.get_device_properties(0).total_memory)
    reserve = max(2 * 1024 ** 3, int(total * 0.14))
    gpu = max(4 * 1024 ** 3, total - reserve)
    try:
        import psutil
        cpu_available = int(psutil.virtual_memory().available)
    except Exception:
        cpu_available = 8 * 1024 ** 3
    cpu = max(2 * 1024 ** 3, int(cpu_available * 0.8))
    return {0: f'{gpu // 1024 ** 2}MiB', 'cpu': f'{cpu // 1024 ** 2}MiB'}

def _prompt_response_schema(batch: Sequence[AssetRequest]) -> dict[str, Any]:
    ids = [asset.asset_id for asset in batch]
    return {'type': 'object', 'additionalProperties': False, 'required': ['assets'], 'properties': {'assets': {'type': 'array', 'minItems': len(batch), 'maxItems': len(batch), 'items': {'type': 'object', 'additionalProperties': False, 'required': ['asset_id', 'prompts'], 'properties': {'asset_id': {'type': 'string', 'enum': ids}, 'prompts': {'type': 'array', 'minItems': _PROMPTS_PER_ASSET, 'maxItems': _PROMPTS_PER_ASSET, 'items': {'type': 'string', 'minLength': 16}}}}}}}

def _validate_prompt_page(value: Any, batch: Sequence[AssetRequest]) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise SpecValidationError('Asset prompt page must be an object.')
    rows = value.get('assets')
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise SpecValidationError('Asset prompt page has no assets array.')
    expected = {asset.asset_id: asset for asset in batch}
    found: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise SpecValidationError('Asset prompt row must be an object.')
        asset_id = str(row.get('asset_id') or '')
        prompts = row.get('prompts')
        if asset_id not in expected or asset_id in found:
            raise SpecValidationError('Asset prompt page contains unknown or duplicate asset_id.')
        if not isinstance(prompts, Sequence) or isinstance(prompts, (str, bytes)):
            raise SpecValidationError(f'Asset {asset_id} prompts must be an array.')
        normalized = tuple(_ensure_trigger(str(prompt).strip()) for prompt in prompts if str(prompt).strip())
        if len(normalized) != _PROMPTS_PER_ASSET or len(set(normalized)) != len(normalized):
            raise SpecValidationError(f'Asset {asset_id} must have exactly {_PROMPTS_PER_ASSET} distinct prompts.')
        asset = expected[asset_id]
        found[asset_id] = {'asset_id': asset_id, 'kind': asset.kind, 'target_path': asset.target_path.replace('\\', '/'), 'width': asset.width, 'height': asset.height, 'prompts': list(normalized)}
    if set(found) != set(expected):
        raise SpecValidationError('Asset prompt page omitted one or more required assets.')
    return [found[asset.asset_id] for asset in batch]

def _valid_plan(value: Any, assets: Sequence[AssetRequest]) -> bool:
    if not isinstance(value, Mapping) or value.get('schema_version') != PLAN_SCHEMA:
        return False
    if value.get('model_id') != FLUX_MODEL_ID or value.get('quantization') != QUANTIZATION or value.get('lora_model_id') != PIXEL_LORA_ID or (value.get('lora_weight_name') != PIXEL_LORA_WEIGHT) or (value.get('lora_trigger') != PIXEL_LORA_TRIGGER) or (value.get('prompt_candidates_per_asset') != _PROMPTS_PER_ASSET):
        return False
    rows = value.get('assets')
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return False
    expected = {asset.asset_id: asset for asset in assets}
    found: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        asset_id = str(row.get('asset_id') or '')
        asset = expected.get(asset_id)
        prompts = row.get('prompts')
        if asset is None or asset_id in found:
            return False
        if str(row.get('kind') or '') != asset.kind or str(row.get('target_path') or '').replace('\\', '/') != asset.target_path.replace('\\', '/') or row.get('width') != asset.width or (row.get('height') != asset.height):
            return False
        if not isinstance(prompts, Sequence) or isinstance(prompts, (str, bytes)) or len(prompts) != _PROMPTS_PER_ASSET:
            return False
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            return False
        if len(set(prompts)) != _PROMPTS_PER_ASSET:
            return False
        found.add(asset_id)
    return found == set(expected)

def _target_receipt(game_design: Mapping[str, Any]) -> dict[str, Any]:
    selection = game_design.get('_platform_selection') if isinstance(game_design, Mapping) else None
    target = selection.get('target') if isinstance(selection, Mapping) else None
    return dict(target) if isinstance(target, Mapping) else {}

def _source_dimensions(width: int, height: int) -> tuple[int, int]:
    scale = max(1.0, 256.0 / max(1, width), 256.0 / max(1, height))
    source_w = min(1024, int(math.ceil(width * scale / 16.0) * 16))
    source_h = min(1024, int(math.ceil(height * scale / 16.0) * 16))
    return (max(256, source_w), max(256, source_h))

def _prepare_candidate(request: AssetRequest, index: int, prompt: str, source_path: Path) -> dict[str, Any]:
    from PIL import Image
    with Image.open(source_path) as opened:
        image = opened.convert('RGBA')
    normalized = image.resize((request.width, request.height), resample=Image.Resampling.NEAREST)
    if request.kind in {'item', 'icon'}:
        normalized = _remove_corner_background(normalized)
    output = source_path.with_name(source_path.stem + '-normalized.png')
    normalized.save(output, format='PNG', optimize=False)
    score = _technical_pixel_score(normalized, request.kind)
    return {'index': index, 'prompt': prompt, 'normalized_path': output, 'score': round(score, 6)}

def _remove_corner_background(image: Any) -> Any:
    rgba = image.convert('RGBA')
    if rgba.getextrema()[3][0] < 255:
        return rgba
    pixels = rgba.load()
    width, height = rgba.size
    corners = [pixels[0, 0], pixels[width - 1, 0], pixels[0, height - 1], pixels[width - 1, height - 1]]
    bg = tuple(sum(color[channel] for color in corners) // 4 for channel in range(3))
    threshold_sq = 24 * 24 * 3
    queue = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
    visited: set[tuple[int, int]] = set()
    while queue:
        x, y = queue.pop()
        if (x, y) in visited or not (0 <= x < width and 0 <= y < height):
            continue
        visited.add((x, y))
        color = pixels[x, y]
        distance = sum((int(color[channel]) - bg[channel]) ** 2 for channel in range(3))
        if distance > threshold_sq:
            continue
        pixels[x, y] = (color[0], color[1], color[2], 0)
        queue.extend(((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))
    return rgba

def _technical_pixel_score(image: Any, kind: str) -> float:
    from PIL import Image, ImageChops, ImageStat
    rgba = image.convert('RGBA')
    alpha = rgba.getchannel('A')
    alpha_values = list(alpha.getdata())
    opaque_ratio = sum(value > 16 for value in alpha_values) / max(1, len(alpha_values))
    occupancy_target = 0.55 if kind in {'item', 'icon'} else 0.95
    occupancy = max(0.0, 1.0 - abs(opaque_ratio - occupancy_target))
    gray = rgba.convert('L')
    dx = ImageChops.difference(gray, gray.transform(gray.size, Image.AFFINE, (1, 0, 1, 0, 1, 0)))
    dy = ImageChops.difference(gray, gray.transform(gray.size, Image.AFFINE, (1, 0, 0, 0, 1, 1)))
    edge = (ImageStat.Stat(dx).mean[0] + ImageStat.Stat(dy).mean[0]) / (2.0 * 255.0)
    colors = rgba.convert('P', palette=Image.Palette.ADAPTIVE, colors=32).getcolors(maxcolors=256) or []
    palette_score = min(1.0, len(colors) / 24.0)
    return 0.5 * occupancy + 0.3 * min(1.0, edge * 4.0) + 0.2 * palette_score

def _validate_png(path: Path, width: int, height: int) -> dict[str, Any]:
    from PIL import Image
    raw = path.read_bytes()
    if not raw.startswith(b'\x89PNG\r\n\x1a\n'):
        raise AssetProductionError(f'Generated asset is not a PNG: {path}')
    with Image.open(path) as image:
        image.load()
        if image.size != (width, height):
            raise AssetProductionError(f'Generated asset dimensions mismatch: {path}')
        mode = image.mode
    return {'sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(), 'mode': mode}

def _safe_target(project_root: Path, raw_path: str) -> Path:
    normalized = raw_path.replace('\\', '/')
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or '..' in pure.parts or pure.suffix.casefold() != '.png':
        raise AssetProductionError(f'Unsafe/non-PNG Minecraft asset target: {raw_path!r}')
    if 'assets' not in pure.parts:
        raise AssetProductionError(f'Minecraft texture target must live under assets/: {raw_path!r}')
    root = Path(project_root).expanduser().resolve()
    target = (root / Path(*pure.parts)).resolve()
    target.relative_to(root)
    return target

def _validate_reference_closure(project_root: Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    missing: set[str] = set()
    model_missing: set[str] = set()
    checked = 0
    model_checked = 0
    assets_roots = tuple(root.glob('assets/*')) + tuple(root.glob('src/main/resources/assets/*'))
    for assets_root in assets_roots:
        if not assets_root.is_dir():
            continue
        namespace = assets_root.name
        for json_path in assets_root.rglob('*.json'):
            try:
                data = json.loads(json_path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                continue
            for location in _texture_locations(data):
                checked += 1
                loc_namespace, loc_path = _split_resource_location(location, namespace)
                if loc_namespace == 'minecraft':
                    continue
                candidate = assets_root.parent / loc_namespace / 'textures' / f'{loc_path}.png'
                if not candidate.is_file():
                    missing.add(f'{loc_namespace}:{loc_path}')
            for location in _model_locations(data):
                model_checked += 1
                loc_namespace, loc_path = _split_resource_location(location, namespace)
                if loc_namespace == 'minecraft':
                    continue
                candidate = assets_root.parent / loc_namespace / 'models' / f'{loc_path}.json'
                if not candidate.is_file():
                    model_missing.add(f'{loc_namespace}:{loc_path}')
    if missing or model_missing:
        parts = []
        if missing:
            parts.append('textures=' + ', '.join(sorted(missing)[:20]))
        if model_missing:
            parts.append('models=' + ', '.join(sorted(model_missing)[:20]))
        raise AssetProductionError('Missing custom resource references: ' + '; '.join(parts))
    return {'ok': True, 'checked_texture_references': checked, 'checked_model_references': model_checked, 'missing': []}

def _texture_locations(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        textures = value.get('textures')
        if isinstance(textures, Mapping):
            for item in textures.values():
                if isinstance(item, str) and (not item.startswith('#')):
                    result.append(item)
        for child in value.values():
            if child is not textures:
                result.extend(_texture_locations(child))
    elif isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
        for child in value:
            result.extend(_texture_locations(child))
    return result

def _model_locations(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {'model', 'parent'} and isinstance(child, str) and (not child.startswith('#')):
                result.append(child)
            else:
                result.extend(_model_locations(child))
    elif isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
        for child in value:
            result.extend(_model_locations(child))
    return result

def _split_resource_location(value: str, default_namespace: str) -> tuple[str, str]:
    text = value.strip()
    if ':' in text:
        namespace, path = text.split(':', 1)
    else:
        namespace, path = (default_namespace, text)
    path = path.removeprefix('textures/').removesuffix('.png')
    return (namespace, path)

def _package_resource_pack(project_root: Path, run_root: Path, *, pack_format: int) -> Path:
    project_root = Path(project_root).resolve()
    pack_meta = {'pack': {'pack_format': pack_format, 'description': 'Generated and verified by M.M.M'}}
    meta_path = project_root / 'pack.mcmeta'
    meta_path.write_text(json.dumps(pack_meta, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    output = Path(run_root).resolve() / 'resource-pack.zip'
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted([meta_path, *project_root.glob('assets/**/*')]):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(project_root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 420 << 16
            archive.writestr(info, path.read_bytes())
    return output

def _candidate_seed(asset_id: str, index: int) -> int:
    raw = hashlib.sha256(f'{asset_id}:{index}'.encode()).digest()
    return int.from_bytes(raw[:4], 'big')

def _ensure_trigger(prompt: str) -> str:
    text = ' '.join(str(prompt).split())
    if PIXEL_LORA_TRIGGER.casefold() in text.casefold():
        return text
    return f'{PIXEL_LORA_TRIGGER}. {text}'.strip()

def _backend_receipt() -> dict[str, Any]:
    return {'model_id': FLUX_MODEL_ID, 'quantization': QUANTIZATION, 'lora_model_id': PIXEL_LORA_ID, 'lora_weight_name': PIXEL_LORA_WEIGHT, 'lora_trigger': PIXEL_LORA_TRIGGER, 'inference_steps': 4}

def _asset_workers() -> int:
    raw = os.environ.get('MMM_ASSET_CPU_WORKERS', '4').strip()
    try:
        value = int(raw)
    except ValueError:
        value = 4
    return max(1, min(value, 8))
__all__ = ['FLUX_MODEL_ID', 'PIXEL_LORA_ID', 'PIXEL_LORA_TRIGGER', 'PIXEL_LORA_WEIGHT', 'PLAN_SCHEMA', 'QUANTIZATION', 'AssetProductionError', 'attach_generation_plan', 'bind_reuse_plan', 'generate_assets', 'install_flux2_q4_image_adapter', 'install_prebootstrap_asset_runtime']
