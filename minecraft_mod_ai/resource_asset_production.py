from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from typing import Any

from .complete_spec import AssetRequest, CompleteProposal, ProductionModule
from .spec import SpecValidationError


class AssetProductionError(RuntimeError):
    pass


_CONTRACT_OWNED_PREFLIGHT = True


def _preflight(proposal: CompleteProposal) -> None:
    from .resource_asset_preflight_contract import validate_asset_generation_inputs
    try:
        validate_asset_generation_inputs(proposal)
    except ValueError as exc:
        raise AssetProductionError(f"Resource asset preflight failed: {exc}") from exc


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
    """Bind reuse through the validated semantic DAG, without similarity routing."""
    try:
        from .evidence_first_planning import validate_evidence_first_plan
        from .evidence_task_receipt_contract import build_execution_receipt_bundle

        validate_evidence_first_plan(evidence_plan, prompt=proposal.requested_prompt)
        receipt_bundle = build_execution_receipt_bundle(evidence_plan)
    except (ImportError, ValueError, TypeError, RecursionError) as exc:
        raise SpecValidationError(
            f'Evidence-first reuse binding rejected an invalid plan: {exc}'
        ) from exc

    plan_sha256 = str(evidence_plan.get('plan_sha256') or '')
    if receipt_bundle.get('plan_sha256') != plan_sha256:
        raise SpecValidationError('Evidence execution receipt bundle is bound to a stale plan hash.')
    raw_receipts = receipt_bundle.get('receipts')
    if not isinstance(raw_receipts, Mapping):
        raise SpecValidationError('Evidence execution receipt bundle has no receipt catalog.')
    expected_receipts = {
        str(task_id): dict(receipt)
        for task_id, receipt in raw_receipts.items()
        if str(task_id) and isinstance(receipt, Mapping)
    }
    if len(expected_receipts) != len(raw_receipts):
        raise SpecValidationError('Evidence execution receipt catalog contains an invalid task receipt.')

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
    if set(expected_receipts) != set(tasks):
        raise SpecValidationError('Evidence execution receipts must map one-to-one to semantic task IDs.')

    modules = {module.module_id: module for module in proposal.modules}
    if len(modules) != len(proposal.modules) or set(modules) != set(tasks):
        raise SpecValidationError(
            'Evidence-first production modules must map one-to-one to semantic task IDs.'
        )

    for task_id, semantic_task in tasks.items():
        module = modules[task_id]
        _validate_evidence_module_binding(
            module=module,
            semantic_task=semantic_task,
            expected_receipt=expected_receipts[task_id],
            evidence_plan_sha256=plan_sha256,
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
    semantic_task: Mapping[str, Any],
    expected_receipt: Mapping[str, Any],
    evidence_plan_sha256: str,
    requirements: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
    components: Mapping[str, Mapping[str, Any]],
) -> None:
    from .evidence_task_receipt_contract import (
        RECEIPT_EXTENSION_FIELDS,
        validate_task_receipt,
    )

    task_id = str(semantic_task.get('task_id') or '')
    if str(expected_receipt.get('task_id') or '') != task_id:
        raise SpecValidationError(f'Evidence task {task_id} execution receipt ID changed.')
    config = module.config
    if config.get('evidence_plan_sha256') != evidence_plan_sha256:
        raise SpecValidationError(
            f'Evidence task {task_id} is bound to a stale plan hash.'
        )
    embedded = config.get('evidence_task')
    if not isinstance(embedded, Mapping):
        raise SpecValidationError(f'Evidence task {task_id} has no host-owned task receipt.')
    try:
        validate_task_receipt(embedded, expected_receipt=expected_receipt)
    except ValueError as exc:
        raise SpecValidationError(str(exc)) from exc

    execution_task = {
        key: value
        for key, value in expected_receipt.items()
        if key not in RECEIPT_EXTENSION_FIELDS
    }
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
        if config.get(key) != execution_task.get(key):
            raise SpecValidationError(
                f'Evidence task {task_id} module binding changed execution field {key!r}.'
            )
    if config.get('batch_id') != task_id:
        raise SpecValidationError(f'Evidence task {task_id} module batch ID does not match.')
    if tuple(module.depends_on) != tuple(
        _strict_string_refs(execution_task.get('depends_on'), f'task {task_id} depends_on')
    ):
        raise SpecValidationError(f'Evidence task {task_id} module dependencies changed.')
    if tuple(module.required_gates) != tuple(
        _strict_string_refs(execution_task.get('required_gates'), f'task {task_id} required_gates')
    ):
        raise SpecValidationError(f'Evidence task {task_id} module gates changed.')

    requirement_refs = _strict_string_refs(
        semantic_task.get('requirement_refs'),
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
        _strict_string_refs(semantic_task.get('reuse_refs'), f'task {task_id} reuse_refs')
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
    """Compatibility hook for old bootstrap callers. Runtime monkey-patching is gone."""
    return


def _plan_row(router: Any, proposal: CompleteProposal, request: AssetRequest) -> dict[str, Any]:
    from .model_adapters.image_diffusion import ImageGenerationConfig
    from .resource_contracts import resolve_asset
    from .resource_prompt_compiler import compile_texture_prompt
    from .resource_visual_spec import resolve_visual_spec
    image_config = router.registry.role(router.profile, "image_generator")
    profile = ImageGenerationConfig.from_adapter_config(image_config)
    spec = proposal.base_proposal.spec
    context = spec.platform.version_context if spec.platform.host_facts_json else None
    owner = next((m for m in proposal.modules if m.module_id == request.owner_module_id), None)
    if request.owner_module_id and owner is None:
        raise AssetProductionError("Asset owner module is unresolved.")
    resolved = resolve_asset(request, namespace=spec.mod_id, minecraft_version=spec.platform.minecraft_version,
                             version_context=context, owner_module=owner)
    visual = resolve_visual_spec(request.visual_spec, request.visual_description)
    visual_bible = proposal.game_design.get("visual_identity", "")
    if not isinstance(visual_bible, str):
        raise AssetProductionError("Visual Bible must be semantic text.")
    purpose = str(owner.config.get("purpose", "")) if owner else ""
    textures = []
    for texture in resolved.textures:
        textures.append({**texture.to_dict(), "prompt": compile_texture_prompt(
            visual_spec=visual, visual_bible=visual_bible, feature_purpose=purpose,
            texture=texture, image_config=image_config)})
    return {
        "asset_id": request.asset_id, "visual_description": request.visual_description,
        "render_kind": resolved.render_kind, "subject_id": resolved.subject_id,
        "container": resolved.container, "candidate_count": profile.candidate_count,
        "visual_spec": visual.to_dict(), "generation_profile": asdict(profile),
        "textures": textures, "documents": [document.to_dict() for document in resolved.documents],
    }


def attach_generation_plan(router: Any, proposal: CompleteProposal) -> CompleteProposal:
    """Persist deterministic resource contracts; Qwen never authors raw backend prompts."""
    from .resource_prompt_compiler import image_profile_fingerprint
    if not proposal.assets:
        return proposal
    _preflight(proposal)
    image_config = router.registry.role(router.profile, "image_generator")
    plan = {
        "schema_version": "mmm/resource-asset-generation-plan-v3",
        "image_profile_sha256": image_profile_fingerprint(image_config),
        "assets": [_plan_row(router, proposal, request) for request in proposal.assets],
    }
    _validate_manifest(plan["assets"])
    if proposal.game_design.get("_asset_generation_plan") == plan:
        return proposal
    updated = replace(proposal, game_design={**proposal.game_design, "_asset_generation_plan": plan}, approval_hash="").with_hash()
    updated.validate()
    return updated


def _safe_target(project_root: Path, relative: str) -> Path:
    normalized = PurePosixPath(str(relative).replace("\\", "/"))
    if normalized.is_absolute() or any(part in {"", ".", ".."} for part in normalized.parts):
        raise AssetProductionError(f"Unsafe resource path: {relative!r}")
    root = project_root.expanduser().resolve()
    target = (root / Path(*normalized.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise AssetProductionError(f"Resource path escaped project root: {relative!r}") from exc
    if target.is_symlink():
        raise AssetProductionError(f"Refusing symlink resource target: {relative!r}")
    return target


def _candidate_seed(asset_id: str, role: str, index: int) -> int:
    digest = hashlib.sha256(f"{asset_id}\0{role}\0{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _edge_error(image: Any) -> float:
    rgb = image.convert("RGB")
    try:
        w, h = rgb.size
        total = 0.0
        samples = 0
        for y in range(h):
            total += sum(abs(a - b) for a, b in zip(rgb.getpixel((0, y)), rgb.getpixel((w - 1, y))))
            samples += 3
        for x in range(w):
            total += sum(abs(a - b) for a, b in zip(rgb.getpixel((x, 0)), rgb.getpixel((x, h - 1))))
            samples += 3
        return total / max(1, samples)
    finally:
        rgb.close()


def _prepare(texture: Mapping[str, Any], source: Path, normalized: Path) -> float:
    from .resource_image_pipeline import postprocess_region, validate_texture
    try:
        from PIL import Image
    except ImportError as exc:
        raise AssetProductionError("Pillow is required for resource post-processing.") from exc
    with Image.open(source) as raw:
        raw.load()
        image = postprocess_region(raw, texture["resource_contract"], (int(texture["width"]), int(texture["height"])))
    try:
        normalized.parent.mkdir(parents=True, exist_ok=True)
        image.save(normalized, format="PNG", optimize=False)
        validate_texture(normalized, texture)
        return 1000.0 - (_edge_error(image) if texture["topology"] == "seamless_tile" else 0.0)
    finally:
        image.close()


def _validate_manifest(rows: Sequence[Mapping[str, Any]]) -> None:
    paths: dict[tuple[str, str], Any] = {}
    asset_ids = set()
    for row in rows:
        if row["asset_id"] in asset_ids:
            raise AssetProductionError("Duplicate asset ID in manifest.")
        asset_ids.add(row["asset_id"])
        for entry in (*row["textures"], *row["documents"]):
            key = row["container"], entry["target_path"]
            if key in paths and ("payload" not in entry or paths[key] != entry):
                raise AssetProductionError(f"Conflicting resource manifest path: {key}.")
            paths[key] = entry


def _validated_plan(router: Any, proposal: CompleteProposal) -> Mapping[str, Any]:
    from .resource_prompt_compiler import image_profile_fingerprint
    plan = proposal.game_design.get("_asset_generation_plan")
    if not isinstance(plan, Mapping) or plan.get("schema_version") != "mmm/resource-asset-generation-plan-v3":
        raise AssetProductionError("Approved proposal has no canonical resource asset plan.")
    config = router.registry.role(router.profile, "image_generator")
    if plan.get("image_profile_sha256") != image_profile_fingerprint(config):
        raise AssetProductionError("Approved resource asset plan is bound to a different image profile.")
    rows = plan.get("assets")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise AssetProductionError("Resource asset plan has no asset rows.")
    ids = [str(row.get("asset_id")) for row in rows if isinstance(row, Mapping)]
    if ids != [request.asset_id for request in proposal.assets]:
        raise AssetProductionError("Resource asset plan no longer matches semantic assets.")
    expected = [_plan_row(router, proposal, request) for request in proposal.assets]
    if json.dumps(rows, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise AssetProductionError("Approved resource contract/manifest differs from current HOST/visual inputs.")
    _validate_manifest(rows)
    return plan


def _container_root(project_root: Path, run_root: Path, container: str) -> Path:
    if container == "mod":
        return project_root.expanduser().resolve()
    if container == "resource_pack":
        root = (run_root / "resource-pack").expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root
    raise AssetProductionError(f"Unsupported resource container: {container!r}")


def _write_documents(project_root: Path, run_root: Path, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    written: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        container = str(row.get("container") or "mod")
        root = _container_root(project_root, run_root, container)
        for document in row.get("documents", ()):
            if not isinstance(document, Mapping) or not isinstance(document.get("payload"), Mapping):
                raise AssetProductionError("Invalid resource document contract.")
            relative = str(document["target_path"])
            encoded = json.dumps(document["payload"], ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            digest = "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()
            key = (container, relative)
            if key in seen and seen[key] != digest:
                raise AssetProductionError(f"Conflicting resource documents target {container}:{relative}.")
            seen[key] = digest
            target = _safe_target(root, relative)
            _atomic_write_bytes(target, encoded.encode("utf-8"))
            written.append({
                "template_id": str(document.get("template_id") or ""),
                "container": container,
                "target_path": relative,
                "resolved_path": str(target),
                "sha256": digest,
            })
    return written


def _resource_reference(value: str) -> tuple[str, str] | None:
    if ":" not in value:
        return None
    namespace, path = value.split(":", 1)
    if not re.fullmatch(r"[a-z0-9_.-]+", namespace) or not re.fullmatch(r"[a-z0-9_./-]+", path):
        return None
    return namespace, path


def _document_references(payload: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    refs: list[tuple[str, str]] = []

    def visit(value: Any, *, key: str = "", in_textures: bool = False) -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                child_name = str(child_key)
                visit(child, key=child_name, in_textures=in_textures or child_name == "textures")
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child, key=key, in_textures=in_textures)
            return
        if not isinstance(value, str):
            return
        ref = _resource_reference(value)
        if ref is None or ref[0] == "minecraft":
            return
        if in_textures:
            refs.append(("texture", value))
        elif key in {"model", "parent"}:
            refs.append(("model", value))

    visit(payload)
    return tuple(refs)


def _reference_target(container: str, kind: str, reference: str) -> str:
    parsed = _resource_reference(reference)
    if parsed is None:
        raise AssetProductionError(f"Invalid generated resource reference: {reference!r}")
    namespace, path = parsed
    if "/" not in path:
        raise AssetProductionError(f"Generated resource reference lacks a resource folder: {reference!r}")
    folder, rest = path.split("/", 1)
    if folder not in {"block", "item", "entity", "gui"}:
        raise AssetProductionError(f"Unsupported generated resource reference folder: {reference!r}")
    prefix = "src/main/resources/" if container == "mod" else ""
    if kind == "model":
        if folder not in {"block", "item"}:
            raise AssetProductionError(f"Model reference has unsupported folder: {reference!r}")
        return f"{prefix}assets/{namespace}/models/{folder}/{rest}.json"
    return f"{prefix}assets/{namespace}/textures/{folder}/{rest}.png"


def _validate_reference_closure(
    project_root: Path,
    run_root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    namespace: str,
) -> dict[str, Any]:
    checked: list[dict[str, str]] = []
    for row in rows:
        container = str(row.get("container") or "mod")
        root = _container_root(project_root, run_root, container)
        for document in row.get("documents", ()):
            if not isinstance(document, Mapping) or not isinstance(document.get("payload"), Mapping):
                raise AssetProductionError("Invalid resource document during graph validation.")
            for kind, reference in _document_references(document["payload"]):
                parsed = _resource_reference(reference)
                if parsed is None or parsed[0] != namespace:
                    continue
                relative = _reference_target(container, kind, reference)
                target = _safe_target(root, relative)
                if not target.is_file() or target.is_symlink():
                    raise AssetProductionError(
                        f"RESOURCE_REFERENCE_UNRESOLVED: {reference} from {document.get('target_path')} -> {relative}"
                    )
                checked.append({
                    "container": container,
                    "kind": kind,
                    "reference": reference,
                    "resolved_path": str(target),
                })
    return {"status": "PASS", "checked_reference_count": len(checked), "references": checked}


def _validate_container_layout(
    proposal: CompleteProposal,
    project_root: Path,
    run_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    containers = {str(row.get("container") or "mod") for row in rows}
    unknown = containers - {"mod", "resource_pack"}
    if unknown:
        raise AssetProductionError(f"Unsupported resource containers: {sorted(unknown)}")
    result: dict[str, Any] = {
        "mod_resources": {"status": "NOT_PRESENT"},
        "standalone_resource_pack": {"status": "NOT_PRESENT"},
    }
    if "mod" in containers:
        bad = [
            str(item["target_path"])
            for row in rows if str(row.get("container") or "mod") == "mod"
            for item in (*row.get("textures", ()), *row.get("documents", ()))
            if not str(item.get("target_path") or "").startswith("src/main/resources/assets/")
        ]
        if bad:
            raise AssetProductionError(f"Mod resources escaped src/main/resources/assets: {bad[:8]}")
        result["mod_resources"] = {
            "status": "PASS",
            "root": str(project_root.expanduser().resolve() / "src/main/resources/assets"),
            "pack_mcmeta_required": False,
        }
    resource_pack_zip = ""
    if "resource_pack" in containers:
        root = _container_root(project_root, run_root, "resource_pack")
        bad = [
            str(item["target_path"])
            for row in rows if row.get("container") == "resource_pack"
            for item in (*row.get("textures", ()), *row.get("documents", ()))
            if not str(item.get("target_path") or "").startswith("assets/")
        ]
        if bad:
            raise AssetProductionError(f"Standalone resource-pack entries escaped assets/: {bad[:8]}")
        pack_format = getattr(proposal.base_proposal.spec.platform, "resource_pack_format", None)
        if type(pack_format) is not int or pack_format < 1:
            raise AssetProductionError("Standalone resource pack requires an admitted positive resource_pack_format.")
        metadata = {"pack": {"pack_format": pack_format, "description": "Generated by M.M.M"}}
        metadata_path = root / "pack.mcmeta"
        _atomic_write_bytes(
            metadata_path,
            (json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        )
        decoded = json.loads(metadata_path.read_text(encoding="utf-8"))
        if decoded != metadata or not (root / "assets").is_dir():
            raise AssetProductionError("Standalone resource-pack container validation failed.")
        archive = run_root / "resource-packs" / "generated-resource-pack.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(root.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    bundle.write(path, path.relative_to(root).as_posix())
        resource_pack_zip = str(archive)
        result["standalone_resource_pack"] = {
            "status": "PASS",
            "root": str(root),
            "pack_mcmeta": str(metadata_path),
            "resource_pack_format": pack_format,
            "zip": resource_pack_zip,
        }
    return result, resource_pack_zip



def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

def generate_assets(router: Any, proposal: CompleteProposal, project_root: Path, run_root: Path) -> dict[str, Any]:
    from .model_adapters.image_diffusion import ImageGenerationConfig
    from .resource_image_pipeline import generate_candidate, validate_texture
    if not proposal.assets:
        return {
            "schema_version": "mmm/resource-production-receipt-v2",
            "status": "TEXTURE_PRODUCTION_PASS",
            "assets": [], "documents": [], "count": 0,
            "resource_graph_validation": {"status": "PASS", "checked_reference_count": 0, "references": []},
            "container_validation": {
                "mod_resources": {"status": "NOT_PRESENT"},
                "standalone_resource_pack": {"status": "NOT_PRESENT"},
            },
            "resource_pack_zip": "",
        }
    _preflight(proposal)
    plan = _validated_plan(router, proposal)
    profile = ImageGenerationConfig.from_adapter_config(router.registry.role(router.profile, "image_generator"))
    rows = [dict(row) for row in plan["assets"]]
    from .resource_image_pipeline import validate_model_consumers
    for row in rows:
        validate_model_consumers(_container_root(project_root, run_root, row["container"]), row["textures"])
        root = _container_root(project_root, run_root, row["container"])
        for texture in row["textures"]:
            if not texture["resource_contract"]["animation"]["mcmeta_required"]:
                stale = _safe_target(root, texture["target_path"] + ".mcmeta")
                if stale.exists():
                    raise AssetProductionError(f"Static texture conflicts with existing animation metadata: {stale}")
    candidate_root = run_root / ".minecraft_ai" / "resource-candidates"
    receipts = []
    with router.image_generation_session("image_generator"):
        for row in rows:
            container = str(row.get("container") or "mod")
            container_root = _container_root(project_root, run_root, container)
            for texture in row.get("textures", ()):
                if not isinstance(texture, Mapping):
                    raise AssetProductionError("Invalid texture contract.")
                asset_id, role, prompt = str(row["asset_id"]), str(texture["role"]), str(texture["prompt"])
                scored = []
                failures = []
                for index in range(profile.candidate_count):
                    normalized = candidate_root / asset_id / role / f"normalized-{index:02d}.png"
                    try:
                        evidence = generate_candidate(
                            lambda **kwargs: router.generate_image("image_generator", **kwargs), texture,
                            prompt=prompt, directory=candidate_root / asset_id / role / f"candidate-{index:02d}",
                            output=normalized, resolution=profile.preferred_generation_resolution,
                            fallback=profile.fallback_generation_resolution, seed=_candidate_seed(asset_id, role, index))
                    except ValueError as exc:
                        failures.append({"candidate": index, "reason": str(exc)})
                        continue
                    scored.append((1000.0, index, normalized, evidence))
                if not scored:
                    raise AssetProductionError(f"No candidate satisfies resource contract for {asset_id}:{role}: {failures}")
                score, index, winner, evidence = min(scored, key=lambda item: (-item[0], item[1]))
                target = _safe_target(container_root, str(texture["target_path"]))
                _atomic_write_bytes(target, winner.read_bytes())
                receipts.append({
                    "asset_id": asset_id, "role": role, "render_kind": row["render_kind"],
                    "container": container, "target": str(target), "target_path": str(texture["target_path"]),
                    "width": int(texture["width"]), "height": int(texture["height"]),
                    "topology": str(texture["topology"]), "alpha_policy": str(texture["alpha_policy"]),
                    "selected_candidate": index, "selected_score": score, "candidate_count": profile.candidate_count,
                    "generation_evidence": evidence, "rejected_candidates": failures,
                    "prompt_sha256": "sha256:" + hashlib.sha256(prompt.encode()).hexdigest(),
                    "sha256": "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest(), "placeholder": False,
                })
    documents = _write_documents(project_root, run_root, rows)
    resource_checks = []
    for row in rows:
        root = _container_root(project_root, run_root, row["container"])
        for texture in row["textures"]:
            try:
                resource_checks.append(validate_texture(_safe_target(root, texture["target_path"]), texture, check_metadata=True))
            except ValueError as exc:
                raise AssetProductionError(f"Final resource validation failed: {exc}") from exc
    graph_validation = _validate_reference_closure(
        project_root, run_root, rows, namespace=proposal.base_proposal.spec.mod_id
    )
    container_validation, resource_pack_zip = _validate_container_layout(
        proposal, project_root, run_root, rows
    )
    return {
        "schema_version": "mmm/resource-production-receipt-v2",
        "status": "TEXTURE_PRODUCTION_PASS",
        "assets": receipts, "documents": documents, "count": len(receipts),
        "resource_graph_validation": graph_validation,
        "container_validation": container_validation,
        "resource_pack_zip": resource_pack_zip,
        "resource_contract_validation": {"status": "PASS", "textures": resource_checks},
        "checks": {
            "semantic_contract_resolved": True,
            "deterministic_prompt_compiler": True,
            "profile_owned_backend": True,
            "reference_closure": graph_validation["status"] == "PASS",
            "container_validated": True,
            "no_placeholder": True,
        },
    }


attach_generation_plan._mmm_resource_asset_preflight = True
generate_assets._mmm_resource_asset_preflight = True

__all__ = ["AssetProductionError", "attach_generation_plan", "bind_reuse_plan", "generate_assets", "install_prebootstrap_asset_runtime"]
