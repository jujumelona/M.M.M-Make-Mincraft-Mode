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

from .complete_spec import AssetRequest, CompleteProposal, CompleteProposalStatus, ProductionModule
from .resource_asset_plan import (
    ResourceAssetPlanError,
    asset_plan_sha256,
    require_asset_plan,
)
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



def _assign_capability_owners(modules: Sequence[ProductionModule], decisions: Sequence[Mapping[str, Any]]) -> dict[int, tuple[Mapping[str, Any], ...]]:
    # Every incoming module has already passed CompleteProposal validation.
    # Do not retain branches for removed production kinds such as audio/custom_java.
    candidates = list(range(len(modules)))
    if not candidates or not decisions:
        return {}
    preferred = next(
        (index for index in candidates if modules[index].kind == "typed_host"),
        candidates[0],
    )
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
    try:
        plan, _selected_rows = require_asset_plan(
            proposal.game_design,
            proposal.assets,
            exact=True,
        )
    except ResourceAssetPlanError as exc:
        raise AssetProductionError(
            f"Approved proposal has no valid canonical resource asset plan: {exc}"
        ) from exc
    config = router.registry.role(router.profile, "image_generator")
    if plan.get("image_profile_sha256") != image_profile_fingerprint(config):
        raise AssetProductionError("Approved resource asset plan is bound to a different image profile.")
    rows = plan["assets"]
    expected = [_plan_row(router, proposal, request) for request in proposal.assets]
    if json.dumps(rows, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise AssetProductionError("Approved resource contract/manifest differs from current HOST/visual inputs.")
    _validate_manifest(rows)
    return plan


def validate_asset_generation_plan(
    router: Any,
    proposal: CompleteProposal,
) -> dict[str, Any]:
    """Validate approved asset authority/profile/host contracts before dispatch."""

    _preflight(proposal)
    if not proposal.assets:
        return {
            "schema_version": "mmm/resource-asset-plan-preflight-v1",
            "status": "PASS",
            "asset_count": 0,
            "asset_plan_sha256": "",
        }
    plan = _validated_plan(router, proposal)
    return {
        "schema_version": "mmm/resource-asset-plan-preflight-v1",
        "status": "PASS",
        "asset_count": len(proposal.assets),
        "asset_plan_sha256": asset_plan_sha256(plan),
    }


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
    *,
    package_resource_pack: bool = True,
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
        if package_resource_pack:
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

def _asset_execution_projection(proposal: CompleteProposal) -> CompleteProposal:
    """Project one approved asset shard to its matching canonical plan rows."""

    # Test/imported callers may provide a proposal-shaped object only for deterministic
    # preflight. Projection is a CompleteProposal-specific optimization and must never
    # pre-empt the actual resource validation contract for those callers.
    if not isinstance(proposal, CompleteProposal):
        return proposal
    if proposal.status is not CompleteProposalStatus.APPROVED or proposal.approval_hash:
        return proposal

    selected_ids = tuple(asset.asset_id for asset in proposal.assets)
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise AssetProductionError(
            "Asset execution projection requires a non-empty unique asset subset."
        )
    try:
        raw_plan, selected_rows = require_asset_plan(
            proposal.game_design,
            proposal.assets,
        )
    except ResourceAssetPlanError as exc:
        raise AssetProductionError(
            "Asset execution projection requires the canonical approved asset plan: "
            f"{exc}"
        ) from exc

    filtered_plan = {
        **raw_plan,
        "assets": selected_rows,
    }
    projected = replace(
        proposal,
        status=CompleteProposalStatus.AWAITING_APPROVAL,
        game_design={
            **proposal.game_design,
            "_asset_generation_plan": filtered_plan,
        },
        approval_hash="",
    ).with_hash()
    projected.validate()
    return projected


def generate_assets(
    router: Any,
    proposal: CompleteProposal,
    project_root: Path,
    run_root: Path,
    *,
    package_resource_pack: bool = True,
) -> dict[str, Any]:
    proposal = _asset_execution_projection(proposal)
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
                failures = []
                selected = None
                for index in range(profile.candidate_count):
                    normalized = candidate_root / asset_id / role / f"normalized-{index:02d}.png"
                    try:
                        evidence = generate_candidate(
                            lambda **kwargs: router.generate_image("image_generator", **kwargs), texture,
                            prompt=prompt, directory=candidate_root / asset_id / role / f"candidate-{index:02d}",
                            output=normalized,
                            resolution=profile.preferred_generation_resolution,
                            seed=_candidate_seed(asset_id, role, index))
                    except ValueError as exc:
                        failures.append({"candidate": index, "reason": str(exc)})
                        continue
                    # Successful candidates all have the same selection score in this
                    # contract. The tie-breaker is the lowest index, so the first successful
                    # candidate is already the mathematically final winner. Generating later
                    # seeds cannot change the selected result.
                    selected = (1000.0, index, normalized, evidence)
                    break
                if selected is None:
                    raise AssetProductionError(f"No candidate satisfies resource contract for {asset_id}:{role}: {failures}")
                score, index, winner, evidence = selected
                target = _safe_target(container_root, str(texture["target_path"]))
                _atomic_write_bytes(target, winner.read_bytes())
                receipts.append({
                    "asset_id": asset_id, "role": role, "render_kind": row["render_kind"],
                    "container": container, "target": str(target), "target_path": str(texture["target_path"]),
                    "width": int(texture["width"]), "height": int(texture["height"]),
                    "topology": str(texture["topology"]), "alpha_policy": str(texture["alpha_policy"]),
                    "selected_candidate": index, "selected_score": score, "candidate_count": profile.candidate_count,
                    "attempted_candidate_count": index + 1,
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
        proposal,
        project_root,
        run_root,
        rows,
        package_resource_pack=package_resource_pack,
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

__all__ = ["AssetProductionError", "attach_generation_plan", "bind_reuse_plan", "generate_assets"]
