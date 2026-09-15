from __future__ import annotations

"""Verified reusable-component/asset registry with an empty-first bootstrap.

The remote repository is allowed to start with no reusable manifest at all.  A 404 or
an absent local manifest means zero candidates, never a production error.  Only
release-ready artifacts with objective verification receipts are eligible for
promotion.
"""

import base64
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DEFAULT_REPOSITORY = "jujumelona/mmm-data"
DEFAULT_BRANCH = "main"
REGISTRY_PATH = "reuse/v1/manifest.json"
SCHEMA_VERSION = "mmm/reuse-registry-v1"


@dataclass(frozen=True)
class VerifiedComponent:
    component_id: str
    capabilities: tuple[str, ...]
    minecraft_version: str
    loader: str
    source_origin: str
    source_commit: str
    license_id: str
    public_symbols: tuple[str, ...]
    required_dependencies: tuple[str, ...]
    test_receipts: tuple[str, ...]
    artifact: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VerifiedComponent | None:
        if value.get("verified") is not True:
            return None
        component_id = str(value.get("component_id") or "").strip()
        capabilities = _strings(value.get("capabilities"))
        target = value.get("target")
        if not component_id or not capabilities or not isinstance(target, Mapping):
            return None
        minecraft_version = str(target.get("minecraft_version") or "").strip()
        loader = str(target.get("loader") or "").strip().casefold()
        receipts = _strings(value.get("test_receipts"))
        if not minecraft_version or not loader or not receipts:
            return None
        artifact = value.get("artifact")
        if not isinstance(artifact, Mapping):
            return None
        return cls(
            component_id=component_id,
            capabilities=capabilities,
            minecraft_version=minecraft_version,
            loader=loader,
            source_origin=str(value.get("source_origin") or ""),
            source_commit=str(value.get("source_commit") or ""),
            license_id=str(value.get("license_id") or ""),
            public_symbols=_strings(value.get("public_symbols")),
            required_dependencies=_strings(value.get("required_dependencies")),
            test_receipts=receipts,
            artifact=dict(artifact),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "capabilities": list(self.capabilities),
            "target": {
                "minecraft_version": self.minecraft_version,
                "loader": self.loader,
            },
            "source_origin": self.source_origin,
            "source_commit": self.source_commit,
            "license_id": self.license_id,
            "public_symbols": list(self.public_symbols),
            "required_dependencies": list(self.required_dependencies),
            "test_receipts": list(self.test_receipts),
            "artifact": dict(self.artifact),
            "verified": True,
        }


def load_verified_components(
    *,
    repository: str = DEFAULT_REPOSITORY,
    branch: str = DEFAULT_BRANCH,
    local_root: str | Path | None = None,
) -> tuple[VerifiedComponent, ...]:
    """Load verified entries. Missing manifests deliberately return an empty tuple."""

    payload: Mapping[str, Any] | None = None
    if local_root is not None:
        local = Path(local_root).expanduser().resolve() / ".minecraft_ai" / REGISTRY_PATH
        if local.is_file() and not local.is_symlink():
            try:
                value = json.loads(local.read_text(encoding="utf-8"))
                if isinstance(value, Mapping):
                    payload = value
            except (OSError, json.JSONDecodeError):
                payload = None
    if payload is None:
        payload = _read_remote_manifest(repository, branch)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        return ()
    entries = payload.get("components")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return ()
    result: list[VerifiedComponent] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            parsed = VerifiedComponent.from_dict(entry)
            if parsed is not None:
                result.append(parsed)
    return tuple(result)


def find_verified_component(
    components: Iterable[VerifiedComponent],
    *,
    capability: str,
    minecraft_version: str,
    loader: str,
) -> VerifiedComponent | None:
    key = capability.casefold()
    candidates = [
        item
        for item in components
        if item.minecraft_version == minecraft_version
        and item.loader == loader.casefold()
        and any(value.casefold() == key for value in item.capabilities)
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (-len(item.test_receipts), item.component_id),
    )[0]


def promotion_records(
    *,
    proposal: Any,
    result: Any,
) -> tuple[dict[str, Any], ...]:
    """Build promotion records only for a fully release-ready execution."""

    if not bool(getattr(result, "release_ready", False)):
        return ()
    unresolved = tuple(getattr(result, "unresolved_gates", ()) or ())
    if unresolved:
        return ()
    if getattr(result, "build_report", None) is None or getattr(result, "jar_validation", None) is None:
        return ()
    game_design = getattr(proposal, "game_design", {})
    selection = game_design.get("_platform_selection") if isinstance(game_design, Mapping) else None
    target = selection.get("target") if isinstance(selection, Mapping) else None
    if not isinstance(target, Mapping):
        return ()
    receipt_values = ["build:PASS", "jar_validation:PASS"]
    if getattr(result, "runtime_receipt", None):
        receipt_values.append("runtime:PASS")
    if getattr(result, "playtest_receipt", None):
        receipt_values.append("playtest:PASS")
    receipts = tuple(receipt_values)
    records: list[dict[str, Any]] = []
    for module in tuple(getattr(proposal, "modules", ()) or ()):
        module_id = str(getattr(module, "module_id", "") or "")
        if not module_id:
            continue
        config = getattr(module, "config", {})
        owned = config.get("_owned_capabilities") if isinstance(config, Mapping) else None
        capability_ids = list(_strings(owned))
        if not capability_ids:
            # A module that owns no capability is support glue, not a reusable feature.
            # Promoting it under every plan capability would poison future reuse matches.
            continue
        owned_plan = config.get("_owned_reuse_plan") if isinstance(config, Mapping) else None
        owned_decisions = []
        if isinstance(owned_plan, Mapping):
            raw = owned_plan.get("capabilities")
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                owned_decisions = [dict(item) for item in raw if isinstance(item, Mapping)]
        records.append(
            {
                "component_id": module_id,
                "capabilities": sorted(set(capability_ids)),
                "target": {
                    "minecraft_version": str(target.get("minecraft_version") or ""),
                    "loader": str(target.get("loader") or ""),
                },
                "source_origin": "mmm_verified_generation",
                "source_commit": "",
                "license_id": "project-owned-or-donor-attribution-preserved",
                "public_symbols": [],
                "required_dependencies": [],
                "test_receipts": list(receipts),
                "artifact": {
                    "project_root": str(getattr(result, "project_root", "")),
                    "jar_path": str(getattr(result, "jar_path", "") or ""),
                    "complete_proposal_hash": str(getattr(result, "complete_proposal_hash", "")),
                    "reuse_decisions": owned_decisions,
                },
                "verified": True,
            }
        )
    asset_receipt = getattr(result, "asset_receipt", None)
    for asset in _verified_asset_rows(asset_receipt):
        asset_id = str(asset.get("asset_id") or "").strip()
        if not asset_id:
            continue
        records.append(
            {
                "component_id": f"asset:{asset_id}",
                "capabilities": [f"asset.{asset.get('kind', 'texture')}.{asset_id}"],
                "target": {
                    "minecraft_version": str(target.get("minecraft_version") or ""),
                    "loader": str(target.get("loader") or ""),
                },
                "source_origin": "mmm_verified_asset",
                "source_commit": "",
                "license_id": "generated-asset",
                "public_symbols": [],
                "required_dependencies": [],
                "test_receipts": ["texture_production:PASS", *receipts],
                "artifact": {
                    "target_path": str(asset.get("target_path") or asset.get("target") or ""),
                    "sha256": str(asset.get("sha256") or ""),
                    "width": asset.get("width"),
                    "height": asset.get("height"),
                    "selected_prompt": str(asset.get("selected_prompt") or ""),
                },
                "verified": True,
            }
        )
    return tuple(records)


def _verified_asset_rows(receipt: Any) -> tuple[Mapping[str, Any], ...]:
    """Flatten direct or sharded texture receipts; only explicit PASS rows qualify."""

    if not isinstance(receipt, Mapping):
        return ()
    rows: list[Mapping[str, Any]] = []
    if receipt.get("status") == "TEXTURE_PRODUCTION_PASS":
        raw = receipt.get("assets")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            rows.extend(item for item in raw if isinstance(item, Mapping))
    shards = receipt.get("shards")
    if isinstance(shards, Sequence) and not isinstance(shards, (str, bytes)):
        for shard in shards:
            rows.extend(_verified_asset_rows(shard))
    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(row.get("asset_id") or "").strip()
        if key:
            unique[key] = row
    return tuple(unique[key] for key in sorted(unique))


def persist_promotions(
    workspace_root: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    repository: str = DEFAULT_REPOSITORY,
    branch: str = DEFAULT_BRANCH,
) -> dict[str, Any]:
    """Persist locally, then sync remotely only when existing consent allows writes."""

    if not records:
        return {"status": "NO_VERIFIED_PROMOTIONS", "count": 0}
    root = Path(workspace_root).expanduser().resolve()
    local = root / ".minecraft_ai" / REGISTRY_PATH
    local.parent.mkdir(parents=True, exist_ok=True)
    existing = _manifest_from_path(local)
    merged = _merge_manifest(existing, records)
    text = json.dumps(merged, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    local.write_text(text, encoding="utf-8")

    remote_status = "NOT_CONFIGURED"
    try:
        from .remote_skill_store_consent import remote_write_allowed

        allowed = bool(remote_write_allowed())
    except Exception:
        allowed = False
    token = os.environ.get("MMM_TRAJECTORY_GITHUB_TOKEN", "").strip() or os.environ.get("GITHUB_TOKEN", "").strip()
    if allowed and token:
        try:
            _write_remote_manifest(repository, branch, text, token)
            remote_status = "SYNCED"
        except Exception as exc:
            remote_status = f"SYNC_FAILED:{type(exc).__name__}"
    return {
        "status": "PROMOTED",
        "count": len(records),
        "local_manifest": str(local),
        "remote_status": remote_status,
    }


def _read_remote_manifest(repository: str, branch: str) -> Mapping[str, Any] | None:
    url = f"https://api.github.com/repos/{repository}/contents/{REGISTRY_PATH}"
    try:
        response = httpx.get(
            url,
            params={"ref": branch},
            headers={"Accept": "application/vnd.github+json", "User-Agent": "mmm-component-registry"},
            timeout=8.0,
        )
    except httpx.HTTPError:
        return None
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        return None
    try:
        value = response.json()
        if not isinstance(value, Mapping) or value.get("encoding") != "base64":
            return None
        raw = base64.b64decode(str(value.get("content") or "").replace("\n", ""), validate=True)
        payload = json.loads(raw.decode("utf-8"))
        return payload if isinstance(payload, Mapping) else None
    except Exception:
        return None


def _write_remote_manifest(repository: str, branch: str, text: str, token: str) -> None:
    url = f"https://api.github.com/repos/{repository}/contents/{REGISTRY_PATH}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "mmm-component-registry",
    }
    current = httpx.get(url, params={"ref": branch}, headers=headers, timeout=10.0)
    sha = ""
    if current.status_code == 200:
        value = current.json()
        if isinstance(value, Mapping):
            sha = str(value.get("sha") or "")
    elif current.status_code != 404:
        current.raise_for_status()
    body = {
        "message": "Promote verified MMM reusable components",
        "content": base64.b64encode(text.encode("utf-8")).decode("ascii"),
        "branch": branch,
    }
    if sha:
        body["sha"] = sha
    response = httpx.put(url, headers=headers, json=body, timeout=15.0)
    response.raise_for_status()


def _manifest_from_path(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _merge_manifest(existing: Mapping[str, Any] | None, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    if isinstance(existing, Mapping) and existing.get("schema_version") == SCHEMA_VERSION:
        raw = existing.get("components")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            for item in raw:
                if isinstance(item, Mapping):
                    identity = _record_identity(item)
                    if identity:
                        by_id[identity] = dict(item)
    for item in records:
        identity = _record_identity(item)
        if identity:
            by_id[identity] = dict(item)
    return {
        "schema_version": SCHEMA_VERSION,
        "storage": "verified-components-and-assets",
        "components": [by_id[key] for key in sorted(by_id)],
    }


def _record_identity(value: Mapping[str, Any]) -> str:
    component_id = str(value.get("component_id") or "").strip()
    target = value.get("target")
    if not component_id or not isinstance(target, Mapping):
        return ""
    stable = {
        "component_id": component_id,
        "minecraft_version": str(target.get("minecraft_version") or ""),
        "loader": str(target.get("loader") or ""),
    }
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


__all__ = [
    "DEFAULT_BRANCH",
    "DEFAULT_REPOSITORY",
    "REGISTRY_PATH",
    "SCHEMA_VERSION",
    "VerifiedComponent",
    "find_verified_component",
    "load_verified_components",
    "persist_promotions",
    "promotion_records",
]
